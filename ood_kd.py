import argparse
from math import gamma
import os
import random
import shutil
import time
import warnings

from tqdm import tqdm

import registry
import datafree
from datafree.criterions import kldiv
# from datafree.utils import AverageMeter

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True, choices=['dfnd', 'mosaic', 'ae_mosaic'])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--recon', default=0, type=float, help='scaling factor for encoder loss')

parser.add_argument('--entropy', default=0, type=float, help='scaling factor for entropy loss')
parser.add_argument('--local', default=0, type=float, help='scaling factor for discriminator loss')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

# parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
# parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
# parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--transfer_set', default='cifar10')
parser.add_argument('--ood_subset', action='store_true',
                    help='use ood subset')
parser.add_argument('--shared_normalizer', default=True,
                    help='Whether to share the same normalization between transfer set and orig train set')
parser.add_argument('--include_raw', action='store_true',
                    help='include unlabeled transfer set as raw data into distillation training')

parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="30,60,120,160", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--z_dim', default=256, type=int, help='latent dimension of generator')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate_teacher', action='store_true',
                    help='evaluate teacher model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
best_acc1 = 0

## Data loading utils
def unwrap(data):
    try:
        data, _ = data
        return data
    except:
        return data

def get_data(iterator, args):
    return unwrap(next(iterator)).cuda(args.gpu) if args.transfer_set else None


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler_s = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) \
        if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/ood_kd-%s/log-%s-%s-%s%s.txt'\
                                                   %(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    args.tb = SummaryWriter('tb_log/ood_kd-%s/tensorboard-%s-%s-%s%s'\
                            %(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset, train_transform, val_transform = registry.get_dataset(name=args.dataset, data_root=args.data_root, return_transform=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.ulb_normalizer = args.normalizer 
    if args.transfer_set is not None and not args.shared_normalizer:
        args.ulb_normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.transfer_set])
    teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)

    cudnn.benchmark = True
    ############################################
    # Setup ulb_dataset
    ############################################
    if args.transfer_set is not None:
        _, raw_dataset, _ = registry.get_dataset(name=args.transfer_set, data_root=args.data_root,train_transform=train_transform)
        _, train_dataset, _ = registry.get_dataset(name=args.transfer_set, data_root=args.data_root,train_transform=val_transform)
    
    if args.ood_subset and args.transfer_set in ['imagenet_32x32', 'places365_32x32']:
        ood_index = prepare_ood_data(train_dataset, teacher, ood_size=len(ori_dataset), args=args)
        train_dataset.samples = [ train_dataset.samples[i] for i in ood_index]
        raw_dataset.samples = [ raw_dataset.samples[i] for i in ood_index]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    raw_sampler = torch.utils.data.distributed.DistributedSampler(raw_dataset) if args.distributed else None
    raw_loader = torch.utils.data.DataLoader(
        raw_dataset, batch_size=args.batch_size, shuffle=(raw_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=raw_sampler)
    

    
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    
    if args.method in ['dfnd', 'mosaic', 'ae_mosaic']:
        # nz = 512 if args.method=='dafl' else 256
        nz = args.z_dim
        generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)

        discriminator = None
        encoder = None
        if args.method == 'degan':
            discriminator = datafree.models.generator.Discriminator(nc=3, img_size=32)
            discriminator = prepare_model(discriminator)
        elif args.method == 'mosaic':
            discriminator = datafree.models.generator.PatchDiscriminator(nc=3, ndf=128)
            discriminator = prepare_model(discriminator)
        elif args.method == 'ae_mosaic':
            discriminator = datafree.models.generator.PatchDiscriminator(nc=3, ndf=128)
            discriminator = prepare_model(discriminator)
            encoder = datafree.models.generator.Encoder(nc=3, nz=nz)
            encoder = prepare_model(encoder)

        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=student, generator=generator, nz=nz, discriminator=discriminator,
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g, recon=args.recon, encoder=encoder,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, entropy=args.entropy,
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, ulb_normalizer=args.ulb_normalizer, device=args.gpu)
    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    # milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        teacher.eval()
        eval_results = evaluator(teacher, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        print('[Eval] Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f}'.format(acc1=acc1, acc5=acc5, loss=val_loss))
        return
    
    if args.evaluate_teacher:
        eval_results = evaluator(teacher, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        print('[Eval] Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f}'.format(acc1=acc1, acc5=acc5, loss=val_loss))
        args.logger.info('[Eval_Teacher] Acc={acc:.4f}'.format(acc=eval_results['Acc']))


    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch

        if args.transfer_set is not None:
            args.ep_steps = len(train_loader)
            train_iter = iter(train_loader)
            #if args.include_raw:
            raw_iter = iter(raw_loader)
        # for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
        for it in tqdm(range( args.ep_steps )): # total kd_steps < ep_steps
            args.n_iter = epoch * args.ep_steps + it

            data = get_data(train_iter, args)
            
            # 1. Data synthesis
            vis_results = synthesizer.synthesize(data, args) # g_steps
            # 2. Knowledge distillation
            data = get_data(raw_iter, args) if args.include_raw else None
            
            train( synthesizer, [student, teacher], criterion, optimizer, data, args) # # kd_steps
        
        for vis_name, vis_image in vis_results.items():
            datafree.utils.save_image_batch( vis_image, 'checkpoints/ood_kd-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )
        
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        args.tb.add_scalar('eval/Acc@1', acc1, epoch)
        args.tb.add_scalar('eval/Acc@5', acc5, epoch)
        args.tb.add_scalar('eval/Loss', val_loss, epoch)

        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/ood_kd-%s/%s-%s-%s%s.pth'%(args.method, args.dataset, args.teacher, args.student, args.log_tag)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)


def train(synthesizer, model, criterion, optimizer, data, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        images = synthesizer.sample()
        if data is not None:
            images = torch.cat([images, data], dim=0)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            if data is not None:
                loss_s = kldiv(s_out, t_out.detach(), reduction='none')
                
                loss_syn = loss_s[:args.synthesis_batch_size]
                loss_data = loss_s[args.synthesis_batch_size:]

                loss_syn = loss_syn.sum()/len(loss_syn)
                loss_data = loss_data.sum()/len(loss_data)
                
                loss_s = loss_s.sum()/len(loss_s)
            else:
                loss_s = criterion(s_out, t_out.detach())
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=len(args.kd_steps), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()

    n_iter = args.n_iter #* args.kd_steps + i
    lr_s = optimizer.param_groups[0]['lr']
    args.tb.add_scalar('train/lr_s', lr_s, n_iter)
    args.tb.add_scalar('train/loss_s', loss_s.data.item(), n_iter)
    
    if data is not None:
        args.tb.add_scalar('train/loss_syn', loss_syn.data.item(), n_iter)
        args.tb.add_scalar('train/loss_data', loss_data.data.item(), n_iter)

def prepare_ood_data(train_dataset, model, ood_size, args):
    if os.path.exists('checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.transfer_set, args.teacher)):
        print("has ood index")
        ood_index = torch.load('checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.transfer_set, args.teacher))
        return ood_index
    model.eval()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    print("no ood index")
    with torch.no_grad():
        entropy_list = []
        model.cuda(args.gpu)
        model.eval()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            p = torch.nn.functional.softmax(output, dim=1)
            ent = -(p*torch.log(p)).sum(dim=1)
            entropy_list.append(ent)
        entropy_list = torch.cat(entropy_list, dim=0)
        ood_index = torch.argsort(entropy_list, descending=True)[:ood_size].cpu().tolist()
        model.cpu()
        os.makedirs('checkpoints/ood_index', exist_ok=True)
        torch.save(ood_index, 'checkpoints/ood_index/%s-%s-%s-ood-index.pth'%(args.dataset, args.unlabeled, args.teacher))
    return ood_index
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()
