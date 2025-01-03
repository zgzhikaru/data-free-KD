import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv, get_image_prior_losses
from datafree.utils import ImagePool, DataIter, clip_images

class GenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0, bn=0, oh=0, ent=0, act=0, balance=0, criterion=None,
                 discriminator=None, local=0, total_steps=None, 
                 normalizer=None, ulb_normalizer=None, device='cpu',
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False):
        super(GenerativeSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.ulb_normalizer = ulb_normalizer

        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ent = ent
        self.balance = balance
        self.act = act
        self.local = local

        #if discriminator is not None:
        self.discriminator = discriminator
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device).train()
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
            #self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d, T_max=total_steps)

        # generator
        self.generator = generator.to(device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device

        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

    def synthesize(self, data, args):
        self.student.eval()
        self.generator.train()
        self.teacher.eval()

        
        output_g = None
        #if data is not None:
        #data_batch_size = data.shape[0]
        z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
        
        #if self.discriminator is not None and data is not None:
        #if self.discriminator and data:
        if not (self.discriminator is None or data is None):
            data_batch_size = data.shape[0]
            synthesis_batch_size = self.synthesis_batch_size

            # Discriminator update
            self.discriminator.train()
            for it in range(self.iterations):
                output_g = self.generator(z)
                inputs_d = self.ulb_normalizer(output_g)
                
                score_f = self.discriminator(inputs_d.detach())
                score_r = self.discriminator(data.detach())
                #loss_d = F.binary_cross_entropy(score_r, torch.ones_like(score_r), reduction='sum')/data_batch_size  \
                #    + F.binary_cross_entropy(score_f, torch.zeros_like(score_f), reduction='sum')/synthesis_batch_size 
                #/(self.synthesis_batch_size + data_batch_size)
                
                loss_d = (F.binary_cross_entropy(score_r, torch.ones_like(score_r), reduction='sum')  \
                    + F.binary_cross_entropy(score_f, torch.zeros_like(score_f), reduction='sum')) \
                        /(data_batch_size + synthesis_batch_size)               

                args.tb.add_scalar('train/loss_d', loss_d.data.item(), args.n_iter)

                loss_d = self.local * loss_d

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()


        # Generator update
        for it in range(self.iterations): 
            #self.optimizer.zero_grad()
            if output_g is None:
                z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device)
                output_g = self.generator(z)
                inputs_d = self.ulb_normalizer(output_g)
            
            #if data is not None:
            if self.discriminator is not None:
                #inputs_d = self.ood_normalizer(output_g)
                score = self.discriminator(inputs_d)
                loss_g = F.binary_cross_entropy(score, torch.ones_like(score), reduction='sum')/synthesis_batch_size
                #loss_g = F.binary_cross_entropy_with_logits(score, torch.ones_like(score), reduction='sum') / len(score)
            else:
                loss_g = 0

            
            inputs = self.normalizer(output_g)
            t_out, t_feat = self.teacher(inputs, return_features=True)

            if args.style:
                all_mean_synth = [h.mean for h in self.hooks]   # (L, C)
                all_feat_synth = [h.feat_mean for h in self.hooks]    # (L, C, H, W)
                bn_feats = [h.feat for h in self.hooks]    # (L, N, C, H, W)  # TODO: Set option to obtain bn_feats from forward
                
                num_channels = [len(h.mean) for h in self.hooks]
                num_layers = len(num_channels)

                apply_weight = True
                all_gt_mean = [h.module.running_mean.data for h in self.hooks]     # (,C)
                all_gt_var = [h.module.running_var.data for h in self.hooks]       # (,C)
                bn_feats = self.teacher(data, return_bn_feat=True)

                if apply_weight:
                    all_mean_ood = [h.mean for h in self.hooks]   # (,C)    # TODO: Obtain from pre-computed over whole dataset
                    all_var_ood = [h.var for h in self.hooks]     # (,C)
                    #feat_ood = [h.feat_mean for h in self.hooks]  # (,C,H,W)
                
                    all_weights = []
                    for l in range(num_layers):
                        feat_ood = bn_feats[l]#.mean(dim=(-1,-2))   # (N, C, H, W) -> (N, C)
                        mean_synth, feat_synth = all_mean_synth[l], all_feat_synth[l]
                        mean_ood, var_ood = all_mean_ood[l].unsqueeze(0), all_var_ood[l].unsqueeze(0)     # (1, C, H, W)
                        gt_mean, gt_var = all_gt_mean[l].unsqueeze(0), all_gt_var[l].unsqueeze(0)     # (1, C)

                        #dist2gt = torch.norm(feat_ood - gt_mean, 2)/gt_var   # (N, C, H)    # TODO Option: Move mean(H,W) after norm; Better interpretation 
                        #dist2ood = torch.norm(feat_ood - mean_ood, 2)/var_ood
                        dist2gt = torch.norm(feat_ood - gt_mean, 2).mean(dim=(-1,-2))/gt_var        # (N, C, H, W) -> (N, C)
                        dist2ood = torch.norm(feat_ood - mean_ood, 2).mean(dim=(-1,-2))/var_ood     # (N, C, H, W) -> (N, C)

                        weight = (dist2gt - dist2ood).exp()     # (N, C)
                        all_weights.append(weight)
                else:
                    all_weights = [torch.ones_like(m) for m in all_mean_synth]  # TODO: Squeeze (N)-dim
        
                loss_feat = 0
                for l in range(num_layers):
                    mean_synth, feat_synth = all_mean_synth[l], all_feat_synth[l]
                    # 1. Compute synthetic sample statistics
                    synth_mean = feat_synth.mean(dim=(-1,-2)).squeeze(0)     # (1, C)
                    synth_std = feat_synth.std(dim=(-1,-2)).squeeze(0)     # (1, C)
                    synth_prod_mean = (feat_synth * feat_synth).mean(dim=(-1,-2)).squeeze(0)

                    # 2. Compute distance to proxy samples
                    #feat_ood = bn_feats[l].mean(dim=(-1,-2))   # (N, C, H, W) -> (N, C)
                    #dist_mean = torch.norm(feat_ood - synth_mean, pow=2)   # (N, C)
                    feat_ood = bn_feats[l]   # (N, C, H, W) 
                    feat_prod_ood = feat_ood * feat_ood   # (N, C, H, W) 
                    dist_mean = torch.norm(feat_ood - synth_mean, pow=2).mean(dim=(-1,-2))   # (N, C, H, W) -> (N, C)
                    dist_var = torch.norm(feat_prod_ood - synth_prod_mean, pow=2).mean(dim=(-1,-2))    # (N, C)

                    # 3. Weight and sum the distance sample-wise
                    weight = all_weights[l].detach()    # (N, C)
                    loss_mean = (weight * dist_mean).sum(dim=0).mean()  #.sum() # (C) # TODO: Design choice for sum or mean over channel
                    loss_var = (weight * dist_var).sum(dim=0).mean()          # (C)

                    loss_feat += loss_mean
                    #loss_feat += loss_var  # Ablation choice
            

            pyx = F.softmax(t_out, dim=1) # p(y|G(z)
            log_softmax_pyx = F.log_softmax(t_out, dim=1)
            loss_ent = -(pyx * log_softmax_pyx).sum(1).mean()

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1] )
            loss_act = - t_feat.abs().mean()
            if self.adv>0:
                s_out = self.student(inputs)
                loss_adv = -self.criterion(s_out, t_out)
            else:
                loss_adv = loss_oh.new_zeros(1)
            #p = F.softmax(t_out, dim=1).mean(0)
            p = pyx.mean(0)
            loss_balance = (p * torch.log(p)).sum() # maximization

            loss_tc = self.bn * loss_bn + self.oh * loss_oh + self.ent * loss_ent \
                + self.adv * loss_adv + self.balance * loss_balance + self.act * loss_act
            
            loss = self.local * loss_g + loss_tc

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.discriminator:
            lr_d = self.optimizer_d.param_groups[0]['lr']
            args.tb.add_scalar('train/loss_g', loss_g.data.item(), args.n_iter)
            args.tb.add_scalar('train/lr_d', lr_d, args.n_iter)
        
        args.tb.add_scalar('train/loss_oh', loss_oh.data.item(), args.n_iter)
        args.tb.add_scalar('train/loss_ent', loss_ent.data.item(), args.n_iter)
        args.tb.add_scalar('train/loss_balance', loss_balance.data.item(), args.n_iter)
    
        args.tb.add_scalar('train/loss_adv', loss_adv.data.item(), args.n_iter)
        args.tb.add_scalar('train/loss_bn', loss_bn.data.item(), args.n_iter)

        args.tb.add_scalar('train/loss_tc', loss_tc.data.item(), args.n_iter)

        lr_g = self.optimizer.param_groups[0]['lr']
        
        args.tb.add_scalar('train/lr_g', lr_g, args.n_iter)
        

        #self.scheduler.step()
        #self.scheduler_d.step()
            
        return { 'synthetic': self.normalizer(inputs.detach(), reverse=True) }
   

    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs