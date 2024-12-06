import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

import registry
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def validate(val_loader, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(1, non_blocking=True)
            target = target.cuda(1, non_blocking=True)
            
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        print(' [Eval]Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f}'
                .format(top1=top1, top5=top5))
    return top1.avg

checkpoints = torch.load('checkpoints/pretrained/cifar100_wrn40_2.pth')

for k, v in checkpoints.items():
    print(k)

print(checkpoints['best_acc1'])

val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( mean=(0.5071, 0.4867, 0.4408),    std=(0.2675, 0.2565, 0.2761) ),
        ])


valset = torchvision.datasets.CIFAR100(root='data/torchdata', \
                                train=False, download=True, \
                                transform=val_transform)

val_loader = torch.utils.data.DataLoader(valset, batch_size=128, \
                    shuffle=False, num_workers=4, pin_memory=True)

### test acc

model = registry.get_model('wrn40_2', num_classes=100, pretrained=True)
model = model.cuda(1)
criterion = nn.CrossEntropyLoss().cuda(1)
model.load_state_dict(checkpoints['state_dict'])
    
acc1 = validate(val_loader, model)
print(acc1)