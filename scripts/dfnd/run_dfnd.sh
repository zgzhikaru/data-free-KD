# python kd_dfnd.py   --batch_size 256 \
#  --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200  \
#  --dataset cifar100 --unlabeled cifar10  --gpu 0 

# python kd_dfnd.py  --batch_size 256 \
#  --teacher resnet34 --student resnet18 --lr 0.1 --epochs 200  \
#  --dataset cifar10 --unlabeled cifar100  --gpu 0 

# python kd_dfnd.py  --batch_size 256 \
#  --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200  \
#  --dataset cifar10 --unlabeled cifar100  --gpu 0 

# python kd_dfnd.py   --batch_size 256 \
#  --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200  \
#  --dataset cifar100 --unlabeled cifar10  --gpu 0 

# python kd_dfnd.py  --batch_size 256 \
#  --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200  \
#  --dataset cifar10 --unlabeled cifar100  --gpu 0 


# python kd_dfnd.py  --batch_size 256 \
#  --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200  \
#  --dataset cifar100 --unlabeled imagenet_32x32 --ood_subset --gpu 0 

# python kd_dfnd.py  --batch_size 256 \
#  --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200  \
#  --dataset cifar100 --unlabeled places365_32x32 --ood_subset --gpu 0 


python kd_dfnd.py  --batch_size 256 \
 --teacher resnet34 --student resnet18 --lr 0.1 --epochs 200  \
 --dataset cifar100 --unlabeled imagenet_32x32 --ood_subset --gpu 0 

python kd_dfnd.py  --batch_size 256 \
 --teacher resnet34 --student resnet18 --lr 0.1 --epochs 200  \
 --dataset cifar100 --unlabeled places365_32x32 --ood_subset --gpu 0 

python kd_dfnd.py   --batch_size 256 \
 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200  \
 --dataset cifar100 --unlabeled imagenet_32x32 --ood_subset  --gpu 0

python kd_dfnd.py   --batch_size 256 \
 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200  \
 --dataset cifar100 --unlabeled places365_32x32 --ood_subset  --gpu 0  