# python ood_kd.py --method mosaic --dataset cifar100 --transfer_set cifar10 \
#  --batch_size 256  --include_raw --shared_normalizer True\
#  --teacher wrn40_2 --student wrn16_1 --lr 0.1 \
#  --kd_steps 5 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --epochs 200 \
#  --lr_decay_milestones 25,30,35 --adv 1  --entropy 1 --local 1\
#  --T 20 --act 0 --balance 10 --T 1 --gpu 0 \

python ood_kd.py --method mosaic --dataset cifar100 --transfer_set cifar10 \
 --batch_size 256  --include_raw --shared_normalizer True\
 --teacher resnet34 --student resnet18 --lr 0.1 \
 --kd_steps 5 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --epochs 200 \
 --lr_decay_milestones 25,30,35 --adv 1  --entropy 1 --local 1\
 --T 20 --act 0 --balance 10 --T 1 --gpu 1 \
 --resume /home/CMI/CMI/checkpoints/ood_kd-mosaic/cifar100-resnet34-resnet18.pth
