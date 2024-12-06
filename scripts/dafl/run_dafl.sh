python datafree_kd.py --method dafl --dataset cifar100 --batch_size 256 \
 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --epochs 200 --kd_steps 5 \
 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 0 --T 20 --bn 0 --oh 1 \
 --act 0.001 --balance 20 --gpu 0 --seed 0 

python datafree_kd.py --method dafl --dataset cifar10 --batch_size 256 \
 --teacher resnet34 --student resnet18 --lr 0.1 --epochs 200 --kd_steps 5 \
 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 0 --T 20 --bn 0 --oh 1 \
 --act 0.001 --balance 20 --gpu 0 --seed 0 

python datafree_kd.py --method dafl --dataset cifar100 --batch_size 256 \
 --teacher resnet34 --student resnet18 --lr 0.1 --epochs 200 --kd_steps 5 \
 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 0 --T 20 --bn 0 --oh 1 \
 --act 0.001 --balance 20 --gpu 0 --seed 0 

python datafree_kd.py --method dafl --dataset cifar10 --batch_size 256 \
 --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200 --kd_steps 5 \
 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 0 --T 20 --bn 0 --oh 1 \
 --act 0.001 --balance 20 --gpu 0 --seed 0 

python datafree_kd.py --method dafl --dataset cifar100 --batch_size 256 \
 --teacher vgg11 --student resnet18 --lr 0.1 --epochs 200 --kd_steps 5 \
 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 0 --T 20 --bn 0 --oh 1 \
 --act 0.001 --balance 20 --gpu 0 --seed 0 
