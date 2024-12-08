python datafree_kd.py --method cmi --dataset cifar10 --batch_size 128 \
--synthesis_batch_size 256 --teacher wrn40_2 --student wrn16_1 --lr 0.1 \
--kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
--lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
--cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 0 \
--cmi_init run/cmi-preinverted-wrn402 --save_dir run/adv_cmi_cifar10 \
--log_tag adv_cmi_cifar10

python datafree_kd.py --method cmi --dataset cifar100 --batch_size 128 \
--synthesis_batch_size 256 --teacher wrn40_2 --student wrn16_1 --lr 0.1 \
--kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
--lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
--cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 0 \
--cmi_init run/cmi-preinverted-wrn402 --save_dir run/adv_cmi_cifar100 \
--log_tag adv_cmi_cifar100

# python datafree_kd.py --method cmi --dataset cifar10 --batch_size 128 \
# --synthesis_batch_size 256 --teacher resnet34 --student resnet18 --lr 0.1 \
# --kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
# --lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
# --cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 1 \
# --cmi_init run/cmi-preinverted-resnet34 --save_dir run/adv_cmi_cifar10 \
# --log_tag adv_cmi_cifar10

# python datafree_kd.py --method cmi --dataset cifar100 --batch_size 128 \
# --synthesis_batch_size 256 --teacher resnet34 --student resnet18 --lr 0.1 \
# --kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
# --lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
# --cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 1 \
# --cmi_init run/cmi-preinverted-resnet34 --save_dir run/adv_cmi_cifar100 \
# --log_tag adv_cmi_cifar100

# python datafree_kd.py --method cmi --dataset cifar10 --batch_size 128 \
# --synthesis_batch_size 256 --teacher vgg11 --student resnet18 --lr 0.1 \
# --kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
# --lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
# --cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 1 \
# --cmi_init run/cmi-preinverted-vgg11 --save_dir run/adv_cmi_cifar10 \
# --log_tag adv_cmi_cifar10

# python datafree_kd.py --method cmi --dataset cifar100 --batch_size 128 \
# --synthesis_batch_size 256 --teacher vgg11 --student resnet18 --lr 0.1 \
# --kd_steps 2000 --ep_steps 2000 --g_steps 200 --lr_g 1e-3 --epochs 40 \
# --lr_decay_milestones 25,30,35 --adv 0.5 --bn 1 --oh 0.5 --cr 0.8 \
# --cr_T 0.1 --T 20 --act 0 --balance 0 --T 20 --gpu 1 \
# --cmi_init run/cmi-preinverted-vgg11 --save_dir run/adv_cmi_cifar100 \
# --log_tag adv_cmi_cifar100
