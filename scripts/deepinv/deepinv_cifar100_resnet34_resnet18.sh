python datafree_kd.py \
--method deepinv \
--dataset cifar100 \
--batch_size 256 \
--teacher resnet34 \
--student resnet18 \
--lr 0.1 \
--epochs 200 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 1000 \
--lr_g 0.1 \
--adv 1 \
--bn 10 \
--oh 1 \
--T 20 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 0 \
--save_dir run/deepinv \
--log_tag deepinv