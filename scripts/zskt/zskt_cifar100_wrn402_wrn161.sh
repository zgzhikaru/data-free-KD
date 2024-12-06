python datafree_kd.py \
--method zskt \
--dataset cifar100 \
--batch_size 256 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--epochs 200 \
--kd_steps 5 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 1e-3 \
--adv 1 \
--T 20 \
--bn 0 \
--oh 0 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 0 \