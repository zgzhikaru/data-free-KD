python datafree_kd.py \
--method cmi \
--dataset cifar100 \
--batch_size 256 \
--teacher wrn40_2 \
--student wrn40_1 \
--lr 0.1 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 200 \
--lr_g 1e-3 \
--adv 0.5 \
--bn 1.0 \
--oh 1.0 \
--cr 0.8 \
--cr_T 0.2 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 0 \
--T 20 \
--save_dir run/cmi \
--log_tag cmi