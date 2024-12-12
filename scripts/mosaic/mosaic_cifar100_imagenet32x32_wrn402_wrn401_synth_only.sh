python datafree_kd.py \
--method mosaic \
--dataset cifar100 \
--transfer_set imagenet_32x32 \
--ood_subset \
--batch_size 256 \
--teacher wrn40_2 \
--student wrn40_1 \
--lr 0.1 \
--epochs 200 \
--kd_steps 5 \
--g_steps 1 \
--lr_g 1e-3 \
--z_dim 256 \
--local 1 \
--adv 1 \
--T 1 \
--bn 0 \
--oh 0 \
--ent 1 \
--act 0 \
--balance 10 \
--gpu 1 \
--seed 0 \
--log_tag 'synth_only'