python ood_kd.py --method mosaic --dataset cifar100 --transfer_set cifar10 \
 --batch_size 256  --include_raw --shared_normalizer True\
 --teacher wrn40_2 --student wrn16_1 --lr 0.1 -j 8 --seed 0\
 --kd_steps 5 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --epochs 200 \
 --adv 1  --entropy 1 --local 1\
 --act 0 --balance 10 --T 1 --gpu 0 --fp16 



 
 python ood_kd.py --method ae_mosaic --dataset cifar100 --transfer_set cifar10 \
 --batch_size 1024  --include_raw --shared_normalizer True\
 --teacher wrn40_2 --student wrn16_1 --lr 0.1 --recon 1 \
 --kd_steps 5 --ep_steps 400 --g_steps 1 --lr_g 1e-3 --epochs 200 \
  --adv 1  --entropy 1 --local 1\
  --act 0 --balance 10 --T 1 --gpu 0 --fp16 --seed 0