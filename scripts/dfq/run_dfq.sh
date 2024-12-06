python datafree_kd.py --method dfq --dataset cifar10 --batch_size 256 \
--teacher wrn40_2 --student wrn16_1 --lr 0.05 --epochs 200 --kd_steps 10 \
--ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 1 --T 1 --bn 1 --oh 1 --act 0 \
--balance 20 --gpu 1 --seed 0 

python datafree_kd.py --method dfq --dataset cifar100 --batch_size 256 \
--teacher wrn40_2 --student wrn16_1 --lr 0.05 --epochs 200 --kd_steps 10 \
--ep_steps 400 --g_steps 1 --lr_g 1e-3 --adv 1 --T 1 --bn 1 --oh 1 --act 0 \
--balance 20 --gpu 1 --seed 0 
