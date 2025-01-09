import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv, get_image_prior_losses
from datafree.utils import ImagePool, DataIter, clip_images

class GenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128, recon = 0,
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None, encoder = None,
                 entropy=0, discriminator=None, local=0, total_steps=None, 
                 normalizer=None, ulb_normalizer=None, device='cpu',
                 feat_loss_w=0, apply_weight=False, eps=1e-3,
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False):
        super(GenerativeSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.ulb_normalizer = ulb_normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act
        self.entropy = entropy
        self.local = local
        self.recon = recon
        self.apply_weight = apply_weight
        self.feat_loss_w = feat_loss_w

        self.eps = eps


        #if discriminator is not None:
        self.discriminator = discriminator
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device).train()
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        
        self.encoder = encoder 
        if self.encoder is not None:
            self.encoder = self.encoder.to(device).train()
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.lr_g, betas=(0.5,0.999))
            
        # generator
        self.generator = generator.to(device).train()
        
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device
        

        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        # self.all_mean_ood_ema = torch.zeros(len(self.hooks))
        # self.all_var_ood_ema = torch.zeros(len(self.hooks))\
        # 在初始化时直接移动到与模型相同的设备
        self.all_mean_ood_ema = [
            None
            for h in self.hooks
        ]
        self.all_var_ood_ema = [
            None
            for h in self.hooks
        ]

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def synthesize(self, data = None, args = None):
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        output_g = None
        #if data is not None:
        #data_batch_size = data.shape[0]
        z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
        data_batch_size = args.batch_size
        synthesis_batch_size = self.synthesis_batch_size
        
        # 判别器的更新
        if self.discriminator is not None:
            # Discriminator update
            self.discriminator.train()
            for it in range(self.iterations):
                output_g = self.generator(z)
                inputs_d = self.ulb_normalizer(output_g)
                
                score_f = self.discriminator(inputs_d.detach())
                score_r = self.discriminator(data.detach())
                #loss_d = F.binary_cross_entropy(score_r, torch.ones_like(score_r), reduction='sum')/data_batch_size  \
                #    + F.binary_cross_entropy(score_f, torch.zeros_like(score_f), reduction='sum')/synthesis_batch_size 
                #/(self.synthesis_batch_size + data_batch_size)
                
                loss_d = (F.binary_cross_entropy(score_r, torch.ones_like(score_r), reduction='sum')  \
                    + F.binary_cross_entropy(score_f, torch.zeros_like(score_f), reduction='sum')) \
                        /(data_batch_size + synthesis_batch_size)               

                args.tb.add_scalar('train/loss_d', loss_d.data.item(), args.n_iter)

                loss_d = self.local * loss_d

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

        # Encoder 更新
        if self.encoder is not None:
            self.encoder.train()
            for it in range(self.iterations):
                mu, log_var = self.encoder(data)
                z_enc, _ = self._sample_gauss(mu, log_var)
                
                score_r = self.discriminator(data.detach(), return_features=True)
                score_enc = self.discriminator(self.ulb_normalizer(self.generator(z_enc)).detach(), return_features=True)

                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
                KLD = KLD.mean(dim=0)
                loss_recon = 0.5 * F.mse_loss(score_r.reshape(synthesis_batch_size, -1), \
                                              score_enc.reshape(synthesis_batch_size, -1), \
                                                reduction="none").sum(dim=-1)
                loss_recon = loss_recon.mean(dim=0)
                # print("KLD: ", KLD)
                # print("loss_recon: ", loss_recon)
                loss_enc =  loss_recon

                self.optimizer_e.zero_grad()
                loss_enc.backward()
                self.optimizer_e.step()

                args.tb.add_scalar('train/loss_KLD', KLD.data.item(), args.n_iter)
                args.tb.add_scalar('train/loss_recon', loss_recon.data.item(), args.n_iter)
                args.tb.add_scalar('train/loss_enc', loss_enc.data.item(), args.n_iter)
                
        # 生成器更新
        for it in range(self.iterations):
            if output_g is None:
                z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
                output_g = self.generator(z)
                output_g = self.ulb_normalizer(output_g)
            if self.discriminator is not None:
                #inputs_d = self.ood_normalizer(output_g)
                score = self.discriminator(inputs_d)
                loss_g = F.binary_cross_entropy(score, torch.ones_like(score), reduction='sum')/synthesis_batch_size
                #loss_g = F.binary_cross_entropy_with_logits(score, torch.ones_like(score), reduction='sum') / len(score)
            else:
                loss_g = 0
                

            inputs = self.normalizer(output_g)
            t_out, t_feat = self.teacher(inputs, return_features=True)

            if self.feat_loss_w > 0:
                all_mean_synth = [h.mean for h in self.hooks]   # (L, C)
                all_feat_synth = [h.feat_mean for h in self.hooks]    # (L, C, H, W)
                
                num_channels = [len(h.mean) for h in self.hooks] # (L)
                num_layers = len(num_channels)
                # print(f"num_layers: {num_layers}")

                # apply_weight = True
                if self.apply_weight:
                    all_gt_mean = [h.module.running_mean.data for h in self.hooks]     # (L,C)
                    all_gt_var = [h.module.running_var.data for h in self.hooks]       # (L,C)

                    self.teacher(data)
                    all_mean_ood = [h.mean for h in self.hooks]   # (L,C)
                    all_var_ood = [h.var for h in self.hooks]     # (L,C)

                    all_feat_ood = [h.feat_mean for h in self.hooks]   # (L,C,H,W)
                    
                    # 计算weights
                    all_weights = []
                    for l in range(num_layers):
                        mean_synth, feat_synth = all_mean_synth[l], all_feat_synth[l] # (C), (C,H,W)

                        mean_ood, var_ood = all_mean_ood[l], all_var_ood[l] # (C), (C)
                        gt_mean, gt_var = all_gt_mean[l], all_gt_var[l] # (C), (C)

                        feat_ood = all_feat_ood[l] # (C,H,W)

                        # new_mean = torch.add(self.all_mean_ood_ema[l] * 0.9 ,  mean_ood * 0.1)
                        # new_var = torch.add(self.all_var_ood_ema[l] * 0.9 ,  var_ood * 0.1)
                        with torch.no_grad():
                            new_mean = self.all_mean_ood_ema[l] * 0.9 + mean_ood * 0.1 if self.all_mean_ood_ema[l] is not None else gt_mean
                            
                            new_var = self.all_var_ood_ema[l] * 0.9 + var_ood * 0.1 if self.all_var_ood_ema[l] is not None else gt_var

                            self.all_mean_ood_ema[l] = new_mean
                            self.all_var_ood_ema[l] = new_var

                        mean_ood, var_ood = self.all_mean_ood_ema[l], self.all_var_ood_ema[l] # (C), (C)
                        
                        # print(feat_ood.shape, gt_mean.shape, gt_var.shape)

                        # dist_syn2gt = torch.norm(mean_synth - gt_mean, 2)/(gt_var + self.eps) # (C)
                        # dist_syn2ood = torch.norm(mean_synth - mean_ood, 2)/(var_ood + self.eps) # (C)
                        # temp = torch.norm(feat_ood - gt_mean.view(-1,1,1), 2)
                        # print(temp.shape)
                        # dist2gt = torch.norm(feat_ood - gt_mean.view(-1,1,1), 2, dim=(-1,-2)).mean(dim=(-1,-2)) / gt_var    # (C, H, W) -> (C)
                        # dist2ood = torch.norm(feat_ood - mean_ood.view(-1,1,1), 2, dim=(-1,-2)).mean(dim=(-1,-2))/var_ood   # (C, H, W) -> (C)
                        dist2gt = torch.sqrt((feat_ood - gt_mean.view(-1,1,1))**2).mean(dim=(-1,-2)) / (gt_var + self.eps)   # (C, H, W) -> (C)
                        dist2ood = torch.sqrt((feat_ood - mean_ood.view(-1,1,1))**2).mean(dim=(-1,-2))/(var_ood + self.eps)   # (C, H, W) -> (C)

                        # weight = (dist_syn2ood - dist_syn2gt).exp()     # (,C)
                        weight = (dist2gt - dist2ood).exp()
                        all_weights.append(weight)
                else:
                    all_weights = [torch.ones_like(m) for m in all_mean_synth]
                # 计算loss_feat
                loss_feat = 0
                for l in range(num_layers):
                    mean_synth, feat_synth = all_mean_synth[l], all_feat_synth[l]  # (C), (C,H,W)

                    # 1. Compute synthetic sample statistics
                    synth_mean = feat_synth.mean(dim=(-1,-2)).squeeze(0)     # (1, C)
                    # synth_std = feat_synth.std(dim=(-1,-2)).squeeze(0)     # (1, C)
                    synth_prod_mean = (feat_synth * feat_synth).mean(dim=(-1,-2)).squeeze(0) # (C)

                    # 2. Compute distance to proxy samples
                    #feat_ood = bn_feats[l].mean(dim=(-1,-2))   # (N, C, H, W) -> (N, C)
                    #dist_mean = torch.norm(feat_ood - synth_mean, pow=2)   # (N, C)
                    # feat_ood = bn_feats[l]   # (N, C, H, W) 

                    feat_ood = all_feat_ood[l] # (C,H,W)
                    feat_prod_ood = feat_ood * feat_ood   # (C, H, W) 
                    # dist_mean = torch.norm(feat_ood - synth_mean.view(-1,1,1), pow=2, dim=(-1,-2)).mean(dim=(-1,-2))   # (C, H, W) -> (C)
                    # dist_var = torch.norm(feat_prod_ood - synth_prod_mean.view(-1,1,1), pow=2, dim=(-1,-2)).mean(dim=(-1,-2))  # (C)
                    dist_mean = torch.sqrt((feat_ood - synth_mean.view(-1,1,1))**2).mean(dim=(-1,-2))   # (C, H, W) -> (C)
                    dist_var = torch.sqrt((feat_prod_ood - synth_prod_mean.view(-1,1,1))**2).mean(dim=(-1,-2))  # (C)
                    # 3. Weight and sum the distance sample-wise
                    weight = all_weights[l].detach()    # (N, C)
                    loss_mean = (weight * dist_mean).sum(dim=0).mean()  #.sum() # (C) # TODO: Design choice for sum or mean over channel
                    loss_var = (weight * dist_var).sum(dim=0).mean()          # (C)
                    loss_feat += loss_mean
                    #loss_feat += loss_var  # Ablation choice
            else:
                loss_feat = 0

            pyx = F.softmax(t_out, dim=1) # p(y|G(z)
            log_softmax_pyx = F.log_softmax(t_out, dim=1)
            loss_ent = -(pyx * log_softmax_pyx).sum(1).mean()

            loss_bn = sum([h.r_feature for h in self.hooks]) if self.bn > 0 else 0
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1] ) if self.oh > 0 else 0
            loss_act = - t_feat.abs().mean() if self.act > 0 else 0
            if self.adv>0:
                s_out = self.student(inputs)
                loss_adv = -self.criterion(s_out, t_out)
            else:
                loss_adv = loss_oh.new_zeros(1)
            p = pyx.mean(0)
            loss_balance = (p * torch.log(p)).sum() # maximization
            loss_tc = self.bn * loss_bn + self.oh * loss_oh + self.entropy * loss_ent \
                + self.adv * loss_adv + self.balance * loss_balance + self.act * loss_act 
                
            
            loss = self.local * loss_g + loss_tc + self.feat_loss_w * loss_feat
            
            if self.encoder is not None:
                score_r = self.discriminator(data.detach(), return_features=True)
                score_enc = self.discriminator(self.ulb_normalizer(self.generator(z_enc)).detach(), return_features=True)

                # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
                # KLD = KLD.mean(dim=0)
                loss_recon = 0.5 * F.mse_loss(score_r.reshape(synthesis_batch_size, -1), \
                                              score_enc.reshape(synthesis_batch_size, -1), \
                                                reduction="none").sum(dim=-1)
                loss_recon = loss_recon.mean(dim=0)
                loss += self.recon * loss_recon 


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                

        if self.local > 0:
            args.tb.add_scalar('train/loss_g', loss_g.data.item(), args.n_iter)
        if self.oh > 0:
            args.tb.add_scalar('train/loss_oh', loss_oh.data.item(), args.n_iter)
        if self.entropy > 0:
            args.tb.add_scalar('train/loss_ent', loss_ent.data.item(), args.n_iter)
        if self.balance > 0:
            args.tb.add_scalar('train/loss_balance', loss_balance.data.item(), args.n_iter)
        if self.adv > 0:
            args.tb.add_scalar('train/loss_adv', loss_adv.data.item(), args.n_iter)
        if self.bn > 0:
            args.tb.add_scalar('train/loss_bn', loss_bn.data.item(), args.n_iter)
        if self.feat_loss_w > 0:
            args.tb.add_scalar('train/loss_feat', loss_feat.data.item(), args.n_iter)
        args.tb.add_scalar('train/loss_tc', loss_tc.data.item(), args.n_iter)
        if self.encoder is not None:
            
            lr_e = self.optimizer_e.param_groups[0]['lr']
            args.tb.add_scalar('train/lr_e', lr_e, args.n_iter)

        lr_g = self.optimizer.param_groups[0]['lr']
        lr_d = self.optimizer_d.param_groups[0]['lr']
        

        args.tb.add_scalar('train/lr_g', lr_g, args.n_iter)
        args.tb.add_scalar('train/lr_d', lr_d, args.n_iter)
        

        return { 'synthetic': self.normalizer(inputs.detach(), reverse=True) }
    
    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs