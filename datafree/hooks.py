import torch
import torch.nn as nn
import torch.nn.functional as F

def register_hooks(modules):
    hooks = []
    for m in modules:
        hooks.append( FeatureHook(m) )
    return hooks

class InstanceMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)

class FeatureHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.output = output
        self.input = input[0]
    
    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanVarHook():
    def __init__(self, module, on_input=True, dim=[0,2,3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim

    def hook_fn(self, module, input, output):
        # To avoid inplace modification
        if self.on_input:
            feature = input[0].clone() 
        else:
            feature = output.clone()
        self.var, self.mean = torch.var_mean( feature, dim=self.dim, unbiased=True )

    def remove(self):
        self.hook.remove()
        self.output=None


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        self.input = input[0]
        nch = self.input.shape[1]
        mean = self.input.mean([0, 2, 3], keepdim=True).squeeze(0)     # (C, 1, 1)
        var = self.input.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)  # (C)

        self.mean, self.var = mean, var
        self.output = output

        #self.r_feat = torch.norm(module.running_mean.data - self.input, 2)  # (C, H, W)
        #self.normed_feat = self.r_feat/self.var
        # NOTE: Since output = normalize(input; running_mean, running_std)
        #self.normed_feat = torch.sum(output * output, dim=(-1,-2))
  
        self.r_feat_mean = F.mse_loss(mean.view(-1), module.running_mean.data)
        self.r_feat_var = F.mse_loss(var.view(-1), module.running_var.data)
        #self.r_feat_mean = torch.norm(module.running_mean.data - mean, 2)
        #self.r_feat_var = torch.norm(module.running_var.data - var, 2)
        self.r_feature = self.r_feat_mean + self.r_feat_var
        #r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #    module.running_mean.data - mean, 2)

    def remove(self):
        self.hook.remove()


class FeatureMatchingHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        #mean = input[0].mean([0, 2, 3])
        mean = input[0].mean([0])
        #var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature =  torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()