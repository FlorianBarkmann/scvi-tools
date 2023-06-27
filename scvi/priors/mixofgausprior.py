import torch
from torch.distributions import Normal
import torch.distributions as dist
from scvi.priors.base_prior import BasePrior
import torch.nn.functional as F


class MixOfGausPrior(BasePrior):
    def __init__(self, n_latent: int, k: int):
        super(MixOfGausPrior, self).__init__()
        self.k = k
        self.w = torch.nn.Parameter(torch.zeros(k,))
        self.mean = torch.nn.Parameter(torch.zeros((n_latent,k)))
        self.logvar = torch.nn.Parameter(torch.ones((n_latent,k)))

    @property
    def distribution(self):
        comp = Normal(self.mean[0],self.logvar[0])
        mix = dist.Categorical(F.softmax(self.w, dim=0))
        return dist.MixtureSameFamily(mix, comp)

    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))

    def log_prob(self, z):
        return self.distribution.log_prob(z)
    
    def description(self):
        return "Mixture of Gaussians with k: " +str(self.k)+ "Prior with means: " + str(self.mean) + " and log variance: " + str(self.logvar)
