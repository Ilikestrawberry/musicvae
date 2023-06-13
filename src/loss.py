import torch
import torch.nn.functional as F


def ELBO_loss(recon_x, x, mu, std, epoch):
    """
    BCE: 입력값 x와 decoder를 거쳐 생성된 x'의 binary 값을 비교한 cross-entropy(Reconstruction loss)
    KLD: encoder를 통해 근사된 q(z)와 실제분포 p(z) 사이의 KL-divergence
    """
    log_var = std.pow(2).log()
    beta = KL_annealing(kl_weight=1.0, kl_weight_start=0.5, rate=0.99995, epoch=epoch)
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + (beta * KLD)


def KL_annealing(kl_weight, kl_weight_start, rate=0.99995, epoch=0):
    return kl_weight - (kl_weight - kl_weight_start) * (rate) ** epoch
