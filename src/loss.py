import torch
import torch.nn.functional as F


def ELBO_loss(recon_x, x, mu, std):
    """
    BCE
    KLD
    """
    log_var = std.pow(2).log()
    BCE = F.binary_cross_entropy(recon_x, x, reduction="mean") * mu.size(0)
    KLD = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / (mu.size(0) * mu.size(1))
    # print(BCE)
    # print(KLD)

    return BCE + KLD
