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

    return BCE + KLD


def rdrop(recon_x1, x, mu1, std1, model):
    """
    Dropout을 적용한 경우 dropout에 의해 나타나는 무작위성이 train 과 inference(generate) 사이 불일치를 초래

    서로 다른 모델에서 얻은 probability 간의 kl divergence를 줄이는 방향으로 학습
    """
    recon_x2, mu2, std2 = model(x)

    BCE = 0.5 * (F.binary_cross_entropy(recon_x1, x, reduction="mean") + F.binary_cross_entropy(recon_x2, x, reduction="mean")) * mu1.size(0)
    KLD = compute_kl_loss(recon_x1, recon_x2)
    print(BCE)
    print(KLD)

    return BCE + KLD


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
