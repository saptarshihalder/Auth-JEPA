import numpy as np
import torch
import torch.nn as nn


def tile_nll(mu, var, P):  # Gaussian negative log-likelihood, summed over 74 measurements
    return 0.5 * (var.log() + (P - mu) ** 2 / var).sum(-1)


def train(net, C_train, P_train, C_val, P_val, epochs=96):  # real photos only
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)  # optimiser
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)  # cosine schedule
    best, state = 1e9, None
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(C_train)).split(48):  # batch size 48
            mu, var = net(C_train[ix])
            loss = tile_nll(mu, var, P_train[ix]).mean()  # training loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # gradient clipping
            opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            vl = tile_nll(*net(C_val), P_val).mean().item()  # loss on held-out real photos
        if vl < best:  # keep the best epoch
            best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net


@torch.no_grad()
def image_score(net, C, P):  # C: (images, 16, 15) inputs, P: (images, 16, 74) measured noise
    mu, var = net(C)
    patch_score = tile_nll(mu, var, P)  # patch score
    return patch_score.mean(1)  # image score = mean of the 16 patch scores


def robust(X):  # scaling fitted on real photos: (x - median) / (1.4826 * MAD)
    f = X.reshape(-1, X.shape[-1]); med = np.median(f, 0)
    mad = np.median(np.abs(f - med), 0) * 1.4826 + 1e-6
    return lambda Y: np.clip(np.nan_to_num((Y - med) / mad), -50, 50).astype(np.float32)
