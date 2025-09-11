import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np




def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H,  D)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        # mean flow settings
        flow_ratio=0.50, # t=r 的比例
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        # cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        # cfg_scale=2.0,
        # experimental
        # cfg_uncond='u',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist

        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32) #形状为 (batch_size, 2)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected] #生成一个从0到batch_size - 1的随机排列
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, y):
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1 ").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1 ").detach().clone()

        # e = torch.randn_like(x)

        z = (1 - t_) * y + t_ * x
        v = x - y
        
        v_hat = v
        
        # forward pass
        # u = model(z, t, r, y=c)
        # model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_onestep(self,  image, model,
                          # tbd: multi-step sampling
                           ):
        # model.eval()


        t = torch.ones((image.shape[0],), device=image.device)
        r = torch.zeros((image.shape[0],), device=image.device)

        z = image - model(image, t, r)
        # z = image + model(image, r, t) # for reverse

        return z
    
    @torch.no_grad()
    def sample_multistep(self, image,model,
                          sample_steps=10):
        # model.eval()

        z=image.clone()  # start from the input image
        all_out=[]    
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=image.device)
        # print(t_vals)
        # print(t_vals)

        for i in range(sample_steps):
            
            # print(i)
            t = torch.full((z.size(0),), t_vals[i], device=image.device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=image.device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1 ").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1 ").detach().clone()

            v = model(z, t, r)
            z = z - (t_-r_) * v
            
            if i % 100 == 0 or (i % 10 == 0 and i > 900):

                all_out.append(z)

        return all_out #后续做保存