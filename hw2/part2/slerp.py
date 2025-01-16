import os
import torch
import numpy as np
from UNet import UNet
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

class DDIM:
    def __init__(self, nn_model, timesteps=1000, beta_schedule=beta_scheduler()):
        self.model = nn_model
        self.timesteps = timesteps
        self.betas = beta_schedule

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # use DDIM to sample
    @torch.no_grad()
    def sample(self, batch_size=10, ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True):
        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            output = (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )
            return output
        def linear(z1, z2, alpha):
            return (z2 * alpha) + (z1 * (1 - alpha))

        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        # start from pure noise (for each example in the batch)
        filenames = [f"{i:02d}.pt" for i in range(0, 9 if batch_size > 9 else batch_size)]
        tensors = [torch.load(os.path.join("hw2_data/face/noise/", filename)) for filename in filenames]
        alpha = [i / 10.0 for i in range(batch_size)]

        slerp_noise = []
        for i in range(len(alpha)):
            slerp_noise.append(slerp(tensors[0], tensors[1], alpha[i]))
        sample_img = torch.cat(slerp_noise, dim=0)

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc="sampling loop time step", total=ddim_timesteps,):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = self.model(sample_img, t)

            # 3. get the predicted x_0
            pred_x0 = (
                (sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            )
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = (torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise)

            # 6. compute x_{t-1} of formula (12)
            x_prev = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
            )

            sample_img = x_prev

        return sample_img.cpu()
    
def output_img(img_num=10, eta=0):
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "part2/Output_folder/"
    UNet_pt_dir = "hw2_data/face/UNet.pt"
    unet_model = UNet()
    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM(nn_model=unet_model.to(device), timesteps=n_T, beta_schedule=beta_scheduler())
    with torch.no_grad():
        x_gen = ddim.sample(batch_size=img_num, ddim_eta=eta)
        concat = []
        for i in range(len(x_gen)):
            img = x_gen[i]
            # min-max normalization
            min_val, max_val = torch.min(img), torch.max(img)
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            concat.append(normalized_x_gen)
        concat = torch.cat(concat, dim=2)
        save_image(concat, save_dir + "slerp.png")

if __name__ == "__main__":
    output_img(img_num=11, eta=0)