import torch
import torch.nn.functional as F
from ex02_helpers import extract, default, exists
from tqdm import tqdm
from einops import rearrange, reduce
from torch import nn
from torch.cuda.amp import autocast
from collections import namedtuple
from functools import partial

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    pass


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    pass


# class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help



        # define alphas
        # TODO
        alphas = 1.0 - self.betas
        self.alphas_comprod = torch.cumprod(alphas, dim=0)
        alphas_comprod_prev = F.pad(self.alphas_comprod[:-1], (1, 0), value=1.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.sqrt_alphas_comprod = torch.sqrt(self.alphas_comprod)
        self.sqrt_one_minus_alphas_comprod = torch.sqrt(1. - self.alphas_comprod)
        self.log_one_minus_alphas_comprod = torch.log(1. - self.alphas_comprod)
        self.sqrt_recip_alphas_comprod = torch.sqrt(1. / self.alphas_comprod)
        self.sqrt_recipm1_alphas_comprod = torch.sqrt(1. / self.alphas_comprod -1)



        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        posterior_variance = self.betas * (1. - alphas_comprod_prev) / (1. - self.alphas_comprod)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        # TODO (2.2): The method should return the image at timestep t-1.
        b, *_, device = *x.shape, self.device

        batched_times = torch.full((b,), t, device = device, dtype = torch.long)

        model_mean, _, model_log_variance, x_zero = model
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return pred_img

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.

        # TODO (2.2): Return the generated images

        img = torch.randn((image_size, image_size, channels), device = self.device)
        imgs = [img]

        x_zero = None

        for t in tqdm(reversed(range(0, int(self.timesteps))), desc = 'sampling loop time step', total = int(self.timesteps)):
            self_cond = x_zero
            img, x_zero = self.p_sample(model, img, t, self_cond)
            imgs.append(img)

        ret = torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        noise = default(noise, lambda: torch.randn_like(x_zero))

        return (
            extract(self.sqrt_alphas_comprod, t, x_zero.shape) * x_zero +
            extract(self.sqrt_one_minus_alphas_comprod, t, x_zero.shape) * noise
        )

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        # if loss_type == 'l1':
        #     # TODO (2.2): implement an L1 loss for this task
        #     loss = None
        # elif loss_type == 'l2':
        #     # TODO (2.2): implement an L2 loss for this task
        #     loss = None
        # else:
        #     raise NotImplementedError()

        b, c, h, w = x_zero.shape

        noise = default(noise, lambda: torch.randn_like(x_zero))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = 0

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_zero.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_zero = x_zero, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_zero from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # predict and take gradient step

        model_out = denoise_model(x, t)

        target = noise

        snr = self.alphas_comprod / (1 - self.alphas_comprod)

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(snr, t, loss.shape)
        return loss.mean()

class Diffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps = 1000,
            sampling_timesteps = None,
            objective = 'pred_noise',
            beta_schedule = 'linear',
            schedule_fn_kwargs = dict(),
            ddim_sampling_eta = 0.,
            auto_normalize = True,
            min_snr_loss_weight = False,
            min_snr_gamma = 5
        ):
            super().__init__()
            assert not (type(self) == Diffusion and model.channels != model.out_dim)
            # assert not model.random_or_learned_sinusoidal_cond

            self.model = model
            self.channels = self.model.channels

            self.image_size = image_size

            self.objective = objective

            assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

            if beta_schedule == 'linear':
                beta_schedule_fn = linear_beta_schedule
            # elif beta_schedule == 'cosine':
            #     beta_schedule_fn = cosine_beta_schedule
            # elif beta_schedule == 'sigmoid':
            #     beta_schedule_fn = sigmoid_beta_schedule
            # else:
            #     raise ValueError(f'unknown beta schedule {beta_schedule}')

            betas = beta_schedule_fn(0.0001, 0.02, timesteps)

            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

            timesteps, = betas.shape
            self.num_timesteps = int(timesteps)

            # sampling related parameters

            self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

            assert self.sampling_timesteps <= timesteps
            self.is_ddim_sampling = self.sampling_timesteps < timesteps
            self.ddim_sampling_eta = ddim_sampling_eta

            # helper function to register buffer from float64 to float32

            register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

            register_buffer('betas', betas)
            register_buffer('alphas_cumprod', alphas_cumprod)
            register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

            # calculations for diffusion q(x_t | x_{t-1}) and others

            register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
            register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
            register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
            register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
            register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

            # calculations for posterior q(x_{t-1} | x_t, x_0)

            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

            # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

            register_buffer('posterior_variance', posterior_variance)

            # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

            register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
            register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
            register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

            # loss weight

            snr = alphas_cumprod / (1 - alphas_cumprod)

            maybe_clipped_snr = snr.clone()
            if min_snr_loss_weight:
                maybe_clipped_snr.clamp_(max = min_snr_gamma)

            if objective == 'pred_noise':
                loss_weight = maybe_clipped_snr / snr
            elif objective == 'pred_x0':
                loss_weight = maybe_clipped_snr
            elif objective == 'pred_v':
                loss_weight = maybe_clipped_snr / (snr + 1)

            register_buffer('loss_weight', loss_weight)

            self.normalize = normalize_to_neg_one_to_one
            self.unnormalize = unnormalize_to_zero_to_one

            # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

            # self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
            # self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) 

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def condition_mean(self, cond_fn, mean,variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **guidance_kwargs)
        new_mean = (
            mean.float() + variance * gradient.float()
        )
        print("gradient: ",(variance * gradient.float()).mean())
        return new_mean

        
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, cond_fn=None, guidance_kwargs=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True
        )
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, batched_times, guidance_kwargs)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond =  None
            img, x_start = self.p_sample(img, t, self_cond, cond_fn, guidance_kwargs)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps, cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)