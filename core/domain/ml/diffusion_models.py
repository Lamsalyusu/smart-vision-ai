from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import copy


class DiffusionSchedule(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"


class SamplingMethod(Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    ANCESTRAL = "ancestral"
    HEUN = "heun"
    EULER = "euler"


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: DiffusionSchedule = DiffusionSchedule.LINEAR
    prediction_type: str = "epsilon"
    clip_denoised: bool = True
    rescale_timesteps: bool = False


@dataclass(frozen=True, slots=True)
class DenoisingStep:
    timestep: int
    x_t: np.ndarray
    noise_pred: np.ndarray
    x_0_pred: np.ndarray
    alpha_bar_t: float
    alpha_t: float
    sigma_t: float


class NoiseSchedule:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.timesteps = config.timesteps
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.maximum(self.posterior_variance, 1e-20))
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _compute_betas(self) -> np.ndarray:
        if self.config.schedule == DiffusionSchedule.LINEAR:
            return np.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)
        elif self.config.schedule == DiffusionSchedule.COSINE:
            s = 0.008
            steps = self.config.timesteps + 1
            x = np.linspace(0, self.config.timesteps, steps)
            alphas_cumprod = np.cos(((x / self.config.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
        elif self.config.schedule == DiffusionSchedule.SIGMOID:
            betas = np.linspace(-6, 6, self.config.timesteps)
            betas = 1 / (1 + np.exp(-betas))
            betas = betas * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
            return betas
        elif self.config.schedule == DiffusionSchedule.QUADRATIC:
            betas = np.linspace(self.config.beta_start**0.5, self.config.beta_end**0.5, self.config.timesteps) ** 2
            return betas
        elif self.config.schedule == DiffusionSchedule.EXPONENTIAL:
            betas = np.exp(np.linspace(np.log(self.config.beta_start), np.log(self.config.beta_end), self.config.timesteps))
            return betas
        return np.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)

    def get_alpha_bar(self, t: int) -> float:
        return self.alphas_cumprod[t]

    def get_alpha(self, t: int) -> float:
        return self.alphas[t]

    def get_beta(self, t: int) -> float:
        return self.betas[t]

    def get_sigma(self, t: int) -> float:
        return np.sqrt(self.posterior_variance[t])

    def q_sample(self, x_0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
        if noise is None:
            noise = np.random.randn(*x_0.shape)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def q_posterior_mean_variance(self, x_0: np.ndarray, x_t: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.posterior_mean_coef1[t] * x_0 + self.posterior_mean_coef2[t] * x_t
        variance = self.posterior_variance[t]
        return mean, variance


class DiffusionModel(ABC):
    @abstractmethod
    def predict_noise(self, x_t: np.ndarray, t: int, cond: Optional[Dict] = None) -> np.ndarray:
        pass

    @abstractmethod
    def predict_x0(self, x_t: np.ndarray, t: int, noise_pred: np.ndarray) -> np.ndarray:
        pass


class UNetModel(DiffusionModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 4,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_fp16: bool = False
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_fp16 = use_fp16

        self.time_embed_dim = model_channels * 4

    def predict_noise(self, x_t: np.ndarray, t: int, cond: Optional[Dict] = None) -> np.ndarray:
        return np.random.randn(*x_t.shape) * 0.1

    def predict_x0(self, x_t: np.ndarray, t: int, noise_pred: np.ndarray) -> np.ndarray:
        schedule = NoiseSchedule(DiffusionConfig())
        sqrt_recip_alpha = schedule.sqrt_recip_alphas[t]
        sqrt_recipm1_alpha_bar = schedule.sqrt_recipm1_alphas_cumprod[t]
        return sqrt_recip_alpha * (x_t - sqrt_recipm1_alpha_bar * noise_pred)


class DiTModel(DiffusionModel):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        class_dropout_prob: float = 0.1
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.learn_sigma = learn_sigma
        self.class_dropout_prob = class_dropout_prob

    def predict_noise(self, x_t: np.ndarray, t: int, cond: Optional[Dict] = None) -> np.ndarray:
        return np.random.randn(*x_t.shape) * 0.1

    def predict_x0(self, x_t: np.ndarray, t: int, noise_pred: np.ndarray) -> np.ndarray:
        schedule = NoiseSchedule(DiffusionConfig())
        sqrt_recip_alpha = schedule.sqrt_recip_alphas[t]
        sqrt_recipm1_alpha_bar = schedule.sqrt_recipm1_alphas_cumprod[t]
        return sqrt_recip_alpha * (x_t - sqrt_recipm1_alpha_bar * noise_pred)


class DDPMSampler:
    def __init__(self, config: DiffusionConfig, schedule: NoiseSchedule):
        self.config = config
        self.schedule = schedule

    def sample(
        self,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        cond: Optional[Dict] = None,
        progress: bool = False
    ) -> List[DenoisingStep]:
        x_t = np.random.randn(*shape)
        steps = []

        for t in reversed(range(self.config.timesteps)):
            noise_pred = model.predict_noise(x_t, t, cond)
            x_0_pred = model.predict_x0(x_t, t, noise_pred)

            if self.config.clip_denoised:
                x_0_pred = np.clip(x_0_pred, -1, 1)

            mean, variance = self.schedule.q_posterior_mean_variance(x_0_pred, x_t, t)

            if t > 0:
                noise = np.random.randn(*x_t.shape)
                x_t = mean + np.sqrt(variance) * noise
            else:
                x_t = mean

            steps.append(DenoisingStep(
                timestep=t,
                x_t=x_t.copy(),
                noise_pred=noise_pred,
                x_0_pred=x_0_pred,
                alpha_bar_t=self.schedule.get_alpha_bar(t),
                alpha_t=self.schedule.get_alpha(t),
                sigma_t=self.schedule.get_sigma(t)
            ))

        return steps

    def sample_loop(self, model: DiffusionModel, shape: Tuple[int, ...], cond: Optional[Dict] = None) -> np.ndarray:
        steps = self.sample(model, shape, cond)
        return steps[-1].x_t if steps else np.zeros(shape)


class DDIMSampler:
    def __init__(
        self,
        config: DiffusionConfig,
        schedule: NoiseSchedule,
        eta: float = 0.0,
        skip_steps: int = 10
    ):
        self.config = config
        self.schedule = schedule
        self.eta = eta
        self.skip_steps = skip_steps
        self.ddim_timesteps = np.arange(0, config.timesteps, skip_steps)[::-1]

    def sample(
        self,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        cond: Optional[Dict] = None,
        progress: bool = False
    ) -> List[DenoisingStep]:
        x_t = np.random.randn(*shape)
        steps = []

        for i, t in enumerate(self.ddim_timesteps):
            noise_pred = model.predict_noise(x_t, t, cond)
            x_0_pred = model.predict_x0(x_t, t, noise_pred)

            if self.config.clip_denoised:
                x_0_pred = np.clip(x_0_pred, -1, 1)

            alpha_bar_t = self.schedule.alphas_cumprod[t]
            alpha_bar_prev = self.schedule.alphas_cumprod[self.ddim_timesteps[i + 1]] if i + 1 < len(self.ddim_timesteps) else 1.0

            sigma_t = self.eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            c1 = sigma_t
            c2 = np.sqrt(1 - alpha_bar_prev - sigma_t**2)

            if i + 1 < len(self.ddim_timesteps):
                noise = np.random.randn(*x_t.shape) if self.eta > 0 else 0
                x_t = np.sqrt(alpha_bar_prev) * x_0_pred + c1 * noise + c2 * noise_pred
            else:
                x_t = x_0_pred

            steps.append(DenoisingStep(
                timestep=t,
                x_t=x_t.copy(),
                noise_pred=noise_pred,
                x_0_pred=x_0_pred,
                alpha_bar_t=alpha_bar_t,
                alpha_t=self.schedule.alphas[t],
                sigma_t=sigma_t
            ))

        return steps


class DPMSSampler:
    def __init__(
        self,
        config: DiffusionConfig,
        schedule: NoiseSchedule,
        order: int = 2,
        steps: int = 20
    ):
        self.config = config
        self.schedule = schedule
        self.order = order
        self.steps = steps
        self.timesteps = np.linspace(config.timesteps - 1, 0, steps + 1).astype(int)

    def sample(
        self,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        cond: Optional[Dict] = None,
        progress: bool = False
    ) -> List[DenoisingStep]:
        x_t = np.random.randn(*shape)
        steps = []
        noise_preds = []

        for i in range(self.steps):
            t = self.timesteps[i]
            t_next = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            noise_pred = model.predict_noise(x_t, t, cond)
            noise_preds.append(noise_pred)

            x_0_pred = model.predict_x0(x_t, t, noise_pred)

            if self.config.clip_denoised:
                x_0_pred = np.clip(x_0_pred, -1, 1)

            alpha_bar_t = self.schedule.alphas_cumprod[t]
            alpha_bar_next = self.schedule.alphas_cumprod[t_next] if t_next >= 0 else 1.0

            if self.order == 1:
                x_t = np.sqrt(alpha_bar_next) * x_0_pred + np.sqrt(1 - alpha_bar_next) * noise_pred
            elif self.order == 2 and len(noise_preds) >= 2:
                h = math.log(alpha_bar_next / alpha_bar_t)
                noise_pred_2 = noise_preds[-2]
                noise_pred_1 = noise_preds[-1]
                x_t = np.sqrt(alpha_bar_next) * x_0_pred + np.sqrt(1 - alpha_bar_next) * (noise_pred_1 + 0.5 * h * (noise_pred_1 - noise_pred_2))

            steps.append(DenoisingStep(
                timestep=t,
                x_t=x_t.copy(),
                noise_pred=noise_pred,
                x_0_pred=x_0_pred,
                alpha_bar_t=alpha_bar_t,
                alpha_t=self.schedule.alphas[t],
                sigma_t=np.sqrt(1 - alpha_bar_t)
            ))

        return steps


class DiffusionPipeline:
    def __init__(
        self,
        model: DiffusionModel,
        config: DiffusionConfig,
        sampler: SamplingMethod = SamplingMethod.DDPM
    ):
        self.model = model
        self.config = config
        self.schedule = NoiseSchedule(config)
        self.sampler = self._create_sampler(sampler)

    def _create_sampler(self, method: SamplingMethod):
        if method == SamplingMethod.DDPM:
            return DDPMSampler(self.config, self.schedule)
        elif method == SamplingMethod.DDIM:
            return DDIMSampler(self.config, self.schedule)
        elif method == SamplingMethod.DPM_SOLVER:
            return DPMSSampler(self.config, self.schedule)
        else:
            return DDPMSampler(self.config, self.schedule)

    def generate(
        self,
        shape: Tuple[int, ...],
        cond: Optional[Dict] = None,
        num_samples: int = 1
    ) -> List[np.ndarray]:
        samples = []
        for _ in range(num_samples):
            steps = self.sampler.sample(self.model, shape, cond)
            samples.append(steps[-1].x_t if steps else np.zeros(shape))
        return samples

    def interpolate(
        self,
        x_0: np.ndarray,
        x_1: np.ndarray,
        t: float,
        cond: Optional[Dict] = None
    ) -> np.ndarray:
        t_idx = int(t * (self.config.timesteps - 1))
        noise_0 = np.random.randn(*x_0.shape)
        noise_1 = np.random.randn(*x_1.shape)

        x_t_0 = self.schedule.q_sample(x_0, t_idx, noise_0)
        x_t_1 = self.schedule.q_sample(x_1, t_idx, noise_1)

        interpolated = (1 - t) * x_t_0 + t * x_t_1

        steps = self.sampler.sample(self.model, x_0.shape, cond)
        return steps[-1].x_t if steps else interpolated


class ConditionalDiffusion:
    def __init__(
        self,
        base_model: DiffusionModel,
        encoder: Callable,
        config: DiffusionConfig
    ):
        self.base_model = base_model
        self.encoder = encoder
        self.config = config
        self.schedule = NoiseSchedule(config)

    def encode_condition(self, condition: Any) -> Dict:
        return self.encoder(condition)

    def sample_conditional(
        self,
        shape: Tuple[int, ...],
        condition: Any,
        sampler: SamplingMethod = SamplingMethod.DDIM
    ) -> np.ndarray:
        cond = self.encode_condition(condition)
        pipeline = DiffusionPipeline(self.base_model, self.config, sampler)
        samples = pipeline.generate(shape, cond, num_samples=1)
        return samples[0]


class DiffusionTrainer:
    def __init__(
        self,
        model: DiffusionModel,
        config: DiffusionConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ema_decay: float = 0.9999
    ):
        self.model = model
        self.config = config
        self.schedule = NoiseSchedule(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.ema_model = None
        self.step = 0

    def training_step(self, x_0: np.ndarray, cond: Optional[Dict] = None) -> Dict[str, float]:
        batch_size = x_0.shape[0]
        t = np.random.randint(0, self.config.timesteps, batch_size)
        noise = np.random.randn(*x_0.shape)

        x_t = np.array([self.schedule.q_sample(x_0[i], t[i], noise[i]) for i in range(batch_size)])
        noise_pred = np.array([self.model.predict_noise(x_t[i], t[i], cond) for i in range(batch_size)])

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "x0":
            target = x_0
        elif self.config.prediction_type == "v":
            alpha_bar = self.schedule.alphas_cumprod[t]
            target = np.sqrt(alpha_bar)[:, np.newaxis, np.newaxis, np.newaxis] * noise - \
                     np.sqrt(1 - alpha_bar)[:, np.newaxis, np.newaxis, np.newaxis] * x_0
        else:
            target = noise

        loss = np.mean((noise_pred - target) ** 2)

        self.step += 1
        return {"loss": float(loss), "step": self.step}

    def validation_step(self, x_0: np.ndarray, cond: Optional[Dict] = None) -> Dict[str, float]:
        batch_size = min(4, x_0.shape[0])
        t = np.random.randint(0, self.config.timesteps, batch_size)
        noise = np.random.randn(*x_0.shape[:batch_size])

        x_t = np.array([self.schedule.q_sample(x_0[i], t[i], noise[i]) for i in range(batch_size)])
        noise_pred = np.array([self.model.predict_noise(x_t[i], t[i], cond) for i in range(batch_size)])

        loss = np.mean((noise_pred - noise) ** 2)
        return {"val_loss": float(loss)}


class InpaintingDiffusion:
    def __init__(self, pipeline: DiffusionPipeline):
        self.pipeline = pipeline

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cond: Optional[Dict] = None
    ) -> np.ndarray:
        shape = image.shape
        known = image * mask
        unknown_mask = 1 - mask

        x_t = np.random.randn(*shape)
        x_t = x_t * unknown_mask + known * mask

        for t in reversed(range(self.pipeline.config.timesteps)):
            noise_pred = self.pipeline.model.predict_noise(x_t, t, cond)
            x_0_pred = self.pipeline.model.predict_x0(x_t, t, noise_pred)

            if self.pipeline.config.clip_denoised:
                x_0_pred = np.clip(x_0_pred, -1, 1)

            x_0_pred = x_0_pred * unknown_mask + known * mask

            mean, variance = self.pipeline.schedule.q_posterior_mean_variance(x_0_pred, x_t, t)

            if t > 0:
                noise = np.random.randn(*shape)
                x_t = mean + np.sqrt(variance) * noise
            else:
                x_t = mean

            x_t = x_t * unknown_mask + known * mask

        return x_t


class SuperResolutionDiffusion:
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        scale_factor: int = 4
    ):
        self.pipeline = pipeline
        self.scale_factor = scale_factor

    def upscale(
        self,
        low_res: np.ndarray,
        cond: Optional[Dict] = None
    ) -> np.ndarray:
        h, w = low_res.shape[:2]
        shape = (h * self.scale_factor, w * self.scale_factor, low_res.shape[2])

        upsampled = cv2.resize(low_res, (w * self.scale_factor, h * self.scale_factor), interpolation=cv2.INTER_CUBIC)

        x_t = np.random.randn(*shape)
        x_t = (x_t + upsampled) / 2

        for t in reversed(range(self.pipeline.config.timesteps)):
            noise_pred = self.pipeline.model.predict_noise(x_t, t, cond)
            x_0_pred = self.pipeline.model.predict_x0(x_t, t, noise_pred)

            if self.pipeline.config.clip_denoised:
                x_0_pred = np.clip(x_0_pred, -1, 1)

            mean, variance = self.pipeline.schedule.q_posterior_mean_variance(x_0_pred, x_t, t)

            if t > 0:
                noise = np.random.randn(*shape)
                x_t = mean + np.sqrt(variance) * noise
            else:
                x_t = mean

        return x_t


class DiffusionMetrics:
    @staticmethod
    def compute_fid(real_images: np.ndarray, generated_images: np.ndarray) -> float:
        try:
            import torch
            from torch.nn.functional import adaptive_avg_pool2d
        except ImportError:
            return 0.0

        return 0.0

    @staticmethod
    def compute_is(images: np.ndarray, splits: int = 10) -> Tuple[float, float]:
        return 0.0, 0.0

    @staticmethod
    def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(1.0 / math.sqrt(mse))

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        return 0.0


import cv2
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet