"""Marigold module

Original code at https://github.com/huggingface/diffusers/blob/main/examples/community/marigold_depth_estimation.py
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.transforms.functional import center_crop

from nerfstudio.utils.rich_utils import CONSOLE

IMG_DIM = 768
CONST_SCALE = 0.18215


class Marigold(nn.Module):
    """Marigold implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    module_name: str = "Marigold"

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        num_train_timesteps: int = 1000,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()

        try:
            from diffusers import DiffusionPipeline, DDIMScheduler

        except ImportError:
            CONSOLE.print("[bold red]Missing Stable Diffusion packages!")
            sys.exit(1)

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.torch_dtype = torch_dtype

        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        CONSOLE.print(f"[bold green]Loading {self.module_name}...")

        pipe = DiffusionPipeline.from_pretrained(
            "Bingxin/Marigold",
            custom_pipeline="marigold_depth_estimation",
            torch_dtype=self.torch_dtype,  # (optional) Run with half-precision (16-bit float).
        )
        assert isinstance(pipe, DiffusionPipeline)  # and hasattr(pipe, "to")
        pipe = pipe.to(self.device)

        pipe.enable_attention_slicing()

        self.pipe = pipe

        self.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        self.unet = pipe.unet
        self.unet.to(memory_format=torch.channels_last)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.auto_encoder = pipe.vae

        CONSOLE.print(f"[bold green]{self.module_name} loaded! :tada:")

        self.empty_text_embed = None

    def _encode_empty_text(self):
        """
        Encode text embedding for empty prompt.
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        with torch.no_grad():
            self.empty_text_embed = self.text_encoder(text_input_ids)[0]

    def sds_loss(
        self,
        image: Float[Tensor, "BS 3 H W"],
        depth: Float[Tensor, "BS 1 H W"],
    ) -> torch.Tensor:
        """Score Distilation Sampling loss proposed in DreamFusion paper (https://dreamfusion3d.github.io/)
        Args:
            image: Rendered image
        Returns:
            The loss
        """
        image = F.interpolate(image, (IMG_DIM, IMG_DIM), mode="bilinear").to(torch.float16)
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        rgb_latent = self.imgs_to_latent(image)
        latents = self.imgs_to_latent(depth)
        if self.empty_text_embed is None:
            self._encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))  # [B, 2, 1024]

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            # pred noise
            latent_model_input = torch.cat([rgb_latent, latents], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=batch_empty_text_embed).sample

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return loss

    def produce_latents(
        self,
        rgb_latent: Float[Tensor, "BS 4 H W"],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: Optional[Float[Tensor, "BS 4 H W"]] = None,
    ) -> Float[Tensor, "BS 4 H W"]:
        """Produce depth latents for a given RGB latent
        Args:
            rgb_latent: RGB latent for conditioning
            height: Height of the image
            width: Width of the image
            num_inference_steps: Number of inference steps
            guidance_scale: How much to weigh the guidance
            latents: Latents to start with
        Returns:
            Latents
        """

        if self.empty_text_embed is None:
            self._encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))  # [B, 2, 1024]

        if latents is None:
            latents = torch.randn(rgb_latent.shape, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)  # type: ignore

        with torch.autocast("cuda"):
            for t in self.scheduler.timesteps:  # type: ignore
                assert latents is not None
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([rgb_latent, latents], dim=1)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t.to(self.device), encoder_hidden_states=batch_empty_text_embed
                    ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]  # type: ignore
        assert isinstance(latents, Tensor)
        return latents

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 1 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        # mean of output channels
        imgs = imgs.mean(dim=1, keepdim=True)

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def img_to_depth(
        self,
        imgs: Float[Tensor, "BS 4 H W"],
        num_inference_steps: int = 50,
        latents=None,
    ) -> Float[Tensor, "BS 1 H W"]:
        """Generate depths from images.
        Args:
            imgs: The image to generate an depth from.
            num_inference_steps: The number of inference steps to perform.
            latents: The latents to start from, defaults to random.
        Returns:
            The generated image.
        """

        shape = imgs.shape
        image = F.interpolate(imgs, (IMG_DIM, IMG_DIM), mode="bilinear").to(self.torch_dtype)
        rgb_latent = self.imgs_to_latent(image)

        latents = self.produce_latents(
            rgb_latent=rgb_latent,
            latents=latents,
            num_inference_steps=num_inference_steps,
        )  # [1, 4, resolution, resolution]

        diffused_img = self.latents_to_img(latents.half())
        diffused_img = F.interpolate(diffused_img, (shape[2], shape[3]), mode="bilinear")

        return diffused_img

    def forward(self, imgs, num_inference_steps=50, latents=None) -> np.ndarray:
        """Generate a depth from an image.

        Args:
            imgs: The images to generate a depth from.
            num_inference_steps: The number of inference steps to perform.
            latents: The latents to start from, defaults to random.
        Returns:
            The generated depth.
        """
        return self.img_to_depth(imgs, num_inference_steps, latents)


def generate_image(image_path: Path, seed: int = 0, steps: int = 50, save_path: Path = Path("test_marigold.png")):
    """Generate an image from a prompt using Stable Diffusion.
    Args:
        image_path: The image path to use.
        seed: The random seed to use.
        steps: The number of steps to use.
        save_path: The path to save the image to.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cuda_device = torch.device("cuda")
    with torch.no_grad():
        marigold = Marigold(cuda_device)

        import mediapy  # Slow to import, so we do it lazily.

        img = torch.from_numpy(mediapy.read_image(image_path)).permute(2, 0, 1).unsqueeze(0).to(cuda_device) / 255.0

        # Resize and center crop image to get IMG_DIM x IMG_DIM
        img_h, img_w = img.shape[2:]
        new_size = (
            (int(IMG_DIM * (img_h / img_w)), IMG_DIM) if img_h > img_w else (IMG_DIM, int(IMG_DIM * (img_w / img_h)))
        )
        img = F.interpolate(img, new_size, mode="bilinear")
        img = center_crop(img, (IMG_DIM, IMG_DIM))
        img_np = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        img_np = (img_np * 255).round().astype("uint8")

        depths = marigold.img_to_depth(img, steps)
        depths = depths.repeat(1, 3, 1, 1)
        depths_np = depths.detach().cpu().permute(0, 2, 3, 1).numpy()
        depths_np = (depths_np * 255).round().astype("uint8")

        mediapy.write_image(str(save_path), np.concatenate([img_np[0], depths_np[0]], axis=1))


if __name__ == "__main__":
    tyro.cli(generate_image)
