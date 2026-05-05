import cv2
import numpy as np
import torch
from PIL import Image
import logging
from thesis_pipeline.components.prompt_utils import cap_prompt_to_token_budget

StableDiffusionInpaintPipeline = None
UNet2DConditionModel = None

class TeleaInpainter:
    """
    Classical Computer Vision Baseline: Telea Inpainting.
    """
    def __init__(self):
        self.method = cv2.INPAINT_TELEA
        self.radius = 3

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        img_np = np.array(image)
        mask_np = np.array(mask)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(img_bgr, mask_np, self.radius, self.method)
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(restored_rgb)

class NavierStokesInpainter:
    """
    Classical Computer Vision Baseline: Navier-Stokes Inpainting.
    """
    def __init__(self):
        self.method = cv2.INPAINT_NS
        self.radius = 3

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        img_np = np.array(image)
        mask_np = np.array(mask)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(img_bgr, mask_np, self.radius, self.method)
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(restored_rgb)

class VanillaSDInpainter:
    """
    Generic AI Baseline: Stable Diffusion 1.5 Inpainting.
    Supports Unconditional, Raw, and Enriched text modes via the 'prompt' argument.

    Reproducibility note
    --------------------
    The generator is re-seeded with the same ``seed`` on every ``inpaint()``
    call.  This is **by design**: it ensures that for a given image+mask
    pair, the only variable that changes across text-conditioning modes is
    the prompt string, producing a fair comparison.

    An observed consequence is that, because this model was never fine-tuned
    with text conditioning, the three prompt variants (Unconditional / Raw /
    Enriched) may produce identical or near-identical outputs.  This is
    documented as a negative finding for RQ-2 (prompt ablation).
    """
    NEGATIVE_PROMPT = "blurry, artifact, distortion, low quality, watermark, text, deformed"

    def __init__(self, device="cuda", num_inference_steps=50, seed=42):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = "runwayml/stable-diffusion-inpainting"
        self.pipeline = None
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        if self.pipeline is None:
            global StableDiffusionInpaintPipeline
            if StableDiffusionInpaintPipeline is None:
                from diffusers import StableDiffusionInpaintPipeline as _StableDiffusionInpaintPipeline

                StableDiffusionInpaintPipeline = _StableDiffusionInpaintPipeline

            self.logger.info(f"Loading Vanilla SD from {self.model_id}...")
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(self.device)
            self.pipeline.safety_checker = None

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        if self.pipeline is None:
            self.load_model()

        if prompt:
            prompt, truncated = cap_prompt_to_token_budget(
                prompt,
                tokenizer=getattr(self.pipeline, "tokenizer", None),
                max_tokens=getattr(getattr(self.pipeline, "tokenizer", None), "model_max_length", 77),
            )
            if truncated:
                # Avoid CLIP truncation warnings and keep key tags front-loaded.
                self.logger.debug("Prompt truncated to CLIP token budget for Vanilla SD.")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        with torch.no_grad():
            output = self.pipeline(
                prompt=prompt, 
                negative_prompt=self.NEGATIVE_PROMPT if prompt else "",
                image=image, 
                mask_image=mask,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        return output

class OursInpainter:
    """
    Our Fine-Tuned Model.

    Reproducibility note
    --------------------
    Same deterministic seed behaviour as ``VanillaSDInpainter``.
    The generator is re-seeded identically for every call so that
    prompt conditioning is the only variable across modes.
    """
    NEGATIVE_PROMPT = "blurry, artifact, distortion, low quality, watermark, text, deformed"

    def __init__(self, model_path, device="cuda", num_inference_steps=50, seed=42):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.pipeline = None
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        if self.pipeline is None:
            global StableDiffusionInpaintPipeline, UNet2DConditionModel
            if StableDiffusionInpaintPipeline is None or UNet2DConditionModel is None:
                from diffusers import (
                    StableDiffusionInpaintPipeline as _StableDiffusionInpaintPipeline,
                    UNet2DConditionModel as _UNet2DConditionModel,
                )

                StableDiffusionInpaintPipeline = _StableDiffusionInpaintPipeline
                UNet2DConditionModel = _UNet2DConditionModel

            self.logger.info(f"Loading Fine-Tuned Model from {self.model_path}...")
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            
            # Load UNet
            unet = UNet2DConditionModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype
            )
            
            # Load Pipeline with Custom UNet
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                unet=unet,
                torch_dtype=dtype,
            ).to(self.device)
            self.pipeline.safety_checker = None

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        if self.pipeline is None:
            self.load_model()

        if prompt:
            prompt, truncated = cap_prompt_to_token_budget(
                prompt,
                tokenizer=getattr(self.pipeline, "tokenizer", None),
                max_tokens=getattr(getattr(self.pipeline, "tokenizer", None), "model_max_length", 77),
            )
            if truncated:
                self.logger.debug("Prompt truncated to CLIP token budget for FT-SD.")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        with torch.no_grad():
            output = self.pipeline(
                prompt=prompt, 
                negative_prompt=self.NEGATIVE_PROMPT if prompt else "",
                image=image, 
                mask_image=mask,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        return output


class TTAInpainter:
    """Test-Time Augmentation wrapper for any SD-based inpainter.

    Applies horizontal flip + identity, averages the (un-flipped) results
    in pixel space.  Improves SSIM and smoothness at near-zero extra cost
    (2x inference instead of 1x).

    Parameters
    ----------
    base_inpainter : VanillaSDInpainter | OursInpainter
        Any inpainter with an ``inpaint(image, mask, prompt)`` interface.
    """

    def __init__(self, base_inpainter):
        self.base = base_inpainter
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        self.base.load_model()

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
        # Original pass
        res_orig = self.base.inpaint(image, mask, prompt=prompt)

        # Horizontally flipped pass
        img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask_flip = mask.transpose(Image.FLIP_LEFT_RIGHT)
        res_flip = self.base.inpaint(img_flip, mask_flip, prompt=prompt)
        res_flip_back = res_flip.transpose(Image.FLIP_LEFT_RIGHT)

        # Average in pixel space
        arr_orig = np.array(res_orig, dtype=np.float32)
        arr_flip = np.array(res_flip_back, dtype=np.float32)
        averaged = np.clip((arr_orig + arr_flip) / 2.0, 0, 255).astype(np.uint8)
        return Image.fromarray(averaged)
