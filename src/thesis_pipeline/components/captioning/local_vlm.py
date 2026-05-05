import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


@dataclass
class CaptionGenerationError(RuntimeError):
    image_path: str
    prompt_label: str
    kind: str
    attempts: int
    retry_count: int
    message: str

    def __str__(self) -> str:
        return (
            f"{self.kind} while generating {self.prompt_label} caption for {self.image_path} "
            f"(attempts={self.attempts}, retries={self.retry_count}): {self.message}"
        )


class LocalVLM:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading VLM: {self.model_name} on {self.device}...")
        try:
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("VLM loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load VLM: {e}")
            raise

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        message = str(exc).lower()
        return "out of memory" in message or "cudaerrormemoryallocation" in message

    def _cleanup_device_memory(self, cleanup_cuda_cache: bool = True) -> None:
        gc.collect()
        if cleanup_cuda_cache and self.device != "cpu" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _prepare_inputs(self, image: Image.Image, prompt: str | None) -> dict[str, Any]:
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")

        prepared: dict[str, Any] = {}
        target_dtype = torch.float16 if self.device != "cpu" else torch.float32
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    prepared[key] = value.to(self.device, dtype=target_dtype)
                else:
                    prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def _generate_once(self, image: Image.Image, prompt: str | None, max_new_tokens: int) -> str:
        prepared_inputs: dict[str, Any] | None = None
        generated_ids = None
        try:
            prepared_inputs = self._prepare_inputs(image, prompt)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **prepared_inputs,
                    max_new_tokens=max_new_tokens,
                )
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()
            return generated_text
        finally:
            if generated_ids is not None:
                del generated_ids
            if prepared_inputs is not None:
                del prepared_inputs
            self._cleanup_device_memory()

    @staticmethod
    def _reduced_token_budget(current_tokens: int, backoff_tokens: int) -> int:
        reduced = max(8, min(backoff_tokens, max(8, current_tokens // 2)))
        if reduced >= current_tokens:
            reduced = max(8, current_tokens - 8)
        return max(8, reduced)

    def _generate_caption_with_report(
        self,
        image: Image.Image,
        image_path: str,
        prompt: str | None,
        *,
        prompt_label: str,
        max_new_tokens: int = 50,
        oom_retry_limit: int = 1,
        oom_backoff_max_new_tokens: int = 24,
        cleanup_cuda_cache: bool = True,
    ) -> dict[str, Any]:
        attempts = 0
        retry_count = 0
        current_max_new_tokens = max_new_tokens
        last_exc: Exception | None = None

        while attempts < max(1, oom_retry_limit + 1):
            attempts += 1
            try:
                text = self._generate_once(image, prompt, current_max_new_tokens)
                self._cleanup_device_memory(cleanup_cuda_cache=cleanup_cuda_cache)
                return {
                    "text": text,
                    "attempts": attempts,
                    "retry_count": retry_count,
                    "max_new_tokens": current_max_new_tokens,
                    "prompt_label": prompt_label,
                }
            except Exception as exc:
                last_exc = exc
                is_oom = self._is_oom_error(exc)
                self._cleanup_device_memory(cleanup_cuda_cache=cleanup_cuda_cache)
                if is_oom and retry_count < oom_retry_limit:
                    retry_count += 1
                    current_max_new_tokens = self._reduced_token_budget(
                        current_max_new_tokens,
                        oom_backoff_max_new_tokens,
                    )
                    self.logger.warning(
                        f"OOM while generating {prompt_label} caption for {image_path}; "
                        f"retrying with max_new_tokens={current_max_new_tokens} "
                        f"(retry {retry_count}/{oom_retry_limit})."
                    )
                    continue

                kind = "cuda_oom" if is_oom else "generation_error"
                raise CaptionGenerationError(
                    image_path=image_path,
                    prompt_label=prompt_label,
                    kind=kind,
                    attempts=attempts,
                    retry_count=retry_count,
                    message=str(exc),
                ) from exc

        raise CaptionGenerationError(
            image_path=image_path,
            prompt_label=prompt_label,
            kind="generation_error",
            attempts=attempts,
            retry_count=retry_count,
            message=str(last_exc or "unknown caption generation failure"),
        )

    def generate_caption(
        self,
        image_path,
        prompt=None,
        *,
        prompt_label: str = "generic",
        max_new_tokens: int = 50,
        oom_retry_limit: int = 1,
        oom_backoff_max_new_tokens: int = 24,
        cleanup_cuda_cache: bool = True,
    ):
        """
        Generates a caption for the image.
        If prompt is provided, it acts as VQA or conditional generation.
        If prompt is None, it generates a generic description.
        """
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        result = self._generate_caption_with_report(
            image,
            str(image_path),
            prompt,
            prompt_label=prompt_label,
            max_new_tokens=max_new_tokens,
            oom_retry_limit=oom_retry_limit,
            oom_backoff_max_new_tokens=oom_backoff_max_new_tokens,
            cleanup_cuda_cache=cleanup_cuda_cache,
        )
        return result["text"]

    def dense_caption_with_report(
        self,
        image_path: str,
        *,
        max_new_tokens: int = 50,
        oom_retry_limit: int = 1,
        oom_backoff_max_new_tokens: int = 24,
        cleanup_cuda_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Generates a rich, multi-aspect description useful for training.
        Uses specific archaeological prompts to avoid generic/tautological output.
        """
        prompt_specs = [
            (
                "description",
                "Describe this ancient Greek pottery vessel, including its shape, color, and visible decorations.",
            ),
            (
                "style",
                "Question: What decorative patterns, painting techniques, and artistic styles are visible on this pottery? Answer:",
            ),
            (
                "details",
                "Question: Describe any figures, scenes, or motifs depicted on this vessel. Answer:",
            ),
        ]

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")

        prompt_reports: list[dict[str, Any]] = []
        prompt_texts: list[tuple[str, str]] = []

        for prompt_label, prompt_text in prompt_specs:
            prompt_report = self._generate_caption_with_report(
                image,
                str(image_path),
                prompt_text,
                prompt_label=prompt_label,
                max_new_tokens=max_new_tokens,
                oom_retry_limit=oom_retry_limit,
                oom_backoff_max_new_tokens=oom_backoff_max_new_tokens,
                cleanup_cuda_cache=cleanup_cuda_cache,
            )
            prompt_reports.append(prompt_report)
            prompt_texts.append((prompt_label, prompt_report["text"]))

        combined = (
            f"{prompt_texts[0][1]}. "
            f"Style: {prompt_texts[1][1]}. "
            f"Details: {prompt_texts[2][1]}."
        ).strip()
        combined = " ".join(combined.split())
        if len(combined.split()) < 6:
            raise CaptionGenerationError(
                image_path=str(image_path),
                prompt_label="dense_caption",
                kind="invalid_caption",
                attempts=sum(int(item["attempts"]) for item in prompt_reports),
                retry_count=sum(int(item["retry_count"]) for item in prompt_reports),
                message=f"Combined caption too short to be scientifically useful: {combined!r}",
            )

        return {
            "caption": combined,
            "prompt_reports": prompt_reports,
            "attempts": sum(int(item["attempts"]) for item in prompt_reports),
            "retry_count": sum(int(item["retry_count"]) for item in prompt_reports),
        }

    def dense_caption(self, image_path):
        return self.dense_caption_with_report(str(image_path))["caption"]
