# src/thesis_pipeline/components/dataset.py
import logging
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class InpaintingDataset(Dataset):
    """
    A PyTorch Dataset for the image inpainting task.
    It loads a ground truth image, its corresponding mask, and its caption.
    """
    def __init__(self, data_dir: Path, image_size: list, split_name: str):
        """
        Args:
            data_dir (Path): Path to the root of the inpainting dataset.
            image_size (list): The target size [height, width] to resize images to.
            split_name (str): The name of the split to load ('train', 'validation', 'test').
        """
        self.split_dir = data_dir / split_name
        self.image_dir = self.split_dir / 'ground_truth'
        self.mask_dir = self.split_dir / 'masks'
        self.caption_dir = self.split_dir / 'captions'
        self.logger = logging.getLogger(__name__)
        
        image_files = sorted([p for p in self.image_dir.glob('*.png') if p.is_file()])
        if not image_files:
            self.logger.warning(f"No images found in {self.image_dir}. This split will be empty.")

        # Build a stable list of valid (image, mask, caption) triples by stem.
        # This prevents noisy training logs and avoids silently re-sampling other
        # items when masks are missing.
        missing_masks = []
        samples = []
        for image_path in image_files:
            stem = image_path.stem
            mask_path = self.mask_dir / f"{stem}.png"
            caption_path = self.caption_dir / f"{stem}.txt"
            if not mask_path.exists():
                missing_masks.append(stem)
                continue
            samples.append((image_path, mask_path, caption_path))

        if missing_masks:
            preview = ", ".join(missing_masks[:10])
            suffix = "" if len(missing_masks) <= 10 else f" (+{len(missing_masks) - 10} more)"
            self.logger.warning(
                f"Missing {len(missing_masks)} masks in '{split_name}' split; "
                f"excluding those samples. Examples: {preview}{suffix}"
            )

        self.samples = samples
        self.logger.info(f"Loaded {len(self.samples)} valid samples from '{split_name}' split.")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.samples:
            raise IndexError("Empty dataset: no valid (image, mask) pairs were found.")

        # Bounded retry in case of a single corrupt image/mask file.
        max_attempts = min(5, len(self.samples))
        for attempt in range(max_attempts):
            real_idx = (idx + attempt) % len(self.samples)
            image_path, mask_path, caption_path = self.samples[real_idx]
            try:
                original_image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                original_image_tensor = self.transform(original_image)
                mask_tensor = self.mask_transform(mask)
                masked_image_tensor = original_image_tensor * (1 - mask_tensor)

                caption = ""
                if caption_path.exists():
                    with open(caption_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()

                return {
                    "original_image": original_image_tensor,
                    "masked_image": masked_image_tensor,
                    "mask": mask_tensor,
                    "caption": caption,
                }
            except Exception as e:
                self.logger.error(
                    f"Error loading sample {real_idx} "
                    f"(image={image_path}, mask={mask_path}). Error: {e}"
                )

        raise RuntimeError("Failed to load a valid sample after retries.")
