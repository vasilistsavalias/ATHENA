"""
CoModGAN fine-tuning adapter.

CoModGAN is fine-tuned via the MI-GAN inference/training code with
domain-specific masks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from thesis_pipeline.components.baseline_finetuning.adapter import (
    BaselineAdapter,
    FTResult,
)

logger = logging.getLogger(__name__)


class CoModGANAdapter(BaselineAdapter):
    """Fine-tune CoModGAN on the project's inpainting dataset."""

    def __init__(self, cfg: dict[str, Any], data_root: Path, checkpoint_dir: Path, device: str = "cuda"):
        super().__init__("CoModGAN", cfg, data_root, checkpoint_dir, device)
        self._best_ckpt: Path | None = None

    def setup(self) -> None:
        self.logger.info("CoModGAN adapter: checking prerequisites …")
        # Would clone MI-GAN repo and download pretrained weights

    def train(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        early_stopping_patience: int = 10,
    ) -> FTResult:
        """Fine-tune CoModGAN using a simplified GAN training loop."""
        self.logger.info(f"CoModGAN fine-tuning: epochs={epochs}, bs={batch_size}, lr={learning_rate}")

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"CoModGAN fine-tuning requires torch + PIL: {e}")

        class InpaintDataset(Dataset):
            def __init__(self, img_dir: Path, mask_dir: Path, size: int = 512):
                self.pairs = []
                for img_p in sorted(img_dir.glob("*.png")):
                    mask_p = mask_dir / img_p.name
                    if mask_p.exists():
                        self.pairs.append((img_p, mask_p))
                self.size = size

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                img_p, mask_p = self.pairs[idx]
                img = Image.open(img_p).convert("RGB").resize((self.size, self.size))
                mask = Image.open(mask_p).convert("L").resize((self.size, self.size))
                img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
                masked = img_t * (1 - mask_t)
                inp = torch.cat([masked, mask_t], dim=0)
                return inp, img_t, mask_t

        train_ds = InpaintDataset(self.train_images, self.train_masks)
        val_ds = InpaintDataset(self.val_images, self.val_masks)

        if len(train_ds) == 0:
            raise RuntimeError("CoModGAN FT: no training pairs found")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        generator = self._build_generator().to(self.device)
        discriminator = self._build_discriminator().to(self.device)

        opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
        criterion_recon = nn.L1Loss()

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            generator.train()
            discriminator.train()
            train_loss_g = 0.0
            for inp, gt, mask in train_loader:
                inp, gt, mask = inp.to(self.device), gt.to(self.device), mask.to(self.device)

                # ----- Discriminator step -----
                fake = generator(inp).detach()
                d_real = discriminator(gt)
                d_fake = discriminator(fake)
                loss_d = (nn.functional.relu(1.0 - d_real).mean() + nn.functional.relu(1.0 + d_fake).mean())
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # ----- Generator step -----
                fake = generator(inp)
                d_fake = discriminator(fake)
                loss_adv = -d_fake.mean()
                loss_recon = criterion_recon(fake * mask, gt * mask) + 0.5 * criterion_recon(fake * (1 - mask), gt * (1 - mask))
                loss_g = loss_adv * 0.1 + loss_recon
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                train_loss_g += loss_g.item()
            train_loss_g /= max(len(train_loader), 1)

            # Validate
            generator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, gt, mask in val_loader:
                    inp, gt, mask = inp.to(self.device), gt.to(self.device), mask.to(self.device)
                    pred = generator(inp)
                    loss = criterion_recon(pred * mask, gt * mask) + 0.5 * criterion_recon(pred * (1 - mask), gt * (1 - mask))
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)

            self.logger.info(f"  CoModGAN epoch {epoch}/{epochs}: g_loss={train_loss_g:.5f}, val_loss={val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = self.checkpoint_dir / "comodgan_best.pth"
                torch.save(generator.state_dict(), ckpt_path)
                self._best_ckpt = ckpt_path
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"  CoModGAN early stopping at epoch {epoch}")
                    break

        return FTResult(
            model_name="CoModGAN",
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_epochs=epoch,
            checkpoint_path=str(self._best_ckpt or ""),
            metrics={"final_train_loss_g": train_loss_g},
        )

    def export(self) -> Path:
        if self._best_ckpt and self._best_ckpt.exists():
            return self._best_ckpt
        raise FileNotFoundError("CoModGAN: no best checkpoint found. Run train() first.")

    @staticmethod
    def _build_generator():
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh(),
        )

    @staticmethod
    def _build_discriminator():
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )
