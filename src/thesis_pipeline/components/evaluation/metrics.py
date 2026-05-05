import warnings
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.feature import local_binary_pattern
from scipy.stats import wasserstein_distance
from transformers import CLIPProcessor, CLIPModel
import logging
import time

class ThesisMetrics:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        
        # Initialize LPIPS
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.use_lpips = True
        except ImportError:
            self.logger.warning("LPIPS not installed. Metric will be skipped.")
            self.use_lpips = False
            
        # Initialize CLIP
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.use_clip = True
        except Exception as e:
            self.logger.warning(f"CLIP not loaded: {e}. Metric will be skipped.")
            self.use_clip = False

    # ------------------------------------------------------------------
    # Masked-region helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        """Return (y_min, y_max, x_min, x_max) bounding box of non-zero mask region."""
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not rows.any():
            return 0, mask.shape[0], 0, mask.shape[1]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return int(y_min), int(y_max + 1), int(x_min), int(x_max + 1)

    # ------------------------------------------------------------------
    # Core metrics — all accept an optional mask for masked-region computation
    #
    # NOTE: When a mask is supplied, PSNR/SSIM/LPIPS/Color/Pattern
    #       operate on the bounding-box crop (not pixel-level masking).
    #       This is intentional: SSIM and LPIPS require spatially
    #       coherent patches, and bbox cropping preserves local
    #       structure while focusing on the damaged region.
    #       Scores may be slightly inflated by surrounding context
    #       pixels; document this limitation in the thesis.
    # ------------------------------------------------------------------

    def calculate_psnr(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            # Crop both images to the mask bounding box for spatial context
            y0, y1, x0, x1 = self._mask_bbox(mask)
            gt_crop = gt[y0:y1, x0:x1]
            pred_crop = pred[y0:y1, x0:x1]
            return skimage_psnr(gt_crop, pred_crop, data_range=255)
        return skimage_psnr(gt, pred, data_range=255)

    def calculate_ssim(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            y0, y1, x0, x1 = self._mask_bbox(mask)
            gt_crop = gt[y0:y1, x0:x1]
            pred_crop = pred[y0:y1, x0:x1]
            # Ensure minimum size for SSIM window
            min_side = min(gt_crop.shape[0], gt_crop.shape[1])
            win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
            if win_size < 3:
                win_size = 3
            if min_side < win_size:
                return skimage_ssim(gt, pred, data_range=255, channel_axis=2)
            return skimage_ssim(gt_crop, pred_crop, data_range=255, channel_axis=2, win_size=win_size)
        return skimage_ssim(gt, pred, data_range=255, channel_axis=2)

    def calculate_lpips(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None):
        if not self.use_lpips:
            warnings.warn("LPIPS not available; returning NaN. Install 'lpips' for real scores.", stacklevel=2)
            return float('nan')
        
        if mask is not None:
            # Crop to mask bounding box for LPIPS (needs spatial structure)
            y0, y1, x0, x1 = self._mask_bbox(mask)
            gt = gt[y0:y1, x0:x1]
            pred = pred[y0:y1, x0:x1]

        # Normalize to [-1, 1] and NCHW
        gt_t = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
        pred_t = torch.tensor(pred).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
        
        with torch.no_grad():
            val = self.lpips_fn(gt_t, pred_t)
        return val.item()

    def calculate_clip_score(self, image: Image.Image, text: str):
        if not self.use_clip or not text:
            if not self.use_clip:
                warnings.warn("CLIP not available; returning NaN.", stacklevel=2)
            return float('nan')
        
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # This calculates similarity
            logits_per_image = outputs.logits_per_image # image-text similarity score
            score = logits_per_image.item() / 100.0 # Normalize roughly to 0-1 range (CLIP logits are scaled)
            
        return score

    def calculate_color_fidelity(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None):
        """
        Computes Wasserstein distance between color distributions in CIE-Lab space.
        Lower distance = better fidelity. Returned as a score in (0, 1] via 1/(1+d).

        When *mask* is provided, only pixels inside the mask are compared.
        """
        if mask is not None:
            y0, y1, x0, x1 = self._mask_bbox(mask)
            gt = gt[y0:y1, x0:x1]
            pred = pred[y0:y1, x0:x1]

        gt_lab = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
        pred_lab = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
        
        # Per-channel Wasserstein distance on the Lab histograms.
        # Weight L channel higher (perceptually dominant) vs. a/b chrominance.
        channel_weights = [0.5, 0.25, 0.25]  # L, a, b
        distances = []
        for i in range(3):  # L, a, b channels
            hist_gt, bin_edges = np.histogram(gt_lab[:, :, i].ravel(), bins=50, range=(0, 255), density=True)
            hist_pred, _ = np.histogram(pred_lab[:, :, i].ravel(), bins=50, range=(0, 255), density=True)
            # Use bin centers as the "values" so wasserstein_distance
            # computes EMD with the correct spatial scale.
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            d = wasserstein_distance(bin_centers, bin_centers, hist_gt, hist_pred)
            distances.append(d * channel_weights[i])
            
        weighted_dist = np.sum(distances)
        fidelity = 1.0 / (1.0 + weighted_dist)
        return fidelity

    def calculate_pattern_preservation(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None):
        """
        Computes LBP Histogram intersection. Higher is better.
        When *mask* is provided, only the mask bounding box region is compared.
        """
        if mask is not None:
            y0, y1, x0, x1 = self._mask_bbox(mask)
            gt = gt[y0:y1, x0:x1]
            pred = pred[y0:y1, x0:x1]

        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        
        radius = 1
        n_points = 8 * radius
        
        lbp_gt = local_binary_pattern(gt_gray, n_points, radius, method='uniform')
        lbp_pred = local_binary_pattern(pred_gray, n_points, radius, method='uniform')
        
        hist_gt, _ = np.histogram(lbp_gt.ravel(), bins=10, range=(0,10), density=True)
        hist_pred, _ = np.histogram(lbp_pred.ravel(), bins=10, range=(0,10), density=True)
        
        # Histogram Intersection
        intersection = np.minimum(hist_gt, hist_pred)
        score = np.sum(intersection)
        return score

    # ------------------------------------------------------------------
    # FID / KID  — Dataset-level distributional metrics
    # ------------------------------------------------------------------
    def _get_inception_features(self, images: list, batch_size: int = 16) -> np.ndarray:
        """Extract Inception-V3 pool3 features (2048-d) for a list of np.ndarray images.

        Falls back to a lightweight CLIP-based feature extractor when
        torchvision InceptionV3 weights are unavailable.
        """
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            from torchvision import transforms

            model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            model.fc = torch.nn.Identity()  # remove classifier head → 2048-d features
            model.eval().to(self.device)

            preprocess = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            all_feats = []
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i + batch_size]
                tensors = torch.stack([
                    preprocess(Image.fromarray(img) if isinstance(img, np.ndarray) else img)
                    for img in batch_imgs
                ]).to(self.device)
                with torch.no_grad():
                    feats = model(tensors)
                all_feats.append(feats.cpu().numpy())
            return np.concatenate(all_feats, axis=0)

        except Exception as e:
            self.logger.warning(f"InceptionV3 unavailable ({e}); using CLIP features for FID/KID.")
            return self._get_clip_features(images, batch_size)

    def _get_clip_features(self, images: list, batch_size: int = 16) -> np.ndarray:
        """Fallback: use CLIP image encoder for 512-d features."""
        if not self.use_clip:
            raise RuntimeError("Neither InceptionV3 nor CLIP available for FID/KID.")
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            pil_imgs = [
                Image.fromarray(img) if isinstance(img, np.ndarray) else img
                for img in batch_imgs
            ]
            inputs = self.clip_processor(images=pil_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    def calculate_fid(self, real_images: list, generated_images: list) -> float:
        """Fréchet Inception Distance between two sets of images.

        FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)
        Lower is better.
        """
        from scipy.linalg import sqrtm

        feats_real = self._get_inception_features(real_images)
        feats_gen = self._get_inception_features(generated_images)

        mu_r, sigma_r = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
        mu_g, sigma_g = feats_gen.mean(axis=0), np.cov(feats_gen, rowvar=False)

        diff = mu_r - mu_g
        covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
        return fid

    def calculate_kid(self, real_images: list, generated_images: list,
                      subset_size: int = 50, n_subsets: int = 10) -> tuple:
        """Kernel Inception Distance (polynomial kernel, degree=3).

        Returns (kid_mean, kid_std) — averaged over random subsets.
        Lower is better.
        """
        feats_real = self._get_inception_features(real_images)
        feats_gen = self._get_inception_features(generated_images)

        n = min(subset_size, len(feats_real), len(feats_gen))
        if n < 2:
            return float('nan'), float('nan')

        def _poly_kernel(x, y, degree=3, gamma=None, coef0=1.0):
            if gamma is None:
                gamma = 1.0 / x.shape[1]
            return (gamma * (x @ y.T) + coef0) ** degree

        kid_values = []
        rng = np.random.RandomState(42)
        for _ in range(n_subsets):
            idx_r = rng.choice(len(feats_real), n, replace=False)
            idx_g = rng.choice(len(feats_gen), n, replace=False)
            fr = feats_real[idx_r]
            fg = feats_gen[idx_g]
            k_rr = _poly_kernel(fr, fr)
            k_gg = _poly_kernel(fg, fg)
            k_rg = _poly_kernel(fr, fg)
            kid = (k_rr.sum() / (n * (n - 1)) + k_gg.sum() / (n * (n - 1))
                   - 2 * k_rg.mean())
            kid_values.append(float(kid))

        return float(np.mean(kid_values)), float(np.std(kid_values))
