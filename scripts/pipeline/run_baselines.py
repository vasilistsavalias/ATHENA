import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

from thesis_pipeline.components.evaluation.baselines import TeleaInpainter, VanillaSDInpainter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_baseline(inpainter, name, test_dir, output_dir, use_prompt=False):
    """
    Runs a baseline on the test set and calculates metrics.
    """
    gt_dir = test_dir / "ground_truth"
    mask_dir = test_dir / "masks"
    caption_dir = test_dir / "captions"
    
    out_images_dir = output_dir / name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(gt_dir.glob("*.png"))
    metrics = []
    
    logger.info(f"Running Baseline: {name} on {len(files)} images...")
    
    for f in tqdm(files):
        try:
            # Load Data
            img = Image.open(f).convert("RGB")
            mask = Image.open(mask_dir / f.name).convert("L")
            
            prompt = "artifact"
            if use_prompt:
                cap_file = caption_dir / f"{f.stem}.txt"
                if cap_file.exists():
                    with open(cap_file, "r") as cf:
                        prompt = cf.read()

            # Run Inference
            if use_prompt:
                restored = inpainter.inpaint(img, mask, prompt=prompt)
            else:
                restored = inpainter.inpaint(img, mask)
            
            # Save Result
            restored.save(out_images_dir / f.name)
            
            # Compute Metrics
            gt_np = np.array(img)
            res_np = np.array(restored)
            
            p_val = psnr(gt_np, res_np, data_range=255)
            s_val = ssim(gt_np, res_np, data_range=255, channel_axis=2)
            
            metrics.append({
                "filename": f.name,
                "method": name,
                "psnr": p_val,
                "ssim": s_val
            })
            
        except Exception as e:
            logger.error(f"Failed on {f.name}: {e}")

    # Save Stats
    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / f"{name}_metrics.csv", index=False)
    logger.info(f"Finished {name}. Avg PSNR: {df['psnr'].mean():.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test split (containing ground_truth/masks)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--skip-sd", action="store_true", help="Skip Stable Diffusion (Slow/GPU needed)")
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    
    # 1. Telea (OpenCV)
    telea = TeleaInpainter()
    evaluate_baseline(telea, "telea", test_dir, out_dir, use_prompt=False)
    
    # 2. Vanilla SD
    if not args.skip_sd:
        sd = VanillaSDInpainter()
        evaluate_baseline(sd, "stable_diffusion_vanilla", test_dir, out_dir, use_prompt=True)

if __name__ == "__main__":
    main()
