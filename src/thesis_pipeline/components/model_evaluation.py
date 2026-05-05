# src/thesis_pipeline/components/model_evaluation.py
import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from thesis_pipeline.visualization import ThesisPlotter
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, config, test_data_dir: Path, output_dir: Path, model_dir: Path, hero_tracker=None):
        self.config = config
        self.test_data_dir = test_data_dir
        self.device = config.device
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.plotter = ThesisPlotter(self.output_dir)
        self.hero_tracker = hero_tracker
        
        try:
            import lpips
            self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
            self.use_lpips = True
        except ImportError:
            self.logger.warning("LPIPS not installed. Skipping perceptual metric.")
            self.use_lpips = False

    def _load_pipeline(self):
        """Loads the trained model into an inpainting pipeline."""
        try:
            model_path = self.model_dir / "unet_final"
            
            # Load the trained UNet explicitly
            unet = UNet2DConditionModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )

            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                unet=unet,
                torch_dtype=torch.float32,
            )
            pipeline.to(self.device)
            self.logger.info(f"Successfully loaded pipeline with UNet from: {model_path}")
            return pipeline
        except Exception as e:
            self.logger.error(f"Failed to load the inpainting pipeline. Error: {e}")
            raise

    def evaluate(self):
        """Runs the full evaluation process."""
        pipeline = self._load_pipeline()
        vae = pipeline.vae
        
        image_dir = self.test_data_dir / 'ground_truth'
        mask_dir = self.test_data_dir / 'masks'
        caption_dir = self.test_data_dir / 'captions'

        image_files = sorted([p for p in image_dir.glob('*.png') if p.is_file()])
        mask_files = sorted([p for p in mask_dir.glob('*.png') if p.is_file()])

        if not image_files or not mask_files:
            self.logger.warning("Test data not found. Skipping evaluation.")
            return

        num_samples = self.config.num_samples_to_evaluate
        if num_samples > 0 and num_samples < len(image_files):
            image_files = image_files[:num_samples]
            mask_files = mask_files[:num_samples]
        
        self.logger.info(f"Evaluating on {len(image_files)} samples.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        results = []
        latent_vectors = []
        
        # Lists for Grid Visualization
        grid_originals = []
        grid_maskeds = []
        grid_restoreds = []
        
        generator = torch.Generator(device=self.device).manual_seed(0)

        for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Evaluating"):
            try:
                # Create per-sample directory
                sample_name = img_path.stem
                sample_dir = samples_dir / sample_name
                sample_dir.mkdir(exist_ok=True)

                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_path).convert("RGB")
                
                # Load Caption
                caption = ""
                cap_path = caption_dir / f"{img_path.stem}.txt"
                if cap_path.exists():
                    with open(cap_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()

                with torch.no_grad():
                    restored_image = pipeline(
                        prompt=caption, image=original_image, mask_image=mask_image,
                        num_inference_steps=self.config.num_inference_steps,
                        generator=generator,
                    ).images[0]
                    
                    # Latent Space Extraction (for t-SNE)
                    # Resize to 512x512 for consistent latent size
                    img_tensor = original_image.resize((512, 512))
                    img_tensor = torch.tensor(np.array(img_tensor)).permute(2,0,1).float().unsqueeze(0).to(self.device) / 127.5 - 1.0
                    latents = vae.encode(img_tensor).latent_dist.sample()
                    latent_vectors.append(latents.cpu().numpy().flatten())

                original_np = np.array(original_image)
                restored_np = np.array(restored_image)
                masked_image_np = np.array(Image.fromarray(original_np * (np.array(mask_image) < 128)))
                
                # Hero Log
                if self.hero_tracker:
                    self.hero_tracker.log_image(restored_image, "11_restored", img_path.name)
                
                # Collect for Grid (First 5)
                if len(grid_originals) < 5:
                    grid_originals.append(original_np)
                    grid_maskeds.append(masked_image_np)
                    grid_restoreds.append(restored_np)
                
                # Metrics
                current_psnr = psnr(original_np, restored_np, data_range=255)
                current_ssim = ssim(original_np, restored_np, data_range=255, channel_axis=2)
                
                # Mask Coverage
                mask_np = np.array(mask_image)
                coverage = np.sum(mask_np > 128) / mask_np.size
                
                metrics = {
                    'filename': img_path.name, 
                    'psnr': current_psnr, 
                    'ssim': current_ssim,
                    'mask_coverage': coverage,
                    'caption_length': len(caption.split())
                }
                
                if self.use_lpips:
                    # LPIPS expects [-1, 1] tensors NCHW
                    orig_t = torch.tensor(original_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 127.5 - 1
                    rest_t = torch.tensor(restored_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 127.5 - 1
                    current_lpips = self.loss_fn_alex(orig_t, rest_t).item()
                    metrics['lpips'] = current_lpips
                
                results.append(metrics)

                # --- Save Per-Sample Artifacts ---
                # 1. Individual Images
                original_image.save(sample_dir / "original.png")
                restored_image.save(sample_dir / "restored.png")
                Image.fromarray(masked_image_np).save(sample_dir / "masked_input.png")
                mask_image.save(sample_dir / "mask.png")

                # 2. Metrics JSON
                import json
                with open(sample_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)

                # 3. Composite View
                masked_image_pil = Image.fromarray(masked_image_np)
                comparison_img = Image.new('RGB', (original_image.width * 3, original_image.height))
                comparison_img.paste(original_image, (0, 0))
                comparison_img.paste(masked_image_pil, (original_image.width, 0))
                comparison_img.paste(restored_image, (original_image.width * 2, 0))
                comparison_img.save(sample_dir / "composite.png")
                
                # 4. Residual Map
                self.plotter.plot_residual_map(
                    original_np, restored_np, 
                    f"Error Map: {img_path.name}", 
                    str(sample_dir / "error_map") # plot_residual_map adds .png
                )

            except Exception as e:
                self.logger.error(f"Failed on sample {img_path.name}. Error: {e}")

        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "evaluation_metrics.csv", index=False)
            
            # Generate Advanced Plots
            self.logger.info("Generating advanced evaluation plots...")
            
            # 0. Comparison Grid
            if grid_originals:
                self.plotter.plot_comparison_grid(grid_originals, grid_maskeds, grid_restoreds, "comparison_grid_random_5")
            
            # 1. Distributions (Violin)
            self.plotter.plot_violin(df['psnr'], "PSNR Distribution", "PSNR (dB)", "psnr_violin", color='primary')
            self.plotter.plot_violin(df['ssim'], "SSIM Distribution", "SSIM", "ssim_violin", color='secondary')
            
            # 2. Bivariate Analysis
            self.plotter.plot_regression_scatter(df['mask_coverage'], df['psnr'], "Impact of Damage Size on Quality", "Mask Coverage (%)", "PSNR", "psnr_vs_coverage")
            self.plotter.plot_regression_scatter(df['caption_length'], df['psnr'], "Impact of Caption Detail on Quality", "Caption Word Count", "PSNR", "psnr_vs_caption")
            
            # 3. Latent Space t-SNE
            if len(latent_vectors) > 5:
                try:
                    X = np.array(latent_vectors)
                    # PCA to 50 dims first
                    pca = PCA(n_components=min(50, len(X)))
                    X_pca = pca.fit_transform(X)
                    # t-SNE to 2 dims
                    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
                    X_embedded = tsne.fit_transform(X_pca)
                    
                    self.plotter.plot_scatter(
                        pd.Series(X_embedded[:,0]), pd.Series(X_embedded[:,1]), 
                        "Latent Space Visualization (t-SNE colored by Quality)", "Dim 1", "Dim 2", "latent_tsne",
                        hue=df['psnr']
                    )
                except Exception as e:
                    self.logger.warning(f"Skipping t-SNE due to error: {e}")

            summary = f"Samples: {len(df)}\nAvg PSNR: {df['psnr'].mean():.4f}\nAvg SSIM: {df['ssim'].mean():.4f}"
            with open(self.output_dir / "summary_report.txt", 'w') as f:
                f.write(summary)
            
            # 4. Stratified Analysis (by Coverage)
            self._analyze_strata(df)
            
            self.logger.info(f"Evaluation Complete. {summary}")
        else:
            self.logger.warning("No results generated during evaluation.")

    def _analyze_strata(self, df: pd.DataFrame):
        """Groups results by difficulty levels."""
        # Define bins for mask coverage
        bins = [0, 0.1, 0.25, 1.1] # Adjusted upper bin to 1.1 to include 1.0
        labels = ['Low (<10%)', 'Medium (10-25%)', 'High (>25%)']
        df['difficulty'] = pd.cut(df['mask_coverage'], bins=bins, labels=labels)
        
        strata_stats = df.groupby('difficulty', observed=True)[['psnr', 'ssim']].agg(['mean', 'std', 'count'])
        strata_stats.to_csv(self.output_dir / "stratified_metrics.csv")
        
        # Visual - Only plot if we have at least one valid category with data
        if not df['difficulty'].isna().all() and len(df) > 1:
            try:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x='difficulty', y='psnr', hue='difficulty', legend=False)
                plt.title("Model Performance vs Damage Severity")
                plt.savefig(self.output_dir / "psnr_by_difficulty.png")
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to plot stratified boxplot: {e}")
        
        self.logger.info("Saved stratified analysis.")
