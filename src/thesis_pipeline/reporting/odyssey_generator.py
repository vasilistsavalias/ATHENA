import logging
from pathlib import Path
import base64

class OdysseyReportGenerator:
    def __init__(self, hero_dir: Path):
        self.hero_dir = hero_dir
        self.logger = logging.getLogger(__name__)

    def _get_base64_image(self, image_path: Path):
        if not image_path.exists():
            return None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def generate_report(self):
        if not self.hero_dir.exists():
            self.logger.warning(f"Hero directory not found: {self.hero_dir}")
            return

        hero_folders = [d for d in self.hero_dir.iterdir() if d.is_dir()]
        
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>The Odyssey: Artifact Journey</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background-color: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
                .container { max-width: 1400px; margin: 0 auto; }
                h1 { text-align: center; color: #e94560; font-size: 3em; margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px; }
                .hero-section { background-color: #16213e; border-radius: 15px; margin-bottom: 50px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
                .hero-title { color: #0f3460; background: #e94560; padding: 10px 20px; border-radius: 5px; display: inline-block; margin-bottom: 20px; font-weight: bold; }
                .journey-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; align-items: start; }
                .stage-card { background: #0f3460; padding: 15px; border-radius: 10px; text-align: center; transition: transform 0.3s; }
                .stage-card:hover { transform: translateY(-5px); }
                .stage-card img { max-width: 100%; border-radius: 5px; border: 2px solid #533483; }
                .stage-title { margin-top: 10px; color: #e94560; font-weight: bold; }
                .stage-desc { font-size: 0.9em; color: #a2a8d3; margin-top: 5px; text-align: left; max-height: 150px; overflow-y: auto; }
                .arrow { font-size: 2em; color: #e94560; align-self: center; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>The Odyssey</h1>
                <p style="text-align: center; margin-bottom: 50px;">Tracing the transformation of ancient artifacts through the restoration pipeline.</p>
        """

        for folder in hero_folders:
            stem = folder.name
            
            # Paths
            p_filtered = folder / "02_filtered.png"
            p_caption = folder / "04_caption.txt"
            p_masked = folder / "08_masked_input.png"
            p_restored = folder / "11_restored.png"
            
            # Read Data
            img_filtered = self._get_base64_image(p_filtered)
            img_masked = self._get_base64_image(p_masked)
            img_restored = self._get_base64_image(p_restored)
            
            caption_text = "No caption generated."
            if p_caption.exists():
                with open(p_caption, "r", encoding="utf-8") as f:
                    caption_text = f.read()

            html_content += f"""
                <div class="hero-section">
                    <div class="hero-title">Artifact: {stem}</div>
                    <div class="journey-grid">
                        
                        <!-- Stage 1/2: Discovery & Filtered -->
                        <div class="stage-card">
                            <div class="stage-title">1. Discovery</div>
                            {f'<img src="data:image/png;base64,{img_filtered}">' if img_filtered else '<p>Missing</p>'}
                            <div class="stage-desc">Identified and cropped from raw collection.</div>
                        </div>

                        <!-- Stage 4: Understanding -->
                        <div class="stage-card">
                            <div class="stage-title">2. Understanding</div>
                            <div class="stage-desc" style="background: #1a1a2e; padding: 10px; border-radius: 5px; font-family: monospace;">
                                {caption_text[:300] + '...' if len(caption_text) > 300 else caption_text}
                            </div>
                        </div>

                        <!-- Stage 8: Damage Simulation -->
                        <div class="stage-card">
                            <div class="stage-title">3. The Damage</div>
                            {f'<img src="data:image/png;base64,{img_masked}">' if img_masked else '<p>Missing</p>'}
                            <div class="stage-desc">Artificial damage applied for training.</div>
                        </div>

                        <!-- Stage 11: Restoration -->
                        <div class="stage-card">
                            <div class="stage-title">4. Restoration</div>
                            {f'<img src="data:image/png;base64,{img_restored}">' if img_restored else '<p>Missing</p>'}
                            <div class="stage-desc">Restored using Stable Diffusion guided by the caption.</div>
                        </div>

                    </div>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """
        
        output_file = self.hero_dir / "odyssey_report.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self.logger.info(f"Odyssey Report generated at: {output_file}")
