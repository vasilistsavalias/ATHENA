import json
from pathlib import Path
import base64

def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_rating_html(output_path, baseline_dir, target_images):
    """
    Generates a standalone HTML file for human rating of restorations.
    Displays: Original, Masked (Input), Telea, Vanilla SD, Ours.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thesis Rating Tool: Artifact Restoration</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f4f4f4; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .row { display: flex; flex-wrap: wrap; margin-bottom: 40px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
            .item { text-align: center; margin: 10px; flex: 1; min-width: 200px; }
            img { width: 100%; border-radius: 4px; border: 1px solid #ccc; }
            h3 { font-size: 14px; color: #555; }
            .rating-controls { margin-top: 10px; background: #eee; padding: 10px; border-radius: 4px; }
            .id-label { font-weight: bold; background: #333; color: white; padding: 2px 8px; border-radius: 4px; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            textarea { width: 100%; height: 100px; margin-top: 20px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ancient Greek Pottery Restoration: Human Evaluation</h1>
            <p>Compare the restoration quality across different methods. Use the CSV area at the bottom to export your ratings.</p>
            <div id="content">
    """

    baseline_path = Path(baseline_dir)
    
    # We loop through unique image names found in baseline results
    # For now, let's assume we use the first 5 for the tool
    for i, img_path in enumerate(target_images):
        stem = Path(img_path).stem
        
        # Prepare Rows
        html_content += f"""
        <div class="row" data-id="{stem}">
            <div class="item">
                <span class="id-label">ID: {stem}</span>
                <h3>Original (GT)</h3>
                <img src="data:image/jpeg;base64,{get_image_base64(img_path)}">
            </div>
            <div class="item">
                <h3>Telea (Classical)</h3>
                <img src="data:image/png;base64,{get_image_base64(baseline_path / 'Telea' / 'Unconditional' / f'{stem}.png')}">
                <div class="rating-controls">
                    Rating (1-5): <input type="number" min="1" max="5" class="rate" data-method="telea">
                </div>
            </div>
            <div class="item">
                <h3>Vanilla SD (Raw Text)</h3>
                <img src="data:image/png;base64,{get_image_base64(baseline_path / 'VanillaSD' / 'Raw' / f'{stem}.png')}">
                <div class="rating-controls">
                    Rating (1-5): <input type="number" min="1" max="5" class="rate" data-method="vanilla_sd">
                </div>
            </div>
            <div class="item">
                <h3>Our Model (Enriched)</h3>
                <p><i>Pending Run...</i></p>
            </div>
        </div>
        """

    html_content += """
            </div>
            <button onclick="exportCSV()">Generate CSV Export</button>
            <textarea id="csvOutput" placeholder="Your rating CSV will appear here..."></textarea>
        </div>

        <script>
            function exportCSV() {
                let csv = "filename,method,rating\n";
                let rows = document.querySelectorAll('.row');
                rows.forEach(row => {
                    let id = row.getAttribute('data-id');
                    let inputs = row.querySelectorAll('.rate');
                    inputs.forEach(input => {
                        let method = input.getAttribute('data-method');
                        let val = input.value || "0";
                        csv += `${id},${method},${val}\n`;
                    });
                });
                document.getElementById('csvOutput').value = csv;
            }
        </script>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Rating tool generated: {output_path}")

if __name__ == "__main__":
    # Example usage
    # This requires Stage 04 to have run already
    print("Rating tool script initialized. Use generate_rating_html in the pipeline.")