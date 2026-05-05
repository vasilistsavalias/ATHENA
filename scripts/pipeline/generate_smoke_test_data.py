import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def generate_mock_data(output_dir, count=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {count} mock images in {output_dir}...")
    
    for i in range(count):
        # Create random image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Draw something recognizable
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 400], fill=(i*10, 100, 200))
        draw.text((200, 250), f"Mock {i}", fill=(255, 255, 255))
        
        filename = f"mock_{i:03d}.png"
        img.save(output_dir / filename)
        
        # Create corresponding caption
        with open(output_dir / f"mock_{i:03d}.txt", "w") as f:
            f.write(f"A photo of a mock ancient vase number {i}")

    print("Mock data generation complete.")

if __name__ == "__main__":
    # Target the processed directory that Stage 7 expects
    target_dir = "smoke_tests/last_run/data/intermediate/06_processed"
    generate_mock_data(target_dir)
