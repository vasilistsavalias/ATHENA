"""
Generate period coverage progression chart showing Raw → BLIP2 → Qwen-refined.

Shows how period/style mentions evolve through the caption pipeline:
- Raw metadata (museum records)
- BLIP2 enriched (vision-only descriptions)
- Qwen-refined (vision + metadata synthesis)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("outputs/07_caption_refinement")
CAPTIONS_RAW = Path("outputs/06_caption_generation/captions_raw.json")
CAPTIONS_BLIP2 = Path("outputs/06_caption_generation/captions_enriched.json")
CAPTIONS_QWEN = Path("outputs/07_caption_refinement")

# Period terminology (archaeological dating + visual style markers)
PERIOD_TERMS = [
    'geometric', 'archaic', 'classical', 'hellenistic',
    'red-figure', 'black-figure', 'white-ground',
    'attic', 'corinthian', 'period', 'century', 'b.c.', 'bc'
]

def count_period_coverage(source_path, is_json=True):
    """Count how many captions contain period/style terms."""
    if is_json:
        with open(source_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        
        matches = sum(
            1 for c in captions.values()
            if any(term in str(c).lower() for term in PERIOD_TERMS)
        )
        total = len(captions)
    else:
        # Directory of .txt files
        caption_files = list(source_path.glob('*.txt'))
        matches = 0
        for cap_file in caption_files:
            try:
                text = cap_file.read_text(encoding='utf-8', errors='ignore').lower()
                if any(term in text for term in PERIOD_TERMS):
                    matches += 1
            except:
                pass
        total = len(caption_files)
    
    return matches, total, (matches / total * 100) if total > 0 else 0

def create_period_coverage_chart():
    """Generate bar chart showing period coverage progression."""
    
    # Calculate coverage at each stage
    print("Calculating period coverage across pipeline stages...")
    
    raw_matches, raw_total, raw_pct = count_period_coverage(CAPTIONS_RAW, is_json=True)
    blip2_matches, blip2_total, blip2_pct = count_period_coverage(CAPTIONS_BLIP2, is_json=True)
    qwen_matches, qwen_total, qwen_pct = count_period_coverage(CAPTIONS_QWEN, is_json=False)
    
    print(f"Raw metadata: {raw_matches}/{raw_total} = {raw_pct:.1f}%")
    print(f"BLIP2 enriched: {blip2_matches}/{blip2_total} = {blip2_pct:.1f}%")
    print(f"Qwen-refined: {qwen_matches}/{qwen_total} = {qwen_pct:.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    stages = ['Raw Metadata\n(Museum Records)', 'BLIP2 Enriched\n(Vision-Only)', 'Qwen-Refined\n(Vision + Metadata)']
    percentages = [raw_pct, blip2_pct, qwen_pct]
    colors = ['#7E7E7E', '#E8A628', '#2E7D32']  # Gray, Orange, Green
    
    bars = ax.bar(stages, percentages, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on bars
    for bar, pct, count, total in zip(bars, percentages, 
                                       [raw_matches, blip2_matches, qwen_matches],
                                       [raw_total, blip2_total, qwen_total]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{pct:.1f}%\n({count:,}/{total:,})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Period/Style Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Caption Pipeline Stage', fontsize=12, fontweight='bold')
    ax.set_title('Period Coverage Progression Through Caption Pipeline', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add annotation explaining the drop at BLIP2
    ax.annotate('Vision models can\'t date pottery\nfrom images alone',
                xy=(1, blip2_pct), xytext=(1, 30),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, color='red', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Add annotation explaining Qwen's recovery
    ax.annotate('Qwen synthesizes vision\n+ metadata context',
                xy=(2, qwen_pct), xytext=(2, 75),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to both Stage 06 and Stage 07 artifacts
    output_path_06 = Path("outputs/06_caption_generation/period_coverage_progression.png")
    output_path_07 = OUTPUT_DIR / "period_coverage_progression.png"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path_06.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path_07, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_06, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Period coverage chart saved to:")
    print(f"   - {output_path_06}")
    print(f"   - {output_path_07}")
    
    plt.close()

if __name__ == "__main__":
    create_period_coverage_chart()
