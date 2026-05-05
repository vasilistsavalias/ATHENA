import pandas as pd
import json
from pathlib import Path
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)

class CaptionQualityAnalyzer:
    """
    Analyzes generated captions for vocabulary richness and consistency.
    """
    def __init__(self, captions_dir: Path, output_dir: Path):
        self.captions_dir = captions_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self):
        caption_files = list(self.captions_dir.glob("*.txt"))
        if not caption_files:
            logger.warning(f"No captions found in {self.captions_dir}")
            return

        all_text = []
        lengths = []
        
        for f in caption_files:
            content = f.read_text(encoding='utf-8')
            all_text.append(content)
            lengths.append(len(content.split()))

        # 1. Vocabulary Analysis
        words = re.findall(r'\w+', " ".join(all_text).lower())
        vocab_counts = Counter(words)
        
        # Focus on archeological/artistic terms (subset)
        key_terms = ["amphora", "vase", "pottery", "greek", "black", "red", "figure", "decoration", "mythology"]
        term_freq = {term: vocab_counts.get(term, 0) for term in key_terms}

        with open(self.output_dir / "caption_vocabulary.json", 'w') as f:
            json.dump({
                "total_words": len(words),
                "unique_words": len(vocab_counts),
                "key_term_frequency": term_freq,
                "top_20_words": vocab_counts.most_common(20)
            }, f, indent=4)

        # 2. Length Statistics
        stats = {
            "count": len(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }
        with open(self.output_dir / "caption_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)

        logger.info(f"Caption quality analysis saved to {self.output_dir}")
