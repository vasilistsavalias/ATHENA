
# src/thesis_pipeline/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Iterable
from matplotlib.colors import ListedColormap, BoundaryNorm
from .style import ThesisStyle

class ThesisPlotter:
    """
    Library of standardized plotting functions for the thesis.
    """
    DEFAULT_MODEL_ORDER = [
        'Telea',
        'Navier-Stokes',
        'Vanilla SD',
        'FT-SD',
        'FT-SD+TTA',
        'LaMa',
        'MAT',
        'CoModGAN',
    ]

    QUALITATIVE_METHOD_ORDER = [
        'Original',
        'Masked',
        'Telea',
        'Navier-Stokes',
        'Vanilla SD',
        'FT-SD',
        'FT-SD+TTA',
        'LaMa',
        'MAT',
        'CoModGAN',
    ]
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ThesisStyle.set_style()

    def _ordered_models(self, present_models: Iterable[str], include_tta: bool = True) -> List[str]:
        present = [str(m) for m in present_models if isinstance(m, str) and m.strip()]
        preferred = [m for m in self.DEFAULT_MODEL_ORDER if include_tta or m != 'FT-SD+TTA']
        ordered = [m for m in preferred if m in present]
        extras = sorted([m for m in present if m not in ordered])
        return ordered + extras

    def save_plot(self, filename: str, close: bool = True):
        """
        Saves the current figure as a high-resolution PNG.
        """
        # Ensure filename has no extension
        name = Path(filename).stem
        
        # Save PNG
        png_path = self.output_dir / f"{name}.png"
        plt.savefig(png_path, format='png', dpi=300)
        
        if close:
            plt.close()
        
        return png_path

    def plot_histogram(self, data: pd.Series, title: str, xlabel: str, filename: str, bins: int = 20, color: str = 'primary'):
        """
        Generates a standardized histogram with density curve (KDE).
        """
        plt.figure()
        color_hex = ThesisStyle.PALETTE.get(color, ThesisStyle.COLORS['indigo'])
        
        sns.histplot(data, bins=bins, kde=True, color=color_hex, edgecolor='white', linewidth=1.2, alpha=0.8)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        
        # Add mean line
        mean_val = data.mean()
        plt.axvline(mean_val, color=ThesisStyle.COLORS['wine'], linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.legend()
        
        self.save_plot(filename)

    def plot_scatter(self, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, hue: Optional[pd.Series] = None):
        """
        Generates a standardized scatter plot.
        """
        plt.figure()
        
        # Use a qualitative palette if hue is provided
        palette = None
        if hue is not None:
             palette = ThesisStyle.get_palette_list()[:len(hue.unique())]

        sns.scatterplot(
            x=x, y=y, hue=hue, 
            palette=palette, 
            alpha=0.7, s=80, edgecolor='white', linewidth=0.5
        )
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if hue is not None:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        self.save_plot(filename)

    def plot_training_curves(self, df: pd.DataFrame, filename: str = "training_curves"):
        """
        Plots Training vs Validation Loss from a log DataFrame.
        Expected columns: 'epoch', 'train_loss', 'val_loss' (optional).
        """
        plt.figure()
        
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', 
                 color=ThesisStyle.PALETTE['train'], marker='o', markersize=4)
        
        if 'val_loss' in df.columns and not df['val_loss'].isna().all():
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', 
                     color=ThesisStyle.PALETTE['val'], marker='s', markersize=4)
            
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        
        self.save_plot(filename)

    def plot_image_grid(self, images: List[np.ndarray], titles: Optional[List[str]] = None, rows: int = 5, cols: int = 5, filename: str = "image_grid"):
        """
        Plots a grid of images.
        """
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < len(images):
                ax.imshow(images[i])
                ax.axis('off')
                if titles and i < len(titles):
                    ax.set_title(titles[i], fontsize=10)
            else:
                ax.axis('off')
        
        plt.tight_layout()
        self.save_plot(filename)

    def plot_violin(self, data: pd.Series, title: str, xlabel: str, filename: str, color: str = 'primary'):
        """
        Generates a standardized violin plot.
        """
        plt.figure()
        color_hex = ThesisStyle.PALETTE.get(color, ThesisStyle.COLORS['indigo'])
        
        sns.violinplot(x=data, color=color_hex, inner="quartile")
        
        plt.title(title)
        plt.xlabel(xlabel)
        self.save_plot(filename)

    def plot_residual_map(self, original: np.ndarray, restored: np.ndarray, title: str, filename: str):
        """
        Plots Original, Restored, and the Residual (Error) Map.
        Residual is calculated as Abs(Original - Restored).
        """
        # Ensure range [0, 1]
        orig_norm = original / 255.0
        rest_norm = restored / 255.0
        residual = np.abs(orig_norm - rest_norm)
        # Average over channels to get intensity map
        residual_intensity = np.mean(residual, axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Restored
        axes[1].imshow(restored)
        axes[1].set_title("Restored")
        axes[1].axis('off')
        
        # Residual
        im = axes[2].imshow(residual_intensity, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title("Residual Error Map (L1)")
        axes[2].axis('off')
        
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_regression_scatter(self, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: str):
        """
        Generates a scatter plot with a regression line.
        """
        paired = pd.DataFrame({"x": x, "y": y}).dropna()
        n = len(paired)
        x_unique = paired["x"].nunique() if n else 0
        y_unique = paired["y"].nunique() if n else 0

        plt.figure()
        if n >= 2 and x_unique > 1 and y_unique > 1:
            sns.regplot(
                x=paired["x"], y=paired["y"],
                scatter_kws={'alpha':0.6, 's':60, 'color': ThesisStyle.PALETTE['primary']},
                line_kws={'color': ThesisStyle.PALETTE['accent']}
            )
            corr = float(paired["x"].corr(paired["y"]))
            legend_label = f'Pearson r: {corr:.2f}'
        else:
            sns.scatterplot(
                x=paired["x"], y=paired["y"],
                alpha=0.6,
                s=60,
                color=ThesisStyle.PALETTE['primary'],
            )
            legend_label = 'Pearson r: n/a (insufficient variation)'
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend([legend_label], loc='upper right')
        
        self.save_plot(filename)

    def plot_heatmap(self, data: np.ndarray, title: str, filename: str, cmap: str = 'viridis'):
        """
        Generates a heatmap from a 2D array.
        """
        plt.figure()
        sns.heatmap(data, cmap=cmap, cbar=True)
        plt.title(title)
        plt.axis('off')
        self.save_plot(filename)

    def plot_bar(self, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, color: str = 'primary'):
        """
        Generates a standardized bar chart.
        """
        plt.figure(figsize=(12, 6))
        color_hex = ThesisStyle.PALETTE.get(color, ThesisStyle.COLORS['indigo'])
        
        sns.barplot(x=x, y=y, color=color_hex)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        self.save_plot(filename)

    def plot_table(self, df: pd.DataFrame, title: str, filename: str):
        """
        Renders a DataFrame as a table.
        """
        fig, ax = plt.subplots(figsize=(8, len(df)*0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        # Style
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(ThesisStyle.PALETTE['primary'])
            else:
                cell.set_facecolor(ThesisStyle.PALETTE['neutral'] if row % 2 else 'white')
        
        plt.title(title, pad=20)
        self.save_plot(filename)

    def plot_execution_time(self, log_path: Union[str, Path], filename: str = "execution_times"):
        """
        Reads execution logs and plots a bar chart of stage durations.
        """
        try:
            df = pd.read_csv(log_path)
            # Filter for last run if multiple runs exist (simple logic: take last N stages)
            # Better: Group by timestamp? For now, just plot all or take tail.
            # Let's assume the log file is cleaned or we want to show history.
            # Actually, let's just plot the latest run.
            
            # Simple approach: Plot last 10 entries (assuming 10 stages)
            df_latest = df.tail(10)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(
                x='duration_seconds', 
                y='stage_name', 
                data=df_latest, 
                hue='stage_name',
                palette=ThesisStyle.get_palette_list(),
            )
            legend = plt.gca().get_legend()
            if legend is not None:
                legend.remove()
            
            plt.title("Pipeline Execution Duration per Stage")
            plt.xlabel("Duration (seconds)")
            plt.ylabel("Stage")
            plt.grid(axis='x')
            
            for i, v in enumerate(df_latest['duration_seconds']):
                plt.text(v + 0.1, i, str(v), color='black', va='center')
                
            self.save_plot(filename)
        except Exception as e:
            print(f"Failed to plot execution times: {e}")

    def plot_comparison_grid(self, original: List[np.ndarray], masked: List[np.ndarray], restored: List[np.ndarray], filename: str = "comparison_grid", max_samples: int = 5):
        """
        Plots a grid of comparisons (Original | Masked | Restored).
        Shows up to max_samples rows.
        """
        n = min(len(original), max_samples)
        fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
        
        # Handle single sample case where axes is 1D
        if n == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n):
            # Original
            axes[i, 0].imshow(original[i])
            axes[i, 0].axis('off')
            if i == 0: axes[i, 0].set_title("Original", fontsize=12, fontweight='bold')
            
            # Masked
            axes[i, 1].imshow(masked[i])
            axes[i, 1].axis('off')
            if i == 0: axes[i, 1].set_title("Input (Masked)", fontsize=12, fontweight='bold')
            
            # Restored
            axes[i, 2].imshow(restored[i])
            axes[i, 2].axis('off')
            if i == 0: axes[i, 2].set_title("Restoration", fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Evaluation chart methods (used by Stage 13 & Stage 15)
    # ------------------------------------------------------------------

    def plot_evaluation_comparison(
        self,
        gt: np.ndarray,
        masked_input: np.ndarray,
        mask: np.ndarray,
        restorations: Dict[str, np.ndarray],
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        filename: str = "comparison_grid",
    ):
        """
        Per-sample side-by-side: Original | Masked | Mask | restorations… | Error map.

        Parameters
        ----------
        gt : Ground-truth image (H, W, 3) uint8.
        masked_input : Image with mask overlay (damaged region tinted red).
        mask : Binary mask (H, W) uint8.
        restorations : {model_name: restored_image_np}.
        metrics : Optional {model_name: {'psnr': …, 'ssim': …}}.
        """
        n_models = len(restorations)
        # Layout: 2 rows — top row = GT/Masked/Mask, bottom row = each model restoration
        top_cols = 3
        bot_cols = max(n_models, 1)
        total_cols = max(top_cols, bot_cols)

        fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8))

        # -- Top row --
        axes[0, 0].imshow(gt)
        axes[0, 0].set_title("Ground Truth", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(masked_input)
        axes[0, 1].set_title("Damaged Input", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title("Mask", fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')

        for c in range(3, total_cols):
            axes[0, c].axis('off')

        # -- Bottom row: restorations --
        model_names = list(restorations.keys())
        for c in range(total_cols):
            if c < n_models:
                name = model_names[c]
                axes[1, c].imshow(restorations[name])
                label = name
                if metrics and name in metrics:
                    m = metrics[name]
                    label += f"\nPSNR {m.get('psnr', 0):.1f} | SSIM {m.get('ssim', 0):.3f}"
                color = ThesisStyle.MODEL_COLORS.get(name, ThesisStyle.COLORS['grey'])
                axes[1, c].set_title(label, fontsize=9, color=color, fontweight='bold')
                axes[1, c].axis('off')
            else:
                axes[1, c].axis('off')

        plt.tight_layout(pad=0.5)
        self.save_plot(filename)

    def plot_metric_bars(
        self,
        df: pd.DataFrame,
        filename: str = "metric_comparison_bars",
    ):
        """
        Grouped bar chart of mean metrics per model.
        Expects df with columns: model, psnr, ssim, lpips, color, pattern.
        Uses only Unconditional condition to avoid duplication.
        """
        metrics_cols = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [c for c in metrics_cols if c in df.columns]
        # Use only Unconditional to get one row per model per sample
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        summary = subset.groupby('model')[present].mean()
        model_order = self._ordered_models(summary.index.tolist())
        summary = summary.reindex(model_order)

        n_metrics = len(present)
        n_models = len(summary)
        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (model, row) in enumerate(summary.iterrows()):
            color = ThesisStyle.MODEL_COLORS.get(model, ThesisStyle.COLORS['grey'])
            ax.bar(x + i * width, row[present].values, width,
                   label=model, color=color, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([m.upper() for m in present])
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison — Mean Metrics (Masked Region)')
        ax.legend(framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_metric_distributions(
        self,
        df: pd.DataFrame,
        filename: str = "metric_distributions",
    ):
        """
        Violin + box plots showing score distribution per model for each metric.
        """
        metrics_cols = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [c for c in metrics_cols if c in df.columns]
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df

        fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 6))
        if len(present) == 1:
            axes = [axes]

        model_order = self._ordered_models(subset['model'].unique().tolist())
        palette = {m: ThesisStyle.MODEL_COLORS.get(m, '#999999') for m in model_order}

        for ax, metric in zip(axes, present):
            sns.violinplot(
                data=subset, x='model', y=metric, order=model_order,
                hue='model', palette=palette, inner='box', ax=ax,
                cut=0, linewidth=0.8, dodge=False,
            )
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.set_title(metric.upper(), fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=30)

        plt.suptitle('Metric Distributions Across Models', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_psnr_vs_coverage(
        self,
        df: pd.DataFrame,
        filename: str = "psnr_vs_coverage",
    ):
        """
        Scatter of PSNR vs mask coverage, colored by model.
        Tests H3 — how models degrade with increasing damage.
        """
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        model_order = self._ordered_models(subset['model'].unique().tolist())
        palette = {m: ThesisStyle.MODEL_COLORS.get(m, '#999999') for m in model_order}

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in model_order:
            sub = subset[subset['model'] == model]
            ax.scatter(
                sub['mask_coverage'], sub['psnr'],
                label=model, color=palette[model],
                alpha=0.6, s=40, edgecolor='white', linewidth=0.3,
            )
            # Trend line
            if len(sub) > 3:
                z = np.polyfit(sub['mask_coverage'], sub['psnr'], 1)
                p = np.poly1d(z)
                xs = np.linspace(sub['mask_coverage'].min(), sub['mask_coverage'].max(), 50)
                ax.plot(xs, p(xs), '--', color=palette[model], alpha=0.8, linewidth=1.5)

        ax.set_xlabel('Mask Coverage')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR vs Damage Coverage — Robustness Analysis')
        ax.legend(framealpha=0.9)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_significance_heatmap(
        self,
        stats_df: pd.DataFrame,
        filename: str = "significance_heatmap",
    ):
        """
        Heatmap: comparisons × metrics. Green = FT-SD better & significant,
        Red = FT-SD worse & significant, Grey = not significant.
        Cell text = Cohen's d.
        """
        if stats_df.empty:
            return

        pivoted = stats_df.pivot(index='comparison', columns='metric', values='cohens_d')
        sig_pivot = stats_df.pivot(index='comparison', columns='metric', values='significant_bonferroni')

        # Build color matrix
        fig, ax = plt.subplots(figsize=(10, max(4, len(pivoted) * 0.8 + 1)))
        # Color: green if significant & d>0 (FT-SD better on higher-is-better),
        #        red if significant & d<0, grey if not significant.
        # For LPIPS, lower is better so flip the sign interpretation.
        lower_better = {'lpips'}
        cell_colors = np.full(pivoted.shape, ThesisStyle.SIG_COLORS['ns'])
        for i, comp in enumerate(pivoted.index):
            for j, metric in enumerate(pivoted.columns):
                d = pivoted.iloc[i, j]
                sig = sig_pivot.iloc[i, j]
                if pd.isna(d) or not sig:
                    continue
                effective_d = -d if metric in lower_better else d
                cell_colors[i, j] = (
                    ThesisStyle.SIG_COLORS['better'] if effective_d > 0
                    else ThesisStyle.SIG_COLORS['worse']
                )

        ax.imshow([[0]], aspect='auto', alpha=0)  # dummy for axes
        ax.set_xlim(-0.5, pivoted.shape[1] - 0.5)
        ax.set_ylim(pivoted.shape[0] - 0.5, -0.5)

        for i in range(pivoted.shape[0]):
            for j in range(pivoted.shape[1]):
                val = pivoted.iloc[i, j]
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor=cell_colors[i, j], edgecolor='white', linewidth=2))
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=11, fontweight='bold',
                            color='white' if cell_colors[i, j] != ThesisStyle.SIG_COLORS['ns'] else '#333')

        ax.set_xticks(range(pivoted.shape[1]))
        ax.set_xticklabels([m.upper() for m in pivoted.columns], fontweight='bold')
        ax.set_yticks(range(pivoted.shape[0]))
        ax.set_yticklabels(pivoted.index, fontsize=10)
        ax.set_title("Statistical Significance — Cohen's d\n(Green = FT-SD better, Red = FT-SD worse, Grey = n.s.)",
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.save_plot(filename)

    @staticmethod
    def build_significance_matrix_pairs() -> Dict[str, List[tuple[str, str]]]:
        big_models = ['LaMa', 'CoModGAN', 'MAT', 'FT-SD']
        baseline_set = ['Telea', 'Navier-Stokes', 'Vanilla SD', 'FT-SD']

        big_vs_base: List[tuple[str, str]] = []
        for anchor in big_models:
            for other in baseline_set:
                if anchor != other:
                    big_vs_base.append((anchor, other))

        big_four_internal: List[tuple[str, str]] = []
        for i in range(len(big_models)):
            for j in range(i + 1, len(big_models)):
                big_four_internal.append((big_models[i], big_models[j]))

        return {
            "big_vs_base": big_vs_base,
            "big_four_internal": big_four_internal,
        }

    @staticmethod
    def _normalize_matrix_config(config: Optional[dict]) -> dict:
        defaults = {
            "enabled": True,
            "scopes": ["unconditional", "condition_expanded"],
            "families": ["big_vs_base", "big_four_internal"],
            "output_subdir": "significance_matrix",
        }
        if config is None:
            return defaults
        merged = defaults.copy()
        for key in defaults.keys():
            try:
                value = config.get(key, defaults[key])
            except Exception:
                value = defaults[key]
            merged[key] = value
        return merged

    @staticmethod
    def _slugify_model_name(name: str) -> str:
        return (
            str(name)
            .strip()
            .lower()
            .replace("+", "plus")
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _ordered_metric_columns(columns: List[str]) -> List[str]:
        order = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        return [metric for metric in order if metric in columns]

    def _extract_pair_scope_tables(
        self,
        stats_df: pd.DataFrame,
        model_anchor: str,
        model_other: str,
        scope: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        required_cols = {
            'model_a', 'condition_a', 'model_b', 'condition_b',
            'metric', 'cohens_d', 'significant_bonferroni',
        }
        if not required_cols.issubset(set(stats_df.columns)):
            return pd.DataFrame(), pd.DataFrame()

        pair_mask = (
            ((stats_df['model_a'] == model_anchor) & (stats_df['model_b'] == model_other))
            | ((stats_df['model_a'] == model_other) & (stats_df['model_b'] == model_anchor))
        )
        pair_df = stats_df[pair_mask].copy()
        if pair_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if scope == "unconditional":
            pair_df = pair_df[
                (pair_df['condition_a'] == 'Unconditional')
                & (pair_df['condition_b'] == 'Unconditional')
            ].copy()
        elif scope == "condition_expanded":
            pass
        else:
            return pd.DataFrame(), pd.DataFrame()

        if pair_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        anchor_is_a = (
            (pair_df['model_a'] == model_anchor) & (pair_df['model_b'] == model_other)
        )
        pair_df['condition_anchor'] = np.where(anchor_is_a, pair_df['condition_a'], pair_df['condition_b'])
        pair_df['condition_other'] = np.where(anchor_is_a, pair_df['condition_b'], pair_df['condition_a'])
        pair_df['row_label'] = pair_df['condition_anchor'].astype(str) + " vs " + pair_df['condition_other'].astype(str)
        pair_df['cohens_d_anchor_minus_other'] = np.where(anchor_is_a, pair_df['cohens_d'], -pair_df['cohens_d'])

        lower_better = {'lpips'}
        better_direction = pair_df['cohens_d_anchor_minus_other'].copy()
        lower_mask = pair_df['metric'].astype(str).isin(lower_better)
        better_direction[lower_mask] = -better_direction[lower_mask]
        sig_mask = pair_df['significant_bonferroni'].astype(bool)
        pair_df['significance_state'] = np.where(
            sig_mask,
            np.where(better_direction > 0, 1, -1),
            0,
        )

        effect_df = pair_df.pivot_table(
            index='row_label',
            columns='metric',
            values='cohens_d_anchor_minus_other',
            aggfunc='mean',
        )
        state_df = pair_df.pivot_table(
            index='row_label',
            columns='metric',
            values='significance_state',
            aggfunc='mean',
        )

        metric_cols = self._ordered_metric_columns(effect_df.columns.tolist())
        if metric_cols:
            effect_df = effect_df.reindex(columns=metric_cols)
        metric_cols_state = self._ordered_metric_columns(state_df.columns.tolist())
        if metric_cols_state:
            state_df = state_df.reindex(columns=metric_cols_state)

        def _row_sort_key(label: str) -> tuple[int, str]:
            return (0, label) if str(label).startswith('Unconditional vs Unconditional') else (1, str(label))

        if len(effect_df.index) > 1:
            effect_df = effect_df.reindex(sorted(effect_df.index.tolist(), key=_row_sort_key))
        if len(state_df.index) > 1:
            state_df = state_df.reindex(sorted(state_df.index.tolist(), key=_row_sort_key))

        return effect_df, state_df

    def _plot_pair_effect_heatmap(self, matrix_df: pd.DataFrame, title: str, output_path: Path) -> bool:
        if matrix_df.empty:
            return False
        vmax = float(np.nanmax(np.abs(matrix_df.values))) if np.isfinite(matrix_df.values).any() else 0.0
        vmax = max(vmax, 0.1)

        fig_h = max(3.0, 1.2 + len(matrix_df.index) * 0.8)
        fig_w = max(6.0, 1.4 + len(matrix_df.columns) * 1.4)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt=".3f",
            cmap=sns.diverging_palette(10, 150, as_cmap=True),
            center=0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.8,
            cbar_kws={'label': "Cohen's d (anchor minus comparator)"},
        )
        plt.title(title, fontweight='bold')
        plt.xlabel("Metric")
        plt.ylabel("Condition Pair (anchor vs comparator)")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True

    def _plot_pair_significance_heatmap(self, state_df: pd.DataFrame, title: str, output_path: Path) -> bool:
        if state_df.empty:
            return False
        values = state_df.fillna(0).to_numpy(dtype=float)
        labels = np.where(values > 0, "sig+", np.where(values < 0, "sig-", "n.s."))

        fig_h = max(3.0, 1.2 + len(state_df.index) * 0.8)
        fig_w = max(6.0, 1.4 + len(state_df.columns) * 1.4)
        plt.figure(figsize=(fig_w, fig_h))
        cmap = ListedColormap([
            ThesisStyle.SIG_COLORS['worse'],
            ThesisStyle.SIG_COLORS['ns'],
            ThesisStyle.SIG_COLORS['better'],
        ])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        sns.heatmap(
            state_df,
            annot=labels,
            fmt='',
            cmap=cmap,
            norm=norm,
            linewidths=0.8,
            cbar=True,
            cbar_kws={'ticks': [-1, 0, 1], 'label': 'Significance state'},
        )
        plt.title(title, fontweight='bold')
        plt.xlabel("Metric")
        plt.ylabel("Condition Pair (anchor vs comparator)")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        cbar = plt.gca().collections[0].colorbar
        cbar.set_ticklabels(['sig-', 'n.s.', 'sig+'])
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True

    def plot_significance_matrix_suite(
        self,
        stats_df: pd.DataFrame,
        config: Optional[dict] = None,
    ) -> pd.DataFrame:
        cfg = self._normalize_matrix_config(config)
        if not bool(cfg.get("enabled", True)):
            return pd.DataFrame()

        output_subdir = str(cfg.get("output_subdir", "significance_matrix")).strip() or "significance_matrix"
        output_root = self.output_dir / output_subdir
        output_root.mkdir(parents=True, exist_ok=True)

        selected_scopes = [str(scope) for scope in cfg.get("scopes", ["unconditional", "condition_expanded"])]
        selected_families = [str(family) for family in cfg.get("families", ["big_vs_base", "big_four_internal"])]
        pairs_by_family = self.build_significance_matrix_pairs()

        manifest_rows: List[dict] = []

        for family in selected_families:
            family_pairs = pairs_by_family.get(family, [])
            for model_anchor, model_other in family_pairs:
                for scope in selected_scopes:
                    effect_df, state_df = self._extract_pair_scope_tables(
                        stats_df=stats_df,
                        model_anchor=model_anchor,
                        model_other=model_other,
                        scope=scope,
                    )
                    for chart_type in ("effect_heatmap", "significance_heatmap"):
                        row = {
                            "family": family,
                            "scope": scope,
                            "model_anchor": model_anchor,
                            "model_other": model_other,
                            "chart_type": chart_type,
                            "status": "skipped",
                            "filepath": "",
                            "skip_reason": "",
                        }
                        matrix_df = effect_df if chart_type == "effect_heatmap" else state_df
                        if matrix_df.empty:
                            row["skip_reason"] = "no_rows_for_pair_scope"
                            manifest_rows.append(row)
                            continue

                        family_dir = output_root / family / scope
                        family_dir.mkdir(parents=True, exist_ok=True)
                        model_slug = f"{self._slugify_model_name(model_anchor)}_vs_{self._slugify_model_name(model_other)}"
                        out_path = family_dir / f"{model_slug}_{chart_type}.png"
                        chart_label = "Cohen's d" if chart_type == "effect_heatmap" else "Bonferroni significance"
                        title = (
                            f"{model_anchor} vs {model_other} — {scope} "
                            f"({chart_label})"
                        )
                        if chart_type == "effect_heatmap":
                            generated = self._plot_pair_effect_heatmap(matrix_df, title, out_path)
                        else:
                            generated = self._plot_pair_significance_heatmap(matrix_df, title, out_path)
                        if generated:
                            row["status"] = "generated"
                            row["filepath"] = str(out_path)
                        else:
                            row["skip_reason"] = "empty_matrix_after_pivot"
                        manifest_rows.append(row)

        manifest_df = pd.DataFrame(manifest_rows, columns=[
            "family", "scope", "model_anchor", "model_other", "chart_type",
            "status", "filepath", "skip_reason",
        ])
        manifest_csv = output_root / "matrix_manifest.csv"
        manifest_json = output_root / "matrix_manifest.json"
        manifest_df.to_csv(manifest_csv, index=False)
        manifest_json.write_text(
            json.dumps(manifest_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        return manifest_df

    def plot_improvement_deltas(
        self,
        df: pd.DataFrame,
        metric: str = 'psnr',
        baseline: str = 'Telea',
        filename: str = "improvement_over_baseline",
    ):
        """
        Per-sample bar chart showing metric improvement of Ours vs a baseline.
        Bars sorted by magnitude. Positive = Ours is better.
        """
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        ours_df = subset[subset['model'] == 'FT-SD'][['sample_id', metric]].set_index('sample_id')
        base_df = subset[subset['model'] == baseline][['sample_id', metric]].set_index('sample_id')

        merged = ours_df.join(base_df, lsuffix='_ours', rsuffix='_base').dropna()
        merged['delta'] = merged[f'{metric}_ours'] - merged[f'{metric}_base']
        merged = merged.sort_values('delta')

        fig, ax = plt.subplots(figsize=(14, max(6, len(merged) * 0.12)))
        colors = [ThesisStyle.COLORS['green'] if d > 0 else ThesisStyle.COLORS['rose']
                  for d in merged['delta']]
        ax.barh(range(len(merged)), merged['delta'], color=colors, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='#333', linewidth=0.8)
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels([s[:30] for s in merged.index], fontsize=6)
        ax.set_xlabel(f'Δ {metric.upper()} (Ours − {baseline})')
        ax.set_title(f'Per-Sample {metric.upper()} Improvement Over {baseline}')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_evaluation_overview(
        self,
        samples: List[Dict],
        filename: str = "evaluation_overview",
    ):
        """
        Summary grid: rows = selected samples, cols = Ground Truth | Damaged | available model outputs.

        Each entry in *samples* is a dict:
          {'gt': ndarray, 'masked': ndarray, '<model_name>': ndarray, 'label': str}
        """
        if not samples:
            return
        present_models = set()
        reserved = {'gt', 'masked', 'label', 'psnr_ours', 'metrics'}
        for sample in samples:
            present_models.update(k for k in sample.keys() if k not in reserved)
        model_cols = self._ordered_models(sorted(present_models))
        col_labels = ['Ground Truth', 'Damaged', *model_cols]
        n_rows = len(samples)
        n_cols = len(col_labels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, s in enumerate(samples):
            imgs = [s.get('gt'), s.get('masked'), *[s.get(model_name) for model_name in model_cols]]
            for j, (img, label) in enumerate(zip(imgs, col_labels)):
                ax = axes[i, j]
                if img is not None:
                    ax.imshow(img)
                ax.axis('off')
                if i == 0:
                    ax.set_title(label, fontsize=11, fontweight='bold')
            # Row label
            tag = s.get('label', '')
            axes[i, 0].set_ylabel(tag, fontsize=9, rotation=0, labelpad=60, va='center')

        plt.suptitle('Evaluation Overview — Best / Median / Worst by PSNR',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_stratified_bars(
        self,
        df: pd.DataFrame,
        group_col: str,
        filename: str = "stratified_analysis",
        title: str = "Stratified Analysis",
    ):
        """
        Grouped bar chart: mean PSNR per model, stratified by *group_col* (e.g. mask_type or coverage_bin).
        """
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        summary = subset.groupby(['model', group_col], observed=False)['psnr'].mean().unstack(fill_value=0)
        model_order = self._ordered_models(summary.index.tolist())
        summary = summary.reindex(model_order)

        summary.plot(
            kind='bar', figsize=(12, 6), edgecolor='white', linewidth=0.5,
            color=[ThesisStyle.COLORS[c] for c in ['teal', 'sand', 'rose']][:len(summary.columns)],
        )
        plt.title(title, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Mean PSNR (dB)')
        plt.xticks(rotation=0)
        plt.legend(title=group_col.replace('_', ' ').title(), framealpha=0.9)
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Individual per-metric charts (one figure per metric)
    # ------------------------------------------------------------------

    def plot_single_metric_bar(
        self,
        df: pd.DataFrame,
        metric: str,
        filename: str = None,
    ):
        """Single-metric grouped bar chart with error bars per model."""
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        model_order = self._ordered_models(subset['model'].unique().tolist())

        means = subset.groupby('model')[metric].mean().reindex(model_order)
        stds = subset.groupby('model')[metric].std().reindex(model_order)
        colors = [ThesisStyle.MODEL_COLORS.get(m, '#999999') for m in model_order]

        higher_better = metric not in ('lpips',)
        best_idx = int(means.values.argmax()) if higher_better else int(means.values.argmin())

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(model_order, means, yerr=stds, capsize=5,
                       color=colors, edgecolor='white', linewidth=0.8)
        # Highlight winner
        bars[best_idx].set_edgecolor('#333333')
        bars[best_idx].set_linewidth(2.5)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        direction = '↑ Higher is better' if higher_better else '↓ Lower is better'
        ax.set_ylabel(f'{metric.upper()} ({direction})')
        ax.set_title(f'{metric.upper()} — Model Comparison', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename or f"bar_{metric}")

    def plot_single_metric_violin(
        self,
        df: pd.DataFrame,
        metric: str,
        filename: str = None,
    ):
        """Single-metric violin + strip plot showing full distribution per model."""
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        model_order = self._ordered_models(subset['model'].unique().tolist())
        palette = {m: ThesisStyle.MODEL_COLORS.get(m, '#999999') for m in model_order}

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            data=subset, x='model', y=metric, order=model_order,
            hue='model', palette=palette, inner=None, cut=0,
            linewidth=0.8, ax=ax, dodge=False,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.stripplot(
            data=subset, x='model', y=metric, order=model_order,
            color='#333333', size=2.5, alpha=0.4, jitter=True, ax=ax,
        )
        # Add median markers
        medians = subset.groupby('model')[metric].median().reindex(model_order)
        for i, med in enumerate(medians):
            ax.plot([i - 0.15, i + 0.15], [med, med], color='white', linewidth=2.5)
            ax.plot([i - 0.15, i + 0.15], [med, med], color='#333', linewidth=1.2)

        higher_better = metric not in ('lpips',)
        direction = '↑ Higher is better' if higher_better else '↓ Lower is better'
        ax.set_ylabel(f'{metric.upper()} ({direction})')
        ax.set_xlabel('')
        ax.set_title(f'{metric.upper()} Distribution Across Models', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self.save_plot(filename or f"violin_{metric}")

    def plot_single_metric_vs_coverage(
        self,
        df: pd.DataFrame,
        metric: str,
        filename: str = None,
    ):
        """Scatter of any metric vs mask coverage, colored by model with trend lines."""
        if 'mask_coverage' not in df.columns:
            return
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        model_order = self._ordered_models(subset['model'].unique().tolist())
        palette = {m: ThesisStyle.MODEL_COLORS.get(m, '#999999') for m in model_order}

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in model_order:
            sub = subset[subset['model'] == model]
            ax.scatter(sub['mask_coverage'], sub[metric],
                       label=model, color=palette[model], alpha=0.5, s=35,
                       edgecolor='white', linewidth=0.3)
            if len(sub) > 3:
                z = np.polyfit(sub['mask_coverage'], sub[metric], 1)
                p = np.poly1d(z)
                xs = np.linspace(sub['mask_coverage'].min(), sub['mask_coverage'].max(), 50)
                ax.plot(xs, p(xs), '--', color=palette[model], alpha=0.8, linewidth=1.5)

        ax.set_xlabel('Mask Coverage (fraction)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Damage Coverage', fontweight='bold')
        ax.legend(framealpha=0.9)
        plt.tight_layout()
        self.save_plot(filename or f"scatter_{metric}_vs_coverage")

    def plot_all_individual_charts(self, df: pd.DataFrame, stats_df: pd.DataFrame = None):
        """Generate all individual per-metric charts in one call."""
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [m for m in metrics if m in df.columns]

        for metric in present:
            self.plot_single_metric_bar(df, metric)
            self.plot_single_metric_violin(df, metric)
            self.plot_single_metric_vs_coverage(df, metric)
            self.plot_improvement_deltas(df, metric=metric, baseline='Telea',
                                        filename=f"deltas_{metric}_vs_telea")

    # ------------------------------------------------------------------
    #  Ablation study charts
    # ------------------------------------------------------------------

    def plot_ablation_bars(
        self,
        df: pd.DataFrame,
        model: str = 'FT-SD',
        filename: str = None,
    ):
        """
        Grouped bar chart showing each metric across 3 text conditions for one model.
        Reveals whether prompt ablation has any effect.
        """
        sub = df[df['model'] == model]
        if sub.empty:
            return
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [m for m in metrics if m in sub.columns]
        conditions = ['Unconditional', 'Raw Text', 'Enriched Text']
        conditions = [c for c in conditions if c in sub['condition'].unique()]

        summary = sub.groupby('condition')[present].mean().reindex(conditions)

        n_metrics = len(present)
        n_conds = len(conditions)
        x = np.arange(n_metrics)
        width = 0.8 / n_conds
        cond_colors = [ThesisStyle.COLORS['teal'], ThesisStyle.COLORS['sand'], ThesisStyle.COLORS['purple']]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, cond in enumerate(conditions):
            vals = summary.loc[cond, present].values
            ax.bar(x + i * width, vals, width, label=cond,
                   color=cond_colors[i % len(cond_colors)],
                   edgecolor='white', linewidth=0.5)
            # Value labels
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, v + 0.003, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels([m.upper() for m in present])
        ax.set_ylabel('Score')
        ax.set_title(f'Prompt Ablation — {model}\n(identical bars = text conditioning has no effect)',
                     fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename or f"ablation_{model.lower().replace(' ', '_')}")

    def plot_ablation_heatmap(
        self,
        df: pd.DataFrame,
        filename: str = "ablation_heatmap",
    ):
        """
        Heatmap: models x metrics, cells show delta(Enriched - Unconditional).
        Near-zero everywhere = text conditioning is inert.
        """
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [m for m in metrics if m in df.columns]
        sd_models = ['Vanilla SD', 'FT-SD']
        sd_models = [m for m in sd_models if m in df['model'].unique()]

        rows = []
        for model in sd_models:
            sub = df[df['model'] == model]
            uncond = sub[sub['condition'] == 'Unconditional'][present].mean()
            enriched = sub[sub['condition'] == 'Enriched Text'][present].mean()
            delta = enriched - uncond
            rows.append(delta)

        delta_df = pd.DataFrame(rows, index=sd_models)
        delta_df = delta_df.reindex(columns=present)
        delta_df.columns = [m.upper() for m in delta_df.columns]

        fig, ax = plt.subplots(figsize=(10, 3 + len(sd_models) * 0.6))
        cmap = sns.diverging_palette(10, 150, as_cmap=True, center='light')
        vmax = max(abs(delta_df.values.min()), abs(delta_df.values.max()), 0.01)
        sns.heatmap(delta_df, annot=True, fmt='.4f', cmap=cmap, center=0,
                    vmin=-vmax, vmax=vmax, linewidths=1, ax=ax,
                    cbar_kws={'label': 'Δ (Enriched − Unconditional)'})
        ax.set_title('Text Conditioning Effect (Δ values)\n'
                     'Values near 0.0000 = no prompt sensitivity',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_ablation_per_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        filename: str = None,
    ):
        """Individual ablation chart for a single metric, both SD models side by side."""
        sd_models = ['Vanilla SD', 'FT-SD']
        sd_models = [m for m in sd_models if m in df['model'].unique()]
        conditions = ['Unconditional', 'Raw Text', 'Enriched Text']
        conditions = [c for c in conditions if c in df['condition'].unique()]

        fig, axes = plt.subplots(1, len(sd_models), figsize=(6 * len(sd_models), 5), sharey=True)
        if len(sd_models) == 1:
            axes = [axes]

        cond_colors = [ThesisStyle.COLORS['teal'], ThesisStyle.COLORS['sand'], ThesisStyle.COLORS['purple']]
        for ax, model in zip(axes, sd_models):
            sub = df[df['model'] == model]
            means = [sub[sub['condition'] == c][metric].mean() for c in conditions]
            stds = [sub[sub['condition'] == c][metric].std() for c in conditions]
            bars = ax.bar(conditions, means, yerr=stds, capsize=5,
                          color=cond_colors[:len(conditions)],
                          edgecolor='white', linewidth=0.8)
            for bar, m_val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{m_val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title(model, fontweight='bold')
            ax.set_ylabel(metric.upper() if ax == axes[0] else '')
            ax.tick_params(axis='x', rotation=25)

        plt.suptitle(f'{metric.upper()} — Prompt Ablation Study',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_plot(filename or f"ablation_{metric}")

    # ------------------------------------------------------------------
    #  Hyperparameter & Training charts
    # ------------------------------------------------------------------

    def plot_hyperparameter_summary(
        self,
        hyperparams: dict,
        filename: str = "hyperparameter_summary",
    ):
        """
        Visual summary of chosen hyperparameters with the search space context.
        Shows selected value within the allowed range where applicable.
        """
        search_space = hyperparams.get('search_space', {})
        lr_range = search_space.get('learning_rate_range', [1e-6, 5e-5])
        selected_lr = hyperparams.get('learning_rate', 1e-5)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Learning rate on log scale with range context
        ax = axes[0]
        ax.set_xscale('log')
        ax.axvspan(lr_range[0], lr_range[1], alpha=0.15, color=ThesisStyle.COLORS['teal'],
                   label='Search range')
        ax.axvline(selected_lr, color=ThesisStyle.COLORS['wine'], linewidth=2.5,
                   label=f'Selected: {selected_lr:.0e}')
        # Literature markers
        lit_points = {'DreamBooth\n(Ruiz 2023)': 1e-5, 'HF Guide': 5e-6, 'LDM base': 1e-4}
        for label, val in lit_points.items():
            if lr_range[0] * 0.5 <= val <= lr_range[1] * 2:
                ax.axvline(val, color='grey', linewidth=0.8, linestyle=':', alpha=0.7)
                ax.text(val, 0.7, label, rotation=90, va='center', ha='right',
                        fontsize=8, color='grey',
                        transform=ax.get_xaxis_transform())
        ax.set_xlim(lr_range[0] * 0.3, lr_range[1] * 3)
        ax.set_title('Learning Rate Selection', fontweight='bold')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.legend()
        ax.set_yticks([])

        # Panel 2: Key hyperparameters table
        ax = axes[1]
        ax.axis('off')
        skip = {'search_space', 'references', 'selection_method'}
        rows = [(k.replace('_', ' ').title(), str(v))
                for k, v in hyperparams.items() if k not in skip]
        table = ax.table(cellText=rows, colLabels=['Parameter', 'Value'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(ThesisStyle.PALETTE['primary'])
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#f8f8f8' if row % 2 else 'white')
        ax.set_title('Selected Hyperparameters', fontweight='bold', pad=20)

        plt.suptitle('Hyperparameter Configuration\n(Literature-Informed Selection)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_training_analysis(
        self,
        training_csv: str,
        filename: str = "training_analysis",
    ):
        """
        Comprehensive training analysis: loss curves + convergence + epoch comparison.
        4 panels: loss curves, train-val gap, pct improvement, epoch-over-epoch delta.
        """
        df = pd.read_csv(training_csv)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Loss curves with annotations
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], 'o-',
                color=ThesisStyle.COLORS['teal'], linewidth=2, label='Train Loss', markersize=6)
        ax.plot(df['epoch'], df['val_loss'], 's-',
                color=ThesisStyle.COLORS['wine'], linewidth=2, label='Val Loss', markersize=6)
        best_epoch = int(df['val_loss'].idxmin()) + 1
        best_val = df['val_loss'].min()
        ax.axvline(best_epoch, color='grey', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: epoch {best_epoch}\n(val={best_val:.5f})',
                    xy=(best_epoch, best_val), xytext=(best_epoch + 1, best_val + 0.001),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')

        # Panel 2: Train-Val gap (generalization)
        ax = axes[0, 1]
        gap = df['val_loss'] - df['train_loss']
        ax.bar(df['epoch'], gap,
               color=[ThesisStyle.COLORS['green'] if g >= 0 else ThesisStyle.COLORS['rose'] for g in gap],
               edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='#333', linewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss − Train Loss')
        ax.set_title('Generalization Gap', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Panel 3: Cumulative pct improvement from epoch 1
        ax = axes[1, 0]
        initial_train = df['train_loss'].iloc[0]
        initial_val = df['val_loss'].iloc[0]
        pct_train = (initial_train - df['train_loss']) / initial_train * 100
        pct_val = (initial_val - df['val_loss']) / initial_val * 100
        ax.plot(df['epoch'], pct_train, 'o-', color=ThesisStyle.COLORS['teal'],
                linewidth=2, label='Train', markersize=5)
        ax.plot(df['epoch'], pct_val, 's-', color=ThesisStyle.COLORS['wine'],
                linewidth=2, label='Val', markersize=5)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Improvement from Epoch 1')
        ax.set_title('Convergence Progress', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')

        # Panel 4: Epoch-over-epoch deltas
        ax = axes[1, 1]
        deltas_val = df['val_loss'].diff()
        colors = [ThesisStyle.COLORS['green'] if d < 0 else ThesisStyle.COLORS['rose']
                  for d in deltas_val.dropna()]
        ax.bar(df['epoch'].iloc[1:], deltas_val.dropna(), color=colors,
               edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='#333', linewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Δ Val Loss (epoch-over-epoch)')
        ax.set_title('Learning Dynamics', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Training Analysis — SD Inpainting Fine-Tuning',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_loss_landscape(
        self,
        training_csv: str,
        filename: str = "loss_landscape",
    ):
        """
        Train loss vs val loss scatter showing the training trajectory.
        Each point is one epoch, connected in sequence.
        """
        df = pd.read_csv(training_csv)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(df['train_loss'], df['val_loss'], 'o-',
                color=ThesisStyle.COLORS['purple'], linewidth=1.5, markersize=8, alpha=0.8)

        # Annotate start and end
        ax.annotate(f'Start (E1)', xy=(df['train_loss'].iloc[0], df['val_loss'].iloc[0]),
                    fontsize=9, fontweight='bold', color=ThesisStyle.COLORS['teal'],
                    xytext=(5, 10), textcoords='offset points')
        ax.annotate(f'End (E{len(df)})', xy=(df['train_loss'].iloc[-1], df['val_loss'].iloc[-1]),
                    fontsize=9, fontweight='bold', color=ThesisStyle.COLORS['wine'],
                    xytext=(5, -15), textcoords='offset points')

        # Best epoch
        best_idx = int(df['val_loss'].idxmin())
        ax.scatter(df['train_loss'].iloc[best_idx], df['val_loss'].iloc[best_idx],
                   s=200, facecolors='none', edgecolors=ThesisStyle.COLORS['wine'],
                   linewidth=2.5, zorder=5, label=f'Best (E{best_idx + 1})')

        # Diagonal (perfect generalization)
        lim_lo = min(df['train_loss'].min(), df['val_loss'].min()) - 0.001
        lim_hi = max(df['train_loss'].max(), df['val_loss'].max()) + 0.001
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], '--', color='grey', alpha=0.4,
                label='Train=Val (no gap)')

        ax.set_xlabel('Train Loss')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Training Trajectory in Loss Space', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_sweep_pareto(
        self,
        trials_df: pd.DataFrame,
        filename: str = "sweep_pareto",
    ):
        """Scatter plot of best val loss vs runtime for sweep trials."""
        required = {'best_val_loss', 'duration_seconds'}
        if not required.issubset(trials_df.columns):
            return

        df = trials_df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['best_val_loss', 'duration_seconds'])
        if df.empty:
            return

        df['duration_min'] = df['duration_seconds'] / 60.0
        best_idx = df['best_val_loss'].astype(float).idxmin()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            df['duration_min'],
            df['best_val_loss'],
            s=90,
            c=ThesisStyle.COLORS['teal'],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8,
        )
        for _, row in df.iterrows():
            trial_id = str(row.get('trial_id', 'trial'))
            ax.annotate(
                trial_id,
                (row['duration_min'], row['best_val_loss']),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

        ax.scatter(
            [df.loc[best_idx, 'duration_min']],
            [df.loc[best_idx, 'best_val_loss']],
            s=180,
            facecolors='none',
            edgecolors=ThesisStyle.COLORS['wine'],
            linewidth=2.2,
            label=f"Selected: {df.loc[best_idx, 'trial_id']}",
            zorder=5,
        )
        ax.set_xlabel("Runtime (minutes)")
        ax.set_ylabel("Best Validation Loss")
        ax.set_title("Mini-Sweep Pareto: Quality vs Runtime", fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()
        plt.tight_layout()
        self.save_plot(filename)

    def plot_lr_wd_response_heatmap(
        self,
        trials_df: pd.DataFrame,
        filename: str = "lr_wd_response_heatmap",
    ):
        """Heatmap of best validation loss across LR x weight decay trials."""
        required = {'learning_rate', 'adam_weight_decay', 'best_val_loss'}
        if not required.issubset(trials_df.columns):
            return

        df = trials_df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['learning_rate', 'adam_weight_decay', 'best_val_loss'])
        if df.empty:
            return

        pivot = df.pivot_table(
            index='adam_weight_decay',
            columns='learning_rate',
            values='best_val_loss',
            aggfunc='min',
        )
        pivot = pivot.sort_index().sort_index(axis=1)
        if pivot.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            linewidths=0.7,
            cbar_kws={'label': 'Best Validation Loss'},
            ax=ax,
        )
        ax.set_title('Learning Rate x Weight Decay Response Surface', fontweight='bold')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Adam Weight Decay')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_trial_learning_curves_panel(
        self,
        sweep_root: Union[str, Path],
        filename: str = "trial_learning_curves_panel",
    ):
        """Panel of train/val loss curves for all sweep trials."""
        sweep_root = Path(sweep_root)
        csv_paths = sorted(sweep_root.glob("trial_*/training_logs.csv"))
        if not csv_paths:
            return

        n = len(csv_paths)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.7 * n_rows), squeeze=False)

        for i, csv_path in enumerate(csv_paths):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col]
            trial_id = csv_path.parent.name

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                ax.set_title(f"{trial_id} (read failed)")
                ax.axis("off")
                continue

            if {'epoch', 'train_loss'}.issubset(df.columns):
                ax.plot(df['epoch'], df['train_loss'], marker='o', markersize=3.5, linewidth=1.6, label='train')
            if {'epoch', 'val_loss'}.issubset(df.columns):
                ax.plot(df['epoch'], df['val_loss'], marker='s', markersize=3.5, linewidth=1.6, label='val')
                best_idx = df['val_loss'].idxmin()
                if pd.notna(best_idx):
                    best_epoch = int(df.loc[best_idx, 'epoch'])
                    best_val = float(df.loc[best_idx, 'val_loss'])
                    ax.axvline(best_epoch, linestyle='--', alpha=0.4, color='grey')
                    ax.text(best_epoch, best_val, f"best={best_val:.4f}", fontsize=8)

            ax.set_title(trial_id, fontweight='bold')
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(fontsize=8, framealpha=0.9)

        for j in range(n, n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            axes[row][col].axis('off')

        plt.suptitle("Sweep Trial Learning Curves", fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_kfold_training_stability(
        self,
        kfold_df: pd.DataFrame,
        filename: str = "kfold_training_stability",
    ):
        """Fold-wise best validation loss with mean +- std stability band."""
        required = {'fold', 'best_val_loss'}
        if not required.issubset(kfold_df.columns):
            return

        df = kfold_df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['best_val_loss'])
        if df.empty:
            return

        df = df.sort_values('fold').reset_index(drop=True)
        x = np.arange(1, len(df) + 1)
        y = df['best_val_loss'].astype(float).to_numpy()
        mu = float(np.mean(y))
        sigma = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(x, y, marker='o', linewidth=2, color=ThesisStyle.COLORS['purple'], label='Fold best val loss')
        ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.18, color=ThesisStyle.COLORS['teal'], label='mean ± std')
        ax.axhline(mu, linestyle='--', linewidth=1.4, color=ThesisStyle.COLORS['wine'], label=f'mean={mu:.4f}')

        ax.set_xticks(x)
        ax.set_xticklabels(df['fold'].tolist(), rotation=0)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Best Validation Loss")
        ax.set_title("K-Fold Training Stability", fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()
        plt.tight_layout()
        self.save_plot(filename)

    def plot_lr_schedule(
        self,
        total_epochs: int = 10,
        lr_max: float = 1e-5,
        warmup_fraction: float = 0.1,
        filename: str = "lr_schedule",
    ):
        """Visualise cosine learning rate schedule with optional warmup."""
        steps = np.arange(total_epochs * 100)
        warmup_steps = int(total_epochs * 100 * warmup_fraction)
        lr = np.zeros_like(steps, dtype=float)
        for i, s in enumerate(steps):
            if s < warmup_steps:
                lr[i] = lr_max * s / max(warmup_steps, 1)
            else:
                progress = (s - warmup_steps) / max(len(steps) - warmup_steps, 1)
                lr[i] = lr_max * 0.5 * (1 + np.cos(np.pi * progress))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps / 100, lr, color=ThesisStyle.COLORS['purple'], linewidth=2)
        if warmup_steps > 0:
            ax.axvspan(0, warmup_steps / 100, alpha=0.1, color=ThesisStyle.COLORS['sand'],
                       label=f'Warmup ({warmup_fraction:.0%})')
        ax.axhline(lr_max, color='grey', linestyle=':', alpha=0.5, label=f'Peak LR: {lr_max:.0e}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Cosine Learning Rate Schedule', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Cross-Validation charts
    # ------------------------------------------------------------------

    def plot_cv_folds(
        self,
        cv_df: pd.DataFrame,
        filename: str = "cv_fold_comparison",
    ):
        """Plot per-fold + mean metric scores across k folds.

        Parameters
        ----------
        cv_df : DataFrame
            Columns: model, metric, fold_1, fold_2, fold_3, mean, std, cv
        """
        metrics = cv_df['metric'].unique()
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
        if n_metrics == 1:
            axes = [axes]

        model_order = self._ordered_models(cv_df['model'].unique().tolist())

        for ax, metric in zip(axes, metrics):
            sub = cv_df[cv_df['metric'] == metric]
            sub = sub.set_index('model').reindex([m for m in model_order if m in sub.index])
            fold_cols = [c for c in sub.columns if c.startswith('fold_')]
            x = np.arange(len(sub))
            width = 0.18

            for i, fc in enumerate(fold_cols):
                colors = [ThesisStyle.MODEL_COLORS.get(m, '#999') for m in sub.index]
                alpha = 0.45
                ax.bar(x + i * width, sub[fc], width, alpha=alpha,
                       color=colors, edgecolor='white', linewidth=0.5,
                       label=fc.replace('_', ' ').title() if i == 0 or True else '')

            # Mean line + error bars
            ax.errorbar(x + width * (len(fold_cols) - 1) / 2, sub['mean'], yerr=sub['std'],
                        fmt='D', color='#333', markersize=7, capsize=5, linewidth=1.5,
                        label='Mean ± Std', zorder=10)

            ax.set_xticks(x + width * (len(fold_cols) - 1) / 2)
            ax.set_xticklabels(sub.index, fontsize=9)
            higher_better = metric not in ('lpips',)
            direction = '↑' if higher_better else '↓'
            ax.set_title(f'{metric.upper()} ({direction})', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        axes[-1].legend(fontsize=8, framealpha=0.9)
        plt.suptitle('3-Fold Cross-Validation Results', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_cv_stability(
        self,
        cv_df: pd.DataFrame,
        filename: str = "cv_stability",
    ):
        """Coefficient of Variation heatmap across models × metrics."""
        metrics = cv_df['metric'].unique()
        models = self._ordered_models(cv_df['model'].unique().tolist())

        pivot = cv_df.pivot_table(index='model', columns='metric', values='cv').reindex(models)

        fig, ax = plt.subplots(figsize=(10, 3 + len(models) * 0.6))
        cmap = sns.light_palette(ThesisStyle.COLORS['wine'], as_cmap=True)
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, linewidths=1, ax=ax,
                    cbar_kws={'label': 'CV (%)'})
        ax.set_title('Cross-Validation Stability\n(lower CV% = more stable results)',
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  FID / KID charts
    # ------------------------------------------------------------------

    def plot_fid_kid(
        self,
        fid_scores: dict,
        kid_scores: dict,
        filename: str = "fid_kid_comparison",
    ):
        """Bar chart comparing FID and KID across models.

        Parameters
        ----------
        fid_scores : dict  {model_name: fid_value}
        kid_scores : dict  {model_name: (kid_mean, kid_std)}
        """
        models = self._ordered_models(fid_scores.keys())
        colors = [ThesisStyle.MODEL_COLORS.get(m, '#999') for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # FID
        ax = axes[0]
        fid_vals = [fid_scores[m] for m in models]
        bars = ax.bar(models, fid_vals, color=colors, edgecolor='white', linewidth=0.8)
        best_idx = int(np.argmin(fid_vals))
        bars[best_idx].set_edgecolor('#333')
        bars[best_idx].set_linewidth(2.5)
        for bar, v in zip(bars, fid_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('FID (↓ lower is better)')
        ax.set_title('Fréchet Inception Distance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # KID
        ax = axes[1]
        kid_means = [kid_scores[m][0] for m in models]
        kid_stds = [kid_scores[m][1] for m in models]
        bars = ax.bar(models, kid_means, yerr=kid_stds, capsize=5,
                      color=colors, edgecolor='white', linewidth=0.8)
        best_idx = int(np.argmin(kid_means))
        bars[best_idx].set_edgecolor('#333')
        bars[best_idx].set_linewidth(2.5)
        for bar, v in zip(bars, kid_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('KID (↓ lower is better)')
        ax.set_title('Kernel Inception Distance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Dataset-Level Distributional Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Qualitative figure grid
    # ------------------------------------------------------------------

    def plot_qualitative_grid(
        self,
        samples: list,
        filename: str = "qualitative_grid",
    ):
        """Publication-ready qualitative comparison grid.

        Parameters
        ----------
        samples : list of dicts
            Each dict has keys: 'name', 'original' (np.ndarray), 'mask',
            and available model outputs (Telea, Navier-Stokes, Vanilla SD,
            FT-SD, FT-SD+TTA, LaMa, MAT, CoModGAN), plus optionally
            'metrics' {model: {metric: val}}.
        """
        methods = list(self.QUALITATIVE_METHOD_ORDER)
        methods = [m for m in methods if m == 'Original' or m == 'Masked' or
                   any(m in s for s in samples)]

        n_rows = len(samples)
        n_cols = len(methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
        if n_rows == 1:
            axes = [axes]

        for row_idx, sample in enumerate(samples):
            for col_idx, method in enumerate(methods):
                ax = axes[row_idx][col_idx]
                ax.axis('off')

                if method == 'Original':
                    img = sample.get('original')
                elif method == 'Masked':
                    img = sample.get('masked_input')
                else:
                    img = sample.get(method)

                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12, color='grey')

                # Column headers on first row
                if row_idx == 0:
                    ax.set_title(method, fontsize=10, fontweight='bold')

                # Row labels
                if col_idx == 0:
                    ax.set_ylabel(sample.get('name', f'Sample {row_idx + 1}'),
                                  fontsize=9, rotation=0, labelpad=60, va='center')

                # PSNR/SSIM annotation for restoration methods
                met = sample.get('metrics', {}).get(method, {})
                if met:
                    label = f"P:{met.get('psnr', 0):.1f} S:{met.get('ssim', 0):.3f}"
                    ax.text(0.5, -0.02, label, ha='center', va='top',
                            transform=ax.transAxes, fontsize=7,
                            color='#333', fontweight='bold')

        plt.suptitle('Qualitative Comparison — Selected Samples',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Mask-type ablation charts
    # ------------------------------------------------------------------

    def plot_mask_type_ablation(
        self,
        df: pd.DataFrame,
        filename: str = "mask_type_ablation",
    ):
        """Grouped bar chart: model performance stratified by mask type.

        Expects df to have a 'mask_type' column.
        """
        if 'mask_type' not in df.columns:
            return
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        mask_types = sorted(subset['mask_type'].unique())
        model_order = self._ordered_models(subset['model'].unique().tolist())
        metrics = ['psnr', 'ssim', 'lpips']
        present = [m for m in metrics if m in subset.columns]

        n_metrics = len(present)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), sharey=False)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, present):
            x = np.arange(len(mask_types))
            width = 0.8 / len(model_order)
            for i, model in enumerate(model_order):
                vals = [subset[(subset['model'] == model) & (subset['mask_type'] == mt)][metric].mean()
                        for mt in mask_types]
                color = ThesisStyle.MODEL_COLORS.get(model, '#999')
                ax.bar(x + i * width, vals, width, label=model, color=color,
                       edgecolor='white', linewidth=0.5)
            ax.set_xticks(x + width * (len(model_order) - 1) / 2)
            ax.set_xticklabels(mask_types)
            higher_better = metric not in ('lpips',)
            direction = '↑' if higher_better else '↓'
            ax.set_title(f'{metric.upper()} ({direction})', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            if ax == axes[0]:
                ax.legend(fontsize=8, framealpha=0.9)

        plt.suptitle('Performance by Mask Type', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_mask_type_heatmap(
        self,
        df: pd.DataFrame,
        filename: str = "mask_type_heatmap",
    ):
        """Heatmap: models × mask_types for each metric."""
        if 'mask_type' not in df.columns:
            return
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [m for m in metrics if m in subset.columns]

        n_metrics = len(present)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, present):
            pivot = subset.pivot_table(index='model', columns='mask_type', values=metric, aggfunc='mean')
            model_order = self._ordered_models(pivot.index.tolist())
            pivot = pivot.reindex([m for m in model_order if m in pivot.index])
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd_r' if metric != 'lpips' else 'YlOrRd',
                        linewidths=1, ax=ax)
            ax.set_title(metric.upper(), fontweight='bold')
            ax.set_ylabel('')
        plt.suptitle('Model × Mask Type Performance', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  Inference timing chart
    # ------------------------------------------------------------------

    def plot_inference_timing(
        self,
        timing_df: pd.DataFrame,
        filename: str = "inference_timing",
    ):
        """Bar chart + table showing inference time per model.

        Parameters
        ----------
        timing_df : DataFrame
            Columns: model, mean_ms, std_ms, images_per_sec
        """
        model_order = self._ordered_models(timing_df['model'].values.tolist())
        timing_df = timing_df.set_index('model').reindex(
            [m for m in model_order if m in timing_df['model'].values]
        ).reset_index()
        colors = [ThesisStyle.MODEL_COLORS.get(m, '#999') for m in timing_df['model']]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart — ms/image (log scale)
        ax = axes[0]
        bars = ax.bar(timing_df['model'], timing_df['mean_ms'], yerr=timing_df['std_ms'],
                      capsize=5, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_yscale('log')
        for bar, v in zip(bars, timing_df['mean_ms']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                    f'{v:.0f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('Inference Time (ms/image, log scale)')
        ax.set_title('Inference Speed', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Speed ratio table
        ax = axes[1]
        ax.axis('off')
        fastest = timing_df['mean_ms'].min()
        rows = []
        for _, r in timing_df.iterrows():
            rows.append([
                r['model'],
                f"{r['mean_ms']:.0f} ± {r['std_ms']:.0f}",
                f"{r['images_per_sec']:.1f}",
                f"{r['mean_ms'] / fastest:.0f}x",
            ])
        table = ax.table(cellText=rows,
                         colLabels=['Model', 'ms/image', 'images/sec', 'Relative'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(ThesisStyle.PALETTE['primary'])
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#f8f8f8' if row % 2 else 'white')
        ax.set_title('Inference Cost Summary', fontweight='bold', pad=20)

        plt.suptitle('Deployment Cost Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  TTA comparison chart
    # ------------------------------------------------------------------

    def plot_tta_comparison(
        self,
        df: pd.DataFrame,
        filename: str = "tta_comparison",
    ):
        """Side-by-side comparison of Ours vs Ours+TTA per metric."""
        subset = df[df['condition'] == 'Unconditional'] if 'condition' in df.columns else df
        tta_models = ['FT-SD', 'FT-SD+TTA']
        tta_models = [m for m in tta_models if m in subset['model'].unique()]
        if len(tta_models) < 2:
            return

        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        present = [m for m in metrics if m in subset.columns]

        means = subset[subset['model'].isin(tta_models)].groupby('model')[present].mean().reindex(tta_models)
        stds = subset[subset['model'].isin(tta_models)].groupby('model')[present].std().reindex(tta_models)

        x = np.arange(len(present))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))

        ours_color = ThesisStyle.MODEL_COLORS.get('FT-SD', '#999')
        tta_color = ThesisStyle.COLORS.get('green', '#66BB6A')

        bars1 = ax.bar(x - width / 2, means.loc['FT-SD'], width, yerr=stds.loc['FT-SD'],
                       capsize=4, label='FT-SD', color=ours_color, edgecolor='white')
        bars2 = ax.bar(x + width / 2, means.loc['FT-SD+TTA'], width, yerr=stds.loc['FT-SD+TTA'],
                       capsize=4, label='FT-SD+TTA', color=tta_color, edgecolor='white')

        # Delta labels
        for i, metric in enumerate(present):
            delta = means.loc['FT-SD+TTA'][metric] - means.loc['FT-SD'][metric]
            direction = '+' if delta > 0 else ''
            ax.text(x[i] + width / 2, means.loc['FT-SD+TTA'][metric] + stds.loc['FT-SD+TTA'][metric] + 0.01,
                    f'{direction}{delta:.4f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color='green' if delta > 0 else 'red')

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in present])
        ax.set_ylabel('Score')
        ax.set_title('Test-Time Augmentation Effect\n(Ours vs Ours + Horizontal Flip Average)',
                     fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename)

    # ------------------------------------------------------------------
    #  V8 scientific completeness charts
    # ------------------------------------------------------------------

    def plot_spatial_contamination_distribution(
        self,
        regeneration_report: dict,
        filename: str = "spatial_contamination_distribution",
    ):
        """Histogram of per-caption contamination ratios with threshold marker."""
        events = regeneration_report.get("events", []) if isinstance(regeneration_report, dict) else []
        if not isinstance(events, list) or not events:
            return

        ratios = []
        for event in events:
            if not isinstance(event, dict):
                continue
            value = event.get("contamination_ratio")
            try:
                ratios.append(float(value))
            except Exception:
                continue
        if not ratios:
            return

        threshold = float(regeneration_report.get("max_contamination_ratio", 0.25) or 0.25)
        clean = sum(1 for x in ratios if x <= threshold)
        contaminated = len(ratios) - clean

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(ratios, bins=20, kde=True, color=ThesisStyle.COLORS['teal'], ax=ax)
        ax.axvline(threshold, color=ThesisStyle.COLORS['wine'], linestyle='--', linewidth=2)
        ax.set_xlabel("Contamination Ratio")
        ax.set_ylabel("Caption Count")
        ax.set_title("Spatial Caption Contamination Distribution", fontweight='bold')
        ax.text(
            0.99,
            0.97,
            f"threshold={threshold:.2f}\\nclean={clean}\\ncontaminated={contaminated}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999"},
        )
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_grounding_validation_panel(
        self,
        grounding_report: dict,
        filename: str = "grounding_validation_panel",
    ):
        """Four-panel Stage06b summary with thresholds and mask-swap baseline."""
        if not isinstance(grounding_report, dict) or not grounding_report:
            return

        metrics = [
            (
                "quadrant_macro_f1",
                "mask_swap_quadrant_macro_f1",
                "min_quadrant_macro_f1",
                "Quadrant Macro-F1",
            ),
            (
                "border_touch_accuracy",
                "mask_swap_border_touch_accuracy",
                "min_border_touch_accuracy",
                "Border-Touch Accuracy",
            ),
            (
                "area_correlation",
                "mask_swap_area_correlation",
                "min_area_correlation",
                "Area Correlation",
            ),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, (real_key, swap_key, thr_key, label) in enumerate(metrics):
            ax = axes[idx]
            real_val = grounding_report.get(real_key)
            swap_val = grounding_report.get(swap_key)
            thr_val = grounding_report.get(thr_key)

            vals = []
            names = []
            colors = []
            for name, value, color in [
                ("Real", real_val, ThesisStyle.COLORS['teal']),
                ("Swap", swap_val, ThesisStyle.COLORS['sand']),
            ]:
                try:
                    vals.append(float(value))
                    names.append(name)
                    colors.append(color)
                except Exception:
                    continue
            if not vals:
                ax.axis('off')
                continue

            ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.7)
            try:
                ax.axhline(float(thr_val), linestyle='--', color=ThesisStyle.COLORS['wine'], linewidth=1.8)
            except Exception:
                pass
            ax.set_title(label, fontweight='bold')
            ax.set_ylim(min(min(vals) - 0.1, -0.1), max(max(vals) + 0.15, 1.0))
            ax.grid(axis='y', alpha=0.25, linestyle='--')

        delta_ax = axes[3]
        delta_payload = grounding_report.get("mask_swap_delta", {})
        delta_rows = []
        for key, label in [
            ("quadrant_macro_f1", "Quadrant F1 Δ"),
            ("border_touch_accuracy", "Border Δ"),
            ("area_correlation", "Area Corr Δ"),
        ]:
            try:
                delta_rows.append((label, float(delta_payload.get(key))))
            except Exception:
                continue
        if delta_rows:
            labels = [x[0] for x in delta_rows]
            values = [x[1] for x in delta_rows]
            colors = [ThesisStyle.COLORS['green'] if v >= 0 else ThesisStyle.COLORS['rose'] for v in values]
            delta_ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.7)
            delta_ax.axhline(0, color='#444', linewidth=1)
            delta_ax.set_title("Real - Swapped Baseline", fontweight='bold')
            delta_ax.tick_params(axis='x', rotation=20)
            delta_ax.grid(axis='y', alpha=0.25, linestyle='--')
        else:
            delta_ax.axis('off')

        passed = grounding_report.get("pass")
        sample_count = grounding_report.get("sample_count", 0)
        plt.suptitle(
            f"Stage06b Grounding Validation (n={sample_count}, pass={passed})",
            fontsize=13,
            fontweight='bold',
            y=1.01,
        )
        plt.tight_layout()
        self.save_plot(filename)

    def plot_regime_source_interaction(
        self,
        interaction_df: pd.DataFrame,
        filename: str = "regime_source_interaction",
    ):
        """2x3 regime-by-source conditioning delta panel with optional CIs."""
        if interaction_df is None or interaction_df.empty:
            return
        required = {"regime", "source_split", "delta_psnr"}
        if not required.issubset(set(interaction_df.columns)):
            return

        reg_order = ["biased", "balanced"]
        src_order = ["wikimedia", "europeana", "combined"]
        working = interaction_df.copy()
        working["regime"] = working["regime"].astype(str).str.lower()
        working["source_split"] = working["source_split"].astype(str).str.lower()

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
        for r, regime in enumerate(reg_order):
            for c, source in enumerate(src_order):
                ax = axes[r, c]
                subset = working[(working["regime"] == regime) & (working["source_split"] == source)]
                if subset.empty:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes, color='#666')
                    ax.set_title(f"{regime.title()} | {source.title()}")
                    ax.axhline(0, color='#444', linewidth=0.8)
                    continue

                delta = float(pd.to_numeric(subset["delta_psnr"], errors="coerce").dropna().mean())
                ci_low = None
                ci_high = None
                if {"ci_low", "ci_high"}.issubset(set(subset.columns)):
                    low_vals = pd.to_numeric(subset["ci_low"], errors="coerce").dropna()
                    high_vals = pd.to_numeric(subset["ci_high"], errors="coerce").dropna()
                    if not low_vals.empty and not high_vals.empty:
                        ci_low = float(low_vals.mean())
                        ci_high = float(high_vals.mean())

                color = ThesisStyle.COLORS['green'] if delta >= 0 else ThesisStyle.COLORS['rose']
                ax.bar(["ΔPSNR"], [delta], color=color, edgecolor='white', linewidth=0.8)
                if ci_low is not None and ci_high is not None and ci_high >= ci_low:
                    err_low = max(0.0, delta - ci_low)
                    err_high = max(0.0, ci_high - delta)
                    ax.errorbar([0], [delta], yerr=[[err_low], [err_high]], fmt='none', ecolor='#333', capsize=5)
                ax.axhline(0, color='#444', linewidth=0.8)
                ax.set_title(f"{regime.title()} | {source.title()}")
                ax.grid(axis='y', alpha=0.25, linestyle='--')

        fig.supylabel("Conditioning Delta (PSNR)")
        plt.suptitle("Regime × Source Interaction: Unconditional − Conditioned", fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_frozen_control_integrity_table(
        self,
        integrity_rows: pd.DataFrame,
        filename: str = "frozen_control_integrity_table",
    ):
        """Compact table-style figure for frozen-control integrity evidence."""
        if integrity_rows is None or integrity_rows.empty:
            return

        df = integrity_rows.copy()
        preferred_cols = [
            "model",
            "samples_hashed",
            "unique_hashes",
            "requires_grad",
            "trainable_params",
            "status",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        if not cols:
            cols = list(df.columns)
        df = df[cols].copy()

        fig, ax = plt.subplots(figsize=(12, max(2.5, 0.6 * len(df) + 1.8)))
        ax.axis('off')
        table = ax.table(
            cellText=df.values,
            colLabels=[c.replace("_", " ").title() for c in df.columns],
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(ThesisStyle.PALETTE['primary'])
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#f7f7f7' if row % 2 else 'white')

        ax.set_title("Frozen Control Integrity Verification", fontweight='bold', pad=12)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_expert_reliability_heatmap(
        self,
        item_level_df: pd.DataFrame,
        reliability_payload: dict | None = None,
        filename: str = "expert_reliability_heatmap",
    ):
        """Heatmap of expert x anchor-item ratings with reliability annotation."""
        if item_level_df is None or item_level_df.empty:
            return
        required = {"participant_id", "item_id", "block", "is_attention_check"}
        if not required.issubset(set(item_level_df.columns)):
            return

        df = item_level_df.copy()
        df = df[df["block"].astype(str) == "A"].copy()
        df = df[df["is_attention_check"].astype(bool)].copy()
        if df.empty:
            return

        for col in ["authenticity_likelihood", "archaeological_plausibility"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if {"authenticity_likelihood", "archaeological_plausibility"}.issubset(set(df.columns)):
            df["anchor_score"] = df[["authenticity_likelihood", "archaeological_plausibility"]].mean(axis=1)
        elif "authenticity_likelihood" in df.columns:
            df["anchor_score"] = df["authenticity_likelihood"]
        else:
            return

        pivot = df.pivot_table(
            index="participant_id",
            columns="item_id",
            values="anchor_score",
            aggfunc="mean",
        )
        if pivot.empty:
            return
        pivot = pivot.sort_index().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(pivot.columns) + 4), max(4, 0.42 * len(pivot.index) + 3)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            linewidths=0.5,
            cbar_kws={'label': 'Mean Anchor Rating (1-5)'},
            ax=ax,
        )
        ax.set_xlabel("Anchor Item ID")
        ax.set_ylabel("Expert / Participant")
        ax.set_title("Expert Anchor Ratings Heatmap", fontweight='bold')

        rel_text = ""
        if isinstance(reliability_payload, dict):
            block_a = reliability_payload.get("block_a", {}) if isinstance(reliability_payload.get("block_a", {}), dict) else {}
            auth = block_a.get("pairwise_within_onepoint_authenticity")
            plaus = block_a.get("pairwise_within_onepoint_plausibility")
            parts = []
            if auth is not None:
                try:
                    parts.append(f"auth±1={float(auth):.3f}")
                except Exception:
                    pass
            if plaus is not None:
                try:
                    parts.append(f"plaus±1={float(plaus):.3f}")
                except Exception:
                    pass
            rel_text = " | ".join(parts)
        if rel_text:
            ax.text(
                1.0,
                1.02,
                rel_text,
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#999"},
            )

        plt.tight_layout()
        self.save_plot(filename)
