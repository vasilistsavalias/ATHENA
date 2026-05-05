
# src/thesis_pipeline/visualization/style.py
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

class ThesisStyle:
    """
    Centralized style configuration for all thesis visualizations.
    Implements Paul Tol's 'Muted' color scheme for accessibility and elegance.
    """
    
    # Paul Tol's Muted Palette (Colorblind Safe)
    # Source: https://personal.sron.nl/~pault/
    COLORS = {
        'rose':    '#CC6677',
        'indigo':  '#332288',
        'sand':    '#DDCC77',
        'green':   '#117733',
        'cyan':    '#88CCEE',
        'wine':    '#882255',
        'teal':    '#44AA99',
        'olive':   '#999933',
        'purple':  '#AA4499',
        'grey':    '#DDDDDD'
    }

    # Semantic Mappings
    PALETTE = {
        'primary': COLORS['indigo'],
        'secondary': COLORS['teal'],
        'accent': COLORS['rose'],
        'neutral': COLORS['grey'],
        'success': COLORS['green'],
        'warning': COLORS['sand'],
        'danger': COLORS['wine'],
        'train': COLORS['indigo'],
        'val': COLORS['rose'],
        'test': COLORS['teal']
    }

    # Model-specific colors — used consistently across all evaluation charts
    MODEL_COLORS = {
        'Telea':          COLORS['olive'],
        'Navier-Stokes':  COLORS['sand'],
        'NavierStokes':   COLORS['sand'],
        'Vanilla SD':     COLORS['rose'],
        'FT-SD':          COLORS['indigo'],
        'FT-SD+TTA':      COLORS['green'],
        'LaMa':           COLORS['teal'],
        'MAT':            COLORS['purple'],
        'CoModGAN':       COLORS['wine'],
        # Condition variants (when showing all matrix entries)
        'FT-SD (Unconditional)':  COLORS['indigo'],
        'FT-SD (Raw Text)':       COLORS['teal'],
        'FT-SD (Enriched Text)':  COLORS['cyan'],
        'Vanilla SD (Unconditional)': COLORS['rose'],
        'Vanilla SD (Raw Text)':      COLORS['wine'],
        'Vanilla SD (Enriched Text)': COLORS['purple'],
    }

    # Significance heatmap colors
    SIG_COLORS = {
        'better':     COLORS['green'],   # FT-SD significantly better
        'worse':      COLORS['rose'],    # FT-SD significantly worse
        'ns':         '#F0F0F0',         # Not significant (light grey)
    }

    @staticmethod
    def set_style():
        """
        Applies the standardized matplotlib style parameters.
        Call this function at the start of any plotting script.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color Cycle
        color_cycle = [
            ThesisStyle.COLORS['indigo'],
            ThesisStyle.COLORS['rose'],
            ThesisStyle.COLORS['teal'],
            ThesisStyle.COLORS['sand'],
            ThesisStyle.COLORS['green'],
            ThesisStyle.COLORS['wine'],
            ThesisStyle.COLORS['cyan'],
            ThesisStyle.COLORS['purple']
        ]
        
        plt.rcParams.update({
            # Figure Size (Assuming A4 paper with margins)
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            
            # Fonts
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
            'font.size': 12,
            
            # Axes
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'axes.prop_cycle': cycler(color=color_cycle),
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            
            # Legend
            'legend.fontsize': 12,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.facecolor': 'white',
            'legend.edgecolor': '#DBDBDB',
            
            # Lines
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            
            # Saving
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'savefig.transparent': False,
            'savefig.format': 'png' # Default to PNG per user request
        })

    @staticmethod
    def get_palette_list():
        """Returns the color cycle as a list."""
        return list(ThesisStyle.COLORS.values())
