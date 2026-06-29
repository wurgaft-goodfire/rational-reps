#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


######### FigureGenerator class for clean plotting #########

class FigureGenerator:
    """Centralized figure generation with consistent styling and configuration."""
    
    def __init__(self):
        self.colors = {
            'blue': "#00BFFF",
            'paleblue': '#b9e0fa',
            'lightgray': '#cccacc',
            'pink': '#eb9bdf', 
            'green': '#abe0a4',
        }
        
        self.font_sizes = {
            'title': 30,
            'axis_label': 30,
            'tick': 22,
            'legend': 20,
            'sideplot_title': 25,
            'sideplot_axis_label': 16,
            'sideplot_tick': 14,
            'sideplot_legend': 15,
            'sideplot_colorbar': 15
        }
        
        self._setup_matplotlib_defaults()
    
    
    def _setup_matplotlib_defaults(self):
        """Configure global matplotlib settings."""
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42 
        
        # Font setup
        avenir_fonts = [font for font in fm.findSystemFonts(fontext='ttc') if 'Avenir' in font]
        if avenir_fonts:
            avenir_path = avenir_fonts[0]
            font_prop = fm.FontProperties(fname=avenir_path)
            font_name = font_prop.get_name()
            fm.fontManager.addfont(avenir_path)
            plt.rcParams['font.family'] = font_name
            plt.rcParams["axes.labelweight"] = "regular"
            print(f"Registered Avenir font from: {avenir_path}")
        else:
            plt.rcParams['font.family'] = 'sans-serif'
            print("Avenir font not found, using serif instead.")
    
    def style_axes(self, ax, show_spines=True, spine_width=3, tick_size=28, tick_length=6, 
                   tick_width=2, tick_pad=10, remove_minor_ticks_x=True, add_arrows=False):
        """Apply consistent styling to axes."""
        if show_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(spine_width)
            ax.spines['bottom'].set_linewidth(spine_width)
            ax.spines['left'].set_zorder(10)
            ax.spines['bottom'].set_zorder(10)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_size, 
                       length=tick_length, width=tick_width, pad=tick_pad)
        
        if remove_minor_ticks_x:
            ax.tick_params(axis='x', which='minor', bottom=False)

    
    def save_figure(self, fig, filepath, format='pdf', dpi=1000):
        """Save figure with publication-quality settings and selective rasterization."""
        for ax in fig.get_axes():
            for collection in ax.collections:
                if collection.__class__.__name__ in ['QuadMesh', 'AxesImage']:
                    collection.set_rasterized(True)
                
                if hasattr(collection, '_offsets') and hasattr(collection._offsets, 'shape'):
                    if collection._offsets.shape[0] > 500:
                        collection.set_rasterized(True)
                        print(f"Rasterized {collection._offsets.shape[0]} points")
        
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=format)
        print(f"Saved figure to {filepath}")