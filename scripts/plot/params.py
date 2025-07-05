import scanpy as sc
import seaborn as sns
from matplotlib import rcParams


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["savefig.bbox"] = "tight"
    sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
    sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)
