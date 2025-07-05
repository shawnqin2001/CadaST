import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

work_dir = "/home/qinxianhan/project/spatial/cadast_demo/"
data_path = f"{work_dir}/output/slide/1126/out_filter.h5ad"
fig_path = f"{work_dir}/figure/visiumLC/"


sns.set_theme(style="white", font="Helvetica", font_scale=1.5)
dpi = 150
output_format = "png"  # Options: "png", "pdf", "both"


def save_figure(fig, filename):
    filename = os.path.join(fig_path, filename)
    print("Saving figure to", filename)
    if output_format in ["png", "both"]:
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if output_format in ["pdf", "both"]:
        fig.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches="tight")
