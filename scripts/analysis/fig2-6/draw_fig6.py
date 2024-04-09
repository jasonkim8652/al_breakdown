import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_list =[]
receptor_names = ["SOS1", "EGFR", "ESR2", "JAK2", "PARP1", "PGR"]
for receptor in receptor_names:
    df = pd.read_csv("/home/jasonkjh/works/data/"+receptor+"/"+receptor+"_inf_ucb_ordering.csv")
    df = df[(df["Dock_order"] < 1e4) & (df["Pred_order"] < 1e4)]
    df_list.append(df)


fig, axs = plt.subplots(
    2, 3, figsize=(6.5, 4), sharex=True, sharey=True, layout="constrained"
)
for i,df in enumerate(df_list):
    ax = axs[i // 3, i % 3]
    *_, hist = ax.hist2d(
        df["Dock_order"],
        df["Pred_order"],
        bins=np.linspace(0, 10000, 11),
        cmap="Blues",
        vmin=0,
        vmax=400,  # adjust if necessary
    )
    for j in range(10):
        pc = plt.Rectangle(
            (1000 * j, 1000 * j),
            1000,
            1000,
            fill=False,
            edgecolor="k",
            linewidth=1,
            alpha=0.5,
        )
        ax.add_patch(pc)
    text = ax.text(0.07, 0.87, receptor_names[i], transform=ax.transAxes)
    text.set_bbox(dict(facecolor="w", alpha=0.7, linewidth=0))
    if i // 3 == 1:
        ax.set_xlabel("Docking score rank")
    if i % 3 == 0:
        ax.set_ylabel("Predicted rank")
    if i == 2:
        ax_cb, hist_cb = ax, hist

fig.canvas.draw()
l, b, w, h = ax_cb.get_position().bounds
cax = fig.add_axes([l + w + 0.03, b, 0.02, h])
fig.colorbar(hist_cb, cax=cax, orientation="vertical")
cax.set_ylabel("Count")

fig.savefig("../figures/fig6.png", dpi=300, facecolor="w", bbox_inches="tight")
