# labplot.py

import os

def save_lab_figure(
    fig,
    axes,
    name,
    mode="both",
    folder="logbook"
):
    """
    Save figure using lab-standard formatting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axes : matplotlib axis or list of axes
    name : filename root
    mode : 'standard', 'presentation', or 'both'
    folder : save directory
    """

    os.makedirs(folder, exist_ok=True)

    # allow single axis input
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    # ---------------- STANDARD ----------------
    if mode in ["standard", "both"]:
        fig.savefig(
            os.path.join(folder, f"{name}_standard.png"),
            dpi=300
        )

    # ---------------- PRESENTATION ----------------
    if mode in ["presentation", "both"]:

        for ax in axes:
            ax.tick_params(
                axis='both',
                labelsize=16,
                width=2,
                length=8
            )

            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)

            if ax.get_legend():
                ax.legend(fontsize=16)

        fig.set_size_inches(14, 8)

        fig.savefig(
            os.path.join(folder, f"{name}_presentation.png"),
            dpi=300
        )