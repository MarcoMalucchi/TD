# labplot.py
from pathlib import Path

#TD root directory automatic detection
TD_ROOT = Path(__file__).resolve().parents[1]


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
    #=====================================
    # BUILD FULL PATH RELATIVE TO TD ROOT.
    #=====================================

    save_path = TD_ROOT / folder
    save_path.mkdir(parents=True, exist_ok=True)

    #=====================================

    # allow single axis input
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    # ---------------- STANDARD ----------------
    if mode in ["standard", "both"]:
        fig.savefig(
            save_path/ f"{name}_standard.png",
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
            save_path / f"{name}_presentation.png",
            dpi=300
        )

import json
import numpy as np
from datetime import datetime

def save_experiment_metadata(fig, axes, name, data=None, metadata=None, folder="logbook"):

    base = TD_ROOT / folder / name
    base.mkdir(parents=True, exist_ok=True)

    #save figure
    save_lab_figure(fig, axes, name, mode="both", folder=f'{folder}/{name}')

    #save data
    if data is not None:
        np.savetxt(base / 'data.txt', data)

    # save metadata
    if metadata is not None:
        metadata['timestamp'] = datetime.now().isoformat()

        with open(base / 'metadata.json', 'w') as f:
            json.dumb(metadata, f, indent=2)
