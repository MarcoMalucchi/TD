# labplot.py
from pathlib import Path
import copy

#TD root directory automatic detection
TD_ROOT = Path(__file__).resolve().parents[1]


def save_lab_figure(
    fig,
    axes,
    name,
    mode="both",
    folder_standard="logbook",
    folder_presentation="presentazione"
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

    # allow single axis input
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    # ---------------- STANDARD ----------------
    if mode in ["standard", "both"]:

        save_path = resolve_save_path(folder_standard)

        fig.set_size_inches(10, 6)

        fig.savefig(
            save_path/ f"{name}_standard.png",
            dpi=300
        )

    # ---------------- PRESENTATION ----------------
    if mode in ["presentation", "both"]:

        save_pres = resolve_save_path(folder_presentation)

        fig_pres = copy.deepcopy(fig)

        axes_pres = fig_pres.axes

        for ax in axes_pres:
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

        fig_pres.set_size_inches(14, 8)

        fig_pres.savefig(
            save_path / f"{name}_presentation.png",
            dpi=300
        )

import json
import numpy as np
from datetime import datetime

def save_experiment_metadata(
        fig=None,
        axes=None,
        name="exp",
        data=None,
        metadata=None):

    base = get_script_dir()

    #save figure
    if fig is not None:
        save_lab_figure(fig, axes, name, mode="both", folder='figures')

    #save data
    if data is not None:
        np.savetxt(base / 'acquisizioni' / f'{name}.txt', data)

    # save metadata
    if metadata is not None:
        metadata['timestamp'] = datetime.now().isoformat()

        with open(base / 'logbook' / f'{name}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def get_script_dir():
    from pathlib import Path
    import inspect

    caller_file = inspect.stack()[2].filename

    return Path(caller_file).resolve().parent

def resolve_save_path(folder):

    script_dir = get_script_dir()
    save_path = script_dir / folder
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path

def next_acquisition_name(
        folder="acquisizioni",
        prefix=None,
        ):
    
    path = resolve_save_path(folder)
    path.mkdir(exist_ok=True)
    existing = list(path.glob(f'{prefix}_*'))
    numbers=[]

    for f in existing:
        try:
            numbers.append(int(f.stem.split('_')[1]))
        except:
            pass

    n = max(numbers, default=0) + 1

    return f'{prefix}_{n:03d}'
