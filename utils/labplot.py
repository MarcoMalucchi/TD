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

        fig_std = copy.deepcopy(fig)
        save_path = resolve_save_path(folder_standard)

        fig_std.set_size_inches(10, 6)

        fig_std.savefig(
            save_path / f"{name}_standard.png",
            dpi=300
        )


    # ---------------- PRESENTATION ----------------
    if mode in ["presentation", "both"]:

        fig_pres = copy.deepcopy(fig)
        save_pres = resolve_save_path(folder_presentation)

        for ax in fig_pres.axes:
            ax.tick_params(axis='both',
                        labelsize=16,
                        width=2,
                        length=8)

            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)

            if ax.get_legend():
                ax.legend(fontsize=16)

        fig_pres.set_size_inches(14, 8)

        fig_pres.savefig(
            save_pres / f"{name}_presentation.png",
            dpi=300
        )

import json
import numpy as np
from datetime import datetime

def save_experiment_metadata(
        fig=None,
        axes=None,
        prefix="exp",
        data=None,
        metadata=None):

    #base = get_script_dir()

    name = next_acquisition_name(prefix=prefix)

    #save figure
    if fig is not None:
        save_lab_figure(fig, axes, name, mode="both", folder_standard='logbook', folder_presentation='presentazione')

    #save data
    if data is not None:
        data_path = resolve_save_path('acquisizioni')

        np.savetxt(data_path / f'{name}.txt', data)

    # save metadata
    if metadata is not None:
        metadata_path = resolve_save_path('logbook')
        metadata['timestamp'] = datetime.now().isoformat()

        with open(metadata_path / f'{name}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

import inspect

def get_script_dir():

    this_file = Path(__file__).resolve()

    for frame in inspect.stack():
        filename = Path(frame.filename).resolve()

        if filename != this_file and 'utils' not in str(filename):
            return filename.parent

def resolve_save_path(folder):

    script_dir = get_script_dir()
    save_path = script_dir / folder
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path

def next_acquisition_name(
        folder="acquisizioni",
        prefix='exp',
        ):
    
    path = resolve_save_path(folder)
    #path.mkdir(exist_ok=True)
    existing = list(path.glob(f'{prefix}_*'))
    numbers=[]

    for f in existing:
        try:
            numbers.append(int(f.stem.split('_')[1]))
        except:
            pass

    n = max(numbers, default=0) + 1

    return f'{prefix}_{n:03d}'
