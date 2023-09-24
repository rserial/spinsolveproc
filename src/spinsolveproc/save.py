"""Plotting save for spinsolveproc."""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.graph_objs._figure import Figure


# Figures
def fig_proton(save_dir: Path, fig_proton: Figure) -> None:
    """
    Save a Proton experiment figure to the specified directory.

    Args:
        save_dir (Path): The directory where the figure will be saved.
        fig_proton (Figure): The Proton experiment figure to save.
    """
    filename_save = "FID_decay_spectrum.jpg"
    pio.write_image(fig_proton, save_dir / filename_save, engine="kaleido")
    print(f"Saved figure: {filename_save}\n")


# Data
def data_proton(
    save_dir: Path,
    time_scale: List[float],
    FIDdecay: List[complex],
    ppm_scale: List[float],
    spectrum: List[complex],
) -> None:
    """
    Save Proton experiment data and related figures to the specified directory.

    Args:
        save_dir (Path): The directory where the data and figures will be saved.
        time_scale (List[float]): Time scale data for the FID decay.
        FIDdecay (List[complex]): FID decay data.
        ppm_scale (List[float]): PPM scale data for the proton spectrum.
        spectrum (List[complex]): Proton spectrum data.
    """
    filename_proton_decay = "Proton_decay"
    filename_proton_spectrum = "Proton_spectrum"
    save_1d_decay_data(save_dir, time_scale, FIDdecay, filename_proton_decay)
    save_1d_spectrum_data(save_dir, ppm_scale, spectrum, filename_proton_spectrum)


def save_1d_decay_data(
    save_dir: Path, time_scale: List[float], decay: List[complex], filename: str
) -> None:
    """
    Save 1D decay data to a text file in the specified directory.

    Args:
        save_dir (Path): The directory where the data file will be saved.
        time_scale (List[float]): Time scale data.
        decay (List[complex]): Decay data.
        filename (str): The name of the data file.
    """
    save_decay = {
        "time [s]": time_scale,
        "decay real [a.u]": np.real(decay),
        "decay imag [a.u]": np.imag(decay),
    }
    df = pd.DataFrame(save_decay, columns=["time [s]", "decay real [a.u]", "decay imag [a.u]"])
    df.to_csv(save_dir / filename, sep="\t", float_format="%10.4f", header=False, index=False)
    print(f"Saved datafile: {filename}\n")


def save_1d_spectrum_data(
    save_dir: Path, ppm_scale: List[float], spectrum: List[complex], filename: str
) -> None:
    """
    Save 1D spectrum data to a text file in the specified directory.

    Args:
        save_dir (Path): The directory where the data file will be saved.
        ppm_scale (List[float]): PPM scale data.
        spectrum (List[complex]): Spectrum data.
        filename (str): The name of the data file.
    """
    save_spectrum = {
        "[ppm]": ppm_scale,
        "spectrum real [a.u]": np.real(spectrum),
        "spectrum imag [a.u]": np.imag(spectrum),
        "spectrum absolute [a.u]": np.abs(spectrum),
    }
    df = pd.DataFrame(
        save_spectrum,
        columns=["[ppm]", "spectrum real [a.u]", "spectrum imag [a.u]", "spectrum absolute [a.u]"],
    )
    df.to_csv(save_dir / filename, sep="\t", float_format="%10.4f", header=False, index=False)
    print(f"Saved datafile: {filename}\n")
