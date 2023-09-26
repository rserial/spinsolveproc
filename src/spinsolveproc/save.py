"""Plotting save for spinsolveproc."""
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.graph_objs._figure import Figure

import spinsolveproc.utils as utils


# Figures
def fig_proton(save_dir: Path, fig_proton: Figure) -> None:
    """
    Save a Proton experiment figure to the specified directory.

    Args:
        save_dir (Path): The directory where the figure will be saved.
        fig_proton (Figure): The Proton experiment figure to save.
    """
    filename_save = "FID_decay_spectrum.html"
    pio.write_html(fig_proton, save_dir / filename_save)
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


def fig_T2(save_dir: Path, fig_T2spec_2Dmap: Figure, fig_T2specdecays_fit: Figure) -> None:
    """
    Save both T2 figures to a specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        fig_T2spec_2Dmap (Figure): The Plotly figure for T2 spec 2D map.
        fig_T2specdecays_fit (Figure): The Plotly figure for T2 spec decays fit.
    """
    save_fig_T2spec_2Dmap(save_dir, fig_T2spec_2Dmap)
    save_fig_T2specdecays_fit(save_dir, fig_T2specdecays_fit)


def save_fig_T2spec_2Dmap(save_dir: Path, fig_T2spec_2Dmap: Figure) -> None:
    """
    Save the T2 spec 2D map figure to a specified directory.

    Args:
        save_dir (Path): The directory where the figure will be saved.
        fig_T2spec_2Dmap (Figure): The Plotly figure to be saved.
    """
    filename_save = "T2spec_2Dmap.html"
    pio.write_html(fig_T2spec_2Dmap, save_dir / filename_save)
    print(f"Saved figure: {filename_save} \n")


def save_fig_T2specdecays_fit(save_dir: Path, fig_T2specdecays_fit: Figure) -> None:
    """
    Save the T2 spec decays fit figure to a specified directory.

    Args:
        save_dir (Path): The directory where the figure will be saved.
        fig_T2specdecays_fit (Figure): The Plotly figure to be saved.
    """
    filename_save = "T2decay.html"
    pio.write_html(fig_T2specdecays_fit, save_dir / filename_save)
    print(f"Saved figure: {filename_save} \n")


def data_T2(
    save_dir: Path,
    ppm_scale: np.ndarray,
    T2_scale: np.ndarray,
    T2spec_2Dmap: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_T2decay: np.ndarray,
) -> None:
    """
    Save T2 data to specified directory.

    Args:
        save_dir (Path): The directory where data will be saved.
        ppm_scale (np.ndarray): An array of ppm scale values.
        T2_scale (np.ndarray): An array of T2 scale values in seconds.
        T2spec_2Dmap (np.ndarray): A 2D map of T2 data.
        peak_ppm_positions (np.ndarray): An array of peak ppm positions.
        peak_T2decay (np.ndarray): A 2D array of peak T2 decay data.
    """
    amplitude, decay, intercept, R2 = utils.fit_monoexponential(
        T2_scale, np.real(peak_T2decay[0, :]), utils.mono_exponential
    )

    T2decay_filename = f"T2decay_Peak_{np.round(peak_ppm_positions[0], 1)}ppm.dat"
    save_1d_decay_data(save_dir, T2_scale, peak_T2decay[0, :], T2decay_filename)

    fitT2decay_filename = f"fitT2decay_Peak_{np.round(peak_ppm_positions[0], 1)}_Monoexp.dat"
    save_T_decay_fit_parameters(save_dir, fitT2decay_filename, amplitude, decay, intercept)

    h5_filename = "T2spec_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "ppm_axis"
    time_axis = "T2_axis"

    # Save processed T2spec 2D map
    save_2d_Tspectrum_and_axes_to_hdf5(
        save_dir,
        T2spec_2Dmap,
        ppm_scale,
        T2_scale,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename} \n")


def save_2d_Tspectrum_and_axes_to_hdf5(
    save_dir: Path,
    Tspectrum_2d: np.ndarray,
    frequency_axis: np.ndarray,
    time_axis: np.ndarray,
    h5_filename: str,
    data_filename: str,
    axis1_filename: str,
    axis2_filename: str,
) -> None:
    """
    Save a 2D Tspectrum and axes to an HDF5 file.

    Args:
        save_dir (Path): The directory where the HDF5 file will be saved.
        Tspectrum_2d (np.ndarray): The 2D Tspectrum data.
        frequency_axis (np.ndarray): The frequency axis data.
        time_axis (np.ndarray): The time axis data.
        h5_filename (str): The name of the HDF5 file.
        data_filename (str): The name of the dataset for Tspectrum_2d.
        axis1_filename (str): The name of the dataset for frequency_axis.
        axis2_filename (str): The name of the dataset for time_axis.
    """
    with h5py.File(save_dir / h5_filename, "w") as h5f:
        h5f.create_dataset(data_filename, data=Tspectrum_2d)
        h5f.create_dataset(axis1_filename, data=frequency_axis)
        h5f.create_dataset(axis2_filename, data=time_axis)


def save_T_decay_fit_parameters(
    save_dir: Path, fitTdecay_filename: str, amplitude: float, decay: float, intercept: float
) -> None:
    """
    Save T decay fit parameters to a CSV file.

    Args:
        save_dir (Path): The directory where the CSV file will be saved.
        fitTdecay_filename (str): The name of the CSV file.
        amplitude (float): The amplitude of the fit.
        decay (float): The decay time in seconds.
        intercept (float): The fit intercept.
    """
    list_fitTdecay = {
        "Amplitude": [amplitude],
        "Time decay [s]": [decay],
        "fit intercept": [intercept],
    }
    df = pd.DataFrame(list_fitTdecay, columns=["Amplitude", "Time decay [s]", "fit intercept"])
    df.to_csv(save_dir / fitTdecay_filename, sep="\t", index=False)


def data_T2Bulk(save_dir: str, T2_scale: np.ndarray, T2decay: np.ndarray) -> None:
    """
    Save T2Bulk decay data and perform exponential fitting.

    Args:
        save_dir (str): Directory to save the data and fitting results.
        T2_scale (np.ndarray): Array containing the time scale for T2Bulk decay.
        T2decay (np.ndarray): Array containing the T2Bulk decay data.
    """
    filename_T2Bulk_decay = "T2Bulkdecay.dat"
    filename_fitting_T2Bulk_decay = "T2Bulkdecay_exp_fitting.dat"

    save_1d_decay_data(save_dir, T2_scale, T2decay, filename_T2Bulk_decay)

    # Fitting
    exponentials = 1
    fitted_parameters, R2 = utils.fit_multiexponential(
        T2_scale, np.real(T2decay), kernel_name="T2", num_exponentials=exponentials
    )

    amplitude = []
    time_decay = []
    intercept = []

    for i in range(exponentials):
        amplitude.append(fitted_parameters[i * 2])
        time_decay.append(1 / fitted_parameters[i * 2 + 1])
    intercept.append(fitted_parameters[-1])

    save_T_decay_fit_parameters(
        save_dir, filename_fitting_T2Bulk_decay, amplitude, time_decay, intercept
    )


def fig_T2Bulk(save_dir: Path, fig_T2Bulkdecays_fit: Figure) -> None:
    """
    Save the T2Bulk decay figure to a specified directory.

    Args:
        save_dir (Path): Directory where the figure will be saved.
        fig_T2Bulkdecays_fit: The figure to be saved.
    """
    filename_save = "T2Bulkdecay.html"
    pio.write_html(fig_T2Bulkdecays_fit, save_dir / filename_save)
    print(f"Saved figure: {filename_save} \n")
