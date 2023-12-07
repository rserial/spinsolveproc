"""Plotting save for spinsolveproc."""
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import spinsolveproc.utils as utils


# Figures
def fig_proton(save_dir: Path, fig_proton: go.Figure) -> None:
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
    time_scale: np.ndarray,
    FIDdecay: np.ndarray,
    ppm_scale: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    """
    Save Proton experiment data and related figures to the specified directory.

    Args:
        save_dir (Path): The directory where the data and figures will be saved.
        time_scale (np.ndarray): Time scale data for the FID decay.
        FIDdecay (np.ndarray): FID decay data.
        ppm_scale (np.ndarray): PPM scale data for the proton spectrum.
        spectrum (np.ndarray): Proton spectrum data.
    """
    filename_proton_decay = "Proton_decay.dat"
    filename_proton_spectrum = "Proton_spectrum.dat"
    save_1d_decay_data(save_dir, time_scale, FIDdecay, filename_proton_decay)
    save_1d_spectrum_data(save_dir, ppm_scale, spectrum, filename_proton_spectrum)


def save_1d_decay_data(
    save_dir: Path, time_scale: np.ndarray, decay: np.ndarray, filename: str
) -> None:
    """
    Save 1D decay data to a text file in the specified directory.

    Args:
        save_dir (Path): The directory where the data file will be saved.
        time_scale (np.ndarray): Time scale data.
        decay (np.ndarray): Decay data.
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
    save_dir: Path, ppm_scale: np.ndarray, spectrum: np.ndarray, filename: str
) -> None:
    """
    Save 1D spectrum data to a text file in the specified directory.

    Args:
        save_dir (Path): The directory where the data file will be saved.
        ppm_scale (np.ndarray): PPM scale data.
        spectrum (np.ndarray): Spectrum data.
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


def fig_T2(save_dir: Path, *figures: go.Figure) -> None:
    """
    Save T2 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        *figures (go.Figure): Variable number of figures to save.
    """
    filename_save = ["T2spec_2Dmap.html", "T2decay.html"]
    for i, figure in enumerate(figures):
        pio.write_html(figure, save_dir / filename_save[i])
        print(f"Saved figure: {filename_save[i]}\n")


def fig_T1(save_dir: Path, *figures: go.Figure) -> None:
    """
    Save T1 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        *figures (go.Figure): Variable number of figures to save.
    """
    filename_save = ["T1spec_2Dmap.html", "T1decay.html"]
    for i, figure in enumerate(figures):
        pio.write_html(figure, save_dir / filename_save[i])
        print(f"Saved figure: {filename_save[i]}\n")


def fig_T1IRT2(save_dir: Path, figure: go.Figure) -> None:
    """
    Save T1 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        figure (go.Figure): figure to save.
    """
    filename_save = "T1IRT2_2Dmap.html"
    pio.write_html(figure, save_dir / filename_save)
    print(f"Saved figure: {filename_save}\n")


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
    data_T2Bulk(save_dir, T2_scale, peak_T2decay.reshape(-1))

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


def data_T1(
    save_dir: Path,
    ppm_scale: np.ndarray,
    T1_scale: np.ndarray,
    T1spec_2Dmap: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_T1decay: np.ndarray,
) -> None:
    """
    Save T1 data to specified directory.

    Args:
        save_dir (Path): The directory where data will be saved.
        ppm_scale (np.ndarray): An array of ppm scale values.
        T1_scale (np.ndarray): An array of T2 scale values in seconds.
        T1spec_2Dmap (np.ndarray): A 2D map of T2 data.
        peak_ppm_positions (np.ndarray): An array of peak ppm positions.
        peak_T1decay (np.ndarray): A 2D array of peak T2 decay data.
    """
    filename_T1decay = "T1decay.dat"
    filename_T1fitting = "T1decay_exp_fitting.dat"
    data_Tdecay(
        save_dir,
        T1_scale,
        peak_T1decay.reshape(-1),
        "T1IR",
        filename_T1decay,
        filename_T1fitting,
    )

    h5_filename = "T1spec_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "ppm_axis"
    time_axis = "T1_axis"

    # Save processed T2spec 2D map
    save_2d_Tspectrum_and_axes_to_hdf5(
        save_dir,
        T1spec_2Dmap,
        ppm_scale,
        T1_scale,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename} \n")


def data_T1IRT2(
    save_dir: Path, timeT1: np.ndarray, timeT2: np.ndarray, T1IRT2array: np.ndarray
) -> None:
    """
    Save T1IRT2 data to an HDF5 file.

    Args:
        save_dir (Path): Directory to save the data.
        timeT1 (np.ndarray): Time axis for T1.
        timeT2 (np.ndarray): Time axis for T2.
        T1IRT2array (np.ndarray): T1IRT2 data.
    """
    h5_filename = "T1IRT2_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "T1_axis"
    time_axis = "T2_axis"

    save_2d_Tspectrum_and_axes_to_hdf5(
        save_dir,
        T1IRT2array,
        timeT1,
        timeT2,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename}\n")


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
    save_dir: Path,
    fitTdecay_filename: str,
    amplitude: List[Any],
    decay: List[Any],
    intercept: List[Any],
) -> None:
    """
    Save T decay fit parameters to a CSV file.

    Args:
        save_dir (Path): The directory where the CSV file will be saved.
        fitTdecay_filename (str): The name of the CSV file.
        amplitude (List[Any]): The amplitude of the fit.
        decay (List[Any]): The decay time in seconds.
        intercept (List[Any]): The fit intercept.
    """
    list_fitTdecay = {
        "Amplitude": [amplitude],
        "Time decay [s]": [decay],
        "fit intercept": [intercept],
    }
    df = pd.DataFrame(list_fitTdecay, columns=["Amplitude", "Time decay [s]", "fit intercept"])
    df.to_csv(save_dir / fitTdecay_filename, sep="\t", index=False)
    print(f"Saved datafile: {fitTdecay_filename}\n")


def data_T2Bulk(save_dir: Path, T2_scale: np.ndarray, T2decay: np.ndarray) -> None:
    """
    Save T2Bulk decay data and perform exponential fitting.

    Args:
        save_dir (Path): Directory to save the data and fitting results.
        T2_scale (np.ndarray): Array containing the time scale for T2Bulk decay.
        T2decay (np.ndarray): Array containing the T2Bulk decay data.
    """
    filename_T2Bulkdecay = "T2Bulkdecay.dat"
    filename_T2Bulkfitting = "T2Bulkdecay_exp_fitting.dat"

    data_Tdecay(
        save_dir,
        T2_scale,
        T2decay,
        "T2",
        filename_T2Bulkdecay,
        filename_T2Bulkfitting,
    )


def data_Tdecay(
    save_dir: Path,
    T_scale: np.ndarray,
    Tdecay: np.ndarray,
    kernel_name: str,
    filename_decay: str,
    filename_fitting: str,
) -> None:
    """
    Save Time decay data and perform exponential fitting.

    Args:
        save_dir (Path): Directory to save the data and fitting results.
        T_scale (np.ndarray): Array containing the time scale for T2Bulk decay.
        Tdecay (np.ndarray): Array containing the T2Bulk decay data.
        kernel_name (str): Kernel name (options are: T1IR, T1ST, T2).
        filename_decay (str): filename of time decay.
        filename_fitting (str): filename of exponential fitting parameters.
    """
    save_1d_decay_data(save_dir, T_scale, Tdecay, filename_decay)

    # Fitting
    exponentials = 1
    fitted_parameters, R2 = utils.fit_multiexponential(
        T_scale, np.real(Tdecay), kernel_name=kernel_name, num_exponentials=exponentials
    )

    amplitude = []
    time_decay = []
    intercept = []

    for i in range(exponentials):
        amplitude.append(fitted_parameters[i * 2])
        time_decay.append(1 / fitted_parameters[i * 2 + 1])
    intercept.append(fitted_parameters[-1])

    save_T_decay_fit_parameters(save_dir, filename_fitting, amplitude, time_decay, intercept)


def fig_T2Bulk(save_dir: Path, fig_T2Bulkdecays_fit: go.Figure) -> None:
    """
    Save the T2Bulk decay figure to a specified directory.

    Args:
        save_dir (Path): Directory where the figure will be saved.
        fig_T2Bulkdecays_fit: The figure to be saved.
    """
    filename_save = "T2Bulkdecay.html"
    pio.write_html(fig_T2Bulkdecays_fit, save_dir / filename_save)
    print(f"Saved figure: {filename_save} \n")
