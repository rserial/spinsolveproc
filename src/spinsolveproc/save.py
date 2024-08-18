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
    """Save a Proton experiment figure to the specified directory.

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
    fid_decay: np.ndarray,
    ppm_scale: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    """Save Proton experiment data and related figures to the specified directory.

    Args:
        save_dir (Path): The directory where the data and figures will be saved.
        time_scale (np.ndarray): Time scale data for the FID decay.
        fid_decay (np.ndarray): FID decay data.
        ppm_scale (np.ndarray): PPM scale data for the proton spectrum.
        spectrum (np.ndarray): Proton spectrum data.
    """
    filename_proton_decay = "Proton_decay.dat"
    filename_proton_spectrum = "Proton_spectrum.dat"
    save_1d_decay_data(save_dir, time_scale, fid_decay, filename_proton_decay)
    save_1d_spectrum_data(save_dir, ppm_scale, spectrum, filename_proton_spectrum)


def save_1d_decay_data(
    save_dir: Path, time_scale: np.ndarray, decay: np.ndarray, filename: str
) -> None:
    """Save 1D decay data to a text file in the specified directory.

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
    """Save 1D spectrum data to a text file in the specified directory.

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


def fig_t2(save_dir: Path, *figures: go.Figure) -> None:
    """Save T2 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        *figures (go.Figure): Variable number of figures to save.
    """
    filename_save = ["T2spec_2Dmap.html", "T2decay.html"]
    for i, figure in enumerate(figures):
        pio.write_html(figure, save_dir / filename_save[i])
        print(f"Saved figure: {filename_save[i]}\n")


def fig_t1(save_dir: Path, *figures: go.Figure) -> None:
    """Save T1 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        *figures (go.Figure): Variable number of figures to save.
    """
    filename_save = ["T1spec_2Dmap.html", "T1decay.html"]
    for i, figure in enumerate(figures):
        pio.write_html(figure, save_dir / filename_save[i])
        print(f"Saved figure: {filename_save[i]}\n")


def fig_t1ir_t2(save_dir: Path, figure: go.Figure) -> None:
    """Save T1 figures to the specified directory.

    Args:
        save_dir (Path): The directory where the figures will be saved.
        figure (go.Figure): figure to save.
    """
    filename_save = "T1IRT2_2Dmap.html"
    pio.write_html(figure, save_dir / filename_save)
    print(f"Saved figure: {filename_save}\n")


def data_t2(
    save_dir: Path,
    ppm_scale: np.ndarray,
    t2_scale: np.ndarray,
    t2_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_t2_decay: np.ndarray,
) -> None:
    """Save T2 data to specified directory.

    Args:
        save_dir (Path): The directory where data will be saved.
        ppm_scale (np.ndarray): An array of ppm scale values.
        t2_scale (np.ndarray): An array of T2 scale values in seconds.
        t2_spec_2d_map (np.ndarray): A 2D map of T2 data.
        peak_ppm_positions (np.ndarray): An array of peak ppm positions.
        peak_t2_decay (np.ndarray): A 2D array of peak T2 decay data.
    """
    data_t2_bulk(save_dir, t2_scale, peak_t2_decay.reshape(-1))

    h5_filename = "T2spec_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "ppm_axis"
    time_axis = "T2_axis"

    # Save processed T2spec 2D map
    save_2d_t_spectrum_and_axes_to_hdf5(
        save_dir,
        t2_spec_2d_map,
        ppm_scale,
        t2_scale,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename} \n")


def data_t1(
    save_dir: Path,
    ppm_scale: np.ndarray,
    t1_scale: np.ndarray,
    t1_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_t1_decay: np.ndarray,
) -> None:
    """Save T1 data to specified directory.

    Args:
        save_dir (Path): The directory where data will be saved.
        ppm_scale (np.ndarray): An array of ppm scale values.
        t1_scale (np.ndarray): An array of T2 scale values in seconds.
        t1_spec_2d_map (np.ndarray): A 2D map of T2 data.
        peak_ppm_positions (np.ndarray): An array of peak ppm positions.
        peak_t1_decay (np.ndarray): A 2D array of peak T2 decay data.
    """
    filename_t1_decay = "T1decay.dat"
    filename_t1_fitting = "T1decay_exp_fitting.dat"
    data_t_decay(
        save_dir,
        t1_scale,
        peak_t1_decay.reshape(-1),
        "T1IR",
        filename_t1_decay,
        filename_t1_fitting,
    )

    h5_filename = "T1spec_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "ppm_axis"
    time_axis = "T1_axis"

    # Save processed T2spec 2D map
    save_2d_t_spectrum_and_axes_to_hdf5(
        save_dir,
        t1_spec_2d_map,
        ppm_scale,
        t1_scale,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename} \n")


def data_pgste(
    save_dir: Path,
    ppm_scale: np.ndarray,
    diff_scale: np.ndarray,
    diff_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_diff_decay: np.ndarray,
) -> None:
    """Save diffusion data to specified directory.

    Args:
        save_dir (Path): The directory where data will be saved.
        ppm_scale (np.ndarray): An array of ppm scale values.
        diff_scale (np.ndarray): An array of diffusion scale values in seconds.
        diff_spec_2d_map (np.ndarray): A 2D map of spectroscopically resolved diffusion data.
        peak_ppm_positions (np.ndarray): An array of peak ppm positions.
        peak_diff_decay (np.ndarray): A 1D array of diffusion decay data.
    """
    filename_pgste_decay = "pgste_decay.dat"
    filename_pgste_fitting = "pgste_decay_1exp_fitting.dat"
    data_diff_decay(
        save_dir,
        diff_scale * 1e-9,
        peak_diff_decay.reshape(-1),
        "PGSTE",
        filename_pgste_decay,
        filename_pgste_fitting,
    )


def data_t1ir_t2(
    save_dir: Path, time_t1: np.ndarray, time_t2: np.ndarray, t1ir_t2_array: np.ndarray
) -> None:
    """Save T1IRT2 data to an HDF5 file.

    Args:
        save_dir (Path): Directory to save the data.
        time_t1 (np.ndarray): Time axis for T1.
        time_t2 (np.ndarray): Time axis for T2.
        t1ir_t2_array (np.ndarray): T1IRT2 data.
    """
    h5_filename = "T1IRT2_2Ddata.h5"
    data_filename = "2Dmap"
    frequency_axis = "T1_axis"
    time_axis = "T2_axis"

    save_2d_t_spectrum_and_axes_to_hdf5(
        save_dir,
        t1ir_t2_array,
        time_t1,
        time_t2,
        h5_filename,
        data_filename,
        frequency_axis,
        time_axis,
    )
    print(f"Saved h5py file: {h5_filename}\n")


def save_2d_t_spectrum_and_axes_to_hdf5(
    save_dir: Path,
    t_spectrum_2d: np.ndarray,
    frequency_axis: np.ndarray,
    time_axis: np.ndarray,
    h5_filename: str,
    data_filename: str,
    axis1_filename: str,
    axis2_filename: str,
) -> None:
    """Save a 2D Tspectrum and axes to an HDF5 file.

    Args:
        save_dir (Path): The directory where the HDF5 file will be saved.
        t_spectrum_2d (np.ndarray): The 2D Tspectrum data.
        frequency_axis (np.ndarray): The frequency axis data.
        time_axis (np.ndarray): The time axis data.
        h5_filename (str): The name of the HDF5 file.
        data_filename (str): The name of the dataset for t_spectrum_2d.
        axis1_filename (str): The name of the dataset for frequency_axis.
        axis2_filename (str): The name of the dataset for time_axis.
    """
    with h5py.File(save_dir / h5_filename, "w") as h5f:
        h5f.create_dataset(data_filename, data=t_spectrum_2d)
        h5f.create_dataset(axis1_filename, data=frequency_axis)
        h5f.create_dataset(axis2_filename, data=time_axis)


def save_t_decay_fit_parameters(
    save_dir: Path,
    fit_t_decay_filename: str,
    amplitude: List[Any],
    err_amplitude: List[Any],
    decay: List[Any],
    err_decay: List[Any],
    intercept: List[Any],
) -> None:
    """Save T decay fit parameters to a CSV file.

    Args:
        save_dir (Path): The directory where the CSV file will be saved.
        fit_t_decay_filename (str): The name of the CSV file.
        amplitude (List[Any]): The amplitude of the fit.
        err_amplitude (List[Any]): The error of the amplitude of the fit.
        decay (List[Any]): The decay time in seconds.
        err_decay (List[Any]): The error of the decay time in seconds.
        intercept (List[Any]): The fit intercept.
    """
    list_fit_t_decay = {
        "Amplitude": [amplitude],
        "Err Amplitude [a.u]": err_amplitude,
        "Time decay [s]": [decay],
        "Err Time decay [s]": err_decay,
        "fit intercept": [intercept],
    }

    df = pd.DataFrame(
        list_fit_t_decay,
        columns=["Amplitude [a.u]", "Err Amplitude [a.u]", "Time decay [s]", "Err Time decay [s]"],
    )
    df.to_csv(save_dir / fit_t_decay_filename, sep="\t", index=False)
    print(f"Saved datafile: {fit_t_decay_filename}\n")


def data_t2_bulk(save_dir: Path, t2_scale: np.ndarray, t2_decay: np.ndarray) -> None:
    """Save T2Bulk decay data and perform exponential fitting.

    Args:
        save_dir (Path): Directory to save the data and fitting results.
        t2_scale (np.ndarray): Array containing the time scale for T2Bulk decay.
        t2_decay (np.ndarray): Array containing the T2Bulk decay data.
    """
    filename_t2_bulk_decay = "T2Bulkdecay.dat"
    filename_t2_bulk_fitting = "T2Bulkdecay_exp_fitting.dat"

    data_t_decay(
        save_dir,
        t2_scale,
        t2_decay,
        "T2",
        filename_t2_bulk_decay,
        filename_t2_bulk_fitting,
    )


def data_t_decay(
    save_dir: Path,
    t_scale: np.ndarray,
    t_decay: np.ndarray,
    kernel_name: str,
    filename_decay: str,
    filename_fitting: str,
) -> None:
    """Save Time decay data and perform exponential fitting.

    Args:
        save_dir (Path): Directory to save the data and fitting results.
        t_scale (np.ndarray): Array containing the time scale for T2Bulk decay.
        t_decay (np.ndarray): Array containing the T2Bulk decay data.
        kernel_name (str): Kernel name (options are: T1IR, T1ST, T2).
        filename_decay (str): filename of time decay.
        filename_fitting (str): filename of exponential fitting parameters.
    """
    save_1d_decay_data(save_dir, t_scale, t_decay, filename_decay)

    # Fitting
    exponentials = 1
    fitted_parameters, r2, cov = utils.fit_multiexponential(
        t_scale, np.real(t_decay), kernel_name=kernel_name, num_exponentials=exponentials
    )
    err = np.sqrt(np.diag(cov))

    amplitude = []
    time_decay = []
    intercept = []

    err_amplitude = []
    err_time_decay = []

    for i in range(exponentials):
        amplitude.append(fitted_parameters[i * 2])
        time_decay.append(1 / fitted_parameters[i * 2 + 1])
        err_amplitude.append(err[i * 2])
        err_time_decay.append(err[i * 2 + 1] / fitted_parameters[i * 2 + 1] ** 2)
    intercept.append(fitted_parameters[-1])

    save_t_decay_fit_parameters(
        save_dir, filename_fitting, amplitude, err_amplitude, time_decay, err_time_decay, intercept
    )


def fig_t2_bulk(save_dir: Path, fig_t2_bulk_decays_fit: go.Figure) -> None:
    """Save the T2Bulk decay figure to a specified directory.

    Args:
        save_dir (Path): Directory where the figure will be saved.
        fig_t2_bulk_decays_fit: The figure to be saved.
    """
    filename_save = "T2Bulkdecay.html"
    pio.write_html(fig_t2_bulk_decays_fit, save_dir / filename_save)
    print(f"Saved figure: {filename_save} \n")


def data_diff_decay(
    save_dir: Path,
    diff_scale: np.ndarray,
    diff_decay: np.ndarray,
    kernel_name: str,
    filename_decay: str,
    filename_fitting: str,
) -> None:
    """Save diffusion decay data and perform exponential fitting.

    Args:
        save_dir (Path): Directory to save the data and fitting results.
        diff_scale (np.ndarray): Array containing the diffusion scale for diffusion decay.
        diff_decay (np.ndarray): Array containing the diffusion decay data.
        kernel_name (str): Kernel name (options are: PGSTE).
        filename_decay (str): filename of time decay.
        filename_fitting (str): filename of exponential fitting parameters.
    """
    save_1d_decay_data(save_dir, diff_scale, diff_decay, filename_decay)

    # Fitting
    exponentials = 1
    fitted_parameters, r2, cov = utils.fit_multiexponential(
        diff_scale,
        np.real(diff_decay),
        kernel_name,
        num_exponentials=exponentials,
    )

    err = np.sqrt(np.diag(cov))

    amplitude = []
    err_amplitude = []
    diffusion_decay = []
    err_diffusion_decay = []

    for i in range(exponentials):
        amplitude.append(fitted_parameters[i * 2])
        diffusion_decay.append(fitted_parameters[i * 2 + 1])
        err_amplitude.append(err[i * 2])
        err_diffusion_decay.append(err[i * 2 + 1])

    list_fit_t_decay = {
        "Amplitude [a.u]": amplitude,
        "Err Amplitude [a.u]": err_amplitude,
        "Diffusion decay [s]": diffusion_decay,
        "Err Diffusion decay [s]": err_diffusion_decay,
    }
    fit_dataframe = pd.DataFrame(
        list_fit_t_decay,
        columns=[
            "Amplitude [a.u]",
            "Err Amplitude [a.u]",
            "Diffusion decay [s]",
            "Err Diffusion decay [s]",
        ],
    )

    save_dataframe_to_text(save_dir, fit_dataframe)


def save_dataframe_to_text(
    save_dir: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Save a dataframe to a text file.

    Args:
        save_dir (Path): Path saving directory
        dataframe (pd.DataFrame): The dataframe to be saved.
    """
    filename = "multiexponential_fit_parameters.dat"
    dataframe.to_csv(save_dir / filename, sep="\t", index=False)
    print(f"Saved fit parameters in: {filename}\n")
