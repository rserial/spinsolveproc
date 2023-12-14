"""Processing functions for spinsolveproc."""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

import spinsolveproc.ngread_modified as ngread_modified
import spinsolveproc.utils as utils

warnings.simplefilter(action="ignore", category=FutureWarning)


def proton(file_path: Path, spinsolve_type: str) -> Tuple:
    """
    Process Spinsolve Proton files from a given file path.

    Args:
        file_path (Path): Path to the Proton file.
        spinsolve_type (str): Type of Spinsolve (e.g., '43 MHz').

    Returns:
        Tuple: A tuple containing time scale, FID decay, ppm scale, and spectrum.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if not (file_path / "data.1d").exists():
        raise FileNotFoundError("Data file not found")

    dic, FIDdecay = utils.read_autophase_data1d(file_path)
    time_scale = dic["spectrum"]["xaxis"]
    ppm_scale = utils.create_ppm_scale(dic)
    spectrum = utils.fft_autophase(file_path, np.array(FIDdecay))

    print("... Done!", "\n")

    return time_scale, FIDdecay, ppm_scale, spectrum


def T2(file_path: Path, spinsolve_type: str) -> tuple:
    """
    Reads and processes SpinSolve spectroscopically resolved T2 files from a file path.

    Finds the peaks in the acquired spectra and calculates the T2 decay associated with each peak.

    Args:
        file_path (Path): The path to the data file.
        spinsolve_type (str): The type of SpinSolve data.

    Returns:
        tuple: A tuple containing the following:
            ppm_scale (ndarray): The chemical shift axis of the 2D spectrum.
            T2_scale (ndarray): The time axis of the T2 decay.
            T2spec_2Dmap (ndarray): The processed 2D spectrum.
            peak_ppm_positions (ndarray): The chemical shift positions of the T2 peaks.
            peak_T2decay (ndarray): The T2 decay associated with each peak.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if not (file_path / "data.2d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par", procpar="proc.par")

    ppm_scale = utils.create_ppm_scale(dic)
    T2_scale = utils.create_time_scale_T2(file_path, dic, spinsolve_type)
    data2D = np.reshape(data, (T2_scale.shape[0], ppm_scale.shape[0]))
    T2spec_2Dmap = utils.fft_autophase(file_path, data2D)
    peak_ppm_positions, peak_T2decay = utils.find_Tpeaks(T2spec_2Dmap, ppm_scale)
    peak_T2decay = peak_T2decay[0]
    print("... Done!!", "\n")
    return ppm_scale, T2_scale, T2spec_2Dmap, peak_ppm_positions, peak_T2decay


def T2Bulk(file_path: Path, spinsolve_type: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Process Spinsolve T2Bulk data and return results.

    Args:
        file_path (Path): Path to the data directory.
        spinsolve_type (str): Type of Spinsolve data.

    Returns:
        Union[Tuple[ndarray, ndarray], None]: T2 time scale and decay data,
        or None if data file is not found.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if not (file_path / "data.1d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.1d", acqupar="acqu.par", procpar="proc.par")

    T2_scale = utils.create_time_scale_T2Bulk(dic, spinsolve_type)
    T2decay = data

    print("... Done!!", "\n")
    return T2_scale, T2decay


def T1(
    file_path: Path,
    spinsolve_type: str,
    integration_center: Optional[float] = None,
    integration_width: Optional[float] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Read and process Spinsolve spectroscopically resolved T1 files.

    Args:
        file_path (Path): Path to Spinsolve T1 data file.
        spinsolve_type (str): Type of Spinsolve data file.
        integration_center (float, optional): Integration center. Defaults to None.
        integration_width (float, optional): Integration width. Defaults to None.

    Returns:
        Tuple:
            ppm_scale (ndarray): Ppm scale of acquired spectra.
            T1_scale (ndarray): T1 time scale of acquired spectra.
            T1spec_2Dmap (ndarray): Processed 2D T1 spectrum.
            peak_ppm_positions (ndarray): Ppm positions of detected peaks.
            peak_T1decay (ndarray): T1 decay values associated with detected peaks.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the integration width is incorrect.
    """
    if not (file_path / "data.2d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par", procpar="proc.par")
    ppm_scale = utils.create_ppm_scale(dic)

    if spinsolve_type == "expert" and dic["acqu"]["delaySpacing"] == "log":
        T1_scale = utils.create_time_scale_T1(dic, log_scale=True)
    else:
        T1_scale = utils.create_time_scale_T1(dic, log_scale=False)

    data2D = np.reshape(data, (T1_scale.shape[0], ppm_scale.shape[0]))

    T1spec_2Dmap = utils.fft_autophase(file_path, data2D)
    peak_ppm_positions, peak_T1decay = utils.find_Tpeaks(
        T1spec_2Dmap, ppm_scale, threshold=0.1, msep_factor=0.2
    )

    ppm_scale = ppm_scale[::-1]

    if integration_width is None:
        integration_width = (ppm_scale[-1] - ppm_scale[0]) / 10
        print("Integration width: ", integration_width, "ppm")
    elif integration_width > np.abs(ppm_scale[-1] - ppm_scale[0]):
        raise ValueError("Incorrect integration width")

    if integration_center is None:
        ppm_start = peak_ppm_positions - np.abs(np.round(integration_width / 2))
        ppm_end = peak_ppm_positions + np.abs(np.round(integration_width / 2))
    elif integration_center is not None:
        ppm_start = integration_center - np.abs(np.round(integration_width / 2))
        ppm_end = integration_center + np.abs(np.round(integration_width / 2))

    T1spec_2Dmap_autophased = utils.autophase_2D(T1spec_2Dmap, 0, -1)
    peak_T1decay = utils.integrate_2D(T1spec_2Dmap_autophased, ppm_scale, ppm_start, ppm_end)
    peak_T1decay = peak_T1decay[0]

    print("... Done!!", "\n")
    print("Peaks ppm positions: ", peak_ppm_positions)
    print("Integration width around peak for calculating signal decay:", ppm_start, ppm_end)
    return ppm_scale, T1_scale, T1spec_2Dmap_autophased, peak_ppm_positions, peak_T1decay


def T1IRT2(file_path: Path, spinsolve_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process Spinsolve T1IRT2 data and return results.

    Args:
        file_path (Path): Path to the data directory.
        spinsolve_type (str): Type of Spinsolve data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Time scales and T1IRT2 data.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    timeT1, timeT2 = utils.load_T1IRT2_timedat_files(file_path)
    data1d_path_list = [
        path for path in sorted((file_path / "1D_T1IRT2").iterdir()) if path.is_dir()
    ]

    T1IRT2 = []
    for path in data1d_path_list:
        if not (path / "data.1d").exists():
            raise FileNotFoundError("Data file not found")
        dic, T2decay = ngread_modified.read(
            path, "data.1d", acqupar="acqu.par", procpar="proc.par", split_opt="no"
        )
        T1IRT2.append(T2decay)

    T1IRT2array = np.array(np.real(T1IRT2), dtype=np.float64)
    print("... Done!!", "\n")
    return timeT1, timeT2, T1IRT2array
