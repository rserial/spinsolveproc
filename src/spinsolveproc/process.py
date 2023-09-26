"""Processing functions for spinsolveproc."""

import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np

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
    """
    if not (file_path / "data.1d").exists():
        print("... Error! Data file not found", "\n")

    dic, FIDdecay = utils.read_autophase_data1d(file_path)
    time_scale = dic["spectrum"]["xaxis"]
    ppm_scale = utils.create_ppm_scale(dic)
    spectrum = utils.fft_autophase(file_path, FIDdecay)

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
            ppm_scale (np.ndarray): The chemical shift axis of the 2D spectrum.
            T2_scale (np.ndarray): The time axis of the T2 decay.
            T2spec_2Dmap (np.ndarray): The processed 2D spectrum.
            peak_ppm_positions (np.ndarray): The chemical shift positions of the T2 peaks.
            peak_T2decay (np.ndarray): The T2 decay associated with each peak.
    """
    if not (file_path / "data.2d").exists():
        print("Error: Data file not found", "\n")
        return None

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par", procpar="proc.par")

    ppm_scale = utils.create_ppm_scale(dic)
    T2_scale = utils.create_time_scale_T2(file_path, dic, spinsolve_type)
    data2D = np.reshape(data, (T2_scale.shape[0], ppm_scale.shape[0]))
    T2spec_2Dmap = utils.fft_autophase(file_path, data2D)
    peak_ppm_positions, peak_T2decay = utils.find_Tpeaks(T2spec_2Dmap, ppm_scale)
    print("... Done!!", "\n")
    return ppm_scale, T2_scale, T2spec_2Dmap, peak_ppm_positions, peak_T2decay


def T2Bulk(file_path: Path, spinsolve_type: str) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Process Spinsolve T2Bulk data and return results.

    Args:
        file_path (Path): Path to the data directory.
        spinsolve_type (str): Type of Spinsolve data.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], None]: T2 time scale and decay data,
        or None if data file is not found.
    """
    if not (file_path / "data.1d").exists():
        print("Error: Data file not found", "\n")
        return None

    dic, data = ngread_modified.read(file_path, "data.1d", acqupar="acqu.par", procpar="proc.par")

    T2_scale = utils.create_time_scale_T2Bulk(dic, spinsolve_type)
    T2decay = data

    print("... Done!!", "\n")
    return T2_scale, T2decay
