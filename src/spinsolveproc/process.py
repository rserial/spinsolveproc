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
    """Process Spinsolve Proton files from a given file path.

    Args:
        file_path (Path): Path to the Proton file.
        spinsolve_type (str): Type of Spinsolve (e.g., '43 MHz').

    Returns:
        A tuple containing time scale, FID decay, ppm scale, and spectrum.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if not (file_path / "data.1d").exists():
        raise FileNotFoundError("Data file not found")

    dic, fid_decay = utils.read_autophase_data1d(file_path)
    time_scale = dic["spectrum"]["xaxis"]
    ppm_scale = utils.create_ppm_scale(dic)
    spectrum = utils.fft_autophase(file_path, np.array(fid_decay))

    print("... Done!", "\n")

    return time_scale, fid_decay, ppm_scale, spectrum


def t2(
    file_path: Path,
    spinsolve_type: str,
    integration_center: Optional[float] = None,
    integration_width: Optional[float] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Reads and processes SpinSolve spectroscopically resolved T2 files from a file path.

    Finds the peaks in the acquired spectra and calculates the T2 decay associated with each peak.

    Args:
        file_path (Path): The path to the data file.
        spinsolve_type (str): The type of SpinSolve data.
        integration_center (float, optional): Integration center. Defaults to None.
        integration_width (float, optional): Integration width. Defaults to None.

    Returns:
        A tuple containing the following:
            ppm_scale (ndarray): The chemical shift axis of the 2D spectrum.
            t2_scale (ndarray): The time axis of the T2 decay.
            t2_spec_2d_map (ndarray): The processed 2D spectrum.
            peak_ppm_positions (ndarray): The chemical shift positions of the T2 peaks.
            peak_t2_decay (ndarray): The T2 decay associated with each peak.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the integration width is incorrect.
    """
    if not (file_path / "data.2d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par", procpar="proc.par")

    ppm_scale = utils.create_ppm_scale(dic)
    t2_scale = utils.create_time_scale_t2(file_path, dic, spinsolve_type)
    data_2d = np.reshape(data, (t2_scale.shape[0], ppm_scale.shape[0]))
    t2_spec_2d_map = utils.fft_autophase(file_path, data_2d)

    ppm_scale = ppm_scale[::-1]  # fix ordering of ppm scale

    peak_ppm_positions, peak_t2_decay = utils.find_time_peaks(
        t2_spec_2d_map, ppm_scale, threshold=0.1, msep_factor=0.2
    )

    if integration_width is None:
        integration_width = np.abs((ppm_scale[-1] - ppm_scale[0]) / 10)
        print("Integration width: ", integration_width, "ppm")
    elif integration_width > np.abs(ppm_scale[-1] - ppm_scale[0]):
        raise ValueError("Incorrect integration width")

    if integration_center is None:
        ppm_start = peak_ppm_positions - np.abs(np.round(integration_width / 2))
        ppm_end = peak_ppm_positions + np.abs(np.round(integration_width / 2))
    elif integration_center is not None:
        ppm_start = integration_center - np.abs(np.round(integration_width / 2))
        ppm_end = integration_center + np.abs(np.round(integration_width / 2))

    t1_spec_2d_map_autophased = utils.autophase_2d(t2_spec_2d_map, 0, -1)
    peak_t2_decay = utils.integrate_2d(t1_spec_2d_map_autophased, ppm_scale, ppm_start, ppm_end)
    peak_t2_decay = peak_t2_decay[0]

    print("... Done!!", "\n")
    print("Peaks ppm positions: ", peak_ppm_positions)
    print("Integration width around peak for calculating signal decay:", ppm_start, ppm_end)
    return ppm_scale, t2_scale, t2_spec_2d_map, peak_ppm_positions, peak_t2_decay


def t2_bulk(file_path: Path, spinsolve_type: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Process Spinsolve T2Bulk data and return results.

    Args:
        file_path (Path): Path to the data directory.
        spinsolve_type (str): Type of Spinsolve data.

    Returns:
        T2 time scale and decay data, or None if data file is not found.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if not (file_path / "data.1d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.1d", acqupar="acqu.par", procpar="proc.par")

    t2_scale = utils.create_time_scale_t2_bulk(dic, spinsolve_type)
    t2_decay = data

    print("... Done!!", "\n")
    return t2_scale, t2_decay


def t1(
    file_path: Path,
    spinsolve_type: str,
    integration_center: Optional[float] = None,
    integration_width: Optional[float] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Read and process Spinsolve spectroscopically resolved T1 files.

    Args:
        file_path (Path): Path to Spinsolve T1 data file.
        spinsolve_type (str): Type of Spinsolve data file.
        integration_center (float, optional): Integration center. Defaults to None.
        integration_width (float, optional): Integration width. Defaults to None.

    Returns:
        A tuple containing:
            ppm_scale (ndarray): Ppm scale of acquired spectra.
            t1_scale (ndarray): T1 time scale of acquired spectra.
            t1_spec_2d_map (ndarray): Processed 2D T1 spectrum.
            peak_ppm_positions (ndarray): Ppm positions of detected peaks.
            peak_t1_decay (ndarray): T1 decay values associated with detected peaks.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the integration width is incorrect.
    """
    if not (file_path / "data.2d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par", procpar="proc.par")
    ppm_scale = utils.create_ppm_scale(dic)

    if spinsolve_type == "expert" and dic["acqu"]["delaySpacing"] == "log":
        t1_scale = utils.create_time_scale_t1(dic, log_scale=True)
    else:
        t1_scale = utils.create_time_scale_t1(dic, log_scale=False)

    data_2d = np.reshape(data, (t1_scale.shape[0], ppm_scale.shape[0]))

    t1_spec_2d_map = utils.fft_autophase(file_path, data_2d)

    ppm_scale = ppm_scale[::-1]  # fix ordering of ppm scale

    peak_ppm_positions, peak_t1_decay = utils.find_time_peaks(
        t1_spec_2d_map, ppm_scale, threshold=0.1, msep_factor=0.2
    )

    if integration_width is None:
        integration_width = np.abs((ppm_scale[-1] - ppm_scale[0]) / 10)
        print("Integration width: ", integration_width, "ppm")
    elif integration_width > np.abs(ppm_scale[-1] - ppm_scale[0]):
        raise ValueError("Incorrect integration width")

    if integration_center is None:
        ppm_start = peak_ppm_positions - np.abs(np.round(integration_width / 2))
        ppm_end = peak_ppm_positions + np.abs(np.round(integration_width / 2))
    elif integration_center is not None:
        ppm_start = integration_center - np.abs(np.round(integration_width / 2))
        ppm_end = integration_center + np.abs(np.round(integration_width / 2))

    t1_spec_2d_map_autophased = utils.autophase_2d(t1_spec_2d_map, 0, -1)
    peak_t1_decay = utils.integrate_2d(t1_spec_2d_map_autophased, ppm_scale, ppm_start, ppm_end)
    peak_t1_decay = peak_t1_decay[0]

    print("... Done!!", "\n")
    print("Peaks ppm positions: ", peak_ppm_positions)
    print("Integration width around peak for calculating signal decay:", ppm_start, ppm_end)
    return ppm_scale, t1_scale, t1_spec_2d_map_autophased, peak_ppm_positions, peak_t1_decay


def t1ir_t2(file_path: Path, spinsolve_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process Spinsolve T1IRT2 data and return results.

    Args:
        file_path (Path): Path to the data directory.
        spinsolve_type (str): Type of Spinsolve data.

    Returns:
        Time scales and T1IRT2 data.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    time_t1, time_t2 = utils.load_t1_ir_t2_timedat_files(file_path)
    data1d_path_list = [
        path for path in sorted((file_path / "1D_T1IRT2").iterdir()) if path.is_dir()
    ]

    t1ir_t2_list = []
    for path in data1d_path_list:
        if not (path / "data.1d").exists():
            raise FileNotFoundError("Data file not found")
        dic, t2_decay = ngread_modified.read(
            path, "data.1d", acqupar="acqu.par", procpar="proc.par", split_opt="no"
        )
        t1ir_t2_list.append(t2_decay)

    t1ir_t2_array = np.array(np.real(t1ir_t2_list), dtype=np.float64)
    print("... Done!!", "\n")
    return time_t1, time_t2, t1ir_t2_array


def pgste(
    file_path: Path,
    spinsolve_type: str,
    integration_center: Optional[float] = None,
    integration_width: Optional[float] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Read and process Spinsolve spectroscopically resolved T1 files.

    Args:
        file_path (Path): Path to Spinsolve T1 data file.
        spinsolve_type (str): Type of Spinsolve data file.
        integration_center (float, optional): Integration center. Defaults to None.
        integration_width (float, optional): Integration width. Defaults to None.

    Returns:
        A tuple containing:
            ppm_scale (ndarray): Ppm scale of acquired spectra.
            t1_scale (ndarray): T1 time scale of acquired spectra.
            t1_spec_2d_map (ndarray): Processed 2D T1 spectrum.
            peak_ppm_positions (ndarray): Ppm positions of detected peaks.
            peak_diff_decay (ndarray): T1 decay values associated with detected peaks.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the integration width is incorrect.
    """
    if not (file_path / "data.2d").exists():
        raise FileNotFoundError("Data file not found")

    dic, data = ngread_modified.read(file_path, "data.2d", acqupar="acqu.par")

    ppm_scale = utils.create_ppm_scale(dic)
    grad_scale = np.loadtxt(file_path / "gradients.par")  # in T/m
    diff_scale = utils.create_diff_scale(dic, grad_scale)  # (s/m²)

    data_2d = np.reshape(data, (grad_scale.shape[0], ppm_scale.shape[0]))

    diff_spec_2d_map = utils.fft_autophase(file_path, data_2d)

    peak_ppm_positions, peak_t1_decay = utils.find_time_peaks(
        diff_spec_2d_map, ppm_scale, threshold=0.1, msep_factor=0.2
    )

    if integration_width is None:
        integration_width = np.abs((ppm_scale[-1] - ppm_scale[0]) / 10)
        print("Integration width: ", integration_width, "ppm")
    elif integration_width > np.abs(ppm_scale[-1] - ppm_scale[0]):
        raise ValueError("Incorrect integration width")

    if integration_center is None:
        ppm_start = peak_ppm_positions - np.abs(np.round(integration_width / 2))
        ppm_end = peak_ppm_positions + np.abs(np.round(integration_width / 2))
    elif integration_center is not None:
        ppm_start = integration_center - np.abs(np.round(integration_width / 2))
        ppm_end = integration_center + np.abs(np.round(integration_width / 2))

    diff_spec_2d_map_autophased = utils.autophase_2d(diff_spec_2d_map, 0, -1)
    peak_diff_decay = utils.integrate_2d(
        diff_spec_2d_map_autophased, ppm_scale, ppm_start, ppm_end
    )
    peak_diff_decay = peak_t1_decay[0]

    print("... Done!!", "\n")
    print("Peaks ppm positions: ", peak_ppm_positions)
    print("Integration width around peak for calculating signal decay:", ppm_start, ppm_end)
    return ppm_scale, diff_scale, diff_spec_2d_map_autophased, peak_ppm_positions, peak_diff_decay
