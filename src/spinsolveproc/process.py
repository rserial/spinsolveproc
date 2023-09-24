"""Processing functions for spinsolveproc."""

from pathlib import Path
from typing import Tuple

from src.spinsolveproc import utils


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
