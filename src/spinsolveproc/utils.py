"""Processing functions for spinsolveproc."""

import re
from typing import Dict, Optional, Tuple, Union

import nmrglue as ng
import numpy as np


def parse_spinsolve_par_line(line: str) -> Tuple[str, Union[str, int, float]]:
    """
    Parse lines in acqu.par and return a tuple (parameter name, parameter value).

    Args:
        line (str): The line to be parsed from the acqu.par file.

    Returns:
        tuple: A tuple containing the parameter name (str) and its value (str, int, or float).
    """
    line = line.strip()  # Drop newline
    name, value = line.split(
        "=", maxsplit=1
    )  # Split at equal sign (and ignore further equals in attribute values)

    # remove spaces
    name = name.strip()
    value = value.strip()

    # Detect value type
    if value[0] == value[-1] == '"':  # When flanked with " always go for string
        return name, str(value[1:-1])  # Drop quote marks
    else:
        try:
            return name, int(
                value
            )  # ValueError when value == str(1.1), but not when value == str(1)
        except ValueError:
            try:
                return name, float(value)
            except ValueError:
                return name, str(value)


def parse_spinsolve_script_line(
    line: str,
) -> Tuple[Optional[str], Optional[Union[str, Tuple[float, int]]]]:
    """
    Parse a line from a Spinsolve script file and extract relevant information.

    Args:
        line (str): The line to be parsed.

    Returns:
        Tuple: A tuple containing the parsed parameter name (str) and its value (various types).
               If the line does not match the expected format, (None, None) is returned.
    """
    # Define regular expression to match the Phase line
    phase_regex = r"Phase\((\d+\.\d+),(\d+)\);"
    if re.match(phase_regex, line):
        value1, value2 = re.findall(phase_regex, line)[0]
        return "Phase", (float(value1), int(value2))
    else:
        return None, None


def get_initial_phase(file_path: str) -> float:
    """
    Read processing.script to get the initial phase.

    Args:
        file_path (str): The path to the directory containing the processing script.

    Returns:
        float: The initial phase value.
    """
    # Read processing.script to get the initial phase
    parameters_script = {}
    with open(file_path / "processing.script", "r") as f:
        info = f.readlines()
    for line in info:
        par_name, par_value = parse_spinsolve_script_line(line)
        parameters_script[par_name] = par_value

    for line in info:
        par_name, par_value = parse_spinsolve_script_line(line)
        if par_name:
            parameters_script[par_name] = par_value
    phase = parameters_script.get("Phase", [0])[0]
    return phase


def read_autophase_data1d(file_path: str) -> Tuple[dict, list]:
    """
    Read data.1d file and perform autophasing on FIDdecay.

    Args:
        file_path (str): The path to the data directory.

    Returns:
        Tuple: A tuple containing the dictionary (dic) and the autophased FIDdecay.
    """
    dic, FIDdecay = ng.spinsolve.read(file_path, "data.1d", acqupar="acqu.par", procpar="proc.par")
    phase = get_initial_phase(file_path)
    # Autophase FIDdecay
    FIDdecay = ng.proc_autophase.autops(FIDdecay, fn="acme", p0=phase, disp=False)
    return dic, FIDdecay


def fft_autophase(file_path: str, FIDdecay: np.ndarray) -> np.ndarray:
    """
    Perform Fourier transformation and autophase on the spectrum.

    Args:
        file_path (str): The path to the data directory.
        FIDdecay (np.ndarray): The FIDdecay data.

    Returns:
        np.ndarray: The autophased spectrum.
    """
    spectrum = ng.proc_base.fft(FIDdecay)  # Fourier transformation
    # Autophase spectrum
    phase = get_initial_phase(file_path)
    spectrum = ng.proc_autophase.autops(spectrum, fn="acme", p0=phase, disp=False)
    return spectrum


def create_ppm_scale(dic: Dict[str, Union[str, int, float]]) -> np.ndarray:
    """
    Creates a PPM scale from the dictionary file of a Spinsolve T1 data file.

    Args:
        dic (Dict[str, Union[str, int, float]]): The dictionary file of the Spinsolve T1 data file.

    Returns:
        np.ndarray: An array containing the PPM scale of the acquired spectra.
    """
    udic = ng.fileiobase.create_blank_udic(1)
    udic[0]["sw"] = (
        float(dic["acqu"]["bandwidth"]) * 1000
    )  # Spectral width in Hz - or width of the whole spectrum

    b1Freq = str(dic["acqu"]["b1Freq"])
    if "d" in b1Freq:
        b1Freq = b1Freq.rstrip("d")
    udic[0]["obs"] = float(b1Freq)

    udic[0]["size"] = float(
        dic["acqu"]["nrPnts"]
    )  # Number of points - from acqu (float(dic["acqu"]["nrPnts"])

    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    return ppm_scale
