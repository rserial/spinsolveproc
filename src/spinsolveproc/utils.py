"""Processing functions for spinsolveproc."""

import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nmrglue as ng
import numpy as np
import scipy.optimize
from numpy import ndarray

warnings.simplefilter(action="ignore", category=FutureWarning)


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


def get_initial_phase(file_path: Path) -> float:
    """
    Read processing.script to get the initial phase.

    Args:
        file_path (Path): The path to the directory containing the processing script.

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
    # Retrieve the "Phase" value from parameters_script, and ensure it's a float
    phase_value = parameters_script.get("Phase")

    if isinstance(phase_value, tuple):
        if len(phase_value) > 0:
            # Use the first element of the tuple as the phase value
            phase = float(phase_value[0])
        else:
            phase = 0.0
    elif isinstance(phase_value, (float, int)):
        phase = float(phase_value)
    elif isinstance(phase_value, str):
        # If "Phase" is a string, attempt to convert it to a float
        try:
            phase = float(phase_value)
        except ValueError:
            # Handle the case where the conversion to float fails
            phase = 0.0
    else:
        # Handle other cases like None or a list
        phase = 0.0  # You can choose a default value here

    return phase


def read_autophase_data1d(file_path: Path) -> Tuple[dict, List]:
    """
    Read data.1d file and perform autophasing on FIDdecay.

    Args:
        file_path (Path): The path to the data directory.

    Returns:
        Tuple: A tuple containing the dictionary (dic) and the autophased FIDdecay.
    """
    dic, FIDdecay = ng.spinsolve.read(file_path, "data.1d", acqupar="acqu.par", procpar="proc.par")
    phase = get_initial_phase(file_path)
    # Autophase FIDdecay
    FIDdecay = ng.proc_autophase.autops(FIDdecay, fn="acme", p0=phase, disp=False)
    return dic, FIDdecay


def fft_autophase(file_path: Path, FIDdecay: ndarray) -> ndarray:
    """
    Perform Fourier transformation and autophase on the spectrum.

    Args:
        file_path (Path): The path to the data directory.
        FIDdecay (ndarray): The FIDdecay data.

    Returns:
        Spectrum (ndarray): The autophased spectrum.
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

    # Ensure that "acqu" key is in the dictionary and that its values are of the expected types
    if "acqu" in dic and isinstance(dic["acqu"], dict):
        acqu = dic["acqu"]

        if "bandwidth" in acqu and isinstance(acqu["bandwidth"], (int, float)):
            udic[0]["sw"] = float(acqu["bandwidth"]) * 1000  # Spectral width in Hz

        if "b1Freq" in acqu and isinstance(acqu["b1Freq"], (str, int, float)):
            b1Freq = str(acqu["b1Freq"]).rstrip("d")
            udic[0]["obs"] = float(b1Freq)

        if "nrPnts" in acqu and isinstance(acqu["nrPnts"], (int, float)):
            udic[0]["size"] = float(acqu["nrPnts"])  # Number of points

    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    return ppm_scale


def create_time_scale_T1(
    dic: Dict[str, Union[str, int, float]], log_scale: bool = True
) -> np.ndarray:
    """
    Create time scale for T1 decay from the dictionary file of a Spinsolve T1 data file.

    Args:
        dic (dict): The dictionary file of the Spinsolve T1 data file.
        log_scale (bool): If True, create a logarithmic scale; otherwise, create a linear scale.

    Returns:
        An array containing the time scale for T1 decay of the acquired spectra.

    Raises:
        ValueError: If the dictionary format is invalid or required values are missing.
    """
    min_delay: Optional[float] = None
    max_delay: Optional[float] = None
    nr_steps: Optional[int] = None

    if "acqu" in dic and isinstance(dic["acqu"], dict):
        acqu = dic["acqu"]

        if "minDelay" in acqu and isinstance(acqu["minDelay"], (float, int)):
            min_delay = float(acqu["minDelay"])
        if "maxDelay" in acqu and isinstance(acqu["maxDelay"], (float, int)):
            max_delay = float(acqu["maxDelay"])
        if "nrSteps" in acqu and isinstance(acqu["nrSteps"], (int, float)):
            nr_steps = int(acqu["nrSteps"])

    if min_delay is not None and max_delay is not None and nr_steps is not None:
        if log_scale:
            t1_scale = np.logspace(
                np.log10(min_delay * 1e-3), np.log10(max_delay * 1e-3), nr_steps
            )
        else:
            t1_scale = np.linspace(min_delay * 1e-3, max_delay * 1e-3, nr_steps)  # Linear scale
        return t1_scale

    # Raise an error if the necessary values are not available
    raise ValueError(
        "Invalid dictionary format or missing required values for time scale creation."
    )


def create_diff_scale(
    dic: Dict[str, Union[str, int, float]],
    grad_scale: np.ndarray,
) -> np.ndarray:
    """
    Create diffusion scale for PGSTE decay from the dictionary file of a Spinsolve PGSTE data file.

    Args:
        dic (dict): The dictionary file of the Spinsolve T1 data file.
        grad_scale(ndarray): array containing gradient intensities.

    Returns:
        An array containing the diffusion axis for PGSTE experiment.

    Raises:
        ValueError: If the dictionary format is invalid or required values are missing.
    """
    gamma_1h = 267.52218744
    small_delta: Optional[float] = None
    big_delta: Optional[float] = None
    nr_steps: Optional[int] = None

    if "acqu" in dic and isinstance(dic["acqu"], dict):
        acqu = dic["acqu"]

        if "lDelta" in acqu and isinstance(acqu["lDelta"], (float, int)):
            small_delta = float(acqu["lDelta"])
        if "bDelta" in acqu and isinstance(acqu["bDelta"], (float, int)):
            big_delta = float(acqu["bDelta"])
        if "nrSteps" in acqu and isinstance(acqu["nrSteps"], (int, float)):
            nr_steps = int(acqu["nrSteps"])

    if small_delta is not None and big_delta is not None and nr_steps is not None:
        diff_scale = (
            gamma_1h**2 * grad_scale**2 * small_delta**2 * (big_delta - small_delta / 3)
        )
        return diff_scale

    # Raise an error if the necessary values are not available
    raise ValueError(
        "Invalid dictionary format or missing required values for time scale creation."
    )


def find_Tpeaks(
    Tspec_2Dmap: np.ndarray,
    ppm_scale: np.ndarray,
    threshold: float = 0.1,
    msep_factor: float = 0.2,
) -> tuple:
    """
    Find peaks in a 2D spectrum.

    Args:
        Tspec_2Dmap (np.ndarray): 2D spectrum map.
        ppm_scale (np.ndarray): PPM scale.
        threshold (float): Peak detection threshold.
        msep_factor (float): Multiplet separation factor.

    Returns:
        tuple: Tuple containing peak ppm positions and peak T2 decay data.
    """
    # Find spectrum peaks
    peaks = ng.peakpick.pick(
        np.abs(Tspec_2Dmap[0, :]),
        threshold * np.max(np.abs(Tspec_2Dmap[0, :])),
        algorithm="thres",
        msep=int(msep_factor * Tspec_2Dmap[0, :].shape[0]),
        table=True,
    )

    # Find peak positions in ppm
    peak_ppm_positions = np.array([ppm_scale[int(peak["X_AXIS"])] for peak in peaks])

    # Construct peaks T2 decay data
    peak_T2decay = np.array([Tspec_2Dmap[:, int(peak["X_AXIS"])] for peak in peaks])

    return peak_ppm_positions, peak_T2decay


def autophase_time_decay(v: np.ndarray) -> np.ndarray:
    """
    Autophase a time decay signal.

    Args:
        v (np.ndarray): Input time decay signal.

    Returns:
        np.ndarray: Autophased time decay signal.
    """
    vsub = v
    s = np.zeros(360, dtype=np.complex128)
    for p in range(360):
        vph = vsub * np.exp(-1j * p / 180 * np.pi)
        s[p] = np.sum(np.real(vph))
    xm = np.argmax(s)
    v = v * np.exp(-1j * xm * np.pi / 180)
    return v


def autophase_2D(v: ndarray, index_left: int, index_right: int) -> ndarray:
    """
    Autophases the input data along the chemical shift axis for each row.

    Args:
        v (ndarray): 2D array containing complex data.
        index_left (int): Index of the left boundary for autophasing.
        index_right (int): Index of the right boundary for autophasing.

    Returns:
        ndarray: Autophased data.
    """
    for x in range(v.shape[0]):
        vsub = v[x, index_left : index_right + 1]
        s = np.zeros(360, dtype=np.complex128)
        for p in range(0, 359 + 1, 1):
            vph = vsub * np.exp(-1j * p / 180 * np.pi)
            s[p] = np.sum((vph))
            # s[p] = np.trapz(np.imag(vph))
        xm = np.argmax(s)
        v[x, :] = v[x, :] * np.exp(-1j * xm * np.pi / 180)
    return v


def integrate_2D(data2D: ndarray, ppm_scale: ndarray, ppm_start: float, ppm_end: float) -> ndarray:
    """
    Integrates a 2D data array within a specified chemical shift range.

    Args:
        data2D (ndarray): 2D array of data.
        ppm_scale (ndarray): Array containing chemical shift values.
        ppm_start (float): Starting point for integration.
        ppm_end (float): Ending point for integration.

    Returns:
        ndarray: Integrated data along the ppm_scale axis.
    """
    # Find the indices corresponding to the start and end positions
    start_idx = np.abs(ppm_scale - ppm_start).argmin()
    end_idx = np.abs(ppm_scale - ppm_end).argmin()

    # Slice the data2D array within the specified range
    data_slice = data2D[:, start_idx : end_idx + 1]

    # Integrate along the ppm_scale axis
    int_Tdecay = np.trapz(data_slice, x=ppm_scale[start_idx : end_idx + 1], axis=1)
    int_Tdecay = int_Tdecay.reshape(1, -1)

    return int_Tdecay


def fit_monoexponential(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    fitting_function: Callable[[np.ndarray, float, float, float], np.ndarray],
) -> Tuple[float, float, float, float]:
    """
    Performs a monoexponential fit on time and signal data.

    Args:
        time_values (np.ndarray): An array of time values.
        signal_values (np.ndarray): An array of signal values.
        fitting_function: A fitting function to use for the exponential fit.

    Returns:
        tuple: A tuple containing the following fitted parameters:
            amplitude (float): The amplitude of the fit.
            decay_time (float): The decay time of the fit.
            intercept (float): The intercept of the fit.
            R2 (float): The coefficient of determination (R-squared) of the fit.
    """
    inverse_decay_time_0 = 1 / (np.max(time_values) / 2)
    p0 = [np.max(signal_values), inverse_decay_time_0, np.min(signal_values)]

    # Perform the fit using the specified function
    fitted_parameters, _ = scipy.optimize.curve_fit(
        fitting_function, time_values, signal_values, p0
    )
    # Get fitted parameters
    amplitude, inverse_decay_time, intercept = fitted_parameters

    # Decay time
    decay_time = 1 / inverse_decay_time

    # Determine quality of the fit
    squaredDiffs = np.square(
        signal_values - fitting_function(time_values, amplitude, inverse_decay_time, intercept)
    )
    squaredDiffsFromMean = np.square(signal_values - np.mean(signal_values))
    R2 = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

    return amplitude, decay_time, intercept, R2


def create_time_scale_T2(
    file_path: Union[str, Path], dic: dict, spinsolve_type: str
) -> np.ndarray:
    """
    Create a time scale for T2 relaxation measurements.

    Args:
        file_path (Union[str, Path]): The file path to the directory.
        dic (dict): A dictionary containing acquisition parameters.
        spinsolve_type (str): The type of SpinSolve measurement ("standard" or "expert").

    Returns:
        np.ndarray: An array representing the T2 time scale.
    """
    if spinsolve_type == "standard":
        # Reads T2 steps from 'delayTimes.txt'
        delayTimes_path = Path(file_path) / "delayTimes.txt"
        with open(delayTimes_path, "r") as f:
            T2_scale = np.array([float(t.replace(",", ".")) for t in f.read().split("\n") if t])
    elif spinsolve_type == "expert":
        T2_scale = np.array(
            [
                echo_number
                * np.array(dic["acqu"]["techo"] * 1e-6)
                * np.array(dic["acqu"]["echoesperstep"])
                for echo_number in range(0, int(dic["acqu"]["nrSteps"]), 1)
            ]
        )
    return T2_scale


def create_time_scale_T2Bulk(dic: dict, spinsolve_type: str) -> Union[np.ndarray, None]:
    """
    Create a time scale for T2 relaxation.

    Args:
        dic (dict): A dictionary containing acquisition parameters.
        spinsolve_type (str): The type of spinsolve data.

    Returns:
        Union[np.ndarray, None]: An array of time values in seconds
        or None if spinsolve_type is not 'expert'.
    """
    if spinsolve_type != "expert":
        print("This function requires a SpinSolve expert file, but a standard file was given.")
        return None
    elif spinsolve_type == "expert":
        T2_scale = np.array(
            [
                echo_number * np.array(dic["acqu"]["echoTime"] * 1e-6)
                for echo_number in range(0, np.array(dic["acqu"]["nrEchoes"]), 1)
            ]
        )
    return T2_scale


def get_fitting_kernel(kernel_name: str, num_exponentials: int) -> Tuple[Callable, int]:
    """
    Get the fitting kernel function and the number of parameters.

    Args:
        kernel_name (str): The name of the kernel.
        num_exponentials (int): The number of exponentials.

    Returns:
        Tuple[callable, int]: A tuple containing the fitting kernel function
        and the number of parameters.

    Raises:
        ValueError: If an invalid kernel name or number of exponentials is provided.
    """
    kernel_names = {
        "T2": ["mono_exponential", "bi_exponential", "tri_exponential"],
        "T1IR": ["IR_mono_exponential", "IR_bi_exponential", "IR_tri_exponential"],
        "PGSTE": ["mono_exponential", "bi_exponential", "tri_exponential"],
    }

    if kernel_name not in kernel_names:
        available_options = ", ".join(kernel_names.keys())
        raise ValueError(
            f"Invalid kernel name '{kernel_name}'. Available options are: {available_options}"
        )

    fitting_kernels = kernel_names[kernel_name]

    if num_exponentials < 1 or num_exponentials > 3:
        raise ValueError("Invalid number of exponentials. Choose a value between 1 and 3.")

    fitting_kernel = globals()[fitting_kernels[num_exponentials - 1]]
    num_params = (
        num_exponentials * 2
    ) + 1  # Calculate the number of parameters based on the number of exponentials
    return fitting_kernel, num_params


def load_T1IRT2_timedat_files(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time data from T1IR and T2 experiments.

    Args:
        file_path (Path): The path to the data directory.

    Returns:
        tuple[np.ndarray, np.ndarray]: Time data for T1 and T2 experiments,
        in seconds.

    Raises:
        FileNotFoundError: If the required time data files are not found.
    """
    data1d_path = file_path / "1D_T1IRT2"

    timeT1_path = data1d_path / "timeT1_s.dat"
    timeT2_path = data1d_path / "timeT2_s.dat"

    if not timeT1_path.exists() or not timeT2_path.exists():
        raise FileNotFoundError("Time data files not found in the specified directory.")

    timeT1 = np.loadtxt(timeT1_path) * 1e-6  # in seconds
    timeT2 = np.loadtxt(timeT2_path)  # in seconds

    return timeT1, timeT2


def fit_multiexponential(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    kernel_name: str,
    num_exponentials: int,
    initial_guesses: Optional[List[float]] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit multiexponential data using the specified kernel.

    Args:
        time_values (np.ndarray): An array of time values.
        signal_values (np.ndarray): An array of signal values.
        kernel_name (str): The name of the kernel.
        num_exponentials (int): The number of exponentials.
        initial_guesses (List[float]): Initial parameter guesses.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the fitted parameters,
        the R-squared value and the covariance 2D array.
    """
    fitting_kernel, num_params = get_fitting_kernel(kernel_name, num_exponentials)

    # Create initial parameter guesses based on the number of exponentials
    if not initial_guesses:
        if fitting_kernel == "PGSTE":
            p0 = [np.max(signal_values) / (i + 1) for i in range(num_exponentials)]
            p0.extend([2.3e-9 / (i + 1) for i in range(num_exponentials)])
            p0.append(np.min(signal_values))
        else:
            inverse_decay_time_0 = 1 / (np.max(time_values) / 2)
            p0 = [np.max(signal_values) / (i + 1) for i in range(num_exponentials)]
            p0.extend([inverse_decay_time_0 / (i + 1) for i in range(num_exponentials)])
            p0.append(np.min(signal_values))
    else:
        p0 = initial_guesses

    # Perform the fit using the specified function
    fitted_parameters, cov = scipy.optimize.curve_fit(
        fitting_kernel, time_values, signal_values, p0
    )

    # Determine quality of the fit
    squared_diffs = np.square(signal_values - fitting_kernel(time_values, *fitted_parameters))
    squared_diffs_from_mean = np.square(signal_values - np.mean(signal_values))
    R2 = 1 - np.sum(squared_diffs) / np.sum(squared_diffs_from_mean)

    return fitted_parameters, R2, cov


def mono_exponential(x: np.ndarray, m: float, t: float, b: float) -> np.ndarray:
    """
    Calculate the result of a monoexponential function.

    Args:
        x (np.ndarray): Input values.
        m (float): Amplitude parameter.
        t (float): Decay time parameter.
        b (float): Intercept parameter.

    Returns:
        np.ndarray: Result of the monoexponential function.
    """
    return m * np.exp(-t * x) + b


def bi_exponential(
    x: np.ndarray, m1: float, t1: float, m2: float, t2: float, b: float
) -> np.ndarray:
    """
    Calculate the bi-exponential function.

    Args:
        x (np.ndarray): An array of input values.
        m1 (float): Amplitude of the first exponential.
        t1 (float): Decay time of the first exponential.
        m2 (float): Amplitude of the second exponential.
        t2 (float): Decay time of the second exponential.
        b (float): Intercept.

    Returns:
        np.ndarray: The calculated bi-exponential function values.
    """
    return m1 * np.exp(-t1 * x) + m2 * np.exp(-t2 * x) + b


def tri_exponential(
    x: np.ndarray, m1: float, t1: float, m2: float, t2: float, m3: float, t3: float, b: float
) -> np.ndarray:
    """
    Calculate the tri-exponential function.

    Args:
        x (np.ndarray): An array of input values.
        m1 (float): Amplitude of the first exponential.
        t1 (float): Decay time of the first exponential.
        m2 (float): Amplitude of the second exponential.
        t2 (float): Decay time of the second exponential.
        m3 (float): Amplitude of the third exponential.
        t3 (float): Decay time of the third exponential.
        b (float): Intercept.

    Returns:
        np.ndarray: The calculated tri-exponential function values.
    """
    return m1 * np.exp(-t1 * x) + m2 * np.exp(-t2 * x) + m3 * np.exp(-t3 * x) + b


def IR(x: np.ndarray, m: float, t: float, b: float) -> np.ndarray:
    """
    Inversion recovery (IR) function with a mono-exponential decay.

    Args:
        x (np.ndarray): Time values.
        m (float): Amplitude.
        t (float): Time constant.
        b (float): Offset.

    Returns:
        np.ndarray: IR function values.
    """
    return m * (1 - 2 * np.exp(-t * x)) + b


def IR_mono_exponential(x: np.ndarray, m: float, t: float, b: float) -> np.ndarray:
    """
    Mono-exponential inversion recovery (IR) function.

    Args:
        x (np.ndarray): Time values.
        m (float): Amplitude.
        t (float): Time constant.
        b (float): Offset.

    Returns:
        np.ndarray: IR function values.
    """
    return m * (1 - 2 * np.exp(-t * x)) + b


def IR_bi_exponential(
    x: np.ndarray, m1: float, t1: float, m2: float, t2: float, b: float
) -> np.ndarray:
    """
    Bi-exponential inversion recovery (IR) function.

    Args:
        x (np.ndarray): Time values.
        m1 (float): Amplitude for the first component.
        t1 (float): Time constant for the first component.
        m2 (float): Amplitude for the second component.
        t2 (float): Time constant for the second component.
        b (float): Offset.

    Returns:
        np.ndarray: IR function values.
    """
    return m1 * (1 - 2 * np.exp(-t1 * x)) + m2 * (1 - 2 * np.exp(-t2 * x)) + b


def IR_tri_exponential(
    x: np.ndarray, m1: float, t1: float, m2: float, t2: float, m3: float, t3: float, b: float
) -> np.ndarray:
    """
    Tri-exponential inversion recovery (IR) function.

    Args:
        x (np.ndarray): Time values.
        m1 (float): Amplitude for the first component.
        t1 (float): Time constant for the first component.
        m2 (float): Amplitude for the second component.
        t2 (float): Time constant for the second component.
        m3 (float): Amplitude for the third component.
        t3 (float): Time constant for the third component.
        b (float): Offset.

    Returns:
        np.ndarray: IR function values.
    """
    return (
        m1 * (1 - 2 * np.exp(-t1 * x))
        + m2 * (1 - 2 * np.exp(-t2 * x))
        + m3 * (1 - 2 * np.exp(-t3 * x))
        + b
    )
