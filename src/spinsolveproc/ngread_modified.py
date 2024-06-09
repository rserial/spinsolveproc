"""This is a wrapper to mofify nmrglue read function to read .2d files."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from nmrglue import jcampdx
from nmrglue.fileio.spinsolve import parse_spinsolve_par_line


def read(  # noqa: C901
    directory: Path,
    specfile: Optional[str] = None,
    acqupar: str = "acqu.par",
    procpar: str = "proc.par",
    split_opt: str = "yes",
) -> Tuple[dict, np.ndarray]:
    """Reads SpinSolve files from a directory.

    When no spectrum filename is given (specfile), the following list is tried, in
    that specific order:
      - "nmr_fid.dx"
      - "data.1d"
      - "fid.1d"
      - "spectrum.1d"
      - "spectrum_processed.1d"

    To use the resolution-enhanced spectrum, use the './Enhanced' folder as input.
    Note that spectrum.1d and spectrum_processed.1d contain only data in the
    frequency domain, so no Fourier transformation is needed. Also, use
    dic["spectrum"]["xaxis"] to plot the x-axis.

    Args:
        directory (Path): Directory to read from.
        specfile (str, optional): Filename to import spectral data from. None uses standard
            filename from ["nmr_fid.dx", "data.1d", "fid.1d", "spectrum.1d",
            "spectrum_processed.1d"].
        acqupar (str): Filename for acquisition parameters. None uses standard name.
        procpar (str): Filename for processing parameters. None uses standard name.
        split_opt (str): Split option.

    Raises:
        IOError: If there is an issue with reading or accessing files in the specified directory.

    Returns:
        A tuple containing the following:
            dic (dict): All parameters that can be present in the data folder:
                dic["spectrum"] - First bytes of spectrum(_processed).1d
                dic["acqu"] - Parameters present in acqu.par
                dic["proc"] - Parameters present in proc.par
                dic["dx"] - Parameters present in the header of nmr_fid.dx
            data (ndarray): Array of NMR data
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise IOError(f"Directory {directory} does not exist")

    # Create empty dic
    dic: Dict[str, Union[dict, np.ndarray]] = {"spectrum": {}, "acqu": {}, "proc": {}, "dx": {}}

    # Read in acqu.par and write to dic
    acqupar_path = directory / acqupar
    if acqupar_path.is_file():
        with open(acqupar_path, "r") as f:
            info = f.readlines()
        for line in info:
            par_name, par_value = parse_spinsolve_par_line(line)

            if par_name is not None:
                dic["acqu"][par_name] = par_value

    # Read in proc.par and write to dic
    procpar_path = directory / procpar
    if procpar_path.is_file():
        with open(procpar_path, "r") as f:
            info = f.readlines()
        for line in info:
            line = line.replace("\n", "")
            k, v = line.split("=")
            dic["proc"][k.strip()] = v.strip()

    # Define which spectrumfile to take, using 'specfile' when defined, otherwise
    # the files in 'priority_list' are tried, in that particular order
    priority_list = ["nmr_fid.dx", "data.1d", "fid.1d", "spectrum.1d", "spectrum_processed.1d"]
    inputfile = None
    if specfile:
        inputfile = directory / specfile
        if not inputfile.is_file():
            raise IOError(f"File {inputfile} does not exist")
    else:
        for priority in priority_list:
            inputfile = directory / priority
            if inputfile.is_file():
                break
    if inputfile is None:
        raise IOError(f"Directory {directory} does not contain spectral data")

    # Detect which file we are dealing with from the extension and read in the spectral data

    # Reading .dx file using existing nmrglue.fileio.jcampdx module
    if inputfile.suffix == ".dx":
        dic["dx"], raw_data = jcampdx.read(inputfile)
        data = raw_data[0][:] + 1j * raw_data[1][:]

    # Reading .1d files
    elif inputfile.suffix == ".1d":
        with open(inputfile, "rb") as f:
            raw_data = f.read()

        # Write out parameters from the first 32 bytes into dic["spectrum"]
        keys = ["owner", "format", "version", "dataType", "xDim", "yDim", "zDim", "qDim"]
        for i, k in enumerate(keys):
            start = i * 4
            end = start + 4
            value = int.from_bytes(raw_data[start:end], "little")
            dic["spectrum"][k] = value
        data = np.frombuffer(raw_data[32:], "<f")

        # The first 1/3 of the file is x-axis data (s or ppm)
        split = data.shape[-1] // 3
        xscale = data[0:split]
        dic["spectrum"]["xaxis"] = xscale

        if split_opt == "yes":
            # The rest is real and imaginary data points interleaved
            data = data[split::2] + 1j * data[1 + split :: 2]
        else:
            # The rest is real and imaginary data points interleaved
            data = data[0::2] + 1j * data[1::2]

    # Reading .2d files
    elif inputfile.suffix == ".2d":
        with open(inputfile, "rb") as f:
            raw_data = f.read()

        # Write out parameters from the first 32 bytes into dic["spectrum"]
        keys = ["owner", "format", "version", "dataType", "xDim", "yDim", "zDim", "qDim"]
        for i, k in enumerate(keys):
            start = i * 4
            end = start + 4
            value = int.from_bytes(raw_data[start:end], "little")
            dic["spectrum"][k] = value
        data = np.frombuffer(raw_data[32:], "<f")

        # Real and imaginary data points interleaved
        data = data[0::2] + 1j * data[1::2]

    else:
        raise IOError(f"File {inputfile} cannot be interpreted, use .dx or .1d instead")

    return dic, data
