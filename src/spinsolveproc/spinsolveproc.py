"""Main functions for spinsolveproc."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import plotly.graph_objects as go

import spinsolveproc.plot as plot
import spinsolveproc.process as process
import spinsolveproc.save as save
import spinsolveproc.utils as utils

logger = logging.getLogger(__name__)


class SpinsolveExperiment:
    """Represents an experiment conducted with a Spinsolve NMR spectrometer.

    This class provides methods for loading experiment parameters, processing data,
    plotting figures, and saving results.

    Attributes:
        acqupar_file (str): The filename for the acquisition parameter file.
        procpar_file (str): The filename for the processing parameter file.

    Methods:
        __init__(self, experiment_path: pathlib.Path) -> None:
            Initializes a SpinsolveExperiment instance for the specified experiment directory.

        load(self) -> None:
            Loads experiment parameters from the acquisition parameter (acqu.par) file.

        process(self) -> Dict[str, Union[Tuple, None]]:
            Processes the experiment data and returns the processed results as a dictionary.

        plot(self, output_dict: Dict[str, Union[Tuple, None]]) -> Tuple[Optional[Tuple],
            Optional[str]]:
            Generates and returns plots for the processed data.

        save_fig(self, figure: Tuple, experiment_name: str) -> None:
            Saves a figure to the processed_data directory.

        save_data(self, output_dict: Dict[str, Union[Tuple, None]],
            experiment_name: str) -> None:
            Saves processed data to the processed_data directory.

        name(self) -> str:
            Gets the name of the experiment.

        type(self) -> str:
            Gets the type of Spinsolve experiment (standard or expert).
    """

    acqupar_file = "acqu.par"
    procpar_file = "proc.par"

    def __init__(self, experiment_path: Path) -> None:
        """Initialize a SpinsolveExperiment instance.

        Args:
            experiment_path (Path): The path to the experiment directory.
        """
        self.experiment_path = experiment_path
        self.parameters: Dict[str, Any] = {}

    def load(self) -> None:
        """Load experiment parameters from the acqu.par file."""
        try:
            with open(self.experiment_path / self.acqupar_file, "r") as f:
                info = f.readlines()
            for line in info:
                par_name, par_value = utils.parse_spinsolve_par_line(line)
                self.parameters[par_name] = par_value
        except FileNotFoundError as e:
            logger.exception("Error loading parameters")
            raise e
        except IOError:
            logger.error("IO Error while loading parameters")
        return

    def process(self, **kwargs: Any) -> Dict:
        """Process the experiment data.

        Parameters:
            **kwargs (Any): Additional optional input parameters.

        Returns:
            A dictionary containing the processed data.

        Raises:
            FileNotFoundError: If the data file is not found.
        """
        print("Processing directory...", self.experiment_path.name, end="")

        processing_functions: Dict[str, Callable[..., Tuple[Any, ...]]] = {
            "Proton": process.proton,
            "1D EXTENDED+": process.proton,
            "T2": process.t2,
            "T2Bulk": process.t2_bulk,
            "T1": process.t1,
            "T1IRT2": process.t1ir_t2,
            "PGSTE": process.pgste,
        }

        if self.name in processing_functions:
            output_dict = {}

            processing_function = processing_functions[self.name]
            output = processing_function(self.experiment_path, self.spinsolve_type, **kwargs)

            output_dict[self.name] = output
            return output_dict
        else:
            raise FileNotFoundError(f"{self.name} data missing from output dictionary")

    def plot(self, output_dict: Dict[str, Any], **kwargs: Any) -> tuple:
        """Generate and return plots for the processed data.

        Args:
            output_dict (Dict[str, Any]): A dictionary containing the processed data.
            **kwargs (Any): Additional optional input parameters.

        Returns:
            Figure and the experiment name.

        Raises:
            NameError: If the experiment name is not found in the output dictionary.
        """
        if self.name not in output_dict:
            raise NameError(f"{self.name} data missing from output dictionary")

        plotting_functions: Dict[str, Callable[..., Tuple[Any, ...]]] = {
            "Proton": plot.setup_fig_proton,
            "1D EXTENDED+": plot.setup_fig_proton,
            "T2": plot.setup_fig_t2,
            "T2Bulk": plot.setup_fig_t2_bulk,
            "T1": plot.setup_fig_t1,
            "T1IRT2": plot.setup_fig_t1ir_t2,
            "PGSTE": plot.setup_fig_pgste,
        }

        if self.name not in plotting_functions:
            raise NameError(f"Plotting function not found for experiment type {self.name}")

        figure = plotting_functions[self.name](
            self.experiment_path.name, *output_dict[self.name], **kwargs
        )

        if not isinstance(figure, tuple):
            figure = (figure,)
        return figure[0], self.name

    def save_fig(self, figure: go.Figure, experiment_name: str) -> None:
        """Save a figure to the processed_data directory.

        Args:
            figure: The figure to be saved.
            experiment_name (str): The name of the experiment.

        Raises:
            NameError: If the saving function for the experiment type.
        """
        save_dir = self.experiment_path / "processed_data"
        if not save_dir.exists():
            save_dir.mkdir()

        saving_functions: Dict[str, Callable[..., None]] = {
            "Proton": save.fig_proton,
            "1D EXTENDED+": save.fig_proton,
            "T2": save.fig_t2,
            "T2Bulk": save.fig_t2_bulk,
            "T1": save.fig_t1,
            "T1IRT2": save.fig_t1ir_t2,
        }
        if experiment_name in saving_functions:
            saving_function = saving_functions[experiment_name]
            if isinstance(figure, tuple):
                saving_function(save_dir, *figure)
            else:
                saving_function(save_dir, figure)
        else:
            raise NameError(f"Saving figure function not found for experiment type {self.name}")

    def save_data(self, output_dict: dict, experiment_name: str) -> None:
        """Save processed data to the processed_data directory.

        Args:
            output_dict (dict): A dictionary containing the processed data.
            experiment_name (str): The name of the experiment.

        Raises:
            NameError: If the saving function for the experiment type.
        """
        save_dir = self.experiment_path / "processed_data"
        if not save_dir.exists():
            save_dir.mkdir()

        if self.name in output_dict:
            saving_functions: Dict[str, Callable[..., None]] = {
                "Proton": save.data_proton,
                "1D EXTENDED+": save.data_proton,
                "T2": save.data_t2,
                "T2Bulk": save.data_t2_bulk,
                "T1": save.data_t1,
                "T1IRT2": save.data_t1ir_t2,
                "PGSTE": save.data_pgste,
            }

            if experiment_name in saving_functions:
                saving_functions[experiment_name](save_dir, *output_dict[experiment_name])
            else:
                raise NameError(f"Saving data function not found for experiment type {self.name}")
        else:
            print(f'Error: "{self.name}" data missing from output dictionary')

    @property
    def name(self) -> Any:
        """Get the name of the experiment.

        Returns:
            The experiment name.
        """
        if "Solvent" in self.parameters:
            name = self.parameters["Protocol"]
        else:
            name = self.parameters["experiment"]
        return name

    @property
    def spinsolve_type(self) -> str:
        """Get the type of Spinsolve experiment.

        Returns:
            The Spinsolve experiment type.
        """
        if "Solvent" in self.parameters:
            spinsolve_type = "standard"
        else:
            spinsolve_type = "expert"
        return spinsolve_type
