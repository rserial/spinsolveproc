"""Main functions for spinsolveproc."""
import logging
from pathlib import Path

import plotly.graph_objects as go

from src.spinsolveproc import plot, process, save, utils

logger = logging.getLogger(__name__)


class SpinsolveExperiment:
    """
    Represents an experiment conducted with a Spinsolve NMR spectrometer.

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
        """
        Initialize a SpinsolveExperiment instance.

        Args:
            experiment_path (Path): The path to the experiment directory.
        """
        self.experiment_path = experiment_path
        self.parameters = {}

    def load(self) -> None:
        """Load experiment parameters from the acqu.par file."""
        try:
            with open(self.experiment_path / self.acqupar_file, "r") as f:
                info = f.readlines()
            for line in info:
                par_name, par_value = utils.parse_spinsolve_par_line(line)
                self.parameters[par_name] = par_value
        except FileNotFoundError as e:
            # Handle the case when the file is not found.
            logger.exception("Error loading parameters")
            raise e
        except IOError:
            # Handle input/output errors.
            logger.error("IO Error while loading parameters")

    def process(self) -> dict:
        """
        Process the experiment data.

        Returns:
            dict: A dictionary containing the processed data.
        """
        print("Processing directory...", self.experiment_path.name, end="")

        processing_functions = {
            "Proton": process.proton,
        }

        if self.name in processing_functions:
            output_dict = {}

            processing_function = processing_functions[self.name]
            output = processing_function(self.experiment_path, self.type)

            output_dict[self.name] = output
            return output_dict
        else:
            print(f'Data not found for experiment type "{self.name}".')

    def plot(self, output_dict: dict) -> tuple:
        """
        Generate and return plots for the processed data.

        Args:
            output_dict (dict): A dictionary containing the processed data.

        Returns:
            tuple: A tuple containing figures and the experiment name.
        """
        if self.name not in output_dict:
            print(f'Error: "{self.name}" data missing from output dictionary')
            return None

        plotting_functions = {
            "Proton": plot.setup_fig_proton,
        }

        if self.name not in plotting_functions:
            print(f'Plotting function not found for experiment type "{self.name}".')
            return None

        figures = plotting_functions[self.name](self.experiment_path.name, *output_dict[self.name])
        if not isinstance(figures, tuple):
            figures = (figures,)
        return figures, self.name

    def save_fig(self, figure: go.Figure(), experiment_name: str) -> None:
        """
        Save a figure to the processed_data directory.

        Args:
            figure: The figure to be saved.
            experiment_name (str): The name of the experiment.
        """
        save_dir = self.experiment_path / "processed_data"
        if not save_dir.exists():
            save_dir.mkdir()

        saving_functions = {
            "Proton": save.fig_proton,
        }
        if experiment_name in saving_functions:
            saving_function = saving_functions[experiment_name]
            if isinstance(figure, tuple):
                saving_function(save_dir, *figure)
            else:
                saving_function(save_dir, figure)
        else:
            print(f'Saving function not found for experiment type "{experiment_name}".')

    def save_data(self, output_dict: dict, experiment_name: str) -> None:
        """
        Save processed data to the processed_data directory.

        Args:
            output_dict (dict): A dictionary containing the processed data.
            experiment_name (str): The name of the experiment.
        """
        save_dir = self.experiment_path / "processed_data"
        if not save_dir.exists():
            save_dir.mkdir()

        if self.name in output_dict:
            saving_functions = {
                "Proton": save.data_proton,
            }

            if experiment_name in saving_functions:
                saving_functions[experiment_name](save_dir, *output_dict[experiment_name])
            else:
                print(f'Saving function not found for experiment type "{experiment_name}".')

        else:
            print(f'Error: "{self.name}" data missing from output dictionary')

    @property
    def name(self) -> str:
        """
        Get the name of the experiment.

        Returns:
            str: The experiment name.
        """
        if "Solvent" in self.parameters:
            name = self.parameters["Protocol"]
        else:
            name = self.parameters["experiment"]
        return name

    @property
    def spinsolve_type(self) -> str:
        """
        Get the type of Spinsolve experiment.

        Returns:
            str: The Spinsolve experiment type.
        """
        if "Solvent" in self.parameters:
            spinsolve_type = "standard"
        else:
            spinsolve_type = "expert"
        return spinsolve_type
