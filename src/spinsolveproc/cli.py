"""Console script for spinsolveproc."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from spinsolveproc import __version__
from spinsolveproc.spinsolveproc import SpinsolveExperiment

app = typer.Typer(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


def version_callback(value: bool) -> None:
    """Callback function for the --version option.

    Parameters:
        - value: The value provided for the --version option.

    Raises:
        - typer.Exit: Raises an Exit exception if the --version option is provided,
        printing the Awesome CLI version and exiting the program.
    """
    if value:
        typer.echo(f"spinsolveproc, version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show the current version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Console script for spinsolveproc."""


@app.command(name="process_exp")
def process_exp(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    process_all: Annotated[
        bool,
        typer.Option(help="Process all experiments in the directory."),
    ] = True,
) -> None:
    """Process Spinsolve data in a directory."""
    path_entry = Path(directory)

    if path_entry.exists():
        if path_entry.joinpath("acqu.par").is_file():
            sample_dir_list = [path_entry]
            typer.echo("Processing current file directory\n")
        elif experiment:
            experiment_names = [f"-{experiment}-", f" {experiment} "]
            sample_dir_list = []
            for experiment_name in experiment_names:
                sample_dir_list += [
                    diracqu.parent for diracqu in path_entry.glob(f"*{experiment_name}*/acqu.par")
                ]
            if not sample_dir_list:
                typer.echo(f"Error: No directories found for experiment {experiment}")
            else:
                typer.echo(f"Number of directories to be processed: {len(sample_dir_list)}\n")
        elif process_all:
            sample_dir_list = [diracqu.parent for diracqu in path_entry.glob("*/acqu.par")]

            if not sample_dir_list:
                typer.echo("Error: No directories found in the specified directory.")
            else:
                typer.echo(f"Number of directories to be processed: {len(sample_dir_list)}\n")
        else:
            typer.echo(
                "Error: Please provide an experiment name or use --all to process all experiments."
            )
            return

        for sample_dir in sample_dir_list:
            experiment_instance = SpinsolveExperiment(sample_dir)
            experiment_instance.load()
            output_dict = experiment_instance.process()
            figure, experiment_name = experiment_instance.plot(output_dict)
            experiment_instance.save_fig(figure, experiment_name)
            experiment_instance.save_data(output_dict, experiment_name)

    else:
        typer.echo("Error! Directory does not exist")


if __name__ == "__main__":
    app()
