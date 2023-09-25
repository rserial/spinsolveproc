"""Console script for spinsolveproc."""
from pathlib import Path

import click

from spinsolveproc.spinsolveproc import SpinsolveExperiment, __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Console script for spinsolveproc."""


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("experiment", required=False)
def main(directory: str, experiment: str) -> None:
    """Process Spinsolve data in a directory."""
    path_entry = Path(directory)

    if path_entry.exists():
        if path_entry.joinpath("acqu.par").is_file():
            sample_dir_list = [path_entry]
            click.echo("Processing current file directory\n")
        elif experiment:
            experiment_names = [f"-{experiment}-", f" {experiment} "]
            sample_dir_list = []
            for experiment_name in experiment_names:
                sample_dir_list += [
                    diracqu.parent for diracqu in path_entry.glob(f"*{experiment_name}*/acqu.par")
                ]
        elif all:
            sample_dir_list = [diracqu.parent for diracqu in path_entry.glob("*/acqu.par")]

            if not sample_dir_list:
                click.echo(f"Error: No directories found for experiment {experiment}")
            else:
                click.echo(f"Number of directories to be processed: {len(sample_dir_list)}\n")
        else:
            click.echo(
                "Error: Please provide an experiment name or use --all to process all experiments."
            )

        for sample_dir in sample_dir_list:
            experiment = SpinsolveExperiment(sample_dir)
            experiment.load()
            output_dict = experiment.process()
            figure, experiment_name = experiment.plot(output_dict)
            experiment.save_fig(figure, experiment_name)
            experiment.save_data(output_dict, experiment_name)

    else:
        click.echo("Error! Directory does not exist")


if __name__ == "__main__":
    main()
