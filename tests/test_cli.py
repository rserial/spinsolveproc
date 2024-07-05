"""Tests for `spinsolveproc`.cli module."""

from typing import List

import pytest
from typer.testing import CliRunner

import spinsolveproc
from spinsolveproc import cli

runner = CliRunner()


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "Console script for spinsolveproc."),
        (["--help"], "Console script for spinsolveproc."),
        (
            ["--version"],
            f"spinsolveproc, version { spinsolveproc.__version__ }\n",
        ),
    ],
)
def test_command_line_interface(options: List[str], expected: str) -> None:
    """Test the CLI."""
    result = runner.invoke(cli.app, options)
    assert result.exit_code == 0
    assert expected in result.stdout
