"""Plotting functions for spinsolveproc."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def setup_fig_proton(
    file_path_name: str,
    time_scale: np.ndarray,
    FIDdecay: np.ndarray,
    ppm_scale: np.ndarray,
    spectrum: np.ndarray,
) -> go.Figure:
    """
    Create a Plotly figure for visualizing proton experiment data.

    Args:
        file_path_name (str):
            The name of the experiment or file path (used in subplot titles).
        time_scale (np.ndarray):
            Time scale data for the FID decay.
        FIDdecay (np.ndarray):
            FID decay data.
        ppm_scale (np.ndarray):
            PPM scale data for the proton spectrum.
        spectrum (np.ndarray):
            Proton spectrum data.

    Returns:
        go.Figure:
            A Plotly figure configured to display proton experiment data with interactive controls.
    """
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Proton decay: " + file_path_name, "Proton Spectrum")
    )

    # Add FID decay traces
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.real(FIDdecay), name="Real FID", line=dict(color="blue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.imag(FIDdecay), name="Imag FID", line=dict(color="red")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_scale,
            y=np.abs(FIDdecay),
            name="Absolute FID",
            line=dict(color="black", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Add Spectrum traces
    fig.add_trace(
        go.Scatter(
            x=ppm_scale, y=np.real(spectrum), name="Real Spectrum", line=dict(color="blue")
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=ppm_scale, y=np.imag(spectrum), name="Imag Spectrum", line=dict(color="red")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ppm_scale,
            y=np.abs(spectrum),
            name="Absolute Spectrum",
            line=dict(color="black", dash="dash"),
        ),
        row=1,
        col=2,
    )

    # Configure axes labels
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Chemical Shift (ppm)", row=1, col=2)
    fig.update_yaxes(title_text="Signal Intensity (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Signal Intensity (a.u.)", row=1, col=2)

    # Configure layout with a white template
    fig.update_layout(
        title="Proton Spectrum and Decay",
        width=1000,
        height=500,
        template="plotly_white",  # Use the white template
    )

    # Add interactive capability to remove real, imag, or absolute parts
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                buttons=list(
                    [
                        dict(
                            label="Real",
                            method="update",
                            args=[
                                {"visible": [True, False, False, True, False, False]},
                                {"title": "Proton Spectrum and Decay - Real"},
                            ],
                        ),
                        dict(
                            label="Imag",
                            method="update",
                            args=[
                                {"visible": [False, True, False, False, True, False]},
                                {"title": "Proton Spectrum and Decay - Imag"},
                            ],
                        ),
                        dict(
                            label="Absolute",
                            method="update",
                            args=[
                                {"visible": [False, False, True, False, False, True]},
                                {"title": "Proton Spectrum and Decay - Absolute"},
                            ],
                        ),
                    ]
                ),
            )
        ]
    )

    return fig
