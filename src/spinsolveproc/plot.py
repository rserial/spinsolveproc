"""Plotting functions for spinsolveproc."""

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spinsolveproc.utils as utils


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
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Proton decay", "Proton Spectrum"))

    # Add FID decay traces
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.real(FIDdecay), name="Real FID", line=dict(color="#2C7FB8")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.imag(FIDdecay), name="Imag FID", line=dict(color="#7FCDBB")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_scale,
            y=np.abs(FIDdecay),
            name="Absolute FID",
            line=dict(color="#636363", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Add Spectrum traces
    fig.add_trace(
        go.Scatter(
            x=ppm_scale, y=np.real(spectrum), name="Real Spectrum", line=dict(color="#2C7FB8")
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ppm_scale, y=np.imag(spectrum), name="Imag Spectrum", line=dict(color="#7FCDBB")
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ppm_scale,
            y=np.abs(spectrum),
            name="Absolute Spectrum",
            line=dict(color="#636363", dash="dash"),
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
        title="Proton Spectrum and Decay" + str(file_path_name),
        width=1200,
        height=500,
        template="plotly_white",
    )
    return fig


def setup_fig_T2(
    file_path_name: str,
    ppm_scale: np.ndarray,
    T2_scale: np.ndarray,
    T2spec_2Dmap: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_T2decay: np.ndarray,
) -> Tuple[go.Figure, go.Figure]:
    """
    Set up figures for T2 experiment.

    Args:
        file_path_name (str): File path name.
        ppm_scale (np.ndarray): Chemical shift axis of the 2D spectrum.
        T2_scale (np.ndarray): Time axis of the T2 decay.
        T2spec_2Dmap (np.ndarray): Processed 2D spectrum.
        peak_ppm_positions (np.ndarray): Chemical shift positions of the T2 peaks.
        peak_T2decay (np.ndarray): T2 decay associated with each peak.

    Returns:
        Tuple[plt.Figure, plt.Figure]: Tuple containing two figures.
    """
    fig_T2spec_2Dmap = setup_fig_Tspec_2Dmap(
        file_path_name,
        ppm_scale,
        T2_scale,
        T2spec_2Dmap,
        peak_ppm_positions,
        peak_T2decay,
        "Spectroscopically resolved T2",
    )

    fig_T2specdecays_fit = setup_fig_Tdecay_fit(
        file_path_name,
        T2_scale,
        peak_T2decay,
        "T2",
        num_exponentials=1,
        plot_title_name="T2 decay",
    )
    return fig_T2spec_2Dmap, fig_T2specdecays_fit


def setup_fig_T1(
    file_path_name: str,
    ppm_scale: np.ndarray,
    T1_scale: np.ndarray,
    T1spec_2Dmap: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_T1decay: np.ndarray,
) -> Tuple[go.Figure, go.Figure]:
    """
    Set up figures for T1 experiment.

    Args:
        file_path_name (str): The name of the file path.
        ppm_scale (np.ndarray): The scale of ppm values.
        T1_scale (np.ndarray): The T1 scale.
        T1spec_2Dmap (np.ndarray): The T1 spectroscopically resolved 2D map.
        peak_ppm_positions (np.ndarray): The peak positions in ppm.
        peak_T1decay (np.ndarray): The peak T1 decay data.

    Returns:
        Tuple[go.Figure, go.Figure]: Two Plotly figures for T1 experiment.
    """
    fig_T1spec_2Dmap = setup_fig_Tspec_2Dmap(
        file_path_name,
        ppm_scale,
        T1_scale,
        T1spec_2Dmap,
        peak_ppm_positions,
        peak_T1decay,
        "Spectroscopically resolved T1",
    )

    fig_T1specdecays_fit = setup_fig_Tdecay_fit(
        file_path_name,
        T1_scale,
        peak_T1decay,
        "T1IR",
        num_exponentials=1,
        plot_title_name="T1 decay",
    )
    return fig_T1spec_2Dmap, fig_T1specdecays_fit


def setup_fig_Tspec_2Dmap(
    file_path_name: str,
    frequency_axis: np.ndarray,
    time_axis: np.ndarray,
    data: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_T1decay: np.ndarray,
    plot_title_name: str,
) -> go.Figure:
    """
    Set up a figure for spectroscopically resolved T1.

    Args:
        file_path_name (str): File path name.
        time_axis (np.ndarray): Time axis.
        frequency_axis (np.ndarray): Frequency axis.
        data (np.ndarray): Data for the heatmap.
        peak_ppm_positions (np.ndarray): Chemical shift positions of the T1 peaks.
        peak_T1decay (np.ndarray): T1 decay associated with each peak.
        plot_title_name (str): Title for the plot.

    Returns:
        go.Figure: A Plotly Figure object.
    """
    fig_Tspec_2Dmap = go.Figure(
        data=go.Heatmap(
            x=np.squeeze(time_axis),
            y=np.squeeze(frequency_axis),
            z=np.real(data),
            colorscale="Blues",
            showscale=False,
        )
    )

    # Set the layout
    fig_Tspec_2Dmap.update_layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis=dict(title="Chemical Shift (ppm)"),
        yaxis=dict(title="Time (s)"),
        showlegend=True,
    )

    # fig_Tspec_2Dmap.show()
    return fig_Tspec_2Dmap


def setup_fig_T2Bulk(
    file_path_name: str,
    T2_scale: np.ndarray,
    T2decay: np.ndarray,
) -> go.Figure:
    """
    Setup a figure for T2Bulk decays fit.

    Args:
        file_path_name (str): File path name.
        T2_scale (np.ndarray): Array containing T2 time scale.
        T2decay (np.ndarray): Array containing T2 decay data.

    Returns:
        go.Figure: A Plotly Figure.
    """
    fig_T2Bulkdecays_fit = setup_fig_Tdecay_fit(
        file_path_name, T2_scale, T2decay, "T2", num_exponentials=1, plot_title_name="T2 decay"
    )
    return fig_T2Bulkdecays_fit


def setup_fig_Tdecay_fit(
    file_path_name: str,
    T_scale: np.ndarray,
    Tdecay: np.ndarray,
    kernel_name: str,
    num_exponentials: int,
    plot_title_name: str,
) -> go.Figure:
    """
    Setup a figure for Tdecay fit.

    Args:
        file_path_name (str): File path name.
        T_scale (np.ndarray): Array containing T scale.
        Tdecay (np.ndarray): Array containing T decay data.
        kernel_name (str): Kernel name.
        num_exponentials (int): Number of exponentials.
        plot_title_name (str): Plot title name.

    Returns:
        go.Figure: A Plotly Figure.
    """
    fitting_kernel, num_params = utils.get_fitting_kernel(kernel_name, num_exponentials)

    fitted_parameters, R2 = utils.fit_multiexponential(
        T_scale, np.real(Tdecay), kernel_name, num_exponentials
    )

    amplitude = []
    time_decay = []

    for i in range(num_exponentials):
        amplitude.append(fitted_parameters[i * 2])
        time_decay.append(1 / fitted_parameters[i * 2 + 1])

    trace1_real = go.Scatter(
        x=T_scale,
        y=np.real(Tdecay) / np.max(np.real(Tdecay)),
        mode="markers",
        name="T Decay - real",
        marker=dict(color="#2C7FB8"),
    )
    trace1_imag = go.Scatter(
        x=T_scale,
        y=np.imag(Tdecay) / np.max(np.real(Tdecay)),
        mode="markers",
        name="T Decay - imag",
        marker=dict(color="#7FCDBB"),
    )
    trace2 = go.Scatter(
        x=T_scale,
        y=fitting_kernel(T_scale, *fitted_parameters[:num_params])
        / np.max(fitting_kernel(T_scale, *fitted_parameters[:num_params])),
        mode="lines",
        name=(
            f"{num_exponentials}exp. fit, Long component T2decay = "
            f"{np.max(np.round(time_decay,3))} s, R² = {np.round(R2, 6)}"
        ),
        marker=dict(color="#636363"),
    )

    layout = go.Layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis_title="Time (s)",
        yaxis_title="Normalized intensity (a.u)",
    )

    fig = go.Figure(data=[trace1_real, trace1_imag, trace2], layout=layout)

    fig.update_layout(height=500, width=800)

    list_fitTdecay = {"Amplitude": amplitude, "Time decay [s]": time_decay}
    df = pd.DataFrame(list_fitTdecay, columns=["Amplitude", "Time decay [s]"])
    print(f"Results {num_exponentials} exp. fit from plot\n{df}")

    return fig


def setup_fig_T1IRT2(
    file_path_name: str,
    time_T1_axis: np.ndarray,
    time_T2_axis: np.ndarray,
    T1IRT2array: np.ndarray,
) -> go.Figure:
    """
    Setup a Plotly figure for T1IRT2 data.

    Args:
        file_path_name (str): Name of the file path.
        time_T1_axis (np.ndarray): Time axis for T1.
        time_T2_axis (np.ndarray): Time axis for T2.
        T1IRT2array (np.ndarray): T1IRT2 data.

    Returns:
        go.Figure: Plotly figure for T1IRT2 data.
    """
    fig_T1IRT2_2Dmap = setup_fig_T1IRT2_2Dmap(
        file_path_name, time_T1_axis, time_T2_axis, T1IRT2array, "T1IRT2 intensity 2D map"
    )
    return fig_T1IRT2_2Dmap


def setup_fig_T1IRT2_2Dmap(
    file_path_name: str,
    time_T1_axis: np.ndarray,
    time_T2_axis: np.ndarray,
    T1IRT2array: np.ndarray,
    title_name: str,
) -> go.Figure:
    """
    Setup a Plotly 2D heatmap for T1IRT2 data.

    Args:
        file_path_name (str): Name of the file path.
        time_T1_axis (np.ndarray): Time axis for T1.
        time_T2_axis (np.ndarray): Time axis for T2.
        T1IRT2array (np.ndarray): T1IRT2 data.
        title_name (str): Title for the plot.

    Returns:
        go.Figure: Plotly figure for the 2D heatmap.
    """
    fig_T1IRT2_2Dmap = go.Figure()

    # Create heatmap
    fig_T1IRT2_2Dmap.add_trace(
        go.Heatmap(z=np.real(T1IRT2array), x=time_T2_axis, y=time_T1_axis, colorscale="Viridis")
    )

    # Update layout
    fig_T1IRT2_2Dmap.update_layout(
        title=title_name + ": " + str(file_path_name),
        xaxis_title="Time T1 (s)",
        yaxis_title="Time T2 (s)",
    )

    return fig_T1IRT2_2Dmap
