"""Plotting functions for spinsolveproc."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spinsolveproc.utils as utils


def setup_fig_proton(
    file_path_name: str,
    time_scale: np.ndarray,
    fid_decay: np.ndarray,
    ppm_scale: np.ndarray,
    spectrum: np.ndarray,
) -> go.Figure:
    """Create a Plotly figure for visualizing proton experiment data.

    Args:
        file_path_name (str):
            The name of the experiment or file path (used in subplot titles).
        time_scale (np.ndarray):
            Time scale data for the FID decay.
        fid_decay (np.ndarray):
            FID decay data.
        ppm_scale (np.ndarray):
            PPM scale data for the proton spectrum.
        spectrum (np.ndarray):
            Proton spectrum data.

    Returns:
        A Plotly figure configured to display proton experiment data with interactive controls.
    """
    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Proton decay", "Proton Spectrum"))

    # Add FID decay traces
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.real(fid_decay), name="Real FID", line={"color": "#2C7FB8"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_scale, y=np.imag(fid_decay), name="Imag FID", line={"color": "#7FCDBB"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_scale,
            y=np.abs(fid_decay),
            name="Absolute FID",
            line={"color": "#636363", "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    # Add Spectrum traces
    fig.add_trace(
        go.Scatter(
            x=ppm_scale, y=np.real(spectrum), name="Real Spectrum", line={"color": "#2C7FB8"}
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ppm_scale, y=np.imag(spectrum), name="Imag Spectrum", line={"color": "#7FCDBB"}
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ppm_scale,
            y=np.abs(spectrum),
            name="Absolute Spectrum",
            line={"color": "#636363", "dash": "dash"},
        ),
        row=1,
        col=2,
    )

    # Configure axes labels
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(
        title_text="Chemical shift (ppm)",
        range=[np.max(ppm_scale), np.min(ppm_scale)],
        row=1,
        col=2,
    )
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


def setup_fig_t2(
    file_path_name: str,
    ppm_scale: np.ndarray,
    t2_scale: np.ndarray,
    t2_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_t2_decay: np.ndarray,
    num_exponentials: Optional[int] = None,
) -> go.Figure:
    """Set up figures for T2 experiment.

    Args:
        file_path_name (str): File path name.
        ppm_scale (np.ndarray): Chemical shift axis of the 2D spectrum.
        t2_scale (np.ndarray): Time axis of the T2 decay.
        t2_spec_2d_map (np.ndarray): Processed 2D spectrum.
        peak_ppm_positions (np.ndarray): Chemical shift positions of the T2 peaks.
        peak_t2_decay (np.ndarray): T2 decay associated with each peak.
        num_exponentials (Optional[int]): number of fitting exponentials (<=3)

    Raises:
        ValueError: If num_exponentials is not an integer or is not between 1 and 3 (inclusive).

    Returns:
        Tuple containing two figures.
    """
    fig_t2_spec_2d_map = setup_fig_t_spec_2d_map(
        file_path_name,
        ppm_scale,
        t2_scale,
        t2_spec_2d_map,
        peak_ppm_positions,
        peak_t2_decay,
        "Spectroscopically resolved T2",
    )

    if num_exponentials is None:
        num_exponentials = 1
    elif not isinstance(num_exponentials, int) or num_exponentials > 3 or num_exponentials < 1:
        raise ValueError("num_exponentials must be an integer between 1 and 3 (inclusive).")

    fig_t2_specdecays_fit = setup_fig_t_decay_fit(
        file_path_name,
        t2_scale,
        peak_t2_decay,
        "T2",
        num_exponentials=num_exponentials,
        plot_title_name="T2 decay",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Spectroscopically resolved T2", "T2 decay"),
    )
    fig.add_trace(fig_t2_spec_2d_map["data"][0], row=1, col=1)
    fig.add_trace(fig_t2_specdecays_fit["data"][0], row=1, col=2)
    fig.add_trace(fig_t2_specdecays_fit["data"][1], row=1, col=2)
    fig.add_trace(fig_t2_specdecays_fit["data"][2], row=1, col=2)

    fig.update_xaxes(
        title_text="Chemical shift (ppm)",
        range=[np.max(ppm_scale), np.min(ppm_scale)],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(
        title_text="Time (s)",
        row=1,
        col=1,
    )

    fig.update_layout(height=500, width=1200, title_text=f"T2 Experiment: {file_path_name}")
    return fig


def setup_fig_t1(
    file_path_name: str,
    ppm_scale: np.ndarray,
    t1_scale: np.ndarray,
    t1_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_t1_decay: np.ndarray,
    num_exponentials: Optional[int] = None,
) -> go.Figure:
    """Set up figures for T1 experiment.

    Args:
        file_path_name (str): The name of the file path.
        ppm_scale (np.ndarray): The scale of ppm values.
        t1_scale (np.ndarray): The T1 scale.
        t1_spec_2d_map (np.ndarray): The T1 spectroscopically resolved 2D map.
        peak_ppm_positions (np.ndarray): The peak positions in ppm.
        peak_t1_decay (np.ndarray): The peak T1 decay data.
        num_exponentials (Optional[int]): number of fitting exponentials (<=3)

    Raises:
        ValueError: If num_exponentials is not an integer or is not between 1 and 3 (inclusive).

    Returns:
        Two Plotly figures for T1 experiment.
    """
    fig_t1_spec_2d_map = setup_fig_t_spec_2d_map(
        file_path_name,
        ppm_scale,
        t1_scale,
        t1_spec_2d_map,
        peak_ppm_positions,
        peak_t1_decay,
        "Spectroscopically resolved T1",
    )

    if num_exponentials is None:
        num_exponentials = 1
    elif not isinstance(num_exponentials, int) or num_exponentials > 3 or num_exponentials < 1:
        raise ValueError("num_exponentials must be an integer between 1 and 3 (inclusive).")

    fig_t1_specdecays_fit = setup_fig_t_decay_fit(
        file_path_name,
        t1_scale,
        peak_t1_decay,
        "T1IR",
        num_exponentials=num_exponentials,
        plot_title_name="T1 decay",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Spectroscopically resolved T1", "T1 decay"),
    )
    fig.add_trace(fig_t1_spec_2d_map["data"][0], row=1, col=1)
    fig.add_trace(fig_t1_specdecays_fit["data"][0], row=1, col=2)
    fig.add_trace(fig_t1_specdecays_fit["data"][1], row=1, col=2)
    fig.add_trace(fig_t1_specdecays_fit["data"][2], row=1, col=2)

    fig.update_xaxes(
        title_text="Chemical shift (ppm)",
        range=[np.max(ppm_scale), np.min(ppm_scale)],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(
        title_text="Time (s)",
        row=1,
        col=1,
    )

    fig.update_layout(height=500, width=1200, title_text=f"T1 Experiment: {file_path_name}")
    return fig


def setup_fig_pgste(
    file_path_name: str,
    ppm_scale: np.ndarray,
    diff_scale: np.ndarray,
    diff_spec_2d_map: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_diff_decay: np.ndarray,
    num_exponentials: Optional[int] = None,
    initial_guesses_expfit: Optional[List[float]] = None,
) -> go.Figure:
    """Set up figures for PGSTE experiment.

    Args:
        file_path_name (str): The name of the file path.
        ppm_scale (np.ndarray): The scale of ppm values.
        diff_scale (np.ndarray): The diffusion scale.
        diff_spec_2d_map (np.ndarray): The diffusion spectroscopically resolved 2D map.
        peak_ppm_positions (np.ndarray): The peak positions in ppm.
        peak_diff_decay (np.ndarray): The diffusion peak decay data.
        num_exponentials (Optional[int]): number of fitting exponentials (<=3)
        initial_guesses_expfit (List[float]): initial guesses for fitting.

    Raises:
        ValueError: If num_exponentials is not an integer or is not between 1 and 3 (inclusive).

    Returns:
        Two Plotly figures for T1 experiment.
    """
    fig_diff_spec_2d_map = setup_fig_diff_spec_2d_map(
        file_path_name,
        ppm_scale,
        diff_scale,
        diff_spec_2d_map,
        peak_ppm_positions,
        peak_diff_decay,
        "Spectroscopically resolved PGSTE",
    )

    if num_exponentials is None:
        num_exponentials = 1
    elif not isinstance(num_exponentials, int) or num_exponentials > 3 or num_exponentials < 1:
        raise ValueError("num_exponentials must be an integer between 1 and 3 (inclusive).")

    fig_diff_specdecays_fit = setup_fig_diff_decay_fit(
        file_path_name,
        diff_scale,
        peak_diff_decay,
        "PGSTE",
        num_exponentials=num_exponentials,
        plot_title_name="Diffusion decay",
        initial_guesses_expfit=initial_guesses_expfit,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Spectroscopically resolved PGSTE", "PGSTE decay"),
    )
    fig.add_trace(fig_diff_spec_2d_map["data"][0], row=1, col=1)
    fig.add_trace(fig_diff_specdecays_fit["data"][0], row=1, col=2)
    fig.add_trace(fig_diff_specdecays_fit["data"][1], row=1, col=2)

    fig.update_xaxes(
        title_text="Chemical shift (ppm)",
        range=[np.max(ppm_scale), np.min(ppm_scale)],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="γ² g² δ² (Δ-δ/3) (10⁹ s/m²)",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="γ² g² δ² (Δ-δ/3) (10⁹ s/m²)", row=1, col=2)
    fig.update_yaxes(
        title_text="Integral amplitude (a.u)",
        row=1,
        col=2,
    )
    fig.update_layout(height=500, width=1200, title_text=f"PGSTE Experiment: {file_path_name}")
    return fig


def setup_fig_diff_spec_2d_map(
    file_path_name: str,
    ppm_axis: np.ndarray,
    diff_axis: np.ndarray,
    data: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_diff_decay: np.ndarray,
    plot_title_name: str,
) -> go.Figure:
    """Set up a figure for spectroscopically resolved T1.

    Args:
        file_path_name (str): File path name.
        diff_axis (np.ndarray): Diffusion axis.
        ppm_axis (np.ndarray): ppm axis.
        data (np.ndarray): Data for the heatmap.
        peak_ppm_positions (np.ndarray): Chemical shift positions of the T1 peaks.
        peak_diff_decay (np.ndarray): Diffusion decay associated with each peak.
        plot_title_name (str): Title for the plot.

    Returns:
        A Plotly Figure object.
    """
    fig_t_spec_2d_map = go.Figure(
        data=go.Heatmap(
            x=np.squeeze(ppm_axis),
            y=np.squeeze(diff_axis) * 1e-9,
            z=np.real(data),
            colorscale="Blues",
            showscale=False,
        )
    )

    # Set the layout
    fig_t_spec_2d_map.update_layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis={"title": "Chemical Shift (ppm)"},
        yaxis={"title": "γ² g² δ² (Δ-δ/3) (10⁹ s/m²)"},
        showlegend=True,
    )

    fig_t_spec_2d_map.update_layout(height=500, width=800)
    return fig_t_spec_2d_map


def setup_fig_t_spec_2d_map(
    file_path_name: str,
    frequency_axis: np.ndarray,
    time_axis: np.ndarray,
    data: np.ndarray,
    peak_ppm_positions: np.ndarray,
    peak_t1_decay: np.ndarray,
    plot_title_name: str,
) -> go.Figure:
    """Set up a figure for spectroscopically resolved T1.

    Args:
        file_path_name (str): File path name.
        time_axis (np.ndarray): Time axis.
        frequency_axis (np.ndarray): Frequency axis.
        data (np.ndarray): Data for the heatmap.
        peak_ppm_positions (np.ndarray): Chemical shift positions of the T1 peaks.
        peak_t1_decay (np.ndarray): T1 decay associated with each peak.
        plot_title_name (str): Title for the plot.

    Returns:
        A Plotly Figure object.
    """
    fig_t_spec_2d_map = go.Figure(
        data=go.Heatmap(
            x=np.squeeze(frequency_axis),
            y=np.squeeze(time_axis),
            z=np.real(data),
            colorscale="Blues",
            showscale=False,
        )
    )

    # Set the layout
    fig_t_spec_2d_map.update_layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis={"title": "Chemical Shift (ppm)"},
        yaxis={"title": "Time (s)"},
        showlegend=True,
    )

    fig_t_spec_2d_map.update_layout(height=500, width=800)
    return fig_t_spec_2d_map


def setup_fig_t2_bulk(
    file_path_name: str,
    t2_scale: np.ndarray,
    t2_decay: np.ndarray,
    num_exponentials: Optional[int] = None,
) -> go.Figure:
    """Setup a figure for T2Bulk decays fit.

    Args:
        file_path_name (str): File path name.
        t2_scale (np.ndarray): Array containing T2 time scale.
        t2_decay (np.ndarray): Array containing T2 decay data.
        num_exponentials (Optional[int]): number of fitting exponentials (<=3)

    Raises:
        ValueError: If num_exponentials is not an integer or is not between 1 and 3 (inclusive).

    Returns:
        A Plotly Figure.
    """
    if num_exponentials is None:
        num_exponentials = 1
    elif not isinstance(num_exponentials, int) or num_exponentials > 3 or num_exponentials < 1:
        raise ValueError("num_exponentials must be an integer between 1 and 3 (inclusive).")

    fig_t2_bulk_decays_fit = setup_fig_t_decay_fit(
        file_path_name,
        t2_scale,
        t2_decay,
        "T2",
        num_exponentials=num_exponentials,
        plot_title_name="T2 decay",
    )
    return fig_t2_bulk_decays_fit


def setup_fig_diff_decay_fit(
    file_path_name: str,
    diff_scale: np.ndarray,
    diff_decay: np.ndarray,
    kernel_name: str,
    num_exponentials: int,
    plot_title_name: str,
    initial_guesses_expfit: Optional[List[float]] = None,
) -> go.Figure:
    """Setup a figure for Tdecay fit.

    Args:
        file_path_name (str): File path name.
        diff_scale (np.ndarray): Array containing diffusion scale.
        diff_decay (np.ndarray): Array containing diffusion decay data.
        kernel_name (str): Kernel name.
        num_exponentials (int): Number of exponentials.
        plot_title_name (str): Plot title name.
        initial_guesses_expfit (List[float]): initial guess for fitting.

    Returns:
        A Plotly Figure.
    """
    fitting_kernel, num_params = utils.get_fitting_kernel(kernel_name, num_exponentials)

    fitted_parameters, r2, cov = utils.fit_multiexponential(
        diff_scale,
        np.real(diff_decay),
        kernel_name,
        num_exponentials,
        initial_guesses_expfit,
    )
    err = np.sqrt(np.diag(cov))

    amplitude = []
    err_amplitude = []
    diffusion_decay = []
    err_diffusion_decay = []

    for i in range(num_exponentials):
        amplitude.append(fitted_parameters[i * 2])
        diffusion_decay.append(fitted_parameters[i * 2 + 1])
        err_amplitude.append(err[i * 2])
        err_diffusion_decay.append(err[i * 2 + 1])

    trace1_real = go.Scatter(
        x=diff_scale,
        y=np.real(diff_decay) / np.max(np.abs(diff_decay)),
        mode="markers",
        name="T Decay - magnitude",
        marker={"color": "#2C7FB8"},
    )
    trace2 = go.Scatter(
        x=diff_scale,
        y=fitting_kernel(diff_scale, *fitted_parameters[:num_params])
        / np.max(fitting_kernel(diff_scale, *fitted_parameters[:num_params])),
        mode="lines",
        name=(
            f"{num_exponentials}exp. fit, Shortest diffusion component = "
            f"{format(np.min(diffusion_decay), '.1e')} s, R² = {np.round(r2, 6)}"
        ),
        marker={"color": "#636363"},
    )

    layout = go.Layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis_title="γ² g² δ² (Δ-δ/3) (10⁹ s/m²)",
        yaxis_title="Normalized integral intensity (a.u)",
    )

    fig = go.Figure(data=[trace1_real, trace2], layout=layout)

    fig.update_layout(height=500, width=800)

    list_fit_t_decay = {
        "Amplitude [a.u]": amplitude,
        "Err Amplitude [a.u]": err_amplitude,
        "Diffusion decay [s]": diffusion_decay,
        "Err Diffusion decay [s]": err_diffusion_decay,
    }
    df = pd.DataFrame(
        list_fit_t_decay,
        columns=[
            "Amplitude [a.u]",
            "Err Amplitude [a.u]",
            "Diffusion decay [s]",
            "Err Diffusion decay [s]",
        ],
    )
    print(f"Results {num_exponentials} exp. fit from plot\n{df}")

    return fig


def setup_fig_t_decay_fit(
    file_path_name: str,
    t_scale: np.ndarray,
    t_decay: np.ndarray,
    kernel_name: str,
    num_exponentials: int,
    plot_title_name: str,
) -> go.Figure:
    """Setup a figure for Tdecay fit.

    Args:
        file_path_name (str): File path name.
        t_scale (np.ndarray): Array containing T scale.
        t_decay (np.ndarray): Array containing T decay data.
        kernel_name (str): Kernel name.
        num_exponentials (int): Number of exponentials.
        plot_title_name (str): Plot title name.

    Returns:
        A Plotly Figure.
    """
    fitting_kernel, num_params = utils.get_fitting_kernel(kernel_name, num_exponentials)

    fitted_parameters, r2, cov = utils.fit_multiexponential(
        t_scale, np.real(t_decay), kernel_name, num_exponentials
    )
    err = np.sqrt(np.diag(cov))

    amplitude = []
    err_amplitude = []
    time_decay = []
    err_time_decay = []

    for i in range(num_exponentials):
        amplitude.append(fitted_parameters[i * 2])
        time_decay.append(1 / fitted_parameters[i * 2 + 1])
        err_amplitude.append(err[i * 2])
        err_time_decay.append(err[i * 2 + 1] / fitted_parameters[i * 2 + 1] ** 2)

    trace1_real = go.Scatter(
        x=t_scale,
        y=np.real(t_decay) / np.max(np.real(t_decay)),
        mode="markers",
        name="T Decay - real",
        marker={"color": "#2C7FB8"},
    )
    trace1_imag = go.Scatter(
        x=t_scale,
        y=np.imag(t_decay) / np.max(np.real(t_decay)),
        mode="markers",
        name="T Decay - imag",
        marker={"color": "#7FCDBB"},
    )
    trace2 = go.Scatter(
        x=t_scale,
        y=fitting_kernel(t_scale, *fitted_parameters[:num_params])
        / np.max(fitting_kernel(t_scale, *fitted_parameters[:num_params])),
        mode="lines",
        name=(
            f"{num_exponentials}exp. fit, Long component time decay = "
            f"{np.max(np.round(time_decay,3))} s, R² = {np.round(r2, 6)}"
        ),
        marker={"color": "#636363"},
    )

    layout = go.Layout(
        title=plot_title_name + ": " + str(file_path_name),
        xaxis_title="Time (s)",
        yaxis_title="Normalized intensity (a.u)",
    )

    fig = go.Figure(data=[trace1_real, trace1_imag, trace2], layout=layout)

    fig.update_layout(height=500, width=800)

    list_fit_t_decay = {
        "Amplitude [a.u]": amplitude,
        "Err Amplitude [a.u]": err_amplitude,
        "Time decay [s]": time_decay,
        "Err Time decay [s]": err_time_decay,
    }
    df = pd.DataFrame(
        list_fit_t_decay,
        columns=["Amplitude [a.u]", "Err Amplitude [a.u]", "Time decay [s]", "Err Time decay [s]"],
    )
    print(f"Results {num_exponentials} exp. fit from plot\n{df}")

    return fig


def setup_fig_t1ir_t2(
    file_path_name: str,
    time_t1_axis: np.ndarray,
    time_t2_axis: np.ndarray,
    t1ir_t2_array: np.ndarray,
) -> go.Figure:
    """Setup a Plotly figure for T1IRT2 data.

    Args:
        file_path_name (str): Name of the file path.
        time_t1_axis (np.ndarray): Time axis for T1.
        time_t2_axis (np.ndarray): Time axis for T2.
        t1ir_t2_array (np.ndarray): T1IRT2 data.

    Returns:
        Plotly figure for T1IRT2 data.
    """
    fig_t1ir_t2_2d_map = setup_fig_t1ir_t2_2d_map(
        file_path_name, time_t1_axis, time_t2_axis, t1ir_t2_array, "T1IRT2 intensity 2D map"
    )
    return fig_t1ir_t2_2d_map


def setup_fig_t1ir_t2_2d_map(
    file_path_name: str,
    time_t1_axis: np.ndarray,
    time_t2_axis: np.ndarray,
    t1ir_t2_array: np.ndarray,
    title_name: str,
) -> go.Figure:
    """Setup a Plotly 2D heatmap for T1IRT2 data.

    Args:
        file_path_name (str): Name of the file path.
        time_t1_axis (np.ndarray): Time axis for T1.
        time_t2_axis (np.ndarray): Time axis for T2.
        t1ir_t2_array (np.ndarray): T1IRT2 data.
        title_name (str): Title for the plot.

    Returns:
        Plotly figure for the 2D heatmap.
    """
    fig_t1ir_t2_2d_map = go.Figure()

    # Create heatmap
    fig_t1ir_t2_2d_map.add_trace(
        go.Heatmap(
            z=np.transpose(np.real(t1ir_t2_array)),
            x=time_t1_axis,
            y=time_t2_axis,
            colorscale="Viridis",
        )
    )

    # Update layout
    fig_t1ir_t2_2d_map.update_layout(
        title=title_name + ": " + str(file_path_name),
        xaxis_title="Time T1 (s)",
        yaxis_title="Time T2 (s)",
        height=500,
        width=800,
    )

    return fig_t1ir_t2_2d_map
