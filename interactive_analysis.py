import json
import csv
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import pandas as pd


# Load data
def load_data(csv_file='eye_tracking_data.csv', json_file='eye_tracking_data.json'):
    """Load eye tracking data from CSV or JSON (skip first 1000 frames, limit to 6000 total frames)"""
    if Path(csv_file).exists():
        data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'frame': int(row['frame']),
                    'timestamp_ms': float(row['timestamp_ms']),
                    'left_horizontal': float(row['left_horizontal']),
                    'left_vertical': float(row['left_vertical']),
                    'right_horizontal': float(row['right_horizontal']),
                    'right_vertical': float(row['right_vertical']),
                })
        data = data[1000:7000]  # Skip first 1000 frames, then take 6000 frames
        print(f"Loaded {len(data)} frames from {csv_file}")
        return data
    elif Path(json_file).exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
        data = data[1000:7000]  # Skip first 1000 frames, then take 6000 frames
        print(f"Loaded {len(data)} frames from {json_file}")
        return data
    else:
        print(f"Error: Could not find {csv_file} or {json_file}")
        return None


def apply_lowpass_filter(data, cutoff_freq=0.1):
    """Apply Butterworth low-pass filter to data"""
    if data is None or len(data) < 5:
        return data
    
    # Design Butterworth filter
    b, a = signal.butter(7, cutoff_freq, btype='low')
    # Apply filter
    filtered = signal.filtfilt(b, a, data)
    return filtered.tolist()


def apply_bandpass_filter(data, low_cutoff_freq=0.1, high_cutoff_freq=0.05):
    """Apply Butterworth bandpass filter to data (high-pass then low-pass)"""
    if data is None or len(data) < 5:
        return data
    
    # Ensure low_cutoff > high_cutoff (low_cutoff is for LPF, high_cutoff is for HPF)
    if low_cutoff_freq <= high_cutoff_freq:
        return data
    
    # Safety check: prevent high-pass filter from being too aggressive (normalized freq must be > 0.003)
    if high_cutoff_freq < 0.003:
        high_cutoff_freq = 0.003
    
    # Design Butterworth filters
    # High-pass filter
    b_hp, a_hp = signal.butter(7, high_cutoff_freq, btype='high')
    # Low-pass filter
    b_lp, a_lp = signal.butter(7, low_cutoff_freq, btype='low')
    
    # Apply both filters
    filtered = signal.filtfilt(b_hp, a_hp, data)
    filtered = signal.filtfilt(b_lp, a_lp, filtered)
    return filtered.tolist()


def compute_fft(data):
    """Compute FFT of data with Hann window (DC removed)"""
    if data is None or len(data) < 2:
        return None, None
    
    data_array = np.array(data)
    # Remove DC component (mean)
    data_array = data_array - np.mean(data_array)
    
    # Apply Hann window to reduce spectral leakage
    window = signal.windows.hann(len(data_array))
    data_array = data_array * window
    
    fft_result = np.fft.fft(data_array)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(data_array))
    
    # Only return positive frequencies
    positive_idx = frequencies >= 0
    return frequencies[positive_idx], magnitude[positive_idx]


def calculate_velocity(position_data):
    """Calculate velocity from position data (degrees per frame)"""
    if not position_data or len(position_data) < 2:
        return None
    
    velocity = []
    
    # Calculate velocity as change in position per frame
    for i in range(1, len(position_data)):
        v = position_data[i] - position_data[i-1]
        velocity.append(v)
    
    # Pad first element to match length
    velocity = [0] + velocity
    
    return velocity


def calculate_acceleration(velocity_data):
    """Calculate acceleration from velocity data (degrees per frame^2)"""
    if not velocity_data or len(velocity_data) < 2:
        return None
    
    acceleration = []
    
    # Calculate acceleration as change in velocity per frame
    for i in range(1, len(velocity_data)):
        a = velocity_data[i] - velocity_data[i-1]
        acceleration.append(a)
    
    # Pad first element to match length
    acceleration = [0] + acceleration
    
    return acceleration


# Load and process data
data = load_data()

# Calculate sampling rate from timestamp data
if data and len(data) > 1:
    time_diffs = [data[i+1]['timestamp_ms'] - data[i]['timestamp_ms'] for i in range(len(data)-1)]
    avg_frame_time_ms = np.mean(time_diffs)
    fps = 1000 / avg_frame_time_ms
    nyquist_freq = fps / 2
    # Convert 25Hz to normalized frequency
    cutoff_hz = 25
    cutoff_normalized = cutoff_hz / nyquist_freq
    print(f"Detected sampling rate: {fps:.2f} Hz")
    print(f"Nyquist frequency: {nyquist_freq:.2f} Hz")
    print(f"25 Hz cutoff in normalized frequency: {cutoff_normalized:.4f}")
else:
    cutoff_normalized = 0.1
    print("Could not calculate sampling rate, using default cutoff of 0.1")

frames = [d['frame'] for d in data]
left_h = [d['left_horizontal'] for d in data]
right_h = [d['right_horizontal'] for d in data]

# Filter position data first (using calculated cutoff)
left_h_filtered = apply_lowpass_filter(left_h, cutoff_freq=cutoff_normalized)
right_h_filtered = apply_lowpass_filter(right_h, cutoff_freq=cutoff_normalized)

# Calculate velocity and acceleration from filtered position
left_velocity = calculate_velocity(left_h_filtered)
right_velocity = calculate_velocity(right_h_filtered)

left_acceleration = calculate_acceleration(left_velocity)
right_acceleration = calculate_acceleration(right_velocity)


def create_main_plot(data_dict):
    """Create the main horizontal analysis plot"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Left Eye - Position (LPF)', 'Left Eye - Velocity', 'Left Eye - Acceleration',
                       'Right Eye - Position (LPF)', 'Right Eye - Velocity', 'Right Eye - Acceleration'),
        specs=[[{}, {}, {}], [{}, {}, {}]]
    )
    
    # Left Eye traces
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['left_h_filtered'], mode='lines', name='Left Position', 
                   line=dict(color='blue', width=2), hovertemplate='Frame: %{x}<br>Angle: %{y:.2f}°'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['left_velocity'], mode='lines', name='Left Velocity',
                   line=dict(color='green', width=2), hovertemplate='Frame: %{x}<br>Velocity: %{y:.4f}°/frame'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['left_acceleration'], mode='lines', name='Left Acceleration',
                   line=dict(color='magenta', width=2), hovertemplate='Frame: %{x}<br>Accel: %{y:.4f}°/frame²'),
        row=1, col=3
    )
    
    # Right Eye traces
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['right_h_filtered'], mode='lines', name='Right Position',
                   line=dict(color='red', width=2), hovertemplate='Frame: %{x}<br>Angle: %{y:.2f}°'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['right_velocity'], mode='lines', name='Right Velocity',
                   line=dict(color='orange', width=2), hovertemplate='Frame: %{x}<br>Velocity: %{y:.4f}°/frame'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=data_dict['frames'], y=data_dict['right_acceleration'], mode='lines', name='Right Acceleration',
                   line=dict(color='purple', width=2), hovertemplate='Frame: %{x}<br>Accel: %{y:.4f}°/frame²'),
        row=2, col=3
    )
    
    # Add horizontal lines at y=0 for all subplots
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (°/frame)", row=1, col=2)
    fig.update_yaxes(title_text="Acceleration (°/frame²)", row=1, col=3)
    fig.update_yaxes(title_text="Angle (degrees)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (°/frame)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (°/frame²)", row=2, col=3)
    
    # Link all x-axes together with range slider on bottom-left
    fig.update_xaxes(title_text="Frame", row=1, col=1, matches='x')
    fig.update_xaxes(title_text="Frame", row=1, col=2, matches='x')
    fig.update_xaxes(title_text="Frame", row=1, col=3, matches='x')
    fig.update_xaxes(title_text="Frame", row=2, col=1, rangeslider_visible=True, rangeslider_thickness=0.05, matches='x')
    fig.update_xaxes(title_text="Frame", row=2, col=2, matches='x')
    fig.update_xaxes(title_text="Frame", row=2, col=3, matches='x')
    
    fig.update_layout(
        title_text='Eye Horizontal Analysis (Position [LPF], Velocity & Acceleration from filtered position)',
        title_font_size=18,
        height=750,
        autosize=True,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


# Prepare data dictionary
data_dict = {
    'frames': frames,
    'left_h': left_h,
    'right_h': right_h,
    'left_h_filtered': left_h_filtered,
    'right_h_filtered': right_h_filtered,
    'left_velocity': left_velocity,
    'right_velocity': right_velocity,
    'left_acceleration': left_acceleration,
    'right_acceleration': right_acceleration
}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Eye Tracking Interactive Analysis", className="mt-4 mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='main-plot', figure=create_main_plot(data_dict))
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Compute FFT on Selected Window", id="fft-button", color="primary", size="lg", className="mt-3 mb-3")
        ], width=6),
        dbc.Col([
            dbc.Button("Export Data", id="export-button", color="success", size="lg", className="mt-3 mb-3")
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Start Frame:", className="mt-3"),
            dcc.Input(
                id='start-frame-input',
                type='number',
                placeholder='Enter start frame',
                value=frames[0],
                style={'width': '100%', 'padding': '10px', 'marginBottom': '10px'}
            )
        ], width=6),
        dbc.Col([
            html.Label("End Frame:", className="mt-3"),
            dcc.Input(
                id='end-frame-input',
                type='number',
                placeholder='Enter end frame',
                value=frames[-1],
                style={'width': '100%', 'padding': '10px', 'marginBottom': '10px'}
            )
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("High-Pass Cutoff (Hz):", className="mt-3"),
            dcc.Slider(
                id='hpf-cutoff-slider',
                min=0,
                max=25,
                step=0.1,
                value=0,
                marks={i: str(i) for i in range(0, 26, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mt-2"
            ),
            html.Div(style={"fontSize": "12px", "color": "gray", "marginTop": "5px"}, 
                    children="(Note: HPF < 0.5 Hz may cause instability)")
        ], width=6),
        dbc.Col([
            html.Label("Low-Pass Cutoff (Hz):", className="mt-3"),
            dcc.Slider(
                id='lpf-cutoff-slider',
                min=5,
                max=50,
                step=1,
                value=25,
                marks={i: str(i) for i in range(5, 51, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mt-2"
            )
        ], width=6)
    ], className="mt-3 mb-2"),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='filter-display', style={"fontSize": "14px", "fontWeight": "bold"})
        ], width=12)
    ], className="mb-4"),
    
    dcc.Store(id='filtered-data-store'),
    
    dbc.Row([
        dbc.Col([
            html.Label("FFT Frequency Range (0-1):", className="mt-3"),
            dcc.RangeSlider(
                id='fft-freq-slider',
                min=0,
                max=1,
                step=0.01,
                value=[0, 1],
                marks={0: '0', 0.5: '0.5', 1: '1'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=12)
    ], className="mt-2 mb-3"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='fft-plot')
        ])
    ])
], fluid=True)


@callback(
    [Output('main-plot', 'figure'),
     Output('filtered-data-store', 'data'),
     Output('filter-display', 'children')],
    Input('hpf-cutoff-slider', 'value'),
    Input('lpf-cutoff-slider', 'value'),
    State('start-frame-input', 'value'),
    State('end-frame-input', 'value')
)
def update_bandpass_filter(hpf_hz, lpf_hz, start_frame, end_frame):
    """Update all plots when bandpass filter cutoffs change"""
    # Convert to normalized frequencies
    hpf_normalized = hpf_hz / nyquist_freq if hpf_hz > 0 else 0
    lpf_normalized = lpf_hz / nyquist_freq
    
    # Apply bandpass filter (HPF then LPF)
    if hpf_hz > 0:
        new_left_h_filtered = apply_bandpass_filter(left_h, low_cutoff_freq=lpf_normalized, high_cutoff_freq=hpf_normalized)
        new_right_h_filtered = apply_bandpass_filter(right_h, low_cutoff_freq=lpf_normalized, high_cutoff_freq=hpf_normalized)
    else:
        # Only LPF if HPF is 0
        new_left_h_filtered = apply_lowpass_filter(left_h, cutoff_freq=lpf_normalized)
        new_right_h_filtered = apply_lowpass_filter(right_h, cutoff_freq=lpf_normalized)
    
    # Recalculate velocity and acceleration
    new_left_velocity = calculate_velocity(new_left_h_filtered)
    new_right_velocity = calculate_velocity(new_right_h_filtered)
    new_left_acceleration = calculate_acceleration(new_left_velocity)
    new_right_acceleration = calculate_acceleration(new_right_velocity)
    
    # Update data dictionary
    updated_data_dict = {
        'frames': frames,
        'left_h': left_h,
        'right_h': right_h,
        'left_h_filtered': new_left_h_filtered,
        'right_h_filtered': new_right_h_filtered,
        'left_velocity': new_left_velocity,
        'right_velocity': new_right_velocity,
        'left_acceleration': new_left_acceleration,
        'right_acceleration': new_right_acceleration
    }
    
    # Create updated plot
    updated_fig = create_main_plot(updated_data_dict)
    
    # Set x-axis range based on start/end frame inputs
    if start_frame is not None and end_frame is not None:
        start_frame = max(frames[0], min(int(start_frame), frames[-1]))
        end_frame = max(frames[0], min(int(end_frame), frames[-1]))
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        
        # Update x-axis range for all subplots
        for row in [1, 2]:
            for col in [1, 2, 3]:
                updated_fig.update_xaxes(range=[start_frame, end_frame], row=row, col=col)
    
    # Store data for FFT and export callbacks
    stored_data = {
        'left_h_filtered': new_left_h_filtered,
        'right_h_filtered': new_right_h_filtered,
        'left_velocity': new_left_velocity,
        'right_velocity': new_right_velocity,
        'left_acceleration': new_left_acceleration,
        'right_acceleration': new_right_acceleration
    }
    
    # Display message
    if hpf_hz > 0:
        display_text = f"Bandpass: {hpf_hz:.1f} - {lpf_hz} Hz | Filter: 7th-order Butterworth (HPF → LPF)"
    else:
        display_text = f"Low-Pass: {lpf_hz} Hz | Filter: 7th-order Butterworth"
    
    return updated_fig, stored_data, display_text


@callback(
    Output('fft-plot', 'figure'),
    Input('fft-button', 'n_clicks'),
    State('start-frame-input', 'value'),
    State('end-frame-input', 'value'),
    State('fft-freq-slider', 'value'),
    State('filtered-data-store', 'data'),
    prevent_initial_call=True
)
def compute_fft_window(n_clicks, start_frame, end_frame, freq_range, stored_data):
    """Compute FFT on the specified frame range (filtered and unfiltered)"""
    if start_frame is None or end_frame is None:
        return {}
    
    # Clamp to valid frame range
    start_frame = max(frames[0], min(int(start_frame), frames[-1]))
    end_frame = max(frames[0], min(int(end_frame), frames[-1]))
    
    # Ensure start < end
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    
    # Find indices for the range
    indices_min = [i for i, f in enumerate(frames) if f >= start_frame]
    indices_max = [i for i, f in enumerate(frames) if f <= end_frame]
    
    if not indices_min or not indices_max:
        start_idx = 0
        end_idx = len(frames)
    else:
        start_idx = min(indices_min)
        end_idx = max(indices_max) + 1
    
    # Extract windowed data (both unfiltered and filtered)
    # Use stored filtered data if available from cutoff slider update
    if stored_data:
        left_h_window_filt = stored_data['left_h_filtered'][start_idx:end_idx]
        right_h_window_filt = stored_data['right_h_filtered'][start_idx:end_idx]
    else:
        left_h_window_filt = left_h_filtered[start_idx:end_idx]
        right_h_window_filt = right_h_filtered[start_idx:end_idx]
    
    left_h_window_raw = left_h[start_idx:end_idx]
    right_h_window_raw = right_h[start_idx:end_idx]
    
    # Compute FFTs
    freq_left_raw, mag_left_raw = compute_fft(left_h_window_raw)
    freq_right_raw, mag_right_raw = compute_fft(right_h_window_raw)
    freq_left_filt, mag_left_filt = compute_fft(left_h_window_filt)
    freq_right_filt, mag_right_filt = compute_fft(right_h_window_filt)
    
    # Filter by frequency range
    if freq_range:
        freq_min, freq_max = freq_range[0], freq_range[1]
        
        # Filter raw
        left_mask_raw = (freq_left_raw >= freq_min) & (freq_left_raw <= freq_max)
        freq_left_raw = freq_left_raw[left_mask_raw]
        mag_left_raw = mag_left_raw[left_mask_raw]
        
        right_mask_raw = (freq_right_raw >= freq_min) & (freq_right_raw <= freq_max)
        freq_right_raw = freq_right_raw[right_mask_raw]
        mag_right_raw = mag_right_raw[right_mask_raw]
        
        # Filter filtered
        left_mask_filt = (freq_left_filt >= freq_min) & (freq_left_filt <= freq_max)
        freq_left_filt = freq_left_filt[left_mask_filt]
        mag_left_filt = mag_left_filt[left_mask_filt]
        
        right_mask_filt = (freq_right_filt >= freq_min) & (freq_right_filt <= freq_max)
        freq_right_filt = freq_right_filt[right_mask_filt]
        mag_right_filt = mag_right_filt[right_mask_filt]
    
    # Create FFT plot with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Left Eye FFT - Raw (Frames {frames[start_idx]}-{frames[end_idx-1]})',
            f'Right Eye FFT - Raw (Frames {frames[start_idx]}-{frames[end_idx-1]})',
            f'Left Eye FFT - Filtered (Frames {frames[start_idx]}-{frames[end_idx-1]})',
            f'Right Eye FFT - Filtered (Frames {frames[start_idx]}-{frames[end_idx-1]})'
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Raw FFTs (top row)
    fig.add_trace(
        go.Scatter(x=freq_left_raw, y=mag_left_raw, mode='lines', name='Left Raw',
                   line=dict(color='blue', width=2), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=freq_right_raw, y=mag_right_raw, mode='lines', name='Right Raw',
                   line=dict(color='red', width=2), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=1, col=2
    )
    
    # Filtered FFTs (bottom row)
    fig.add_trace(
        go.Scatter(x=freq_left_filt, y=mag_left_filt, mode='lines', name='Left Filtered',
                   line=dict(color='green', width=2), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=freq_right_filt, y=mag_right_filt, mode='lines', name='Right Filtered',
                   line=dict(color='orange', width=2), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=2, col=2
    )
    
    # Update all axes labels and use log scale for y-axis
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Frequency (normalized)", row=row, col=col)
            fig.update_yaxes(title_text="Magnitude (log scale)", type="log", row=row, col=col)
    
    main_title = f'FFT Comparison: Unfiltered vs Filtered (Frames {frames[start_idx]}-{frames[end_idx-1]}, Total: {end_idx-start_idx} frames)'
    fig.update_layout(
        title_text=main_title,
        title_font_size=16,
        height=800,
        autosize=True,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


@callback(
    Input('export-button', 'n_clicks'),
    State('start-frame-input', 'value'),
    State('end-frame-input', 'value'),
    State('filtered-data-store', 'data'),
    prevent_initial_call=True
)
def export_data(n_clicks, start_frame, end_frame, stored_data):
    """Export data and plots to files for the specified frame range"""
    if start_frame is None or end_frame is None:
        return
    
    # Clamp to valid frame range
    start_frame = max(frames[0], min(int(start_frame), frames[-1]))
    end_frame = max(frames[0], min(int(end_frame), frames[-1]))
    
    # Ensure start < end
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    
    # Find indices for the range
    indices_min = [i for i, f in enumerate(frames) if f >= start_frame]
    indices_max = [i for i, f in enumerate(frames) if f <= end_frame]
    
    if not indices_min or not indices_max:
        start_idx = 0
        end_idx = len(frames)
    else:
        start_idx = min(indices_min)
        end_idx = max(indices_max) + 1
    
    # Create export directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = f'export_{timestamp}_frames_{frames[start_idx]}-{frames[end_idx-1]}'
    os.makedirs(export_dir, exist_ok=True)
    
    # Extract windowed data
    frame_range = list(range(start_idx, end_idx))
    frame_nums = [frames[i] for i in frame_range]
    left_h_win = [left_h[i] for i in frame_range]
    right_h_win = [right_h[i] for i in frame_range]
    
    # Use stored filtered data if available, otherwise use initial filter
    if stored_data:
        left_h_filt_win = [stored_data['left_h_filtered'][i] for i in frame_range]
        right_h_filt_win = [stored_data['right_h_filtered'][i] for i in frame_range]
        left_vel_win = [stored_data['left_velocity'][i] for i in frame_range]
        right_vel_win = [stored_data['right_velocity'][i] for i in frame_range]
        left_acc_win = [stored_data['left_acceleration'][i] for i in frame_range]
        right_acc_win = [stored_data['right_acceleration'][i] for i in frame_range]
    else:
        left_h_filt_win = [left_h_filtered[i] for i in frame_range]
        right_h_filt_win = [right_h_filtered[i] for i in frame_range]
        left_vel_win = [left_velocity[i] for i in frame_range]
        right_vel_win = [right_velocity[i] for i in frame_range]
        left_acc_win = [left_acceleration[i] for i in frame_range]
        right_acc_win = [right_acceleration[i] for i in frame_range]
    
    # Export time-domain data
    time_domain_df = pd.DataFrame({
        'Frame': frame_nums,
        'Left_Position_Raw': left_h_win,
        'Right_Position_Raw': right_h_win,
        'Left_Position_Filtered': left_h_filt_win,
        'Right_Position_Filtered': right_h_filt_win,
        'Left_Velocity': left_vel_win,
        'Right_Velocity': right_vel_win,
        'Left_Acceleration': left_acc_win,
        'Right_Acceleration': right_acc_win
    })
    time_domain_df.to_csv(f'{export_dir}/time_domain_data.csv', index=False)
    
    # Create and export time-domain plots
    fig_time = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Left Eye - Position (LPF)', 'Left Eye - Velocity', 'Left Eye - Acceleration',
                       'Right Eye - Position (LPF)', 'Right Eye - Velocity', 'Right Eye - Acceleration'),
        specs=[[{}, {}, {}], [{}, {}, {}]]
    )
    
    # Left Eye traces
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=left_h_filt_win, mode='lines', name='Left Position', 
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=left_vel_win, mode='lines', name='Left Velocity',
                   line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=left_acc_win, mode='lines', name='Left Acceleration',
                   line=dict(color='magenta', width=2)),
        row=1, col=3
    )
    
    # Right Eye traces
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=right_h_filt_win, mode='lines', name='Right Position',
                   line=dict(color='red', width=2)),
        row=2, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=right_vel_win, mode='lines', name='Right Velocity',
                   line=dict(color='orange', width=2)),
        row=2, col=2
    )
    fig_time.add_trace(
        go.Scatter(x=frame_nums, y=right_acc_win, mode='lines', name='Right Acceleration',
                   line=dict(color='purple', width=2)),
        row=2, col=3
    )
    
    # Add horizontal lines at y=0
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Update labels
    fig_time.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
    fig_time.update_yaxes(title_text="Velocity (°/frame)", row=1, col=2)
    fig_time.update_yaxes(title_text="Acceleration (°/frame²)", row=1, col=3)
    fig_time.update_yaxes(title_text="Angle (degrees)", row=2, col=1)
    fig_time.update_yaxes(title_text="Velocity (°/frame)", row=2, col=2)
    fig_time.update_yaxes(title_text="Acceleration (°/frame²)", row=2, col=3)
    
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig_time.update_xaxes(title_text="Frame", row=row, col=col)
    
    fig_time.update_layout(
        title_text=f'Time Domain Analysis (Frames {frames[start_idx]}-{frames[end_idx-1]})',
        title_font_size=16,
        height=800,
        width=1600,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Save time-domain plot as HTML and PNG
    fig_time.write_html(f'{export_dir}/time_domain_plot.html')
    try:
        fig_time.write_image(f'{export_dir}/time_domain_plot.png', width=1600, height=800)
    except:
        print("  Note: PNG export requires kaleido. Install with: pip install kaleido")
    
    # Compute and export FFT data
    left_h_win_raw = [left_h[i] for i in frame_range]
    right_h_win_raw = [right_h[i] for i in frame_range]
    
    # Use stored filtered data if available
    if stored_data:
        left_h_win_filt = [stored_data['left_h_filtered'][i] for i in frame_range]
        right_h_win_filt = [stored_data['right_h_filtered'][i] for i in frame_range]
    else:
        left_h_win_filt = [left_h_filtered[i] for i in frame_range]
        right_h_win_filt = [right_h_filtered[i] for i in frame_range]
    
    freq_left_raw, mag_left_raw = compute_fft(left_h_win_raw)
    freq_right_raw, mag_right_raw = compute_fft(right_h_win_raw)
    freq_left_filt, mag_left_filt = compute_fft(left_h_win_filt)
    freq_right_filt, mag_right_filt = compute_fft(right_h_win_filt)
    
    # Export FFT data
    fft_left_raw_df = pd.DataFrame({
        'Frequency': freq_left_raw,
        'Magnitude': mag_left_raw
    })
    fft_left_raw_df.to_csv(f'{export_dir}/fft_left_raw.csv', index=False)
    
    fft_right_raw_df = pd.DataFrame({
        'Frequency': freq_right_raw,
        'Magnitude': mag_right_raw
    })
    fft_right_raw_df.to_csv(f'{export_dir}/fft_right_raw.csv', index=False)
    
    fft_left_filt_df = pd.DataFrame({
        'Frequency': freq_left_filt,
        'Magnitude': mag_left_filt
    })
    fft_left_filt_df.to_csv(f'{export_dir}/fft_left_filtered.csv', index=False)
    
    fft_right_filt_df = pd.DataFrame({
        'Frequency': freq_right_filt,
        'Magnitude': mag_right_filt
    })
    fft_right_filt_df.to_csv(f'{export_dir}/fft_right_filtered.csv', index=False)
    
    # Create and export FFT plot
    fig_fft = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Left Eye FFT - Raw',
            f'Right Eye FFT - Raw',
            f'Left Eye FFT - Filtered',
            f'Right Eye FFT - Filtered'
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    fig_fft.add_trace(
        go.Scatter(x=freq_left_raw, y=mag_left_raw, mode='lines', name='Left Raw',
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig_fft.add_trace(
        go.Scatter(x=freq_right_raw, y=mag_right_raw, mode='lines', name='Right Raw',
                   line=dict(color='red', width=2)),
        row=1, col=2
    )
    fig_fft.add_trace(
        go.Scatter(x=freq_left_filt, y=mag_left_filt, mode='lines', name='Left Filtered',
                   line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig_fft.add_trace(
        go.Scatter(x=freq_right_filt, y=mag_right_filt, mode='lines', name='Right Filtered',
                   line=dict(color='orange', width=2)),
        row=2, col=2
    )
    
    for row in [1, 2]:
        for col in [1, 2]:
            fig_fft.update_xaxes(title_text="Frequency (normalized)", row=row, col=col)
            fig_fft.update_yaxes(title_text="Magnitude (log scale)", type="log", row=row, col=col)
    
    fig_fft.update_layout(
        title_text=f'FFT Comparison (Frames {frames[start_idx]}-{frames[end_idx-1]})',
        title_font_size=16,
        height=800,
        width=1400,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig_fft.write_html(f'{export_dir}/fft_plot.html')
    try:
        fig_fft.write_image(f'{export_dir}/fft_plot.png', width=1400, height=800)
    except:
        pass
    
    print(f"\n✓ Data exported to: {export_dir}/")
    print(f"  CSV Files:")
    print(f"    - time_domain_data.csv")
    print(f"    - fft_left_raw.csv, fft_right_raw.csv")
    print(f"    - fft_left_filtered.csv, fft_right_filtered.csv")
    print(f"  Plots:")
    print(f"    - time_domain_plot.html & .png")
    print(f"    - fft_plot.html & .png\n")


if __name__ == '__main__':
    app.run(debug=True)
