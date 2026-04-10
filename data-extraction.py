import json
import csv
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

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

def apply_lowpass_filter(data, cutoff_freq=0.1):
    """Apply Butterworth low-pass filter to data"""
    if data is None or len(data) < 5:
        return data
    
    # Design Butterworth filter
    b, a = signal.butter(3, cutoff_freq, btype='low')
    # Apply filter
    filtered = signal.filtfilt(b, a, data)
    return filtered.tolist()

def compute_fft(data):
    """Compute FFT of data and return frequencies and magnitude spectrum"""
    if data is None or len(data) < 2:
        return None, None
    
    data_array = np.array(data)
    fft_result = np.fft.fft(data_array)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(data_array))
    
    # Only return positive frequencies
    positive_idx = frequencies >= 0
    return frequencies[positive_idx], magnitude[positive_idx]

def plot_horizontal_analysis(data):
    """Plot interactive horizontal position, velocity, and acceleration for both eyes"""
    if not data:
        return
    
    frames = [d['frame'] for d in data]
    left_h = [d['left_horizontal'] for d in data]
    right_h = [d['right_horizontal'] for d in data]
    
    # Filter position data first
    left_h_filtered = apply_lowpass_filter(left_h)
    right_h_filtered = apply_lowpass_filter(right_h)
    
    # Calculate velocity and acceleration from filtered position
    left_velocity = calculate_velocity(left_h_filtered)
    right_velocity = calculate_velocity(right_h_filtered)
    
    left_acceleration = calculate_acceleration(left_velocity)
    right_acceleration = calculate_acceleration(right_velocity)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Left Eye - Position (LPF)', 'Left Eye - Velocity', 'Left Eye - Acceleration',
                       'Right Eye - Position (LPF)', 'Right Eye - Velocity', 'Right Eye - Acceleration'),
        specs=[[{}, {}, {}], [{}, {}, {}]]
    )
    
    # Left Eye traces
    fig.add_trace(
        go.Scatter(x=frames, y=left_h_filtered, mode='lines', name='Left Position', 
                   line=dict(color='blue', width=2), hovertemplate='Frame: %{x}<br>Angle: %{y:.2f}°'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=left_velocity, mode='lines', name='Left Velocity',
                   line=dict(color='green', width=2), hovertemplate='Frame: %{x}<br>Velocity: %{y:.4f}°/frame'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=frames, y=left_acceleration, mode='lines', name='Left Acceleration',
                   line=dict(color='magenta', width=2), hovertemplate='Frame: %{x}<br>Accel: %{y:.4f}°/frame²'),
        row=1, col=3
    )
    
    # Right Eye traces
    fig.add_trace(
        go.Scatter(x=frames, y=right_h_filtered, mode='lines', name='Right Position',
                   line=dict(color='red', width=2), hovertemplate='Frame: %{x}<br>Angle: %{y:.2f}°'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=right_velocity, mode='lines', name='Right Velocity',
                   line=dict(color='orange', width=2), hovertemplate='Frame: %{x}<br>Velocity: %{y:.4f}°/frame'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=frames, y=right_acceleration, mode='lines', name='Right Acceleration',
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
    
    # Update layout with synchronized range slider
    fig.update_layout(
        title_text='Eye Horizontal Analysis (Position [LPF], Velocity & Acceleration from filtered position)',
        title_font_size=18,
        height=750,
        autosize=True,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.show()
    print("✓ Interactive plot displayed (Butterworth LPF applied to position before differentiation). Use the range slider at the bottom to navigate through the video.")

def plot_fft_analysis(data):
    """Plot FFT comparison of position data before and after LPF"""
    if not data:
        return
    
    left_h = [d['left_horizontal'] for d in data]
    right_h = [d['right_horizontal'] for d in data]
    
    # Filter position data
    left_h_filtered = apply_lowpass_filter(left_h)
    right_h_filtered = apply_lowpass_filter(right_h)
    
    # Compute FFTs
    freq_left_raw, mag_left_raw = compute_fft(left_h)
    freq_left_filt, mag_left_filt = compute_fft(left_h_filtered)
    freq_right_raw, mag_right_raw = compute_fft(right_h)
    freq_right_filt, mag_right_filt = compute_fft(right_h_filtered)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Left Eye - Raw Position FFT', 'Left Eye - Filtered Position FFT',
                       'Right Eye - Raw Position FFT', 'Right Eye - Filtered Position FFT'),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Left Eye - Raw FFT
    fig.add_trace(
        go.Scatter(x=freq_left_raw, y=mag_left_raw, mode='lines', name='Left Raw',
                   line=dict(color='blue', width=1), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=1, col=1
    )
    
    # Left Eye - Filtered FFT
    fig.add_trace(
        go.Scatter(x=freq_left_filt, y=mag_left_filt, mode='lines', name='Left Filtered',
                   line=dict(color='green', width=1), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=1, col=2
    )
    
    # Right Eye - Raw FFT
    fig.add_trace(
        go.Scatter(x=freq_right_raw, y=mag_right_raw, mode='lines', name='Right Raw',
                   line=dict(color='red', width=1), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=2, col=1
    )
    
    # Right Eye - Filtered FFT
    fig.add_trace(
        go.Scatter(x=freq_right_filt, y=mag_right_filt, mode='lines', name='Right Filtered',
                   line=dict(color='orange', width=1), hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.2f}'),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Frequency (normalized)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (normalized)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (normalized)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (normalized)", row=2, col=2)
    
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)
    
    fig.update_layout(
        title_text='FFT Analysis: Position Data Before and After LPF',
        title_font_size=18,
        height=800,
        autosize=True,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.show()
    print("✓ FFT analysis plot displayed in separate window.")

def print_statistics(data):
    """Print statistics about eye tracking data"""
    if not data:
        return
    
    left_h = [d['left_horizontal'] for d in data]
    left_v = [d['left_vertical'] for d in data]
    right_h = [d['right_horizontal'] for d in data]
    right_v = [d['right_vertical'] for d in data]
    
    print("\n" + "="*60)
    print("EYE TRACKING STATISTICS")
    print("="*60)
    print(f"Total frames: {len(data)}")
    print(f"\nLeft Eye - Horizontal (Yaw):")
    print(f"  Range: [{min(left_h):.2f}°, {max(left_h):.2f}°]")
    print(f"  Mean: {np.mean(left_h):.2f}°, Std: {np.std(left_h):.2f}°")
    print(f"\nLeft Eye - Vertical (Pitch):")
    print(f"  Range: [{min(left_v):.2f}°, {max(left_v):.2f}°]")
    print(f"  Mean: {np.mean(left_v):.2f}°, Std: {np.std(left_v):.2f}°")
    print(f"\nRight Eye - Horizontal (Yaw):")
    print(f"  Range: [{min(right_h):.2f}°, {max(right_h):.2f}°]")
    print(f"  Mean: {np.mean(right_h):.2f}°, Std: {np.std(right_h):.2f}°")
    print(f"\nRight Eye - Vertical (Pitch):")
    print(f"  Range: [{min(right_v):.2f}°, {max(right_v):.2f}°]")
    print(f"  Mean: {np.mean(right_v):.2f}°, Std: {np.std(right_v):.2f}°")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    if data:
        # Print statistics
        print_statistics(data)
        
        # Plot horizontal analysis
        plot_horizontal_analysis(data)
        
        # Plot FFT analysis
        plot_fft_analysis(data)
