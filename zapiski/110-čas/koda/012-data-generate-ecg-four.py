"""
This script generates synthetic ECG signals with four classes:
- Class 0 (Black): Normal ECG signals
- Class 1 (Red): ECG signals with Premature Ventricular Contraction (PVC)
- Class 2 (Blue): ECG signals with Atrial Fibrillation (AFib)
- Class 3 (Green): ECG signals with Ventricular Tachycardia (VTach)

PVC is an abnormal heartbeat that occurs when the heart's ventricles contract earlier than expected.
AFib is characterized by highly irregular RR intervals and absence of clear P waves.
VTach is characterized by a fast, regular rhythm with wide QRS complexes and often reduced amplitude.
"""

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Configuration
count = 300
class_counts = {
    "normal": count,
    "pvc": count,
    "afib": count,
    "vtach": count
}

duration = 3  # seconds
sampling_rate = 150
signal_length = duration * sampling_rate

n_normal = class_counts["normal"]
n_pvc = class_counts["pvc"]
n_afib = class_counts["afib"]
n_vtach = class_counts["vtach"]

# Generate healthy signals with more variety
X_normal = []
for i in tqdm(range(n_normal), desc="Generating Normal"):
    # Use a consistent heart rate for all three beats
    heart_rate = np.random.randint(60, 101)
    # Lower noise for clearer waves
    noise = np.random.uniform(0.005, 0.02)
    # Consistent amplitude
    amplitude = np.random.uniform(0.9, 1.1)
    
    # Generate one long signal and then split it into three parts
    full_signal = nk.ecg_simulate(duration=3, sampling_rate=sampling_rate,
                                 heart_rate=heart_rate, noise=noise, random_state=i)
    
    # Split into three equal parts
    part_length = len(full_signal) // 3
    beats = []
    for j in range(3):
        start_idx = j * part_length
        end_idx = start_idx + part_length
        beat = full_signal[start_idx:end_idx]
        beats.append(beat)
    
    signal = np.concatenate(beats)
    
    # Apply amplitude variation
    signal = signal * amplitude
    
    # Add very slight baseline drift
    baseline_drift = np.sin(np.linspace(0, 2*np.pi, signal_length)) * np.random.uniform(0.02, 0.05)
    signal = signal + baseline_drift
    
    X_normal.append(signal)
X_normal = np.array(X_normal)

# Generate PVC signals with more variety
X_pvc = []
for i in tqdm(range(n_pvc), desc="Generating PVC"):
    # Choose which of the three beats will be PVC
    pvc_beat_idx = np.random.randint(0, 3)
    beats = []
    for j in range(3):
        if j == pvc_beat_idx:
            # PVC beat: wider, low-amplitude with random variations
            heart_rate = np.random.randint(100, 140)
            noise = np.random.uniform(0.02, 0.06)
            pvc_base = nk.ecg_simulate(duration=1, sampling_rate=sampling_rate,
                                      heart_rate=heart_rate, noise=noise, random_state=i*10+j+2000)
            width_factor = np.random.uniform(1.5, 2.5)
            pvc_like = np.interp(np.linspace(0, len(pvc_base), int(len(pvc_base)*width_factor)),
                                np.arange(len(pvc_base)), pvc_base)
            pvc_like = pvc_like[:len(pvc_base)]
            pvc_amplitude = np.random.uniform(0.3, 0.8)
            pvc_like = pvc_like * pvc_amplitude
            beats.append(pvc_like)
        else:
            # Normal beat
            heart_rate = np.random.randint(60, 90)
            noise = np.random.uniform(0.01, 0.05)
            amplitude = np.random.uniform(0.7, 1.3)
            beat = nk.ecg_simulate(duration=1, sampling_rate=sampling_rate,
                                  heart_rate=heart_rate, noise=noise, random_state=i*10+j)
            beat = beat * amplitude
            beats.append(beat)
    signal = np.concatenate(beats)
    # Ensure exact length by padding or truncating if necessary
    if len(signal) < signal_length:
        signal = np.pad(signal, (0, signal_length - len(signal)))
    else:
        signal = signal[:signal_length]
    # Add some random baseline drift to the entire signal
    baseline_drift = np.sin(np.linspace(0, 4*np.pi, signal_length)) * np.random.uniform(0.1, 0.3)
    signal = signal + baseline_drift
    X_pvc.append(signal)
X_pvc = np.array(X_pvc)

# Generate AFib signals
X_afib = []
for i in tqdm(range(n_afib), desc="Generating AFib"):
    # Highly irregular RR intervals
    rr_intervals = np.random.normal(loc=0.8, scale=0.2, size=6)  # seconds
    rr_intervals = np.clip(rr_intervals, 0.4, 1.5)
    signal = np.array([])
    for j, rr in enumerate(rr_intervals):
        heart_rate = 60 / rr
        # Convert duration to samples
        samples = int(rr * sampling_rate)
        seg = nk.ecg_simulate(length=samples, sampling_rate=sampling_rate,
                             heart_rate=heart_rate,
                             noise=np.random.uniform(0.01, 0.03),
                             random_state=i*10+j)
        signal = np.concatenate([signal, seg])
    if len(signal) < signal_length:
        signal = np.pad(signal, (0, signal_length - len(signal)))
    else:
        signal = signal[:signal_length]
    X_afib.append(signal)
X_afib = np.array(X_afib)

# Generate VTach signals
X_vtach = []
for i in tqdm(range(n_vtach), desc="Generating VTach"):
    heart_rate = np.random.randint(150, 221)
    noise = np.random.uniform(0.01, 0.04)
    signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate,
                            heart_rate=heart_rate, noise=noise, random_state=i+5000)
    width_factor = np.random.uniform(1.5, 2.0)
    vtach_like = np.interp(np.linspace(0, len(signal), int(len(signal)*width_factor)),
                          np.arange(len(signal)), signal)
    vtach_like = vtach_like[:signal_length]
    vtach_like = vtach_like * np.random.uniform(0.3, 0.6)
    X_vtach.append(vtach_like)
X_vtach = np.array(X_vtach)

# Combine and label
X = np.vstack([X_normal, X_pvc, X_afib, X_vtach])
y = np.array([0]*n_normal + [1]*n_pvc + [2]*n_afib + [3]*n_vtach)

# Save to file
np.savez("ecg-four-classes.npz", X=X, y=y)

# Plot 10 random signals
plt.figure(figsize=(12, 8))
indices = random.sample(range(len(X)), 10)
color_map = {0: 'black', 1: 'red', 2: 'blue', 3: 'green'}
label_map = {0: 'Normal', 1: 'PVC', 2: 'AFib', 3: 'VTach'}
used_labels = set()
for i, idx in enumerate(indices):
    color = color_map[y[idx]]
    label = label_map[y[idx]]
    plot_label = label if label not in used_labels else ""
    plt.plot(X[idx] + i*2, color=color, label=plot_label)
    used_labels.add(label)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (offset for clarity)")
plt.grid(True)
plt.legend()
plt.savefig("ecg-four-classes.svg")
plt.show()