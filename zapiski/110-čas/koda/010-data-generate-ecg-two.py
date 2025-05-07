"""
Generiramo sintetične signale EKG z dvema razredoma:
- razred 0 (črna): normalni EKG signali
- razred 1 (rdeča): signali EKG s prezgodnjo ventrikularno kontrakcijo (pvc)

pvc je nenormalni srčni utrip, ki se pojavi, ko se srčne komore skrčijo prej, kot je pričakovano.
"""

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# konfiguracija
count = 300
class_counts = {
    "normal": count,
    "pvc": count
}

duration = 3  # sekunde
sampling_rate = 150
signal_length = duration * sampling_rate

n_normal = class_counts["normal"]
n_pvc = class_counts["pvc"]

# generiranje zdravih signalov z več različicami
X_normal = []
for i in tqdm(range(n_normal), desc="Generating Normal"):
    # uporabi konstantno srčno frekvenco za vse tri utripe
    heart_rate = np.random.randint(60, 101)
    # manj šuma za jasnejše valove
    noise = np.random.uniform(0.005, 0.02)
    # konstantna amplituda
    amplitude = np.random.uniform(0.9, 1.1)
    
    # generiraj en dolg signal in ga nato razdel na tri dele
    full_signal = nk.ecg_simulate(duration=3, sampling_rate=sampling_rate,
                                 heart_rate=heart_rate, noise=noise, random_state=i)
    
    # razdel na tri enake dele
    part_length = len(full_signal) // 3
    beats = []
    for j in range(3):
        start_idx = j * part_length
        end_idx = start_idx + part_length
        beat = full_signal[start_idx:end_idx]
        beats.append(beat)
    
    signal = np.concatenate(beats)
    
    # uporabi variacijo amplitude
    signal = signal * amplitude
    
    # dodaj zelo rahlo variacijo osnovne črte
    baseline_drift = np.sin(np.linspace(0, 2*np.pi, signal_length)) * np.random.uniform(0.02, 0.05)
    signal = signal + baseline_drift
    
    X_normal.append(signal)
X_normal = np.array(X_normal)

# generiranje pvc signalov z več različicami
X_pvc = []
for i in tqdm(range(n_pvc), desc="Generating PVC"):
    # izberi, kateri od treh utripov bo pvc
    pvc_beat_idx = np.random.randint(0, 3)
    beats = []
    for j in range(3):
        if j == pvc_beat_idx:
            # pvc utrip: širši, nizka amplituda z naključnimi variacijami
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
            # normalen utrip
            heart_rate = np.random.randint(60, 90)
            noise = np.random.uniform(0.01, 0.05)
            amplitude = np.random.uniform(0.7, 1.3)
            beat = nk.ecg_simulate(duration=1, sampling_rate=sampling_rate,
                                  heart_rate=heart_rate, noise=noise, random_state=i*10+j)
            beat = beat * amplitude
            beats.append(beat)
    signal = np.concatenate(beats)
    # zagotovi točno dolžino z dopolnitvijo ali skrajšanjem po potrebi
    if len(signal) < signal_length:
        signal = np.pad(signal, (0, signal_length - len(signal)))
    else:
        signal = signal[:signal_length]
    # dodaj nekaj naključne variacije osnovne črte celotnemu signalu
    baseline_drift = np.sin(np.linspace(0, 4*np.pi, signal_length)) * np.random.uniform(0.1, 0.3)
    signal = signal + baseline_drift
    X_pvc.append(signal)
X_pvc = np.array(X_pvc)

# združi in označi
X = np.vstack([X_normal, X_pvc])
y = np.array([0]*n_normal + [1]*n_pvc)

# shrani v datoteko
np.savez("ecg-two-classes.npz", X=X, y=y)

# nariši 10 naključnih signalov
plt.figure(figsize=(12, 8))
indices = random.sample(range(len(X)), 10)
color_map = {0: 'black', 1: 'red'}
label_map = {0: 'Normal', 1: 'PVC'}
used_labels = set()
for i, idx in enumerate(indices):
    color = color_map[y[idx]]
    label = label_map[y[idx]]
    plot_label = label if label not in used_labels else ""
    plt.plot(X[idx] + i*2, color=color, label=plot_label)
    used_labels.add(label)
plt.xlabel("indeks vzorca")
plt.ylabel("amplituda (premik za jasnost)")
plt.grid(True)
plt.legend()
plt.savefig("ecg-two-classes.svg")
plt.show()