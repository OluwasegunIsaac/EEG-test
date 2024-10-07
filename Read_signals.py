import mne
import numpy as np
import tempfile
import streamlit as st

# Function to normalize EEG data
def normalize_eeg(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal

# Function to slice EEG signal into 30-second windows
def slice_signal(signal, sfreq, window_size=30):
    slices = []
    step = int(sfreq * window_size)
    for i in range(0, signal.shape[1], step):
        slice_chunk = signal[:, i:i + step]
        slices.append(slice_chunk)
    return slices

# Function to read and segment EEG data from .edf file
def read_edf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Read the temporary file with MNE
        raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)
        data, _ = raw[:19, :]  # Use the first 19 channels (standard EEG channels)
        sfreq = int(raw.info['sfreq'])  # Sampling frequency
    except Exception as e:
        st.error(f"Error reading the .edf file: {e}")
        return None, None
    
    # Normalize the signal
    normalized_signal = normalize_eeg(data)
    
    # Slice the normalized signal into 30-second windows
    segments = slice_signal(normalized_signal, sfreq, 30)
    
    return segments, sfreq
