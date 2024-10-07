import streamlit as st
import tensorflow as tf
from Read_signals import read_edf  # Import the signal reading module
from generate_scalograms import generate_scalogram_image  # Import the scalogram generation module
from tensorflow.keras.models import load_model 
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import logging

# Set the page layout to "wide" to cover the full screen width
st.set_page_config(layout="centered")

# Logging configuration
logging.basicConfig(filename="app.log", level=logging.DEBUG)

# Helper function to get the correct path of resources (for standalone executables)
def resource_path(relative_path):
    """ Get the absolute path to resource files when running as a standalone executable. """
    try:
        base_path = sys._MEIPASS  # For PyInstaller and Nuitka standalone mode
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Load the model and logo using resource_path
model_path = resource_path("model.h5")
logo_path = resource_path("logo.png")

# Load the model
try:
    # Load the model
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.error(f"Error loading model: {e}", exc_info=True)  # Add exc_info=True to log the full traceback

# Define the class labels
class_labels = ['Epilepsy Detected', 'No Epilepsy Detected']

# Function to preprocess image for model prediction
def preprocess_image(image):
    img_height, img_width = 224, 224
    image = Image.fromarray(image).resize((img_width, img_height))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Function to predict whether the EEG segment is epileptic
def predict_epilepsy(segment_image):
    preprocessed_image = preprocess_image(segment_image)
    prediction = model.predict(preprocessed_image)
    return class_labels[int(np.round(prediction[0][0]))]

# Load and display the logo at the top
try:
    logo = Image.open(logo_path)
    st.image(logo, use_column_width=True)
except Exception as e:
    st.error(f"Error loading logo: {e}")
    logging.debug(f"Error loading logo: {e}")

# Streamlit app layout
st.title("Epilepsy Detection from EEG (.edf) Files")
st.write("Upload a .edf file to analyze 30-minute EEG data for epilepsy detection.")

# Custom CSS for styling the Detect button and result box
st.markdown(
    """
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    div.stButton > button {
        background-color: #007bff;
        color: white;
        padding: 15px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        width: 200px;
        text-align: center;
    }

    div.stButton > button:hover {
        background-color: #0056b3;
    }
    .result-box {
        margin: 20px auto;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        width: 30%;
        background-color: #f1f1f1;
    }
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to plot EEG signal
def plot_eeg_signal(concatenated_channels, sfreq):
    
    time = np.arange(len(concatenated_channels)) / sfreq  # Time in seconds
    
    # Plot the signal
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, concatenated_channels, label='EEG Signal')
    ax.axis('off')
    # ax.set_xlabel('Time (seconds)')
    # ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Store prediction results in session state to persist across interactions
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# File uploader for .edf files
uploaded_file = st.file_uploader("Choose an .edf file", type=["edf"])

# Reset session state when a new file is uploaded
if uploaded_file is not None:
    # Clear the previous prediction result when a new file is uploaded
    st.session_state.prediction = None  # Reset prediction
    st.session_state.prediction_result = None  # Reset result text
    
    # Read and process the .edf file
    segments, sfreq = read_edf(uploaded_file)
    concatenated_segments = np.concatenate(segments, axis=1)
    concatenated_channels = np.concatenate(concatenated_segments, axis=0)
    
    # Call the plotting function to display the EEG signals
    plot_eeg_signal(concatenated_channels, sfreq)

    if segments is None or sfreq is None:
        st.error("Could not process the uploaded EEG file.")
    else:
        segment_predictions = []

        # Center the Predict button using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Detect"):
                with st.spinner("Wait for result..."):  # Show spinner during prediction
                    try:
                        # Predict epilepsy for each 30-second segment
                        for i, segment in enumerate(segments):
                            flattened_segment = np.concatenate(segment, axis=0)
                            scalogram_image = generate_scalogram_image(flattened_segment, sfreq)
                            prediction = predict_epilepsy(scalogram_image)
                            segment_predictions.append(prediction)

                        # Count the predictions for each class
                        epilepsy_count = segment_predictions.count("Epilepsy Detected")
                        non_epilepsy_count = segment_predictions.count("No Epilepsy Detected")

                        # Majority vote for final result
                        if epilepsy_count > len(segments) // 2:
                            st.session_state.prediction = "Epilepsy Detected"
                        else:
                            st.session_state.prediction = "No Epilepsy Detected"
                    except Exception as e:
                        st.error(f"Error in prediction process: {e}")

        # Display prediction result in a styled box
        if st.session_state.prediction is not None:
            if st.session_state.prediction == "Epilepsy Detected":
                text_color = "#8B0000"  # Dark red for epilepsy detected
            else:
                text_color = "#4CAF50"  # Green for no epilepsy detected

            st.markdown(f"""
            <div class="result-box">
                <p class="result-title">Result:</p>
                <p class="result-text" style="color:{text_color};">{st.session_state.prediction}</p>
            </div>
            """, unsafe_allow_html=True)
