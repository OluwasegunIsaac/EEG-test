import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from Read_signals import read_edf  # Import the signal reading module
from generate_scalograms import generate_scalogram_image  # Import the scalogram generation module
import os
import sys
import logging
import matplotlib.pyplot as plt

# Import your custom modules if they are in the same directory
# from Read_signals import read_edf  # Module to read .edf files
# from generate_scalograms import generate_scalogram_image  # Module to generate scalograms from signals

# If these modules are not available, comment out the import statements and make sure to include the necessary functions in the code.

# Set the page layout
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

# Function to preprocess the uploaded scalogram image
def preprocess_uploaded_image(image):
    img_height, img_width = 224, 224
    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to match model input shape
        image = image.resize((img_width, img_height))
        
        # Convert the image to a numpy array and normalize it
        image_array = np.array(image) / 255.0
        
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

# Function to predict the class using the pre-trained model for uploaded images
def predict_eeg(image):
    preprocessed_image = preprocess_uploaded_image(image)
    if preprocessed_image is None:
        return None
    try:
        # Get the predictions from the model
        logits = model.predict(preprocessed_image)
        
        # Since it's binary classification, use sigmoid for binary classification
        predicted_class_index = int(np.round(logits[0][0]))  # Assuming logits are raw predictions
        predicted_class = class_labels[predicted_class_index]
        
        return predicted_class
    except Exception as e:
        st.error(f"Error in making prediction: {e}")
        return None

# Load and display the logo at the top
try:
    logo = Image.open(logo_path)
    st.image(logo, use_column_width=True)
except Exception as e:
    st.error(f"Error loading logo: {e}")
    logging.debug(f"Error loading logo: {e}")


# Add vertical space to move the radio buttons down
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Create columns to center the radio buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Add a radio button for selecting the prediction method, centered horizontally
    st.markdown('<h2>Prediction Method</h2>', unsafe_allow_html=True)
    option = st.radio(
        "",
        ('EEG Signal', 'EEG Scalogram Image'),
        horizontal=True,
        label_visibility='collapsed'
    )
# Custom CSS for radio buttons and other styling
st.markdown(
    """
    <style>
    /* Horizontal radio buttons */
    div[data-baseweb="radio-group"] {
        display: inline-flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    div[data-baseweb="radio"] {
        display: inline-flex;
        align-items: center;
        margin-right: 20px; /* Space between the radio buttons */
    }

    /* Change the fill color of the radio button when selected */
    div[data-baseweb="radio"] input[type="radio"]:checked + div {
        background-color: black;
        border-radius: 50%;
        width: 15px;
        height: 15px;
        border: 2px solid black;
    }

    /* Style the labels next to the radio buttons */
    div[data-baseweb="radio"] label {
        margin-left: 8px;
        color: black;
        font-weight: bold;
    }

    /* Style for the Detect button */
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

    /* Hover effect for the Detect button */
    div.stButton > button:hover {
        background-color: #0056b3;
    }

    /* Style for the result box */
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

    /* Style for the result title */
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Style for the result text */
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

# --- Main Application Logic Based on Selected Option ---

if option == 'EEG Signal':
    # --- Code for Predicting with .edf File ---
    st.title("Epilepsy Detection from EEG (.edf) Files")
    st.write("Upload a .edf file to analyze 30-minute EEG data for epilepsy detection.")

    # File uploader for .edf files
    uploaded_file = st.file_uploader("Choose an .edf file", type=["edf"])

    # Reset session state when a new file is uploaded
    if uploaded_file is not None:
        # Clear the previous prediction result when a new file is uploaded
        st.session_state.prediction = None  # Reset prediction
        st.session_state.prediction_result = None  # Reset result text

        # Read and process the .edf file
        # Make sure you have the 'read_edf' function defined or imported
        segments, sfreq = read_edf(uploaded_file)  # Replace with your actual function

        if segments is None or sfreq is None:
            st.error("Could not process the uploaded EEG file.")
        else:
            concatenated_segments = np.concatenate(segments, axis=1)
            concatenated_channels = np.concatenate(concatenated_segments, axis=0)

            # Call the plotting function to display the EEG signals
            plot_eeg_signal(concatenated_channels, sfreq)

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

                                # Generate scalogram image
                                scalogram_image = generate_scalogram_image(flattened_segment, sfreq)  # Replace with your actual function

                                # Predict using the model
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

elif option == 'EEG Scalogram Image':
    # --- Code for Predicting with EEG Scalogram Image ---
    st.title("Epilepsy Detection from EEG Scalograms")
    st.write("Upload an EEG scalogram image for epilepsy prediction.")

    # File uploader to allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    # Reset session state when a new file is uploaded
    if uploaded_file is not None:
        # Clear the previous prediction result when a new file is uploaded
        st.session_state.prediction = None  # Reset prediction

        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Scalogram', use_column_width=True)

            # Center the Predict button using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Detect"):
                    with st.spinner("Please wait..."):
                        try:
                            # Get the prediction and store it in the session state
                            prediction = predict_eeg(image)
                            st.session_state.prediction = prediction
                        except Exception as e:
                            st.error(f"Error in prediction process: {e}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # Display the result only if a prediction has been made
    if st.session_state.prediction is not None:
        if st.session_state.prediction == "Epilepsy Detected":
            text_color = "#8B0000"  # Dark red for epilepsy detected
        else:
            text_color = "#4CAF50"  # Green for no epilepsy detected

        # Display the result in a visually appealing, styled result box
        st.markdown(f"""
        <div class="result-box">
            <p class="result-title">Result:</p>
            <p class="result-text" style="color:{text_color};">{st.session_state.prediction}</p>
        </div>
        """, unsafe_allow_html=True)

