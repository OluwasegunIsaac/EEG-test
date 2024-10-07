import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import sobel

# Function to create a scalogram from a segment of EEG signal
def create_stft_scalogram(signal, fs, nperseg=256, noverlap=None, nfft=None):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    scalogram = np.abs(Zxx)
    log_scalogram = np.log1p(scalogram)
    return  f, t, log_scalogram

# Function to apply Sobel filter to a scalogram
def apply_sobel_filter(scalogram):
    # Apply Sobel filter in both x and y directions (time and frequency axes)
    sobel_x = sobel(scalogram, axis=1)  # Sobel filter along the time axis
    sobel_y = sobel(scalogram, axis=0)  # Sobel filter along the frequency axis
    
    # Combine the Sobel filters
    sobel_combined = np.hypot(sobel_x, sobel_y)  # Magnitude of the gradient
    
    return sobel_combined


# Function to preprocess EEG segment and generate scalogram image
def generate_scalogram_image(segment, sfreq):
    # Create the STFT scalogram
    f, t, scalogram = create_stft_scalogram(segment, sfreq)
    
    # Apply Sobel filter to the scalogram
    sobel_scalogram = apply_sobel_filter(scalogram)
    
    # Plot the scalogram using imshow
    fig, ax = plt.subplots()
    ax.imshow(sobel_scalogram, aspect='auto', extent=[t.min(), t.max(), f.min(), f.max()], cmap='jet')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert the plot to a NumPy array (scalogram image)
    fig.canvas.draw()  # Draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # Get RGB image as a NumPy array
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to match the image dimensions
    
    plt.close(fig)  # Close the figure to release memory
    
    return image  # Return the scalogram as an image (NumPy array)
