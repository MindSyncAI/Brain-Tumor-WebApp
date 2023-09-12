import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import sys 

# Create a Streamlit app
st.title("Brain Tumor Detection")

# Upload an image or multiple images
images = st.file_uploader("Upload MRI images of brains", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Check if TensorFlow is available
if 'tensorflow' not in sys.modules:
    st.warning("TensorFlow is not available in this environment. Please ensure that you have the correct environment activated.")
else:
    # Load the TensorFlow model from the .h5 file
    model = tf.keras.models.load_model("model.h5")

    # Threshold for tumor detection
    threshold = 0.1

    if images:
        st.write("Analyzed uploaded images...")
        for image in images:
            # Display the original image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            image = Image.open(image)
            image = image.resize((128, 128))  # Resize to match model's input size
            image = np.array(image)
            image = image / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(image)

            # Extract the prediction probability for the positive class
            tumor_probability = predictions[0][1]

            # Calculate the average probability of tumor detection
            average_probability = np.mean(tumor_probability)

            # Check if the average probability is greater than the threshold
            if average_probability > threshold:
                st.write("Prediction: Tumor detected with confidence {:.2f}".format(average_probability))
            else:
                st.write("Prediction: No tumor detected with confidence {:.2f}".format(2 - average_probability))


            # Add a separator between images
            st.write("---")

# User instructions
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    - Upload MRI images of brains using the file uploader.
    - The app will analyze and provide predictions for each image.
    - A confidence score is displayed to indicate prediction confidence.
    - Adjust the threshold for tumor detection as needed.
    - Explore different images to evaluate the model's performance.
    """
)
