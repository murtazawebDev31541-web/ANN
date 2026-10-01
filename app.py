import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"Could not load the model: {e}")


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🔢 MNIST Digit Classifier")

st.write(
    "Upload a handwritten digit image and let the model predict it."
)


# ==========================================
# IMAGE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# IMAGE PROCESSING + PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict Digit"):

        if not model_loaded:
            st.error("Model could not be loaded.")
            st.stop()

        try:
            # ----------------------------------
            # Convert to grayscale
            # ----------------------------------

            image = image.convert("L")

            # ----------------------------------
            # Resize to MNIST size
            # ----------------------------------

            image = image.resize((28, 28))

            # ----------------------------------
            # Convert image to NumPy array
            # ----------------------------------

            image_array = np.array(image)

            # ----------------------------------
            # Normalize pixel values
            # 0-255 → 0-1
            # ----------------------------------

            image_array = image_array.astype("float32") / 255.0

            # ----------------------------------
            # Flatten image
            # 28 × 28 → 784
            # ----------------------------------

            image_array = image_array.reshape(1, 784)

            # ----------------------------------
            # Make prediction
            # ----------------------------------

            predictions = model.predict(
                image_array,
                verbose=0
            )

            # ----------------------------------
            # Get predicted digit
            # ----------------------------------

            predicted_digit = int(
                np.argmax(predictions[0])
            )

            # ----------------------------------
            # Get confidence
            # ----------------------------------

            confidence = float(
                np.max(predictions[0])
            ) * 100

            # ----------------------------------
            # Display result
            # ----------------------------------

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            # ----------------------------------
            # Probability distribution
            # ----------------------------------

            st.subheader("Prediction Probabilities")

            probabilities = predictions[0]

            probability_dict = {
                str(i): float(probabilities[i])
                for i in range(10)
            }

            st.bar_chart(probability_dict)

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )