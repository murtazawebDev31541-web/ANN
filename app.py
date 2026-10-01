import os
import numpy as np
import streamlit as st
import keras

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ==========================================
# 1. FASTAPI BACKEND CONFIGURATION
# ==========================================

api_app = FastAPI(title="Keras Model Inference API")

# Global model container
model = None


# ==========================================
# 2. LOAD KERAS MODEL
# ==========================================

def load_model():
    global model

    model_path = "mnist_model.keras"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")


# Load model when Streamlit starts
load_model()


# ==========================================
# 3. FASTAPI REQUEST MODEL
# ==========================================

class PredictRequest(BaseModel):
    features: list[float]


# ==========================================
# 4. FASTAPI PREDICTION ENDPOINT
# ==========================================

@api_app.post("/predict")
def predict(request: PredictRequest):

    global model

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    # Convert input features to NumPy array
    data = np.array(request.features, dtype=np.float32)

    # Check that we have exactly 784 features
    if data.shape[0] != 784:
        raise HTTPException(
            status_code=400,
            detail="Expected feature vector length 784."
        )

    # Add batch dimension
    input_data = np.expand_dims(data, axis=0)

    # Make prediction
    predictions = model.predict(input_data, verbose=0)

    # Get predicted class
    predicted_class = int(
        np.argmax(predictions, axis=1)[0]
    )

    # Convert probabilities to list
    probabilities = predictions[0].tolist()

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }


# ==========================================
# 5. STREAMLIT FRONTEND
# ==========================================

st.set_page_config(
    page_title="Model Inference Dashboard",
    layout="centered"
)


st.title("Sequential Neural Network Inference")

st.write(
    "FastAPI backend & Streamlit UI running together "
    "on Streamlit Community Cloud."
)


# ==========================================
# 6. INPUT FEATURES
# ==========================================

st.subheader("Input Features")

input_type = st.radio(
    "Choose Input Method:",
    (
        "Generate Random Sample",
        "Manual Zero Array"
    )
)


if input_type == "Generate Random Sample":

    sample_input = np.random.rand(784).tolist()

else:

    sample_input = np.zeros(784).tolist()


# ==========================================
# 7. PREDICTION BUTTON
# ==========================================

if st.button("Run Prediction"):

    payload = {
        "features": sample_input
    }

    try:

        # Call FastAPI prediction function directly
        result = predict(
            PredictRequest(**payload)
        )

        st.success(
            f"**Predicted Class:** "
            f"{result['predicted_class']}"
        )

        st.subheader(
            "Class Probabilities Distribution"
        )

        st.bar_chart(
            result["probabilities"]
        )

    except HTTPException as e:

        st.error(
            f"API Error ({e.status_code}): "
            f"{e.detail}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )