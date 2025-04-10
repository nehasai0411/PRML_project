import streamlit as st
import joblib
import cv2
import numpy as np

# Load trained model and PCA transformer
svm = joblib.load("svm_model.pkl")
pca = joblib.load("pca_transformer.pkl")

st.title("CIFAR-10 Image Classifier with SVM")
# Class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Decode and resize image to 32x32 (RGB)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize and flatten
    img_rgb = img_rgb.astype('float32') / 255.0
    flat_img = img_rgb.flatten().reshape(1, -1)  # (1, 3072)

    # Apply PCA transformation
    img_pca = pca.transform(flat_img)  # (1, 100)

    # Predict using SVM
    prediction = svm.predict(img_pca)

    predicted_class = class_names[prediction[0]]
    st.image(img_rgb, caption='Uploaded Image (Resized to 32x32)', use_column_width=True)
    st.write("Prediction:", predicted_class)
