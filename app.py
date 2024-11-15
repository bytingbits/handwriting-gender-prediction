from PIL import Image, ImageEnhance
import cv2
import numpy as np
import pickle
import streamlit as st

N_ANGLE_BINS_HINGE = 40
BIN_SIZE_HINGE = 360 // N_ANGLE_BINS_HINGE
LEG_LENGTH = 25
sharpness_factor = 10
bordersize = 3

def preprocess_image(img_file, sharpness_factor=10, bordersize=3):
    im = Image.open(img_file)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(sharpness_factor)

    # Resize image to double its size
    (width, height) = (im.width * 2, im.height * 2)
    im_s_1 = im_s_1.resize((width, height))
    image = np.array(im_s_1)

    # Add white border
    image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert to binary image using Otsu's thresholding
    _, bw_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw_image

def get_contour_pixels(bw_image):
    contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Sort contours by area in descending order and exclude the largest one (background)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

    return contours

def get_hinge_features(img_file):
    bw_image = preprocess_image(img_file, sharpness_factor, bordersize)
    contours = get_contour_pixels(bw_image)
    hist = np.zeros((N_ANGLE_BINS_HINGE, N_ANGLE_BINS_HINGE))

    for cnt in contours:
        n_pixels = len(cnt)
        if n_pixels <= LEG_LENGTH:
            continue

        points = np.array([point[0] for point in cnt])
        xs, ys = points[:, 0], points[:, 1]
        point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
        point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
        x1s, y1s = point_1s[:, 0], point_1s[:, 1]
        x2s, y2s = point_2s[:, 0], point_2s[:, 1]

        phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
        phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)

        indices = np.where(phi_2s > phi_1s)[0]

        for i in indices:
            phi1 = int(phi_1s[i] // BIN_SIZE_HINGE) % N_ANGLE_BINS_HINGE
            phi2 = int(phi_2s[i] // BIN_SIZE_HINGE) % N_ANGLE_BINS_HINGE
            hist[phi1, phi2] += 1

    # Normalize histogram
    normalised_hist = hist / np.sum(hist) if np.sum(hist) != 0 else hist
    feature_vector = normalised_hist[np.triu_indices_from(normalised_hist, k=1)]

    return feature_vector.reshape(1, -1)


with open('model_hinge_poly.pkl', 'rb') as model_file:
    model_hinge = pickle.load(model_file)

with open('model_cold_poly.pkl', 'rb') as model_file:
    model_cold = pickle.load(model_file)

with open('model_combined_poly.pkl', 'rb') as model_file:
    model_combined = pickle.load(model_file)

# Streamlit app UI
st.title("✍️ Handwriting-based Gender Classification")
st.write("Upload your handwriting sample image to get a gender prediction.")

# Dropdown menu to select the model
model_choice = st.selectbox(
    "Select the model to use for prediction:",
    options=["Hinge", "Cold", "Combined"],
    index=0
)

uploaded_file = st.file_uploader("Choose a handwriting sample image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    hinge_features = get_hinge_features(uploaded_file)
    # Add functionality to process features for other models if necessary
else:
    st.info("Please upload an image file to process.")

if st.button("Predict Gender"):
    # Select the appropriate model
    if model_choice == "Hinge":
        prediction = model_hinge.predict(hinge_features)
    elif model_choice == "Cold":
        prediction = model_cold.predict(hinge_features)  # Replace with actual cold model feature processing if different
    elif model_choice == "Combined":
        prediction = model_combined.predict(hinge_features)  # Replace with combined model feature processing if different
    
    gender = "Male" if prediction == 1 else "Female"
    st.write(f"Predicted Gender: {gender}")

