import numpy as np
import os
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def main():
    st.title("Hinge and Cold Data Classifier")

    # Load data
    x_hinge = np.load('hinge_features.npy')
    x_cold = np.load('cold_features.npy')
    y = np.load('labels.npz')['label']

    # Split data into training and testing
    x_hinge_train, x_hinge_test, y_hinge_train, y_hinge_test = train_test_split(
        x_hinge, y, random_state=104, test_size=0.1, shuffle=True
    )
    x_cold_train, x_cold_test, y_cold_train, y_cold_test = train_test_split(
        x_cold, y, random_state=104, test_size=0.1, shuffle=True
    )

    # Combine features
    x_combined_train = np.hstack((x_hinge_train, x_cold_train))
    x_combined_test = np.hstack((x_hinge_test, x_cold_test))

    # Allow user to choose the kernel
    kernel = st.selectbox("Select Kernel", ["rbf", "poly"])

    # Train the model
    if kernel == "rbf":
        clf_combined = SVC(kernel='rbf', verbose=True, C=200)
    else:
        clf_combined = SVC(kernel='poly', verbose=True, C=200)
    clf_combined.fit(x_combined_train, y_hinge_train)

    # Evaluate the model
    y_pred_combined_train = clf_combined.predict(x_combined_train)
    accuracy_combined_train = accuracy_score(y_hinge_train, y_pred_combined_train)
    confusion_combined_train = confusion_matrix(y_hinge_train, y_pred_combined_train)
    f1_combined_train = f1_score(y_hinge_train, y_pred_combined_train, average='weighted')

    y_pred_combined_test = clf_combined.predict(x_combined_test)
    accuracy_combined_test = accuracy_score(y_hinge_test, y_pred_combined_test)
    confusion_combined_test = confusion_matrix(y_hinge_test, y_pred_combined_test)
    f1_combined_test = f1_score(y_hinge_test, y_pred_combined_test, average='weighted')

    # Display results
    st.subheader("Training Metrics")
    st.write(f"Accuracy: {accuracy_combined_train:.2f}")
    st.write(f"Confusion Matrix:\n{confusion_combined_train}")
    st.write(f"F1 Score: {f1_combined_train:.2f}")

    st.subheader("Testing Metrics")
    st.write(f"Accuracy: {accuracy_combined_test:.2f}")
    st.write(f"Confusion Matrix:\n{confusion_combined_test}")
    st.write(f"F1 Score: {f1_combined_test:.2f}")

if __name__ == "__main__":
    main()