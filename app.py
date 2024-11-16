import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Path to your local data folder
LOCAL_DATA_PATH = './'  # Adjust this if your data is in a different folder
MODEL_SAVE_PATH = './models/'  # Folder to save trained models

# Function to load data from local folder
def load_data_from_local():
    # Load the data from local files
    x_hinge = np.load(os.path.join(LOCAL_DATA_PATH, 'hinge_features.npy'))
    x_cold = np.load(os.path.join(LOCAL_DATA_PATH, 'cold_features.npy'))
    y = np.load(os.path.join(LOCAL_DATA_PATH, 'labels.npz'))['label']
    
    return x_hinge, x_cold, y

# Load the data
x_hinge, x_cold, y = load_data_from_local()

# Splitting dataset into Training and Testing for both hinge and cold features
x_hinge_train, x_hinge_test, y_hinge_train, y_hinge_test = train_test_split(
    x_hinge, y, random_state=104, test_size=0.1, shuffle=True
)

x_cold_train, x_cold_test, y_cold_train, y_cold_test = train_test_split(
    x_cold, y, random_state=104, test_size=0.1, shuffle=True
)

# Hyperparameter optimization function
def hyperparameter_optimization(x_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100, 200],  # Range of values for C
        'gamma': ['scale', 'auto', 0.1, 1, 10]  # Range of values for gamma
    }

    svc = SVC(kernel='poly', verbose=True)
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Function to train and evaluate the SVM model
def train_and_evaluate(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    confusion_train = confusion_matrix(y_train, y_pred_train)
    confusion_test = confusion_matrix(y_test, y_pred_test)

    return accuracy_train, accuracy_test, confusion_train, confusion_test, y_pred_test

# Streamlit UI
st.title("SVM Hyperparameter Optimization and Model Training")

# Feature selection
feature_set = st.sidebar.selectbox("Choose Feature Set", ["Hinge", "Cold"])

# Choose optimization method
optimization_method = st.sidebar.selectbox("Choose Hyperparameter Optimization Method", 
                                           ["None", "GridSearchCV"])

# C and gamma sliders for manual tuning
if optimization_method == "None":
    C_value = st.sidebar.slider("Select C", 0.1, 2000.0, 0.1)
    gamma_value = st.sidebar.slider("Select Gamma", 0.001, 10.0, 0.1)

# Load the correct feature set
if feature_set == "Hinge":
    x_train, y_train, x_test, y_test = x_hinge_train, y_hinge_train, x_hinge_test, y_hinge_test
else:
    x_train, y_train, x_test, y_test = x_cold_train, y_cold_train, x_cold_test, y_cold_test

# Train and evaluate with hyperparameter optimization or manual tuning
if optimization_method == "GridSearchCV":
    clf, best_params = hyperparameter_optimization(x_train, y_train)
    st.write(f"Best Parameters: {best_params}")
else:
    clf = SVC(kernel='poly', C=C_value, gamma=gamma_value)
    accuracy_train, accuracy_test, confusion_train, confusion_test, y_pred_test = train_and_evaluate(
        clf, x_train, y_train, x_test, y_test
    )

# Display results
if optimization_method == "None":
    # Display donut charts for accuracy
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Training", "Testing"], 
        values=[accuracy_train*100, accuracy_test*100],
        hole=0.3,
        hoverinfo="label+percent"
    ))
    st.plotly_chart(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix (Testing Data)")
    sns.heatmap(confusion_test, annot=True, fmt='d', cmap="Blues")
    st.pyplot()

    # Optionally display predicted vs expected
    if st.checkbox("Show Predicted vs Expected DataFrame"):
        df = pd.DataFrame({'True': y_test, 'Predicted': y_pred_test})
        st.write(df)

# Saving and downloading the model
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model_filename = os.path.join(MODEL_SAVE_PATH, f'model_{feature_set.lower()}.pkl')
pickle.dump(clf, open(model_filename, 'wb'))

# Provide a download link for the model
st.download_button("Download Model", model_filename)
