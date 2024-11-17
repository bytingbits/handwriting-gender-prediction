import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
snow_cnt=0
# Path to your local data folder
LOCAL_DATA_PATH = './'  # Adjust this if your data is in a different folder
MODEL_SAVE_PATH = './models/'  # Folder to save trained models

def create_animated_credit_bar():
    # Custom CSS for the container
    st.markdown("""
        <style>
        .credit-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(90deg, #ff9ecd, #ffb8de, #ffd1ec);
            padding: 10px;
            text-align: center;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        .floating-circle {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin: 0 10px;
            animation: float 2s infinite ease-in-out;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .credit-text {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: #333;
            margin: 0 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create the animated bar with circles and text
    st.markdown(f"""
        <div class="credit-container">
            <div class="floating-circle" style="background-color: #ff9ecd; animation-delay: 0s;"></div>
            <div class="floating-circle" style="background-color: #ffb8de; animation-delay: 0.3s;"></div>
            <div class="floating-circle" style="background-color: #ffd1ec; animation-delay: 0.6s;"></div>
            <span class="credit-text">Made by Sreya, Shakthi and Sharada</span>
            <div class="floating-circle" style="background-color: #ffd1ec; animation-delay: 0.9s;"></div>
            <div class="floating-circle" style="background-color: #ffb8de; animation-delay: 1.2s;"></div>
            <div class="floating-circle" style="background-color: #ff9ecd; animation-delay: 1.5s;"></div>
        </div>
    """, unsafe_allow_html=True)

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
        'C': [1, 10, 50, 200, 400],  # Range of values for C
        'gamma': [0.001, 0.01, 0.1, 1, 7.6, 10, 'scale'],  # Range of values for gamma
        'kernel': ['poly']
        
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
create_animated_credit_bar()
# Feature selection
feature_set = st.sidebar.selectbox("Choose Feature Set", ["Hinge", "Cold"])

# Choose optimization method
optimization_method = st.sidebar.selectbox("Choose Hyperparameter Optimization Method", 
                                           ["Manual", "GridSearchCV"])

# C and gamma sliders for manual tuning    

# Load the correct feature set
if feature_set == "Hinge":
    x_train, y_train, x_test, y_test = x_hinge_train, y_hinge_train, x_hinge_test, y_hinge_test
    if optimization_method == "Manual":
        C_value = st.sidebar.slider("Select C", 0.01, 50.0, 0.1)
        gamma_value = 'scale'
else:
    if snow_cnt==0:
        st.snow()
        snow_cnt=1
    x_train, y_train, x_test, y_test = x_cold_train, y_cold_train, x_cold_test, y_cold_test
    if optimization_method == "Manual":
        C_value = st.sidebar.slider("Select C", 0.01, 500.0, 1.0)
        gamma_value = st.sidebar.slider("Select Gamma", 0.001, 10.0, 0.1)
   
    

# Train and evaluate with hyperparameter optimization or manual tuning
if optimization_method == "GridSearchCV":
    clf, best_params = hyperparameter_optimization(x_train, y_train)
    #st.write(f"Best Parameters: {best_params}")
    cols = st.columns(len(best_params))  # Create a column for each item in the dictionary

# Iterate over the dictionary and display key-value pairs in separate columns
    for i, (key, value) in enumerate(best_params.items()):
        cols[i].metric(key, value)  # Display key as heading
        
    st.write("Formula for scaled Gamma:")
    st.latex(r"""
    \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}
""")
else:
    clf = SVC(kernel='poly', C=C_value, gamma=gamma_value)
    accuracy_train, accuracy_test, confusion_train, confusion_test, y_pred_test = train_and_evaluate(
        clf, x_train, y_train, x_test, y_test
    )

# Display results
if optimization_method == "Manual":
    # Display donut chart for training accuracy
    fig_train = go.Figure()
    fig_train.add_trace(go.Pie(
        labels=["Training Accuracy", "Remaining"], 
        values=[accuracy_train * 100, 100 - accuracy_train * 100],
        hole=0.3,
        hoverinfo="label+percent"
    ))
    fig_train.update_layout(title="Training Accuracy")
    st.plotly_chart(fig_train)

    # Display donut chart for testing accuracy
    fig_test = go.Figure()
    fig_test.add_trace(go.Pie(
        labels=["Testing Accuracy", "Remaining"], 
        values=[accuracy_test * 100, 100 - accuracy_test * 100],
        hole=0.3,
        hoverinfo="label+percent"
    ))
    fig_test.update_layout(title="Testing Accuracy")
    st.plotly_chart(fig_test)

    # Confusion Matrix
    st.subheader("Confusion Matrix (Testing Data)")
    sns.heatmap(confusion_test, annot=True, fmt='d', cmap="Blues")
    st.pyplot()

    # Optionally display predicted vs expected
    if st.checkbox("Show Predicted vs Expected DataFrame"):
        df = pd.DataFrame({'True': y_test, 'Predicted': y_pred_test})
        st.write(df)

# Saving and downloading the model
#os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
#model_filename = os.path.join(MODEL_SAVE_PATH, f'model_{feature_set.lower()}.pkl')
#pickle.dump(clf, open(model_filename, 'wb'))

# Provide a download link for the model
#st.download_button("Download Model", model_filename)
