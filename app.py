import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Your existing make_donut function
def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(domain=[input_text, ''], range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(domain=[input_text, ''], range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
    
    return plot_bg + plot + text
x_hinge = np.load('hinge_features.npy')
x_cold = np.load('cold_features.npy') #uncomment later
y = np.load('labels.npz')['label']
#split into training and testing 
x_hinge_train, x_hinge_test, y_hinge_train, y_hinge_test = train_test_split(
    x_hinge, y, random_state=104, test_size=0.1, shuffle=True
)

x_cold_train, x_cold_test, y_cold_train, y_cold_test = train_test_split(
    x_cold, y, random_state=104, test_size=0.1, shuffle=True
)
#combined
x_combined_train = np.hstack((x_hinge_train, x_cold_train))
x_combined_test = np.hstack((x_hinge_test, x_cold_test))
clf_combined = SVC(kernel='poly', verbose=True, C=200) #changing kernel to poly improves training and testing accuracy. 
clf_combined.fit(x_combined_train, y_hinge_train)
#training
y_pred_combined_train = clf_combined.predict(x_combined_train)

accuracy_combined_train = accuracy_score(y_hinge_train, y_pred_combined_train)
confusion_combined_train = confusion_matrix(y_hinge_train, y_pred_combined_train)
f1_combined_train = f1_score(y_hinge_train, y_pred_combined_train, average='weighted')
print("Training Accuracy:", accuracy_combined_train)
print("Confusion Matrix:\n", confusion_combined_train)
print("F1 Score:", f1_combined_train)
y_pred_combined_test = clf_combined.predict(x_combined_test)

accuracy_combined_test = accuracy_score(y_hinge_test, y_pred_combined_test)
confusion_combined_test = confusion_matrix(y_hinge_test, y_pred_combined_test)
f1_combined_test = f1_score(y_hinge_test, y_pred_combined_test, average='weighted')

# Display donut charts using Streamlit
st.title("Model Performance")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Training accuracy donut chart
with col1:
    st.altair_chart(make_donut(accuracy_combined_train * 100, "Training Accuracy", "blue"))

# Testing accuracy donut chart
with col2:
    st.altair_chart(make_donut(accuracy_combined_test * 100, "Testing Accuracy", "green"))
