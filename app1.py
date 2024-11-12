import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
print("Testing Accuracy:", accuracy_combined_test)
print("Confusion Matrix:\n", confusion_combined_test)
print("F1 Score:", f1_combined_test)