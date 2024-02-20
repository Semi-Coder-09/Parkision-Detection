#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Data Collection & Analysis

# In[2]:


# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('./merged_dataset.csv')


# In[3]:


# printing the first 5 rows of the dataframe
parkinsons_data.head()


# In[4]:


# number of rows and columns in the dataframe
parkinsons_data.shape


# In[5]:


# getting more information about the dataset
parkinsons_data.info()


# In[6]:


# checking for missing values in each column
parkinsons_data.isnull().sum()


# In[7]:


# getting some statistical measures about the data
parkinsons_data.describe()


# In[8]:


# distribution of target Variable
parkinsons_data['status'].value_counts()


# 1  --> Parkinson's Positive
# 
# 0 --> Healthy
# 

# In[9]:


# grouping the data bas3ed on the target variable
#parkinsons_data.groupby('status').mean()
parkinsons_data['status'] = pd.to_numeric(parkinsons_data['status'], errors='coerce')

# Now you can apply the mean function
mean_value = parkinsons_data['status'].mean()


# Data Pre-Processing

# Separating the features & Target

# In[10]:


X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']


# In[11]:


print(X)


# In[12]:


print(Y)


# Splitting the data to training data & Test data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# Data Standardization

# In[15]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


scaler.fit(X_train)


# In[17]:


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[18]:


print(X_train)


# Model Training

# Support Vector Machine Model

# In[19]:


knn_pipe=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=9))
knn_pipe.fit(X_train, Y_train)
k = 3  # You can choose the value of k based on experimentation
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_scaled, Y_train)
y_pred = knn_classifier.predict(X_test_scaled)


# Model Evaluation

# Accuracy Score

# In[20]:


# accuracy score on training data
X_test_prediction = knn_classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

y_pred = knn_classifier.predict(X_test)
PRF_Micro = precision_recall_fscore_support(Y_test, y_pred, average='micro')
PRF_Macro = precision_recall_fscore_support(Y_test, y_pred, average='macro')

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Precision, Recall, F1-Score (Micro)")
print("Precision:", PRF_Micro[0])
print("Recall:", PRF_Micro[1])
print("F1-Score:", PRF_Micro[2])

conf_matrix = confusion_matrix(Y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)



import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Building a Predictive System

# In[21]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
#input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569

#changing input data to a numpy array
'''input_data = input()
input_list = input_data.split(",")
input_data_as_numpy_array = np.array(input_list)

#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = knn_classifier.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
 print("The Person has Parkinsons")
'''


# In[22]:


import tkinter as tk
import re
from tkinter import filedialog
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to handle model prediction
# Function to handle model prediction
def run_prediction():
    input_data = entry_var.get()
    if input_data:
        # Split the input data based on both commas and newlines
        input_list = [float(value) for value in re.split(r',|\n', input_data) if value.strip()]
        input_data_as_numpy_array = np.array(input_list)

        # Reshape the numpy array
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize the data
        std_data = scaler.transform(input_data_reshaped)

        prediction = knn_classifier.predict(std_data)

        if prediction[0] == 0:
            output_label.config(text="The Person does not have Parkinson's Disease")
        else:
            output_label.config(text="The Person has Parkinson's")

        # Plot confusion matrix in GUI
        plot_confusion_matrix()

    else:
        output_label.config(text="Please enter data first.")



# Function to plot the confusion matrix
def plot_confusion_matrix():
    # Assuming you have the `conf_matrix` variable available
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Embed the plot into Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()


# Create the main window
root = tk.Tk()
root.title("Parkinson's Disease Prediction")

# Input label and entry field
input_label = tk.Label(root, text="Enter data (comma-separated):")
input_label.pack()
entry_var = tk.StringVar()
input_entry = tk.Entry(root, textvariable=entry_var)
input_entry.pack()

# Predict button
predict_button = tk.Button(root, text="Predict", command=run_prediction)
predict_button.pack()

# Output label for displaying prediction result
output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()


