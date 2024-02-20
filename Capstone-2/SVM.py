#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE DEPENDENCIES

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score



# # DATA COLLECTION AND ANALYSIS

# In[2]:


df = pd.read_csv('merged_dataset.csv')

# Basic statistics of the dataset
print(df.describe())


# In[3]:


# number of rows and columns in the dataframe
df.shape


# In[4]:


# getting more information about the dataset
df.info()


# In[5]:


# checking for missing values in each column
df.isnull().sum()


# In[6]:


# getting some statistical measures about the data
df.describe()


# In[7]:


# distribution of target Variable
df['status'].value_counts()


# In[8]:


df['status'] = pd.to_numeric(df['status'], errors='coerce')

# Now you can apply the mean function
mean_value = df['status'].mean()


# In[9]:


X = df.drop(columns=['name','status'], axis=1)
Y = df['status']
print(X)


# In[10]:


print(Y)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)


# # MODEL TRAINING (SVM)

# In[12]:


model = svm.SVC(kernel='linear')
# training the SVM model with training data
model.fit(X_train, y_train)


# In[13]:


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[14]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[15]:


# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

y_pred = model.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
PRF_Micro = precision_recall_fscore_support(y_test, y_pred, average='micro')
PRF_Macro = precision_recall_fscore_support(y_test, y_pred, average='macro')


print("Precision, Recall, F1-Score (Micro)")
print("Precision:", PRF_Micro[0])
print("Recall:", PRF_Micro[1])
print("F1-Score:", PRF_Micro[2])

conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)



import warnings
warnings.filterwarnings("ignore", category=UserWarning)
'''
#input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569
input_data = input()
input_list = input_data.split(",")
input_data_as_numpy_array = np.array(input_list)

# changing input data to a numpy array
#input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")'''


# In[16]:


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[17]:


# import tkinter as tk
# from tkinter import filedialog



# # Function to handle model prediction
# def run_prediction():
#     input_data = entry_var.get()
#     if input_data:
#         input_list = input_data.split(",")
#         input_data_as_numpy_array = np.array(input_list)

#         # Reshape the numpy array
#         input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#         # Standardize the data
#         std_data = scaler.transform(input_data_reshaped)

#         prediction = model.predict(std_data)

#         if prediction[0] == 0:
#             output_label.config(text="The Person does not have Parkinson's Disease")
#         else:
#             output_label.config(text="The Person has Parkinson's")
#     else:
#         output_label.config(text="Please enter data first.")


# # Create the main window
# root = tk.Tk()
# root.title("Parkinson's Disease Prediction")

# # Input label and entry field
# input_label = tk.Label(root, text="Enter data (comma-separated):")
# input_label.pack()
# entry_var = tk.StringVar()
# input_entry = tk.Entry(root, textvariable=entry_var)
# input_entry.pack()

# # Predict button
# predict_button = tk.Button(root, text="Predict", command=run_prediction)
# predict_button.pack()

# # Output label for displaying prediction result
# output_label = tk.Label(root, text="")
# output_label.pack()

# root.mainloop()


# In[18]:


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

        prediction = model.predict(std_data)

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

