import subprocess
import tkinter as tk

def run_model(selected_model, input_data):
    if selected_model and input_data:
        notebook_files = {
            'SVM': 'SVM.ipynb',
            'KNN': 'KNN.ipynb',
            'Decision Tree': 'Decision_tree.ipynb',
            'Bagging': 'Bagging.ipynb'
        }

        notebook_file = notebook_files.get(selected_model)
        if notebook_file:
            # Provide the full path to the jupyter executable
            subprocess.run([r"C:\Python312\python.exe", "-m", "nbconvert", "--to", "script", notebook_file])
        else:
            print("Invalid model selected.")
    else:
        print("Please enter both model and input data.")

# Create the main window
root = tk.Tk()
root.title("Model Runner")

# Model selection dropdown
model_var = tk.StringVar(root)
model_var.set("SVM")  # default value
model_options = ['SVM', 'KNN', 'Decision Tree', 'Bagging']
model_dropdown = tk.OptionMenu(root, model_var, *model_options)
model_dropdown.pack()

# Input label and entry field
input_label = tk.Label(root, text="Enter data:")
input_label.pack()
input_string = tk.StringVar()
input_entry = tk.Entry(root, textvariable=input_string)
input_entry.pack()

# Run Model button
run_button = tk.Button(root, text="Run Model", command=lambda: run_model(model_var.get(), input_string.get()))
run_button.pack()

root.mainloop()
