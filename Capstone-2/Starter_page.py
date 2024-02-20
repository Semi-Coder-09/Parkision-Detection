import tkinter as tk
import subprocess
import os

class NotebookExecutor:
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path

    def execute_notebook(self):
        try:
            subprocess.Popen(["jupyter", "nbconvert", "--to", "script", self.notebook_path], stdout=subprocess.PIPE)
            subprocess.Popen(["python", f"{os.path.splitext(os.path.basename(self.notebook_path))[0]}.py"])
        except Exception as e:
            print(f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Notebook Executor")

# Define the NotebookExecutor instances for each button
executor_1 = NotebookExecutor("./SVM.ipynb")
executor_2 = NotebookExecutor("notebook2.ipynb")
executor_3 = NotebookExecutor("notebook3.ipynb")
executor_4 = NotebookExecutor("notebook4.ipynb")

# Function to execute the first notebook
def execute_notebook_1():
    executor_1.execute_notebook()

# Function to execute the second notebook
def execute_notebook_2():
    executor_2.execute_notebook()

# Function to execute the third notebook
def execute_notebook_3():
    executor_3.execute_notebook()

# Function to execute the fourth notebook
def execute_notebook_4():
    executor_4.execute_notebook()

# Create buttons for each notebook
button_1 = tk.Button(root, text="Execute Notebook 1", command=execute_notebook_1)
button_1.pack()

button_2 = tk.Button(root, text="Execute Notebook 2", command=execute_notebook_2)
button_2.pack()

button_3 = tk.Button(root, text="Execute Notebook 3", command=execute_notebook_3)
button_3.pack()

button_4 = tk.Button(root, text="Execute Notebook 4", command=execute_notebook_4)
button_4.pack()

# Run the Tkinter main loop
root.mainloop()
