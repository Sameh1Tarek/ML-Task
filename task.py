#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create the main window
root = tk.Tk()
root.title("Ridge Regression")
root.geometry("800x600")

# Initialize a variable to keep track of whether data has been loaded
data_loaded = False

# Add a label and button to choose a file
label = tk.Label(root, text="Choose a file:")
label.grid(row=0, column=0, padx=10, pady=10)


def choose_file():
    load_data()


choose_file_button = tk.Button(root, text="Choose File", command=choose_file)
choose_file_button.grid(row=0, column=2, padx=10, pady=10)

# Add a treeview widget to display the CSV data
table = ttk.Treeview(root)
table.grid(row=1, column=0, columnspan=5, padx=10, pady=10)

# Add a text widget to display the file description
file_description = tk.Text(root, width=50, height=10)
file_description.grid(row=2, column=0, padx=10, pady=10, rowspan=3)

# Add a label to display the test size
test_size_label = tk.Label(root, text="Test Size: 0.1")
test_size_label.grid(row=2, column=1, padx=10, pady=10)

# Add a slider bar to select the test size
test_size_slider = tk.Scale(root, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, length=200,
                            command=lambda x: test_size_label.config(text=f"Test Size: {x}"))
test_size_slider.set(0.1)
test_size_slider.grid(row=2, column=2, padx=10, pady=10)

# Add a label and entry field to input the regularization strength
alpha_label = tk.Label(root, text="Regularization Strength:")
alpha_label.grid(row=3, column=1, padx=10, pady=10)

alpha_entry = tk.Entry(root)
alpha_entry.grid(row=3, column=2, padx=10, pady=10)

# Add a label to display the mean squared error
mse_label = tk.Label(root, text="Mean Squared Error (Test): N/A / (Train): N/A")
mse_label.grid(row=7, column=0, padx=10, pady=10)

# Add a label to display the root mean squared error
rmse_label = tk.Label(root, text="Root Mean Squared Error (Test): N/A / (Train): N/A")
rmse_label.grid(row=8, column=0, padx=10, pady=10)

# Add a label to display the mean absolute error
mae_label = tk.Label(root, text="Mean Absolute Error (Test): N/A / (Train): N/A")
mae_label.grid(row=9, column=0, padx=10, pady=10)


# Add a function to load data from a CSV file
def load_data():
    global data_loaded, x_train, x_test, y_train, y_test

    # Open a file dialog to choose a file
    file_path = filedialog.askopenfilename()

    # Load the data from the file
    data = pd.read_csv(file_path)
    # Impute missing values with the mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(data.iloc[:, 1:3])
    data.iloc[:, 1:3] = imputer.transform(data.iloc[:, 1:3])

    # Encode the categorical variables as numerical data
    le = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Display the data in the table widget
    table["columns"] = list(data.columns)
    table["show"] = "headings"
    for column in table["columns"]:
        table.heading(column, text=column)
    for row in data.to_numpy().tolist():
        table.insert("", "end", values=row)

    # Display the file description
    file_description.delete("1.0", tk.END)
    file_description.insert(tk.END, f"File Path: {file_path}\n\n")
    file_description.insert(tk.END, f"Number of Rows: {data.shape[0]}\n")
    file_description.insert(tk.END, f"Number of Columns: {data.shape[1]}")

    # Split the data into training and test sets
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_slider.get(), random_state=42)

    data_loaded = True


# Add a function to clear the loaded data
def clear_data():
    global data_loaded

    data_loaded = False

    # Clear the table widget
    table.delete(*table.get_children())

    # Clear the file description widget
    file_description.delete("1.0", tk.END)

    # Clear the evaluation metric labels
    mse_label.config(text="Mean Squared Error (Test): N/A / (Train): N/A")
    rmse_label.config(text="Root Mean Squared Error (Test): N/A / (Train): N/A")
    mae_label.config(text="Mean Absolute Error (Test): N/A / (Train): N/A")

    # Clear the entry field for the regularization strength
    alpha_entry.delete(0, tk.END)


# Add a function to train the Ridge model
def train_model():
    global data_loaded, x_train, x_test, y_train, y_test

    # Check if data has been loaded
    if not data_loaded:
        return

    # Get the regularization strength from the entry field
    alpha = float(alpha_entry.get())

    # Train the Ridge model
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred_test = model.predict(x_test)

    # Compute the evaluation metrics on the test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Make predictions on the train set
    y_pred_train = model.predict(x_train)

    # Compute the evaluation metrics on the train set
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    # Update the labels to display the evaluation metrics
    mse_label.config(text=f"Mean Squared Error (Test): {mse_test:.2f} / (Train): {mse_train:.2f}")
    rmse_label.config(text=f"Root Mean Squared Error (Test): {rmse_test:.2f} / (Train): {rmse_train:.2f}")
    mae_label.config(text=f"Mean Absolute Error (Test): {mae_test:.2f} / (Train): {mae_train:.2f}")


# Add a button to train the model
train_model_button = tk.Button(root, text="Train Model", command=train_model)
train_model_button.grid(row=6, column=1, padx=10, pady=10)

# Add a button to clear the loaded data
clear_data_button = tk.Button(root, text="Clear Data", command=clear_data)
clear_data_button.grid(row=6, column=2, padx=10, pady=10)

# Run the main loop
root.mainloop()


# In[ ]:





# In[ ]:




