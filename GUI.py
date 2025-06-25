import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Women Prediction System")
root.geometry("1300x850")
root.configure(bg='#ffebee')

# Title
title_label = tk.Label(root, text="Women Prediction System", font=("Helvetica", 24, "bold"), 
                       bg='#ffebee', fg='#c0392b')
title_label.pack(pady=20)

# Frame for graph and map
visual_frame = tk.Frame(root, bg='#ffebee')
visual_frame.pack(pady=10)

# Create figures for graph and map
fig_graph, ax_graph = plt.subplots(figsize=(6, 4))
fig_map, ax_map = plt.subplots(figsize=(6, 4))
ax_graph.set_facecolor('#ffffff')
ax_map.set_facecolor('#ffffff')
fig_graph.patch.set_facecolor('#ffffff')
fig_map.patch.set_facecolor('#ffffff')

canvas_graph = FigureCanvasTkAgg(fig_graph, master=visual_frame)
canvas_graph.get_tk_widget().grid(row=0, column=0, padx=10)

canvas_map = FigureCanvasTkAgg(fig_map, master=visual_frame)
canvas_map.get_tk_widget().grid(row=0, column=1, padx=10)

# Global variables
df = None
model = None
scaler = None
feature_names = None
crime_types = ['rape', 'kidnapping_and_abduction', 'dowry_deaths', 
               'assault_on_women_with_intent_to_outrage_her_modesty', 
               'insult_to_modesty_of_women', 'cruelty_by_husband_or_his_relatives', 
               'importation_of_girls']
X_train = {}
X_test = {}
y_train_dict = {}
y_test_dict = {}

def upload_file():
    global df, scaler, feature_names, X_train, X_test, y_train_dict, y_test_dict
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df = pd.read_excel(file_path)
        X = df[crime_types]
        y_dict = {crime: (df[crime] > 0).astype(int) for crime in crime_types}
        X_train.clear()
        X_test.clear()
        y_train_dict.clear()
        y_test_dict.clear()
        for crime in crime_types:
            X_train[crime], X_test[crime], y_train_dict[crime], y_test_dict[crime] = train_test_split(
                X, y_dict[crime], test_size=0.2, random_state=42)
        scaler = StandardScaler()
        for crime in crime_types:
            X_train[crime] = scaler.fit_transform(X_train[crime])
            X_test[crime] = scaler.transform(X_test[crime])
        feature_names = X.columns.tolist()
        result_label.config(text="Data loaded successfully!")

def train_model():
    global model
    if df is not None and X_train and y_train_dict:
        model = {crime: LogisticRegression(max_iter=1000) for crime in crime_types}
        for crime in crime_types:
            model[crime].fit(X_train[crime], y_train_dict[crime])
        result_label.config(text="Models trained successfully!")
    else:
        result_label.config(text="Please upload an Excel file first!")

def predict_crime(crime_type): 
    global model, scaler, df, feature_names
    if model and df is not None and feature_names is not None:
        if df[crime_types].iloc[-1].isnull().all():
            result_label.config(text="Error: No valid data for prediction.")
            return
        latest_data = df[crime_types].iloc[-1].values
        latest_data_df = pd.DataFrame([latest_data], columns=feature_names)
        latest_data_scaled = scaler.transform(latest_data_df)
        probability = model[crime_type].predict_proba(latest_data_scaled)[0][1]
        prediction = model[crime_type].predict(latest_data_scaled)[0]
        result = "Yes" if prediction == 1 else "No"
        result_label.config(text=f"{crime_type.capitalize()} Prediction: {result} (Probability: {probability:.2f})")

        # Bar chart for all crimes
        ax_graph.clear()
        probabilities = [model[crime].predict_proba(latest_data_scaled)[0][1] for crime in crime_types]
        bars = ax_graph.bar(crime_types, probabilities, color=plt.cm.Reds(probabilities))
        for bar in bars:
            height = bar.get_height()
            ax_graph.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')
        ax_graph.set_title("Crime Prediction Probabilities", fontsize=12)
        ax_graph.set_ylabel("Probability")
        plt.setp(ax_graph.get_xticklabels(), rotation=45, ha='right')
        canvas_graph.draw()

        # Simulated map by state
        ax_map.clear()
        if 'state/ut' in df.columns:
            state_data = df.groupby('state/ut')[crime_types].mean()
            state_probs = state_data.mean(axis=1)
            ax_map.bar(state_probs.index, state_probs.values, color=plt.cm.Reds(state_probs.values))
            ax_map.set_title("Average Crime Risk by State", fontsize=12)
            ax_map.set_ylabel("Average Crime Probability")
            ax_map.set_xticklabels(state_probs.index, rotation=45, ha='right')
        else:
            ax_map.text(0.5, 0.5, "No 'state/ut' column found.\nPlease check your Excel file.",
                        ha='center', va='center', color='#c0392b', fontsize=10)
        canvas_map.draw()
    else:
        result_label.config(text="Please upload a file and train the model first!")

def clear_all():
    ax_graph.clear()
    ax_map.clear()
    canvas_graph.draw()
    canvas_map.draw()
    result_label.config(text="Cleared all visualizations and result.")

# Frame for buttons
button_frame = tk.Frame(root, bg='#ffebee')
button_frame.pack(pady=10)

upload_button = ttk.Button(button_frame, text="Upload Excel File", command=upload_file)
upload_button.grid(row=0, column=0, padx=10)

train_button = ttk.Button(button_frame, text="Train Models", command=train_model)
train_button.grid(row=0, column=1, padx=10)

clear_button = ttk.Button(button_frame, text="Clear All", command=clear_all)
clear_button.grid(row=0, column=2, padx=10)

# Prediction Buttons
predict_buttons_frame = tk.Frame(root, bg='#ffebee')
predict_buttons_frame.pack(pady=10)

for idx, crime in enumerate(crime_types):
    btn = ttk.Button(predict_buttons_frame, text=f"Predict {crime.replace('_', ' ').capitalize()}",
                     command=lambda c=crime: predict_crime(c), width=30)
    btn.grid(row=idx // 2, column=idx % 2, padx=10, pady=5)

# Result Label
result_label = tk.Label(root, text="Prediction Result will appear here.", font=("Helvetica", 12), 
                        bg='#ffebee', fg='#c0392b', wraplength=1100)
result_label.pack(pady=10)

root.mainloop()
# END OF APPLICATION 