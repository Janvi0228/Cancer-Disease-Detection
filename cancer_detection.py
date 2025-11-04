# cancer_detection.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox

def train_model():
    data = {
        'fatigue': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'cough': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        'chest_pain': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'weight_loss': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'bleeding': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'fever': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'loss_of_appetite': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'nausea': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'difficulty_breathing': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'skin_changes': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'swelling': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'night_sweats': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'cancer_detected': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)
    feature_cols = [
        'fatigue', 'cough', 'chest_pain', 'weight_loss', 'bleeding',
        'fever', 'loss_of_appetite', 'nausea', 'difficulty_breathing',
        'skin_changes', 'swelling', 'night_sweats'
    ]
    X = df[feature_cols]
    y = df['cancer_detected']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, feature_cols

def main():
    model, scaler, feature_cols = train_model()

    root = tk.Tk()
    root.title("Cancer Detection (Educational Demo)")

    frm = ttk.Frame(root, padding=16)
    frm.grid()

    title_lbl = ttk.Label(frm, text="Cancer Detection (Educational Demo)")
    title_lbl.grid(column=0, row=0, columnspan=3, pady=(0, 8))

    ttk.Label(frm, text="Select your symptoms:").grid(column=0, row=1, sticky="w", pady=(0, 8))

    vars_map = {}
    for i, feature in enumerate(feature_cols):
        var = tk.BooleanVar(value=False)
        vars_map[feature] = var
        label = feature.replace("_", " ").capitalize()
        col = i % 3
        row = 2 + i // 3
        ttk.Checkbutton(frm, text=label, variable=var).grid(column=col, row=row, sticky="w", padx=8, pady=4)

    # Controls row just below the last checkbox row
    controls_row = 2 + (len(feature_cols) + 2) // 3

    # Result area
    result_frame = ttk.Frame(frm)
    result_frame.grid(column=0, row=controls_row + 1, columnspan=3, sticky="we", pady=(12, 0))
    result_label = ttk.Label(result_frame, text="Result will appear here.")
    result_label.grid(column=0, row=0, sticky="w")
    prob_bar = ttk.Progressbar(result_frame, orient="horizontal", length=280, mode="determinate", maximum=100)
    prob_bar.grid(column=0, row=1, sticky="we", pady=(6, 0))

    def update_prediction(show_dialog=False):
        user_inputs = {k: int(v.get()) for k, v in vars_map.items()}
        user_df = pd.DataFrame([user_inputs])
        user_scaled = scaler.transform(user_df)
        pred = model.predict(user_scaled)[0]
        proba = float(model.predict_proba(user_scaled)[0, 1])
        prob_bar["value"] = int(proba * 100)
        if pred == 1:
            result_label.config(text=f"Possible signs of cancer detected. Probability: {proba:.2f}")
            if show_dialog:
                messagebox.showerror("Result", f"Possible signs of cancer detected. Probability: {proba:.2f}\n\nThis is for educational use only, not medical advice.")
        else:
            result_label.config(text=f"No cancer detected. Probability of cancer: {proba:.2f}")
            if show_dialog:
                messagebox.showinfo("Result", f"No cancer detected. Probability of cancer: {proba:.2f}\n\nThis is for educational use only, not medical advice.")

    def on_predict():
        update_prediction(show_dialog=True)

    def on_select_all():
        for v in vars_map.values():
            v.set(True)
        if auto_update_var.get():
            update_prediction(show_dialog=False)

    def on_clear_all():
        for v in vars_map.values():
            v.set(False)
        result_label.config(text="Result will appear here.")
        prob_bar["value"] = 0

    # Auto update option
    auto_update_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text="Auto update", variable=auto_update_var).grid(column=2, row=controls_row, sticky="e")

    # Buttons
    predict_btn = ttk.Button(frm, text="Predict", command=on_predict)
    predict_btn.grid(column=0, row=controls_row, pady=(12, 0), sticky="w")
    select_all_btn = ttk.Button(frm, text="Select all", command=on_select_all)
    select_all_btn.grid(column=0, row=controls_row, padx=(90, 0), pady=(12, 0), sticky="w")
    clear_btn = ttk.Button(frm, text="Clear", command=on_clear_all)
    clear_btn.grid(column=0, row=controls_row, padx=(170, 0), pady=(12, 0), sticky="w")

    # Trace variables for auto update
    def variable_changed(*_):
        if auto_update_var.get():
            update_prediction(show_dialog=False)
    for v in vars_map.values():
        v.trace_add('write', variable_changed)

    root.mainloop()


if __name__ == "__main__":
    main()
