import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import re


# Helper function to clean user inputs identically to training pipeline
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ReviewClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Review Authenticity Detector")
        self.root.geometry("550x450")
        self.root.configure(bg="#f4f6f9")

        # Load ML Pipeline models
        try:
            self.model = joblib.load('random_forest_model.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model artifacts not found!\nPlease run train_model.py first.")
            self.root.destroy()
            return

        self.setup_ui()

    def setup_ui(self):
        # Title Styling
        title_label = tk.Label(
            self.root,
            text="Review Fraud Detector",
            font=("Helvetica", 18, "bold"),
            bg="#f4f6f9",
            fg="#2c3e50"
        )
        title_label.pack(pady=15)

        # Text Prompt
        instruction_label = tk.Label(
            self.root,
            text="Enter the text review below to assess its authenticity:",
            font=("Helvetica", 11),
            bg="#f4f6f9",
            fg="#7f8c8d"
        )
        instruction_label.pack(anchor="w", padx=30)

        # Scrolled Text Box alternative frame
        self.text_area = tk.Text(
            self.root,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            width=55,
            height=10,
            bd=2,
            relief="groove"
        )
        self.text_area.pack(pady=10, padx=30)

        # Frame for action buttons
        button_frame = tk.Frame(self.root, bg="#f4f6f9")
        button_frame.pack(pady=10)

        # Analyze Action Button
        self.btn_analyze = tk.Button(
            button_frame,
            text="Verify Review",
            font=("Helvetica", 11, "bold"),
            bg="#27ae60",
            fg="white",
            padx=15,
            pady=5,
            command=self.classify_review,
            activebackground="#219653",
            activeforeground="white",
            cursor="hand2"
        )
        self.btn_analyze.grid(row=0, column=0, padx=10)

        # Clear Input Button
        self.btn_clear = tk.Button(
            button_frame,
            text="Clear Text",
            font=("Helvetica", 11),
            bg="#7f8c8d",
            fg="white",
            padx=15,
            pady=5,
            command=self.clear_input,
            activebackground="#95a5a6",
            activeforeground="white",
            cursor="hand2"
        )
        self.btn_clear.grid(row=0, column=1, padx=10)

        # Dynamic Output Display
        self.result_label = tk.Label(
            self.root,
            text="Classification Status: Awaiting Input",
            font=("Helvetica", 13, "bold"),
            bg="#f4f6f9",
            fg="#34495e"
        )
        self.result_label.pack(pady=20)

    def classify_review(self):
        user_input = self.text_area.get("1.0", tk.END).strip()

        if not user_input:
            messagebox.showwarning("Input Needed", "Please provide review text before analysis.")
            return

        # Preprocess, Transform and Predict
        cleaned = clean_text(user_input)
        vectorized_text = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized_text)[0]

        # Calculate Confidence/Probability metrics
        probabilities = self.model.predict_proba(vectorized_text)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = probabilities[class_idx] * 100

        # Update visual states
        if prediction == "Real":
            self.result_label.config(
                text=f"Result: AUTHENTIC REVIEW ({confidence:.1f}% Confidence)",
                fg="#27ae60"
            )
        else:
            self.result_label.config(
                text=f"Result: SUSPECTED FAKE REVIEW ({confidence:.1f}% Confidence)",
                fg="#c0392b"
            )

    def clear_input(self):
        self.text_area.delete("1.0", tk.END)
        self.result_label.config(text="Classification Status: Awaiting Input", fg="#34495e")


if __name__ == "__main__":
    root = tk.Tk()
    app = ReviewClassifierApp(root)
    root.mainloop()