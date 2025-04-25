import joblib
import fitz  # PyMuPDF
import re
import numpy as np

# -----------------------------
# Tokenizer (used in model)
# -----------------------------
def skill_tokenizer(text):
    return text.split(', ')

# -----------------------------
# Clean input for TF-IDF
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9, ]', '', text)
    return text

# -----------------------------
# Load Model & Label Encoder
# -----------------------------
model = joblib.load("career_classifier_advanced.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# -----------------------------
# Match known skills from resume
# -----------------------------
def extract_skills(resume_text, known_skills):
    resume_text = resume_text.lower()
    return [skill for skill in known_skills if re.search(rf'\b{re.escape(skill)}\b', resume_text)]

# -----------------------------
# Predict Career from Resume
# -----------------------------
def predict_from_resume(pdf_path):
    try:
        resume_text = extract_text_from_pdf(pdf_path)
        known_skills = list(model.named_steps['tfidf'].vocabulary_.keys())
        extracted = extract_skills(resume_text, known_skills)

        if not extracted:
            return "No recognizable skills found."

        cleaned_input = clean_text(", ".join(extracted))
        prediction = model.predict([cleaned_input])
        return label_encoder.inverse_transform(prediction)[0]

    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# CLI interface
# -----------------------------
if __name__ == "__main__":
    print("🔍 Career Predictor from Resume PDF")
    path = "test.pdf"

    result = predict_from_resume(path)
    print(f"\n✅ Prediction: {result}")


