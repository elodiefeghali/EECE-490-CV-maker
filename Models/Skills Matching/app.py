import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# ---------------------------------------------
# 1. Load and Clean Dataset
# ---------------------------------------------
df = pd.read_csv("data/resumes.csv")
df['cleaned_resume'] = df['cleaned_resume'].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9, ]', '', text)
    return text

X_cleaned = df['cleaned_resume'].apply(clean_text)
y_raw = df['Category']

# ---------------------------------------------
# 2. Label Encode Targets
# ---------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ---------------------------------------------
# 3. Split Data
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------
# 4. Tokenizer Function (No Lambda!)
# ---------------------------------------------
def skill_tokenizer(text):
    return text.split(', ')

# ---------------------------------------------
# 5. Model Pipeline + Tuning Setup
# ---------------------------------------------
vectorizer = TfidfVectorizer(
    tokenizer=skill_tokenizer,
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9
)

feature_selector = SelectKBest(score_func=chi2, k=1000)

logreg = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=200)
knn = KNeighborsClassifier(metric='cosine')
svm = SVC(probability=True)

ensemble = VotingClassifier(estimators=[
    ('lr', logreg),
    ('rf', rf),
    ('knn', knn),
    ('svm', svm)
], voting='soft')

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('select', feature_selector),
    ('clf', ensemble)
])

param_grid = {
    'select__k': [500, 1000],
    'clf__rf__max_depth': [10, 20],
    'clf__lr__C': [0.1, 1, 10],
    'clf__knn__n_neighbors': [3, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------
# 6. Train and Tune
# ---------------------------------------------
print("🚀 Starting Grid Search...")
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# ---------------------------------------------
# 7. Evaluate on Test Set
# ---------------------------------------------
y_pred = grid.best_estimator_.predict(X_test)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------------------------------------------
# 8. Save Model and Encoder
# ---------------------------------------------
joblib.dump(grid.best_estimator_, 'career_classifier_advanced.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model and encoder saved to disk!")

# ---------------------------------------------
# 9. Function for Future Use
# ---------------------------------------------
def predict_career_from_text(resume_text: str) -> str:
    model = joblib.load('career_classifier_advanced.pkl')
    label_enc = joblib.load('label_encoder.pkl')
    cleaned = clean_text(resume_text)
    prediction = model.predict([cleaned])
    return label_enc.inverse_transform(prediction)[0]
