import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

# --- Step 1: Load the Data ---
# Load the CSV data from the "data" folder.
df = pd.read_csv("data/career_path_in_all_field.csv")

# --- Step 2: Define Features and Target ---
# "Career" is the target variable.
X = df.drop("Career", axis=1)    # X = the input features we know (e.g., GPA, internships, skills, etc.)
y = df["Career"]                 # y = the output we want to predict (in this case, the 'Career' field)

# Encode the target variable ("Career") using LabelEncoder.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Step 3: Define Preprocessing ---
# Specify numeric and categorical features.
numeric_features = [
    "GPA", "Extracurricular_Activities", "Internships", "Projects",
    "Leadership_Positions", "Field_Specific_Courses", "Research_Experience",
    "Coding_Skills", "Communication_Skills", "Problem_Solving_Skills",
    "Teamwork_Skills", "Analytical_Skills", "Presentation_Skills",
    "Networking_Skills", "Industry_Certifications"
]
categorical_features = ["Field"]

# Create a ColumnTransformer to scale numeric features and one-hot encode categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --- Step 4: Build the Pipeline ---
# Create the pipeline with preprocessing and a RandomForestClassifier.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# --- Step 5: Stratified Train-Test Split ---
# Use stratification to ensure that the distribution of classes is preserved.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Step 6: Train the Model ---
pipeline.fit(X_train, y_train)

# --- Step 7: Evaluate the Model on Test Data ---
y_pred = pipeline.predict(X_test)
print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Step 8: Stratified Cross-Validation ---
# Use StratifiedKFold for a more representative cross-validation.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
print("Stratified Cross-validation Accuracy Scores:", cv_scores)
print("Average Stratified Cross-validation Accuracy:", cv_scores.mean())

# --- Step 9: Export the Model ---
# Save the trained pipeline and the LabelEncoder for later use.
joblib.dump(pipeline, "carrer_model.pkl")
joblib.dump(le, "career_label_encoder.pkl")
