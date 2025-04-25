import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load the dataset
file_path = '/mnt/data/job_recommendation_dataset 3.csv'
df = pd.read_csv(file_path)

# Step 2: Define features and target variable
X = df[['Location', 'Experience Level', 'Salary', 'Industry', 'Required Skills']]
y = df['Job Title']

# Step 3: Preprocess the data
# Identify categorical and numeric features
numeric_features = ['Salary']
categorical_features = ['Location', 'Experience Level', 'Industry', 'Required Skills']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
# Step 4: Build the pipeline and set up GridSearchCV
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

# Set up the hyperparameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],  # Regularization strength
    'classifier__solver': ['liblinear', 'saga'],  # Solvers for logistic regression
    'classifier__penalty': ['l2', 'elasticnet']  # Regularization penalties
}

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
