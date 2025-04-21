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
