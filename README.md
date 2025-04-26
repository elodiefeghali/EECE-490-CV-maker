# EECE-490-CV-maker



Model 1: Machine Learning Model- Career Advisor
Objective of model 1: To train a machine learning classifier that predicts the most suitable career path for a candidate based on various academic, skill, and activity features, including GPA, coding skills, internships, and more.
We will walk through guided steps; first, we load the data, it will:
•	Reads the dataset from a CSV file.
•	Assumes the file is in a data directory.
•	Each row is a candidate; each column is a feature, except Career (the label).
Then we defined Features and Target: 
•	Drops the "Career" column to get the feature set X.
•	Encodes the categorical target Career into numerical form using LabelEncoder.
We cleaned our data and preprocessed by using StandardScaler which scale numerical features (important for many ML models) and OneHotEncoder which is applied to the "Field" column (e.g., "Engineering", "Biology").
Then we builded a pipeline: we combine preprocessing and model into one clean pipeline We also used RandomForestClassifier which is a robust ensemble model that handles mixed data types well.
After that, we split the data into 80% training and 20% testing and we use stratify to maintain the proportion of career classes across both sets.
Then we train the entire pipeline (both preprocessing and model) on the training data. We evaluate the model on unseen test data and printed out a classification_report that shows precision, recall, and F1-score for each career label. We used StratifiedKFold for cross-validation, maintaining class balance in each fold and we measure accuracy across 5 splits and prints the average. Finally, we save the entire pipeline (preprocessing + model) as well as the LabelEncoder to later decode predictions (e.g., 0 → "Data Scientist").










