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

Model 2: Machine learning model - Skills Matching

Objective of model 2: Predict the career domain that best matches a résumé’s skill list, using a text-processing pipeline and an ensemble of classifiers.

Step 1: 
- Loads `resumes.csv` from the `data/` folder. 
- Fills any missing skill strings with `''`, converts text to lowercase, and strips non-alphanumeric characters so the corpus is noise-free.

Step 2: Encode target 
- Uses `LabelEncoder` to convert categorical career labels (e.g., “Data Science”) into integers that ML algorithms can learn from.

Step 3: Vectorise & Tokenise Skills
- Defines a custom tokenizer `skill_tokenizer` that splits comma-separated skills.
- Applies `TfidfVectorizer` with `ngram_range=(1, 2)` and `max_df=0.9`, creating sparse TF-IDF vectors that capture single skills and two-word phrases such as “data analysis”.

Step 4: Select Best Features 
- Runs `SelectKBest(chi2, k = 1000)` to keep the 1 000 skill features most correlated with the career labels, trimming noise and curbing over-fitting.

Step 5: Build and ensemble classifier 
Wraps four complementary learners in a soft-voting ensemble:  
  – Logistic Regression (interpretable linear baseline)  
  – Random Forest (captures non-linear feature interactions)  
  – K-Nearest Neighbours with cosine distance (leverages skill similarity)  
  – Support Vector Machine (robust in high-dimensional TF-IDF space)
In soft voting, each model outputs a probability distribution over all career classes; the ensemble averages these probabilities and selects the class with the highest resulting score, delivering a smoother, more reliable prediction than any single learner could achieve on its own

Step 6: Hyperparameter Tuning
- Uses `GridSearchCV` wrapped around the full pipeline and explores 72 parameter combinations:  
  `select__k ∈ {500, 1000}`  `lr__C ∈ {0.1, 1, 10}`  
  `rf__max_depth ∈ {10, 20}` `knn__n_neighbors ∈ {3, 5}`  
- Employs `StratifiedKFold(n_splits = 5)` so every fold preserves class balance.
- Best configuration selected:  `k = 1000`, `C = 1`, `max_depth = 20`, `n_neighbors = 5`.  
Hyperparameter tuning is essential for squeezing maximum predictive power from our models, and GridSearchCV gives us a principled, repeatable way to do that without manual guess-and-check tuning. GridSearchCV systematically tries every candidate combination and scores each with cross-validation, giving an unbiased estimate of generalisation.

Step 7: Persist the model 
Saves the tuned pipeline as `career_classifier_advanced.pkl` and the `LabelEncoder` as `label_encoder.pkl` with `joblib`, making the entire workflow reloadable in a single line for real-time inference.

Outcome: A fast, 93 %-accurate resume classifier that marries classic TF-IDF text features with an ensemble of tuned algorithms, ready for deployment in a Flask or FastAPI service to recommend career paths at scale.








