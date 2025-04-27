# EECE-490-CV-maker
Many job seekers, especially new graduates, struggle with the process of creating an optimized CV, identifying the right career path, and finding job opportunities that match their skills. 
This platform fills a significant gap by providing users with intelligent tools to enhance their job applications, receive personalized career advice. As the demand for personalized career development services increases.
The goal of this project is to develop a web application that allows users to upload their CVs and receive personalized career path predictions, job recommendations, and skills matching based on their academic and professional information. The application integrates machine learning models and OpenAI's GPT-4 to provide detailed analysis and recommendations.
Key Features:
The application provides several features to users. First, users can securely register, log in, and log out. The application stores and encrypts profile information (username, email, password) using Flask-SQLAlchemy and Werkzeug's password hashing mechanism.
Once users are logged in, they can upload their CVs in various formats including PDF, DOCX, or TXT. The uploaded CVs are stored in the database and saved to the server file system for further processing.
The application allows users to analyze their CVs. After uploading a CV, users can trigger an analysis using OpenAI's GPT-4 model. This analysis provides key information such as skills, years of experience, important keywords, an overall score, strengths and weaknesses, and recommendations. Users can download an improved version of their CV based on the analysis or suggested fixes.
Another key feature is career prediction. Users input their academic and skill-related data, such as GPA, coding skills, internships, leadership experience, and more. A machine learning model then predicts the most suitable career path for the user based on their profile.
Finally, the application includes a skills matching feature. Users can upload a resume in PDF format, and the application extracts skills from the document. A machine learning model then predicting suitable career paths based on the extracted skills.

Technological Stack:
The web application uses Flask, a lightweight Python web framework, for routing, session management, and template rendering. Flask-SQLAlchemy is used for handling database operations, allowing interaction with the database through an Object-Relational Mapping (ORM) system. OpenAI's GPT-4 API is integrated into the application to perform CV analysis, career predictions, and job recommendations.
The application uses Joblib to save and load machine learning models, such as the career prediction model and label encoder. Pandas and Scikit-learn are used for data manipulation, preprocessing, and building the machine learning models for career prediction and skills matching.
For password management and session security, the application uses Werkzeug's password hashing and session management tools. Additionally, PyMuPDF (fitz) and PyPDF2 libraries are used to extract text from PDF documents.
The python-docx library is used to create and manipulate DOCX files for generating CVs from user input.

Detailed Functional Flow:
When a user visits the application, they are first prompted to register. The registration form requires the user to provide their username, email, and password. Upon successful registration, users can log in to access the platform. After logging in, users are taken to the homepage.
On the homepage, users can upload their CVs. The application supports DOCX, PDF, and TXT formats. Once a CV is uploaded, users can view and manage their CVs in the profile section.
After uploading a CV, users can request an analysis of the document. The analysis provides insights into the CV content, including the key skills, years of experience, strengths, and weaknesses. The user receives a recommendation for improving the CV based on the analysis. Users can choose to download the improved version of the CV or make changes manually.
For career predictions, users input their academic details, including their GPA, extracurricular activities, internships, and leadership positions. The machine learning model uses this data to predict a suitable career path. The user is provided with a recommendation based on their profile.
The skills matching feature allows users to upload a resume in PDF format. The application extracts the key skills from the document and matches them with job opportunities using a machine learning model. The application provides career recommendations based on the matched skills.

Database Schema:
The database consists of two key tables. The User Table stores user-specific data such as the username, email, and password hash. It also maintains a relationship with the CV Table, which stores each uploaded CV along with the filename and timestamp of when it was created. The CV table has a foreign key reference to the user who uploaded it, allowing each user to have multiple CVs.

Security:
The application uses Flask sessions to manage user authentication. The login required decorator ensures that only authenticated users can access certain pages, such as the profile and job recommendations. The application also uses password hashing to securely store user passwords and prevent them from being stored in plain text. Flask’s session management ensures that users remain logged in during their session.

Challenges Encountered:
Throughout the development of this application, a few challenges arose. One of the primary challenges was handling different file formats (PDF, DOCX, TXT) for CV uploads. This required integrating multiple libraries to extract text from various document types. Another challenge was ensuring that the career prediction and skills matching models accurately return relevant predictions and recommendations. Fine-tuning the model and making sure it handled diverse user data was crucial.
We initially explored the idea of implementing job matching functionality, but unfortunately, we encountered challenges due to the unavailability of the LinkedIn API and the lack of suitable datasets for job matching. Additionally, we considered implementing a candidate recommendation system for companies; meaning that based on many candidate resume, the model will predict who is the best suited for a specific job. However, upon further evaluation, we determined that this feature was not aligned with the core objectives of our project since it targets more the market of companies and therefore, we decided not to pursue it.


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

Model 2: Machine Learning Model- Skills Matching
Objective of model 2: Predict the career domain that best fits a résumé’s skill list. It begins by loading *resumes.csv* from the *data/* folder, filling missing skill strings with empty text, converting everything to lowercase, and stripping non-alphanumeric characters so the corpus is noise-free. Career labels are then numerically encoded with `LabelEncoder`, after which a custom tokenizer (`skill_tokenizer`) splits comma-separated skills and a `TfidfVectorizer` with `ngram_range=(1, 2)` and `max_df = 0.9` turns the skills into sparse TF-IDF vectors that capture both single terms and two-word phrases such as “data analysis.” To reduce noise and over-fitting, `SelectKBest(chi2, k = 1000)` keeps the 1 000 features most correlated with the career labels. The refined vectors flow into a soft-voting ensemble that blends four complementary learners—Logistic Regression for a linear baseline, Random Forest for non-linear feature interactions, cosine-distance K-Nearest Neighbours for skill similarity, and an SVM that excels in high-dimensional TF-IDF space; each model outputs class probabilities, the ensemble averages them, and the class with the highest averaged score becomes the prediction, yielding smoother and more reliable results than any single learner. GridSearchCV, wrapped around the full pipeline, then explores 72 hyperparameter combinations (`select__k {500, 1000}`, `lr__C {0.1, 1, 10}`, `rf__max_depth {10, 20}`, and `knn__n_neighbors {3, 5}`) using `StratifiedKFold(n_splits = 5)` to preserve class balance in every fold; the best configuration—`k = 1000`, `C = 1`, `max_depth = 20`, `n_neighbors = 5`—delivers roughly 93 % accuracy. Finally, the tuned pipeline is saved as *career_classifier_advanced.pkl* and its label encoder as *label_encoder.pkl* via `joblib`, so the entire workflow can be reloaded in one line for real-time inference, making it a deployable, high-accuracy résumé classifier for Flask or FastAPI services.

Future Improvements:
We plan to revisit the job matching feature in the future once we have access to the necessary resources, such as a reliable job dataset or API. This feature will enhance the platform by providing users with more accurate and personalized job recommendations based on their skills and experience, further improving their career development journey.


Google drive link: https://drive.google.com/drive/folders/1gzwaywwUhl_27op2vBN_Sv2FWJ8j6R8N?usp=sharing

**Running the CV Maker Application**:
**Direct Method**
Prerequisites:

Python 3.8 or higher
Required Python packages (listed in requirements.txt)
Valid OpenAI API key
Downloaded model files from Google Drive

Steps:

1. Clone or download the repository
   
   git clone https://github.com/elodiefeghali/EECE-490-CV-maker
   cd cv-maker

3. Install dependencies
   
  pip install -r requirements.txt

5. Create necessary directories:
   
  mkdir -p static/temp static/cvs

7. Ensure model files are in the correct location
   
  Copy the model files from Google Drive to these locations:
  
  Models/Career Advisor/carrer_model.pkl
  Models/Career Advisor/career_label_encoder.pkl
  Models/Skills Matching/career_classifier_advanced.pkl
  Models/Skills Matching/label_encoder.pkl

5. Run the application
   
   python app.py

7. Access the application
   
  Open your browser and navigate to http://localhost:5001

We tried to dockerize the project but encountered many problems:
First the building process was so long we spent 3 days trying to solve the errors, then when the build was successful we encountered a module not found error (scipy.sparse._csr) and tried to install the compatible versions but that didn't work and gave us more errors during the rebuilding process. The error might be related to our system configuration or python version that's why we attached the dockerfile in case you want to make sure it's working. Thank you for your consideration and we apoligize for the late docker submission.

**Docker Setup (Alternative)**:
1. Build the Docker image
   
   docker build -t cv-maker-app .
3. Run the container 
   
   docker run -d -p 5001:5001 --name cv-maker cv-maker-app
5. Access the application
   
   Open your browser and navigate to http://localhost:5001
7. If you encounter SciPy issues
   
   If you see errors related to scipy.sparse._csr or other SciPy components, you may need to fix the dependencies inside the container:

   docker exec -it cv-maker
   
   pip uninstall -y scikit-learn scipy
   
   pip install scikit-learn==1.0.2 scipy==1.7.3
   
   exit
   
   docker restart cv-maker




