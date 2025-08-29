🩺 Diabetes Prediction using Machine Learning

📖 Project Overview

This project applies machine learning techniques to predict the likelihood of diabetes in patients based on health measurements such as glucose level, BMI, and age.

The dataset used is the Pima Indians Diabetes Dataset, available on Kaggle.

The goal is to compare different machine learning models and evaluate their performance in predicting diabetes, while also highlighting the importance of appropriate evaluation metrics for medical problems.

⸻

📂 Project Workflow
	1.	Data Exploration (EDA)
	•	Checked dataset distribution, missing values, and correlations.
	•	Visualized class imbalance (diabetic vs. non-diabetic cases).
	2.	Data Preprocessing
	•	Replaced invalid zero values (in columns like Blood Pressure, Insulin, Skin Thickness).
	•	Scaled features using StandardScaler (important for Logistic Regression).
	3.	Model Training
	•	Logistic Regression: Achieved ~75% accuracy.
	•	Random Forest Classifier: Achieved ~72% accuracy with default parameters.
	4.	Model Evaluation
	•	Compared performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
	•	Highlighted that in medical diagnosis, Recall (Sensitivity) is often more critical than raw accuracy.

⸻

🛠️ Technologies Used
	•	Python 3.9+
	•	Google Colab (development environment)
	•	Libraries:
	•	Pandas, NumPy (data handling)
	•	Matplotlib, Seaborn (visualization)
	•	Scikit-learn (ML models & evaluation)

🚀 How to Run
	1.	Clone the repository:
 git clone https://github.com/beetwin1/diabetes-prediction.git 
cd diabetes-prediction
  2.	Install dependencies:
pip install -r requirements.txt
  3.	Open the Jupyter notebook (or Colab):
jupyter notebook notebooks/diabetes_prediction.ipynb

📊 Results
	•	Logistic Regression: ~75% accuracy, strong baseline model.
	•	Random Forest Classifier: ~72% accuracy (default parameters).
	•	Key insight: Logistic Regression performed slightly better, but further tuning of Random Forest could improve results.

 🔮 Next Steps
	•	Hyperparameter tuning for Random Forest (GridSearchCV / RandomizedSearchCV).
	•	Test additional algorithms (XGBoost, LightGBM, SVM).
	•	Address class imbalance with SMOTE or class-weight adjustments.
	•	Build a simple web app (Streamlit/Flask) for predictions.

 📜 License

This project is released under the MIT License.



🙌 Acknowledgements
	•	Dataset: Kaggle – Pima Indians Diabetes Dataset
	•	Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
