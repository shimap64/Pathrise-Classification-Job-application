Pathrise Machine Learning Project-Classification
Objective
This project aims to develop and evaluate machine learning models to predict whether a candidate is "placed" using a dataset with mixed numerical and categorical features. The workflow includes data preprocessing, feature scaling, model optimization, and performance evaluation through various metrics.
________________________________________
1. Data Preprocessing
Handling Missing Values
•	Missing categorical values were replaced with defaults based on domain knowledge.
•	Missing numerical values were filled using the mean or grouped means.
Feature Scaling
•	StandardScaler was applied to normalize numerical features, ensuring consistent scaling for better model performance.
________________________________________
2. Modeling
Classifiers
The following models were used:
1.	K-Nearest Neighbors (KNN)
2.	Support Vector Machine (SVM)
3.	Decision Tree (DT)
4.	Random Forest (RF)
5.	Logistic Regression (LR)
Hyperparameter Optimization
•	Performed GridSearchCV with cross-validation to identify the optimal parameters for each model.
•	The best hyperparameters were used for final model training.
________________________________________
3. Evaluation
Metrics
1.	Accuracy: Measures overall correctness of predictions.
2.	Confusion Matrices: Provide insights into model-specific true positives, true negatives, false positives, and false negatives.
3.	ROC-AUC: Analyzes the discriminative ability of models.
4. Results
4.1 Accuracy Scores
  
Model	Accuracy	Key Observations
KNN	55%	Balanced performance but struggles with recall.
SVM	66%	Improved balance between precision and recall.
DT	71%	Achieved good overall performance with a higher recall for class 1.
RF	73%	Best accuracy, strong recall for class 1 but lower precision for class 0.
LR	60%	Moderate accuracy with decent recall for class 0.
4.2 Confusion Matrices
•	Each confusion matrix shows how well the models classified each class:
o	Rows represent the actual classes.
o	Columns represent the predicted classes.
 
![image](https://github.com/user-attachments/assets/c481e32c-13f8-42d6-acb6-fed5eacbb6e8)

 ![image](https://github.com/user-attachments/assets/af234e50-62e0-492a-9237-2486f498618a)

 ![image](https://github.com/user-attachments/assets/7288a8a6-fb8a-4606-b31a-dad8db95eb24)

![image](https://github.com/user-attachments/assets/84516b1f-f2ae-43cb-a045-ef9f016dad4f)
![image](https://github.com/user-attachments/assets/fd266281-e541-4186-af79-3721c6d0ea63)

 
 


1.	KNN Confusion Matrix
o	Confusion matrix for KNN showing a balanced misclassification rate between classes 0 and 1.
2.	SVM Confusion Matrix
o	Confusion matrix for SVM illustrating improved balance in predicting class 0 and class 1.
3.	DT Confusion Matrix
o	Confusion matrix for Decision Tree showing high recall for class 1 and decent precision for class 0.
4.	RF Confusion Matrix
o	Caption: Confusion matrix for Random Forest with the highest recall for class 1 but some misclassification for class 0.
5.	LR Confusion Matrix
o	Caption: Confusion matrix for Logistic Regression showing moderate performance with an edge in class 0 recall.
________________________________________
4.3 ROC Curves
•	ROC curves provide a visual representation of the trade-off between sensitivity (True Positive Rate) and specificity (False Positive Rate).
•	AUC values demonstrate the overall discriminative power of each model.
![image](https://github.com/user-attachments/assets/23554a77-e31f-4666-a76c-4fbc7048e0cd)

 
•	ROC curves for KNN, SVM, DT, RF, and LR, with AUC scores highlighting the superior performance of Random Forest and Decision Tree classifiers.
________________________________________
5. Saving Models
The final trained models were saved as pickle files for deployment:
•	pathrise-knn.pkl
•	pathrise-svm.pkl
•	pathrise-dt.pkl
•	pathrise-rf.pkl
•	pathrise-lr.pkl
