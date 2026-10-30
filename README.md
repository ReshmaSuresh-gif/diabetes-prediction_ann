# Diabetes Prediction using an Artificial Neural Network

Neural Network mini project. An ANN (scikit-learn `MLPClassifier`, hidden layers 64 and 32) predicts whether a
patient has diabetes from 8 clinical measurements. It is compared with Logistic Regression and Random Forest,
and a Streamlit app shows the predicted probability for a new patient.


## Dataset
Pima Indians Diabetes Database: 768 records, 8 features, binary target `Outcome`
(1 = diabetes). The CSV is in `data/raw/diabetes.csv` (copy from the public plotly/datasets repository).
Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible, so they are treated as missing
and filled with the median.

## Results 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| ANN (64, 32)| 0.721 | 0.617 | 0.537 | 0.574 | 0.818 |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.814 |



## Project structure

app.py                       Streamlit app
src/                         preprocessing, training, model loading, prediction, UI
data/raw/diabetes.csv        dataset
Diabetes_ANN_Simple.ipynb    simple step-by-step notebook
report/                      full project report (Word)
images/                      figures used in this README


