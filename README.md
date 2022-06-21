<a><img alt='python' src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
<a><img alt='numpy' src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"></a>
<a><img alt='pandas' src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"></a>
<a><img alt='sk-learn' src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"></a>

# Heart-Attack-Disease Predictor
**Description** : Trained with patient datasets based on the data dictionary to predict heart attack

**ML Model** : Pipeline of 3 models which are logistic regression,decision tree and random forest

**Preprocessing step** : Train split data,gridsearchcv fine tuner

**Objective** : To perform EDA and predict if a person is prone to a heart attack or not
                To achieve model with validation accuracy of more than 70%

**Data Dictionary:**

age - Age of the patient

sex - Sex of the patient

cp - Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
trtbps - Resting blood pressure (in mm Hg)

chol - Cholestoral in mg/dl fetched via BMI sensor

fbs - (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False

restecg - Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy

thalachh - Maximum heart rate achieved

oldpeak - Previous peak

slp - Slope

caa - Number of major vessels

thall - Thalium Stress Test result ~ (0,3)

exng - Exercise induced angina ~ 1 = Yes, 0 = No

output - Target variable

### Exploratory Data Analysis (EDA)
1) Data Loading
2) Data Inspection
3) Data Cleaning
4) Features Selection
5) Pre-Processing


**Discussion** :

 🟠There are no NaN values in the data
 
 🟠There are certain outliers in all the continuous features from the boxplot graph
 
 🟠Duplicated data only 1 and drop method use to remove it
 
 🟠Train split test is used to train and split the xtrain,xtest,ytrain,ytest
 
 🟠For features selection, used logistic and cramers v, to select the highest accuracy respective to the output
 
 🟠Features select are: age,thalachh,oldpeak and thall
 
 🟠ML development used logistic classifier, decision tree, and random forest
 
 🟠MinMaxScaler+RF Classifier has the best accuracy model for training and deployment which is 73%, fine tuner used: 76%
 
 🟠Streamlit is used for model deployment, to demonstrate prediction of heart attack: Heart.app.py


**Dataset** :

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

[Datasets](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)


**Enjoy Coding!** 🚀















