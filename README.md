# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1 Start
 
 2 Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.
 
 3 Split the data into training and test sets using train_test_split.
 
 4 Create and fit a logistic regression model to the training data.
 
 5 Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.
 
 6 Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

## Program:

## Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Developed by: P PARTHIBAN
## RegisterNumber:  212223230145
```python
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## Output:

<img width="845" alt="image" src="https://github.com/user-attachments/assets/4933a87c-36fb-4b8d-83c9-203c0aa6f0d4">

<img width="692" alt="image" src="https://github.com/user-attachments/assets/c218e631-0495-456c-baf5-137fb6739991">

<img width="646" alt="image" src="https://github.com/user-attachments/assets/6c4107ff-d963-484d-8416-aa1484ce6bbd">


<img width="218" alt="image" src="https://github.com/user-attachments/assets/0a02c7ac-0b51-4c9f-9d40-49d034cba75a">


<img width="630" alt="image" src="https://github.com/user-attachments/assets/37dc26a0-bace-45c8-b286-1b8656937418">


<img width="161" alt="image" src="https://github.com/user-attachments/assets/67383bc2-ce42-42e1-8b2e-d51f2720450f">

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
