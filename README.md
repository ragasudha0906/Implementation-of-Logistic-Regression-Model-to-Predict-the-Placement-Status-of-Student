# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the placement dataset and remove unnecessary columns such as serial number and salary.
2. Convert all categorical attributes into numerical values using label encoding. 
3. Split the dataset into training and testing sets using train–test split.
4. Train a Logistic Regression model using the training data.
5. Predict placement status and evaluate the model using accuracy and confusion matrix.
6. 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAGASUDHA R
RegisterNumber:  212224230215
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")
print(data.head())

data1 = data.copy()

data1.drop(['sl_no', 'salary'], axis=1, inplace=True)

print("\nMissing values:\n", data1.isnull().sum())
print("\nDuplicate values:", data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])

x = data1.iloc[:, :-1]
y = data1['status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print("\nAccuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", confusion)

from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn import metrics

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Not Placed', 'Placed']
)

cm_display.plot()
plt.show()
```

## Output:

<img width="1108" height="893" alt="Screenshot 2026-01-29 153845" src="https://github.com/user-attachments/assets/0dcd32db-88f1-4a2c-a5f6-238aefb72c97" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
