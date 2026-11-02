# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sushil Shiva R
RegisterNumber:212224250017
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Salary.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
# Accuracy
print("\nAccuracy:", accuracy_score(y, y_pred)*100)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()
```

## Output:


<img width="1320" height="755" alt="Screenshot 2026-11-02 111857" src="https://github.com/user-attachments/assets/e68be858-f686-4b90-8d4f-fb9a7437a6f6" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
