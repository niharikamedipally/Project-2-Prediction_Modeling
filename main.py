import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)

print(df.isnull().sum())

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print(df.head())

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

LR = LogisticRegression(max_iter=1000)

LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print("Predictions Completed!")

accuracy = accuracy_score(y_test, y_pred)
print("LR Model Accuracy:", accuracy)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

cm1 = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm1)

DT = DecisionTreeClassifier(random_state=42)

DT.fit(X_train, y_train)
y_pred1 = DT.predict(X_test)
print("DT Predictions Completed!")

accuracy = accuracy_score(y_test, y_pred1)
print("Model Accuracy:", accuracy)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

cm2 = confusion_matrix(y_test, y_pred1)
print("Confusion Matrix:",cm2)

RF = RandomForestClassifier(n_estimators=100, random_state=42)

RF.fit(X_train, y_train)
y_pred2 = RF.predict(X_test)
print("RF Predictions Completed!")

accuracy = accuracy_score(y_test, y_pred2)
print("Model Accuracy:", accuracy)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

cm3 = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:",cm3)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.figure(figsize=(6,4))
sns.heatmap(cm3, annot=True, fmt="d", cmap="Blues")
plt.show()