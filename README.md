# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebooka

## Algorithm
1. import pandas module and import the required data set
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARESH.R
RegisterNumber:  212223240104
*/
import pandas as pd
df = pd.read_csv("/content/Employee.csv")
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# print(df['left'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# df['Departments'] = le.fit_transform(df['Departments'])
df['salary'] = le.fit_transform(df['salary'])
# print(df)

x = df.iloc[:,[0,1,2,3,4,5,7,9]]
# x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_acciden','promotion_last_5years','salary']]
print("X :",x)
y = df['left']
print("Y :",y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)
# print("X - Train : \n",x_train)
# print("X - Test : \n",x_test)
# print("Y - Train : \n",y_train)
# print("Y - Test : \n",y_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n",y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy :",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![Screenshot 2024-10-05 131618](https://github.com/user-attachments/assets/9d51103b-9d88-4715-abdd-2dfbcdd944ca)

![Screenshot 2024-10-05 131631](https://github.com/user-attachments/assets/e4144af6-4d67-44fb-bd7b-cb2ea70e167d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
