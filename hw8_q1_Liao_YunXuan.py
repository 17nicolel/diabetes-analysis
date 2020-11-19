# YunXuan Liao
# ITP 449 Fall 2020 # HW8
# Q1

#1. Create a DataFrame “diabetes_knn” to store the diabetes data and set option to display all columns
# without any restrictions on the number of columns displayed.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.max_columns',None)
diabetes_knn=pd.read_csv('diabetes(1).csv')
print(diabetes_knn)

#2. Determine the dimensions of the “diabetes_knn” dataframe.
outcome=diabetes_knn['Outcome']
pregnancies=diabetes_knn['Pregnancies']
glucose=diabetes_knn['Glucose']
bloodpressure=diabetes_knn['BloodPressure']
skinthickness=diabetes_knn['SkinThickness']
insulin=diabetes_knn['Insulin']
bmi=diabetes_knn['BMI']
diabetespedigreefunction=diabetes_knn['DiabetesPedigreeFunction']
age=diabetes_knn['Age']

print(diabetes_knn.groupby(outcome).mean())

plt.figure(1)
sb.countplot(x=outcome,hue=pregnancies,data=diabetes_knn)
plt.figure(2)
sb.countplot(x=outcome,hue=glucose,data=diabetes_knn)
plt.figure(3)
sb.countplot(x=outcome,hue=bloodpressure,data=diabetes_knn)
plt.figure(4)
sb.countplot(x=outcome,hue=skinthickness,data=diabetes_knn)
plt.figure(5)
sb.countplot(x=outcome,hue=insulin,data=diabetes_knn)
plt.figure(6)
sb.countplot(x=outcome,hue=bmi,data=diabetes_knn)
plt.figure(7)
sb.countplot(x=outcome,hue=diabetespedigreefunction,data=diabetes_knn)
plt.figure(8)
sb.countplot(x=outcome,hue=age,data=diabetes_knn)
plt.show()



#3. Update the DataFrame to account for missing values.
print(diabetes_knn.isnull().sum())


#4. Create the Feature Matrix and Target Vector.
X=diabetes_knn.iloc[:,0:8]
y=diabetes_knn.iloc[:,8]

#5. Standardize the attributes of Feature Matrix
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
print(X.head())

#6. Split the Feature Matrix and Target Vector into training and testing sets, reserving 30%
#of the data for testing. random_state = 2020, stratify = y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=2020,stratify=y)

#Develop a KNN based model and obtain KNN score (accuracy) for train and test data for
#k’s values ranging between 1 to 15.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

neighbors=range(1,16)
accuracyTrain=[]
accuracyTest=[]

for k in neighbors:
    KNN=KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,y_train)
    accuracy1=KNN.score(X_train,y_train)
    accuracyTrain.append(accuracy1)
    accuracy2=KNN.score(X_test,y_test)
    accuracyTest.append(accuracy2)

#8. Plot a graph of train and test score and determine the best value of k
plt.plot(neighbors,accuracyTrain,label='Training Accuracy')
plt.plot(neighbors,accuracyTest,label='Testing Accuracy')
plt.xticks(neighbors)
plt.legend()
plt.title('KNN: Varying Number of Neighbors')
plt.xlabel('k= Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#choose 5 for best value of k

#9. Display the test score of the model with best value of k and print the confusion matrix for it.
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
print(confusionmatrix)
plot_confusion_matrix(model, X_test, y_test)
plt.show()

#10. Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22
#skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age.
test = [[2,150,85,22,200,30,0.3,55]]
result = model.predict(test)
print(result)
#the outcome is 1.
