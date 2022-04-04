import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

datainput = pd.read_csv("drug.csv", delimiter=",")
X = datainput[['age', 'sex', 'bp', 'cholesterol', 'Na_to_K']].values

# Data Preprocessing
from sklearn import preprocessing

label_gender = preprocessing.LabelEncoder()
label_gender.fit(['F', 'M'])
X[:, 1] = label_gender.transform(X[:, 1])
# print(X[:, 1])

label_BP = preprocessing.LabelEncoder()
label_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = label_BP.transform(X[:, 2])
# print(X[:, 2])

label_Chol = preprocessing.LabelEncoder()
label_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = label_Chol.transform(X[:, 3])
# print(X[:, 3])

y = datainput["drug"]
# print(y)

# train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit(X_train, y_train)
predicted = drugTree.predict(X_test)

print(predicted)
print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predicted))

"""
Decision tree is an algorithm which is mainly applied to data classification scenarios. 
It is a tree structure where each node represents the features and each edge represents the decision taken. 
Starting from the root node we go on evaluating the features for classification and take a decision to follow 
a specific edge. Whenever a new data point comes in , this same method is applied again and again and then the 
final conclusion is taken when all the required features are studied or applied to the classification scenario. 
So Decision tree algorithm is a supervised learning model used in predicting a dependent variable with a series 
of training variables.
"""