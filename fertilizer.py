import pandas as pd
df = pd.read_csv('Crop_recommendation.csv')

# print(df.head(6))

# print(df.isnull().sum())

# print(df.info())

df['label'] = df['label'].astype('object')

# print(df.info())

# print(df['label'].value_counts())

# print(df.tail())

x = df.drop('label', axis=1)
y = df['label']

# print(x.info())
# print("-------------------------")
# print(y.info())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, test_size=0.2)

# print("Test---------------------")
# print(x_test.info())
# print("Train--------------------")
# print(x_train.info())

# ------------logestic regration, decission tree, random regration------------

# --------------- Logestic regration
"""
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)

from sklearn.metrics import accuracy_score
logistic_reg_acc = accuracy_score(y_test, y_pred1)
print("Logistic Accuracy is: "+str(logistic_reg_acc))
# Logistic Accuracy is: 0.9704545454545455

"""

# ---------------- Decision Tree
"""
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

from sklearn.metrics import accuracy_score
decision_acc = accuracy_score(y_test, y_pred2)
print("Decision Tree Accuracy is: "+str(decision_acc))
# Decision Tree Accuracy is: 0.9954545454545455
"""

# ---------------- Random fores

'''
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier()
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)

from sklearn.metrics import accuracy_score
random_acc = accuracy_score(y_test, y_pred3)
print("Random Forest Accuracy is: "+str(random_acc))
# Random Forest Accuracy is: 0.9977272727272727
'''
# --------- Making Model -----------
import joblib
# filename = 'crop_predict_model'
# joblib.dump(model3, 'crop_predict_model')

app = joblib.load('crop_predict_model')
arr = [[90,42,43,20.879744,82.002744,6.502985,202.935536]]
my_predict = app.predict(arr)
print("Predict: ",my_predict)