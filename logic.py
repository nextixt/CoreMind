"""This is 1st vrsion of logic engine of CoreMind, this version has small dataset.
    It trains on dataset and predict value 'y' which means 'need jacket' or not"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
features = ['Snowing', 'Cold']
X = df[features]
y = df.NeedJacket
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier(random_state = 1, max_depth = 3)
model.fit(X_train, y_train)
err = mean_absolute_error(y_test, model.predict(X_test))
print(f'Model error: {err}')
pred = pd.DataFrame([[0, 1]], columns=features)
print(model.predict(pred)) 
