import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
#CART graphing
#from subprocess import call
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

#set seed
random.seed(54)
#import pydot
#pydot.find_graphviz("e:/python/lib/site-packages/graphviz")

submission = pd.read_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/gender_submission.csv")
test = pd.read_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/test.csv")
train = pd.read_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/train.csv")

log_reg = LogisticRegression()


train.groupby('Parch')['Age'].nunique()

X = train.drop(['Survived', 'Name', 'PassengerId'], axis=1)
y = train['Survived']

X = train.drop(['Survived', 'Name'], axis=1)
y = train['Survived']

# Prep data for xgboost
# dtrain = xgb.DMatrix(data=X_train, label=y_train)
# dtest = xgb.DMatrix(data=X_test, label=y_test)


# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.columns.values)
# Feature processing
num_vars = ['Age', 'Fare']
num_steps = [('num_impute', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())]
num_transform = Pipeline(steps=num_steps)

cat_vars = ['Pclass', 'Sex', 'Embarked']
cat_steps = [('cat_impute', SimpleImputer(strategy='most_frequent')),
             ("ohe", OneHotEncoder())]
cat_transform = Pipeline(steps=cat_steps)

preprocs = ColumnTransformer(
                            transformers=[
                                ('num', num_transform, num_vars),
                                ('cat', cat_transform, cat_vars)
                            ])


#pipeline steps set up
final_steps_lr = [
               ('preprocess', preprocs),
               ('log_reg', LogisticRegression(solver='lbfgs'))
              ]

final_steps_cart = [
               ('preprocess', preprocs),
               ('cart', DecisionTreeClassifier())
]

final_steps_cart2 = [
               ('preprocess', preprocs),
               ('cart', DecisionTreeClassifier(max_depth=3))
]

final_steps_rf = [
               ('preprocess', preprocs),
               ('rf', RandomForestClassifier())
                ]

final_steps_xgb = [
               ('preprocess', preprocs),
               ('xgb', xgb.XGBClassifier(objective="binary:logistic"))
                ]

#CART
cart_pipe = Pipeline(final_steps_cart)
cart_pipe2 = Pipeline(final_steps_cart2)
cart_grid = {'cart__max_depth':  [int(x) for x in np.linspace(start=3, stop=10, num=7, endpoint=True)],
             'cart__random_state': [21]
             }

cart_cv = GridSearchCV(estimator=cart_pipe, param_grid=cart_grid, cv=5, n_jobs=-1)
cart_cv.fit(X_train, y_train)
cart_cv.predict(X_test)

print("CV cart:", cart_cv.score(X_test, y_test))

print("Best CART params", cart_cv.best_params_)

best_cart = cart_cv.best_estimator_
best_cart.fit(X_train, y_train)

print("Best_Cart:", best_cart.score(X_test, y_test))


cart_pipe2.fit(X_train, y_train)

print("CART manual fit:", cart_pipe2.score(X_test, y_test))

cart_pred = best_cart.predict(X_test)

#RandomForest
rf_pipe = Pipeline(final_steps_rf)
#need to pre-append the model with the model name set in pipline
rf_grid = {'rf__n_estimators': [int(x) for x in np.linspace(start=10, stop=50, num=10)],
           'rf__max_depth': [int(x) for x in np.linspace(start=2, stop=7, num=5, endpoint=True)],
           'rf__random_state': [21]}
#grid search
rf_cv = GridSearchCV(estimator=rf_pipe, param_grid=rf_grid, cv=5, n_jobs=-1)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_
best_params = rf_cv.best_params_
best_score = rf_cv.best_score_

print(best_rf, best_score)
print(best_params)
print(rf_cv.score(X_test, y_test))

#rf_pred = rf_cv.predict(X_test)
#rf_cv.named_steps['rf'].feature_importances_


# Logistic Regression
pipeline_lr = Pipeline(steps=final_steps_lr)
model_1 = pipeline_lr.fit(X_train, y_train)
LR_pred = pipeline_lr.predict(X_test)
print(classification_report(y_test, LR_pred))

# XGBOOST
pipeline_xgb = Pipeline(steps=final_steps_xgb)
pipeline_xgb.fit(X_train, y_train)
xgb_pred = pipeline_xgb.predict(X_test)

#comparisons

print("Logistic Regression: ", model_1.score(X_test, y_test))
print("CART: ", best_cart.score(X_test, y_test))
print("RandomForest: ", best_rf.score(X_test, y_test))
print("XGBoost: ", pipeline_xgb.score(X_test, y_test))

ensembled = VotingClassifier(estimators=[
                            ('lr', model_1),
                            ('CART', best_cart),
                            ('xgb', pipeline_xgb),
                            ('rf', best_rf)
                            ],
                            voting='hard'
)

ensembled.fit(X_train, y_train)
print("Ensembled: ", ensembled.score(X_test, y_test))

#Ensemble method2

X_test2 = X_test

#X_test2.join(pd.DataFrame(best_rf.predict(X_test)))
#X_test2['RF_Predict'] = pd.concat(X_test, pd.DataFrame(best_rf.predict(X_test)))
X_test2['RF_Predict'] = best_rf.predict(X_test)
X_test2['LR_Predict'] = model_1.predict(X_test)
X_test2['CART_Predict'] = best_cart.predict(X_test)
X_test2['XGB_Predict'] = pipeline_xgb.predict(X_test)


final_steps_cart3 = [
               ('preprocess', preprocs),
               ('cart', DecisionTreeClassifier())
]

#CART ensemble
cart_pipe3 = Pipeline(final_steps_cart3)
cart_grid = {'cart__max_depth':  [int(x) for x in np.linspace(start=2, stop=10, num=8, endpoint=True)],
             'cart__random_state': [21]
             }

cart_ensemble_cv = GridSearchCV(estimator=cart_pipe3, param_grid=cart_grid, cv=5, n_jobs=-1)
cart_ensemble_cv.fit(X_test, y_test)

print(cart_ensemble_cv.best_score_)

#cart_ensemble_cv.score(X_train,y_train)

#Cleaning and export
ensemble_answer = ensembled.predict(test)
cart_ensemble_answer = cart_ensemble_cv.predict(test)
logistic_answer = model_1.predict(test)
xgb_answer = pipeline_xgb.predict(test)
cart_answer = best_cart.predict(test)
rf_answer = best_rf.predict(test)


ensemble_answer2 = np.column_stack((test['PassengerId'], ensemble_answer))
cart_ensemble_answer2 = np.column_stack((test['PassengerId'], cart_ensemble_answer))
logistic_answer2 = np.column_stack((test['PassengerId'], logistic_answer))
xgb_answer2 = np.column_stack((test['PassengerId'], xgb_answer))
cart_answer2 = np.column_stack((test['PassengerId'], cart_answer))
rf_answer2 = np.column_stack((test['PassengerId'], rf_answer))


ensemble_answer3 = pd.DataFrame({'PassengerId': ensemble_answer2[:, 0], 'Survived': ensemble_answer2[:, 1]})
cart_ensemble_answer3 = pd.DataFrame({'PassengerId': cart_ensemble_answer2[:, 0], 'Survived': cart_ensemble_answer2[:, 1]})
logistic_answer3 = pd.DataFrame({'PassengerId': logistic_answer2[:, 0], 'Survived': logistic_answer2[:, 1]})
xgb_answer3 = pd.DataFrame({'PassengerId': xgb_answer2[:, 0], 'Survived': xgb_answer2[:, 1]})
cart_answer3 = pd.DataFrame({'PassengerId': cart_answer2[:, 0], 'Survived': cart_answer2[:, 1]})
rf_answer3 = pd.DataFrame({'PassengerId': rf_answer2[:, 0], 'Survived': rf_answer2[:, 1]})

pd.DataFrame(ensemble_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/ensemble_pred.csv", index=False)
pd.DataFrame(cart_ensemble_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/cart ensemble_pred.csv", index=False)
pd.DataFrame(logistic_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/logistic_pred.csv", index=False)
pd.DataFrame(xgb_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/xgb_pred.csv", index=False)
pd.DataFrame(cart_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/cart.csv", index=False)
pd.DataFrame(rf_answer3).to_csv("C:/Users/theGameTrader/Documents/Python Scripts/Titanic Kaggle/predictions/rf.csv", index=False)

# WIP

# CART graphs
dot_data = StringIO()

export_graphviz(cart_pipe2.named_steps['cart'],
                out_file=dot_data,
                feature_names=X_train.columns.values,
                class_names=y_train.name,
                rounded=True,
                proportion=False,
                precision=2,
                filled=True)

tree_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(tree_graph.create_png())

#convert dot file to png via command line package
#call(['dot'], '-Tpng', 'titanic_cart.dot', '-o', 'titanic_cart.png', '-Gdpi=600')