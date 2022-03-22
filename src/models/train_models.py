import os
import numpy as np
import pandas as pd
import seaborn as sns

# import merged_data as df
def get_csv_data(filename, x = 4):

    folderpath = r'' #file path deleted
    path = os.path.join(folderpath, filename)
    df = pd.read_csv(path, skiprows = x)
    return df

df = get_csv_data('merged_data.csv')

#X_train has 08farm number
X_train = df.drop(['state', '08party', '12party', '2012'], axis= 1)
Y_train = df['08party']

#X_test has 12farm number
X_test = df.drop(['state', '08party', '12party', '2008'], axis = 1)
Y_test = df['12party']



from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

def test_model(models, X_train, Y_train):

    #using Professor Levy's codes...
    results = []

    for name, model in models:

        kf = StratifiedKFold(n_splits=10)
        res = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
        res_mean = round(res.mean(), 4)
        res_std  = round(res.std(), 4)
        results.append((name, res_mean, res_std))

    print('The three models:')
    for line in results:
        print(line[0].ljust(10), str(line[1]).ljust(6), str(line[2]))

models = [('Dec Tree', DecisionTreeClassifier()),
          ('Lin Disc', LinearDiscriminantAnalysis()),
          ('SVC', SVC(gamma='auto'))]

test_model(models, X_train, Y_train)
print('SVC appears to be the best model with the smallest standard deviation.')


def SVC_test(X_train, Y_train, X_test):

    #using SVC model
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predict = model.predict(X_test)

    return predict

predict = SVC_test(X_train, Y_train, X_test)
predict = df['12predict']
print('The predicted winners presented in the original dataframe as "12predict":\n')
print(df[['state', '08party', '12party', '12predict']])



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('The accuracy score is:\n', accuracy_score(Y_test, predict))
print('The confusion matrix is:\n', confusion_matrix(Y_test, predict))
print('The classification report is:\n', classification_report(Y_test, predict))

#My model uses the per capita farm income in each state to predict 2012
# presidential election. My model was not very successful in predicting the
# results. My model predicts the democrats to win a literal landslide with
# 49 states. I am not entirely sure about how to understand the predictions.
#It appears that no matter what level of farm in come you are on, democrats
# should win the state. My model has an accuracy score of 55%, by comparing
# '12party' and '12predict'. Having more variables to predict from will greatly
# increase the accuracy of my model. I used the get_csv function from previous
# code to help get csv file.
