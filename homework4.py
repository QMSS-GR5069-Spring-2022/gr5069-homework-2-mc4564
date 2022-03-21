import os
import numpy as np
import pandas as pd
import seaborn as sns

def get_csv_data(filename, x = 4):
    
    folderpath = r'' #file path deleted
    path = os.path.join(folderpath, filename)
    df = pd.read_csv(path, skiprows = x)
    return df

farm = get_csv_data('farm.csv')
gdp  = get_csv_data('gdp.csv')
ele  = get_csv_data('1976-2016-president.csv', x = 0)



def clean_election(ele):
    
    #clean election data to choose winners
    ele = ele[(ele['year'] == 2008) | (ele['year'] == 2012)]
    ele = ele[['year', 'state', 'party', 'candidatevotes']]
    
    winners = ele.sort_values('candidatevotes', ascending = False).groupby(['year', 'state']).first()
    #citation: https://stackoverflow.com/questions/30486417/pandas-how-do-i-select-first-row-in-each-group-by-group#:~:text=The%20pandas%20groupby%20function%20could%20be%20used%20for,key%2C%20you%20should%20pass%20as%20the%20subset%3D%20variable
    winners = winners.iloc[:,0]
    winners = winners.reset_index()
    winners['party'] = winners['party'].str.replace('democratic-farmer-labor', 'democrat')
    
    #two columns with 08 winners and 12 winners
    winners08 = winners[winners['year'] == 2008]
    winners12 = winners[winners['year'] == 2012]
    winners08 = winners08.rename(columns = {'party': '08party'})
    winners12 = winners12.rename(columns = {'party': '12party'})
    
    winners08 = winners08.merge(winners12, left_on = 'state', right_on = 'state', how = 'outer')
    
    winners08 = winners08.drop(['year_x'], axis = 1)
    winners08 = winners08.drop(['year_y'], axis = 1)

    return winners08

ele = clean_election(ele)



def clean_farm(farm):
    
    #farm is a dataset of per person farm income
    farm = farm.drop(['GeoFips'], axis = 1)
    farm['GeoName'] = farm['GeoName'].str.replace(' *', '')
    farm['GeoName'] = farm['GeoName'].str.replace('*', '')
    #citation: https://stackoverflow.com/questions/28986489/how-to-replace-text-in-a-column-of-a-pandas-dataframe
    farm = farm.rename(columns = {'GeoName': 'state'})
    #citation: https://www.geeksforgeeks.org/python-change-column-names-and-row-indexes-in-pandas-dataframe/
    farm = farm.dropna()
    
    return farm

farm = clean_farm(farm)



def merge_data(ele, farm):
    
    #merge election and farm
    ele = ele.merge(farm, left_on = 'state', right_on = 'state', how = 'outer')
    df = ele.dropna()

    return df

df = merge_data(ele, farm)


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
print('SVC appears to be the best model, with the smallest standard deviation.')


def SVC_test(X_train, Y_train, X_test):
    
    #using SVC model
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predict = model.predict(X_test)
    
    return predict

predict = SVC_test(X_train, Y_train, X_test)
df['12predict'] = predict
print('The predicted winners presented in the original dataframe as "12predict":\n')
print(df[['state', '08party', '12party', '12predict']])



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('The accuracy score is:\n', accuracy_score(Y_test, predict))
print('The confusion matrix is:\n', confusion_matrix(Y_test, predict))
print('The classification report is:\n', classification_report(Y_test, predict))

#My model uses the per capita farm income in each state to predict 2012 presidential election.
#My model was not very successful in predicting the results. My model predicts the democrats to win a literal landslide with 49 states.
#I am not entirely sure about how to understand the predictions.
#It appears that no matter what level of farm in come you are on, democrats should win the state.
#My model has an accuracy score of 55%, by comparing '12party' and '12predict'.
#Having more varialbes to predict from will greatly increase the accuracy of my model. 
#I used the get_csv function from previous codes to help get csv file. 