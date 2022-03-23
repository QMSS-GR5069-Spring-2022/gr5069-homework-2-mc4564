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

    # clean election data to choose winners
    ele = ele[(ele['year'] == 2008) | (ele['year'] == 2012)]
    ele = ele[['year', 'state', 'party', 'candidatevotes']]

    winners =
    ele.sort_values('candidatevotes', ascending = False).
        groupby(['year', 'state']).first()
#citation:
# https://stackoverflow.com/questions/30486417/pandas-how-do-i-select-first-row-in-each-group-by-group#:~:text=The%20pandas%20groupby%20function%20could%20be%20used%20for,key%2C%20you%20should%20pass%20as%20the%20subset%3D%20variable
    winners = winners.iloc[:,0]
    winners = winners.reset_index()
    winners['party'] =
        winners['party'].str.replace('democratic-farmer-labor', 'democrat')

    #two columns with 08 winners and 12 winners
    winners08 = winners[winners['year'] == 2008]
    winners12 = winners[winners['year'] == 2012]
    winners08 = winners08.rename(columns = {'party': '08party'})
    winners12 = winners12.rename(columns = {'party': '12party'})

    winners08 =
        winners08.merge(winners12, left_on = 'state', right_on = 'state', how = 'outer')

    winners08 = winners08.drop(['year_x'], axis = 1)
    winners08 = winners08.drop(['year_y'], axis = 1)

    return winners08

ele = clean_election(ele)



def clean_farm(farm):

    #farm is a dataset of per person farm income
    farm = farm.drop(['GeoFips'], axis = 1)
    farm['GeoName'] = farm['GeoName'].str.replace(' *', '')
    farm['GeoName'] = farm['GeoName'].str.replace('*', '')
#citation:
# https://stackoverflow.com/questions/28986489/how-to-replace-text-in-a-column-of-a-pandas-dataframe
    farm = farm.rename(columns = {'GeoName': 'state'})
#citation:
# https://www.geeksforgeeks.org/python-change-column-names-and-row-indexes-in-pandas-dataframe/
    farm = farm.dropna()

    return farm

farm = clean_farm(farm)



def merge_data(ele, farm):

    #merge election and farm
    ele = ele.merge(farm, left_on = 'state', right_on = 'state', how = 'outer')
    df = ele.dropna()

    return df

df = merge_data(ele, farm)

# export to
df.to_csv(r'data\processed_data\merged_data.csv')
