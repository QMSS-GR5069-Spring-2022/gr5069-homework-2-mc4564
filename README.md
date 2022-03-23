# Data Skills 2: Homework 4 (ML)
## Voting classification
## Author: mc4564
__Due date: Sunday November 29th before midnight__


In this project, I use presidential annual state-level data employment data from [NAICS](https://www.naics.com/search/) and from the [Bureau of Economic Analysis](https://apps.bea.gov/iTable/iTable.cfm?reqid=70&step=1&isuri=1) and supervised machine learning to classify states as voting for the Republican candidate or the Democratic candidate for president.

I decided to use data from the 2008 election to predict the 2012 election. I did this by first cleaning up my election data (data/1976-2016-president.csv) to choose election winners by state in 2008 and 2012. Then I cleaned the farm data (data/farm.csv) to join in the dataset of per person farm income. I trained my supervised learning models on the 2008 winner data and tested my supervised learning models on the 2012 winner data.

My model uses the per capita farm income in each state to predict 2012 presidential election.
My model was not very successful in predicting the results. My model predicts the democrats to win a literal landslide with 49 states. It appears that no matter what level of farm in come you are on, democrats should win the state. My model has an accuracy score of 55%, by comparing '12party' and '12predict'. Having more variables to predict from will greatly increase the accuracy of my model.
