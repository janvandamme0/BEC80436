# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:40:59 2024

@author: jdamme
"""
# import relevant packages
import pandas as pd
import statsmodels.api as sm
import math
import os
import matplotlib.pyplot as plt

#make connection to datasource
path="C:/Users/jdamme/Desktop/thesis1/data/python/final_script.py"
current_dir = os.path.dirname(path)
estimation_window_csv_path = os.path.join(current_dir, "data_policyestimationwindow.csv")
event_window_csv_path = os.path.join(current_dir, "data_policyeventwindow.csv")
#import relevant data
estdata = pd.read_csv(estimation_window_csv_path, delimiter=';', decimal=',')
eventdata = pd.read_csv(event_window_csv_path, delimiter=';', decimal=',')

#%% regression
#dataformating
date_format='%d.%m.%Y'
estdata['Date']=pd.to_datetime(estdata['Date'], format=date_format, dayfirst=True)
eventdata['Date']=pd.to_datetime(eventdata['Date'],format=date_format, dayfirst=True)

#calculate daily returns
eventdata['Rmt']=eventdata['BCI'].pct_change()

#results dataframe
all_results=[]
#start loop
for column in estdata.columns[1:-1]:
    
    #remove missing dates
    valid_indices=estdata[column].notnull()
    estdata['Rmt'] = estdata['BCI'][valid_indices].pct_change()
    valid_values=estdata[column][valid_indices]
    valid_returns=valid_values.pct_change().iloc[1:]
    valid_bci=estdata['Rmt'][valid_indices].iloc[1:]
#regression
    X1 = sm.add_constant(valid_bci)
    model = sm.OLS(valid_returns, X1)
    results=model.fit()
    print(results.summary())

    evalue= sm.add_constant(eventdata[['Rmt']])
    pred_values=results.predict(evalue)

# abnormal returns
    returns=pd.DataFrame()
    returns['Date']=eventdata['Date']
    returns[f'{column} AcReturn']= eventdata[column].pct_change()
    returns[f'{column} ExReturn']= pred_values
    returns[f'{column} AbReturn']= returns[f'{column} AcReturn']-returns[f'{column} ExReturn']
    returns[f'{column} Error']= eventdata[column].pct_change() - pred_values
    all_results.append(returns)

#make and export final document (for possible manual validation)
all_results_df = pd.concat(all_results, axis=1)
output_file_path = os.path.join(current_dir, 'output2.csv')
all_results_df.to_csv(output_file_path, index=False)
# end loop here

#scatterplot for returns of last location, not neccessary
plt.scatter(valid_bci, valid_returns, s=1)
plt.plot(valid_bci, results.params[1]*valid_bci+results.params[0], color='black')
plt.xlabel('Marketreturn')
plt.ylabel('return for US_PNW')
#%% BREAK

#select relevant data for simplified dataframe, also excluding BCI
date_column = all_results_df.iloc[:,0]
datecolumnonce= pd.DataFrame(date_column)
abreturn_columns = [col for col in all_results_df.columns if 'AbReturn' in col] #if date included, here [:-1]
Error_columns = [col for col in all_results_df.columns if 'Error' in col]

#create dataframe
date_abreturn_df = pd.concat([datecolumnonce] + [all_results_df[abreturn_columns + Error_columns]], axis=1)
date_abreturn_df.set_index('Date', inplace=True)

#%% from here on work per event window

#cumulate, define event window dates
date_abreturn_df.index = pd.to_datetime(date_abreturn_df.index)
start_date = '2022.05.25'
end_date = '2022.05.31'
event_date = '2022.05.25'
sel_range=date_abreturn_df.loc[start_date:end_date]
CAR=sel_range[abreturn_columns].sum()
print('Cumulative AR:',CAR)
#determine N
num_loc=len(CAR)
num_dat=len(sel_range)
print('Number of locations:', num_loc)
print('Number of time observations:', num_dat)

#robustness tests
#variance average cumulative abnormal return
vari_df=date_abreturn_df.loc[start_date:end_date]
var_pl=vari_df[Error_columns].var(axis=0) * num_dat
#average cumulative abnormal return
CAAR=CAR.sum()/num_loc
var_caar=var_pl.sum()/(num_loc**2)
#test value for first robustness test
O1=CAAR/math.sqrt(var_caar)
print('Robustness test1:',O1)

#second robustness test
from scipy.stats import rankdata
from scipy.stats import norm
import numpy as np
#rank returns for each location in the specified event window
rankk_it=pd.concat([sel_range[abreturn_columns]], axis=1)
rankk_it=rankk_it.apply(rankdata)
#export ranks for possible manual validation
output_file_path = os.path.join(current_dir, 'rank2.csv')
rankk_it.to_csv(output_file_path, index=False)
#total
rankk_it_value=rankk_it-((len(rankk_it)+1)/2)
rankk_it_total=rankk_it_value.sum(axis=1)
value1=(rankk_it_total/num_loc)**2
sK= np.sqrt((sum(value1)/(len(rankk_it))))
#test value for second robustness test
O2=(1/num_loc)*sum(rankk_it.loc[event_date]-(len(rankk_it)+1)/2)/sK
print('Robustness test2:',O2)
#determine p-values
p_value1=2*(1-norm.cdf(abs(O1)))
p_value2=2*(1-norm.cdf(abs(O2)))
print('p-value test1:',p_value1)
print('p-value test2:',p_value2)