#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 01:19:31 2021

@author: yumingchang
"""


import numpy as np
import pandas as pd

df = pd.read_csv('/Users/yumingchang/Documents/python advance_ML/lending club/loan_data_2007_2014.csv',low_memory=False)
df_c = df.copy()
pd.options.display.max_columns = None

#print(df_c['emp_length'])

#### prepressess

df_c['emp_length'] = df_c['emp_length'].str.replace('\+ years', '')
df_c['emp_length'] = df_c['emp_length'].str.replace('< 1 years', str(0))
df_c['emp_length'] = df_c['emp_length'].str.replace('n/a', str(0))
df_c['emp_length'] = df_c['emp_length'].str.replace(' years', '')
df_c['emp_length'] = df_c['emp_length'].str.replace(' year', '')


#print(df_c['term'].unique())
df_c['term_int'] = df_c['term'].str.replace(' months', '').astype('int')

df_c['earliest_cr_line'] = pd.to_datetime(df_c['earliest_cr_line'], format = '%b-%y')

#timedelta = 除以m代表,用m表示 ,delta代表時間單位 可用來加減時間
df_c['m_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01')-
    df_c['earliest_cr_line'])/ np.timedelta64(1, 'M')))

df_c['earliest_cr_line'].describe()

df_c.loc[:, ]

print(df_c.loc[:,['earliest_cr_line', 'earliest_cr_line_date','m_earliest_cr_line']]
[df_c['m_earliest_cr_line']<0])

df_c['m_earliest_cr_line'][df_c['m_earliest_cr_line']<0] = df_c['m_earliest_cr_line'].max()

df_c['m_earliest_cr_line'].describe()


df_c['issue_d'] = pd.to_datetime(df_c['issue_d'], format = '%b-%y')
#time --> timestamp -> numeric
df_c['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01')  - df_c['issue_d'])/ np.timedelta64(1, 'M')))

np.datetime64('2010-10-01') +np.timedelta64(10,'D')

###encoding
df_c.info()

#prefix gives colume a name
loan_data_dummies = [pd.get_dummies(df_c['grade'], prefix = 'GRE'),
                     pd.get_dummies(df_c['sub_grade'], prefix = 'sub_GRE'),
                     pd.get_dummies(df_c['home_ownership'], prefix = 'home_ownership'),
                     pd.get_dummies(df_c['verification_status'], prefix = 'verification_status'),
                     pd.get_dummies(df_c['loan_status'], prefix = 'loan_stats'),
                     pd.get_dummies(df_c['purpose'], prefix = 'purpose'),
                     pd.get_dummies(df_c['addr_state'], prefix = 'add_State'),
                     pd.get_dummies(df_c['initial_list_status'], prefix = 'ils')]
                     
##data imputation
df_c.isnull()
df_c.isnull().sum()
    
df_c['total_rev_hi_lim'].fillna(df_c['funded_amnt'], inplace = True)

pd.options.display.max_rows = 30
df_c['annual_inc'] = df_c['annual_inc'].replace('nan',round(df_c['annual_inc'].mean()))

df_c['annual_inc'].fillna(round(df_c['annual_inc'].mean()),inplace = True)




df_c['acc_now_delinq'].fillna(0 ,inplace = True)

df_c['m_earliest_cr_line'].fillna(0 ,inplace = True)

df_c['total_acc'].fillna(0 ,inplace = True)

df_c['pub_rec'].fillna(0 ,inplace = True)


#explore data of dependent variable y
df_c['loan_status'].unique()
df_c['loan_status'].value_counts()/ len(df_c['loan_status'])

#list  'Charged Off', 'Default', and late to default (0)

df_c['good_bad'] = np.where(df_c['loan_status'].isin(['Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged off','Late (16-30 days)']),0,1)


df_c['good_bad']

#turning continued variable to category

###train, test
from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(df_c.drop('good_bad', axis = 1), df_c['good_bad'] ,test_size = 0.2, random_state =42)

df_inputs_prepr = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train

dfx = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis = 1) 

#####group
#1
dfx.groupby(dfx.columns.values[0], as_index = False).count()
#2
dfc = dfx.groupby(dfx.columns.values[0], as_index = False)[dfx.columns.values[1]].count()



#mean
dfm = dfx.groupby(dfx.columns.values[0],  as_index = False)[dfx.columns.values[1]].mean()

dfc.columns = ['grade', 'c_gb']
dfm.columns = ['grade','prob_g']

df_new = pd.concat([dfc,dfm], axis = 1)

df_new = df_new.iloc[:,[0,1,3]]

#proportion
df_new['n_proportion'] = df_new['c_gb'] / df_new['c_gb'].sum()


## number of good and bad  in each category
# c_gb count of good and bad
df_new['n_good'] = df_new['c_gb'] * df_new['prob_g']
df_new['n_bad'] = df_new['c_gb'] *  (1 - df_new['prob_g']) 


#good in each category / good in all
df_new['prop_good'] = df_new['n_good']/ df_new['n_good'].sum()
df_new['prop_bad'] = df_new['n_bad'] / df_new['n_bad'].sum()

###WoE
df_new['WoE'] = np.log(df_new['prop_good'] / df_new['prop_bad']) 

df_new = df_new.sort_values(['WoE'])
df_new = df_new.reset_index(drop = True)

#info reveal the overall value
df_new['info_ratio'] = (df_new['prop_good'] - df_new['prop_bad']) * df_new['WoE']
df_new['info_ratio'] = df_new['info_ratio'].sum()

###automation
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis =1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis =1)
    df =df.iloc[:,[0, 1, 3]]
    #group by類別: count of good and bad
    #prop_good = good in every category =  (1 | category) /  sum in each
    df.columns = [df.columns.values[0],'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1- df['prop_good']) * df['n_obs']
    #prop_n_good : good in cerntain caterogies / overall good
    df['prop_n_good'] = df['n_good'] /  df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] =np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) *df['WoE']
    df['IV'] = df['IV'].sum()
    return df
 

df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

###visualization
#x: grade, y: debt
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 40):
    x = np.array(df_WoE.iloc[:,0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize = (10,4))
    plt.plot(x,y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels )
    #don't know how to use ax to rotate

plot_by_woe(df_temp )


###part1 preprocessing discrete variables: dummy variables

df_temp = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp

dl = pd.DataFrame(np.zeros(1))


dl['n_obs'] = df_temp.iloc[2,1] + sum(df_temp.iloc[[0,1,5],1])
dl['n_good'] =  df_temp.iloc[2,4] + sum(df_temp.iloc[[0,1,5],4])
dl['n_bad'] = df_temp.iloc[2,5] + sum(df_temp.iloc[[0,1,5],5])
dl['prop_n_good'] = df_temp.iloc[2,6] + sum(df_temp.iloc[[0,1,5],6])
dl['prop_n_bad'] = df_temp.iloc[2,7] + sum(df_temp.iloc[[0,1,5],7])
dl['WoE'] =np.log(dl['prop_n_good'] / dl['prop_n_bad'])
ndl = dl.iloc[0,1:]


#compile the column
df_temp.drop([0,1,5], axis = 0, inplace = True)
df_temp.iloc[0,1] = ndl.iloc[0]


df_temp.iloc[0,4] = ndl.iloc[1]
df_temp.iloc[0,5] = ndl.iloc[2]
df_temp.iloc[0,6] = ndl.iloc[3]
df_temp.iloc[0,7] = ndl.iloc[4]
df_temp.iloc[0,8] = ndl.iloc[5]

df_temp.iloc[0,2] = df_temp.iloc[0,4] / df_temp.iloc[0,1]
df_temp.iloc[0,3] = df_temp.iloc[0,5] / df_temp.iloc[0,1]


####woe and n_obs is correct merged#



df_temp1 = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)



plot_by_woe(df_temp1)


# exclude the low obs variables
plot_by_woe(df_temp1.iloc[2: -2,:])

plot_by_woe(df_temp1.iloc[6: -6,:])

#preprocessing hw
#1. verification_status
df_temp2 = woe_discrete(df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp2 = df_temp2.sort_values(['WoE'])
df_temp2 = df_temp2.reset_index(drop = True)

plot_by_woe(df_temp2)

#2. purpose
df_temp3 = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp3 = df_temp3.sort_values(['WoE'])
df_temp3 = df_temp3.reset_index(drop = True)

plot_by_woe(df_temp3)

#interval
df_temp3.iloc[:5,:]
df_temp3.iloc[5:9,:]
df_temp3.iloc[9,:]
df_temp3.iloc[10,:]
df_temp3.iloc[11:13,:]
df_temp3.iloc[13,:]

#???why other and medical are in the same group and edu and other small one are in a group

#3. initial_list_status
df_temp4 = woe_discrete(df_inputs_prepr, 'initial_list_status', df_targets_prepr)


###continueous variable
def woe_discrete_continuous(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis =1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis =1)
    df =df.iloc[:,[0, 1, 3]]
    #group by類別: count of good and bad
    #prop_good = good in every category =  (1 | category) /  sum in each
    df.columns = [df.columns.values[0],'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1- df['prop_good']) * df['n_obs']
    #prop_n_good : good in cerntain caterogies / overall good
    df['prop_n_good'] = df['n_good'] /  df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] =np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    #sort by the original value
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) *df['WoE']
    df['IV'] = df['IV'].sum()
    return df

df_inputs_prepr['term_int'].unique()
df_temp5 = woe_discrete_continuous(df_inputs_prepr, 'term_int', df_targets_prepr)


def plot_by_woec(df_WoE, rotation_of_x_axis_labels = 40):
    x = np.array(df_WoE.iloc[:,0])
    y = df_WoE['WoE']
    fig, ax = plt.subplots(1,1)
    ax.plot(x,y, marker = 'o', linestyle = '--', color = 'k')
    ax.set_xlabel(df_WoE.columns[0])
    ax.set_ylabel('Weight of Evidence')
    ax.set_title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    ax.set_xticks(df_WoE.iloc[:,0])
    ax.set_xticklabels(df_WoE.iloc[:,0], rotation = rotation_of_x_axis_labels)
    #don't know how to use ax to rotate
plot_by_woec(df_temp5)


df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)



#df_inputs_prepr['emp_length_int'].unique()

#create course fine
df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_lengh_int'].isin([0],1,0))


###continuous variable
#split into 200 variable
df_inputs_prepr['mths_since_issue_d'].unique()
# 50 is the category numbers cut, every cell (373028) is being put in to this
df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'],50)


df_temp6 = woe_discrete_continuous(df_inputs_prepr,'mths_since_issue_d_factor',df_targets_prepr)

#woe graph
plot_by_woe(df_temp6,90)

#create new category  for each interval
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)),1,0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)),1,0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)),1,0)
#....etc
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85,int(df_inputs_prepr['mths_since_issue_d'].max()))),1,0)


#find classify
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)

df_temp7 = woe_discrete_continuous(df_inputs_prepr, 'int_rate_factor', df_targets_prepr)

plot_by_woe(df_temp7, 90)
#the higher the interest rate, the lower the woe, the higher the probability of default

#coarse classify 
df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548),1,0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate']<=12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate']<= 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) &(df_inputs_prepr['int_rate'] <= 20.281), 1, 0) 
df_inputs_prepr['int_rate:20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281),1,0)


#funded amnt_factor
#fine classifier
df_inputs_prepr['funded_amnt_factor'] =pd.cut(df_inputs_prepr['funded_amnt'], 50)
df_temp8 = woe_discrete_continuous(df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)

#the woe varies a lot, so it is sufficient not to use funded amnt in the model. That means different interval
# will gives different conclusion in very small period.

plot_by_woe(df_temp8,90)

#preprocessing: m_earliest_cr_line
df_inputs_prepr['m_earliest_cr_line'] = pd.cut(df_inputs_prepr['m_earliest_cr_line'], 50)
df_temp10 = woe_discrete_continuous(df_inputs_prepr,'m_earliest_cr_line',df_targets_prepr )
plot_by_woe(df_temp10,90)


#WoE close and obs are close the mean
df_inputs_prepr['m_earliest_cr_line:<140'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['m_earliest_cr_line:140-164'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(140,164)), 1, 0)
df_inputs_prepr['m_earliest_cr_line:164-246'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(164,246 )), 1, 0)
df_inputs_prepr['m_earliest_cr_line:246-270'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(246,271)), 1, 0)
df_inputs_prepr['m_earliest_cr_line:270-294'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(271,294)), 1, 0)
df_inputs_prepr['m_earliest_cr_line:294-352'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(294,352)), 1, 0)
df_inputs_prepr['m_earliest_cr_line:>352'] = np.where(df_inputs_prepr['m_earliest_cr_line'].isin(range(362, 587)), 1, 0)


#preprocessing: installment

df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
df_temp9 = woe_discrete_continuous(df_inputs_prepr,'installment_factor',df_targets_prepr )
#last part is not able to set a representative dummies.
plot_by_woe(df_temp9,90)

#preprocessing: delinq_2yrs
df_temp11 = woe_discrete_continuous(df_inputs_prepr,'delinq_2yrs',df_targets_prepr )
plot_by_woe(df_temp11,90)

#preprocessing: inq_last_6mths
df_inputs_prepr['inq_last_6mths_f'] = pd.cut(df_inputs_prepr['inq_last_6mths'], 30)
df_temp12 = woe_discrete_continuous(df_inputs_prepr,'inq_last_6mths_f',df_targets_prepr )
df_inputs_prepr['inq_last_6mths:~1.1'] = np.where(df_inputs_prepr['inq_last_6mths'] >0 & (df_inputs_prepr['inq_last_6mths'] <1.1), 1, 0)
df_inputs_prepr['inq_last_6mths:1.1-6.6'] = np.where(df_inputs_prepr['inq_last_6mths'] >1.1 & (df_inputs_prepr['inq_last_6mths'] <6.6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6.6'] = np.where(df_inputs_prepr['inq_last_6mths'] <6.6 , 1, 0)


plot_by_woe(df_temp12,90)

#open_acc

df_inputs_prepr['open_acc_f'] = pd.cut(df_inputs_prepr['open_acc'], 50)
df_temp13 = woe_discrete_continuous(df_inputs_prepr,'open_acc',df_targets_prepr )
plot_by_woe(df_temp13,90)



df_inputs_prepr['open_acc:1-34'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 34), 1, 0)
df_inputs_prepr['open_acc:>34'] = np.where((df_inputs_prepr['open_acc'] > 34), 1, 0)


#pub_rec
df_temp14 = woe_discrete_continuous(df_inputs_prepr, 'pub_rec', df_targets_prepr)
plot_by_woe(df_temp14,90)

#annual_income
#obs centered in low income
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
df_temp15 = woe_discrete_continuous(df_inputs_prepr,'annual_inc_factor',df_targets_prepr )
plot_by_woe(df_temp15,90)

#income > 140000 is high income
#create dummies on high income
#select all the column but income less than or equal to 140000
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000,:]
df_inputs_prepr_temp['annual_inc_factor'] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
df_temp16 = woe_discrete_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])
plot_by_woe(df_temp16.iloc[2:,:],90)

#monotonous
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)

df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
#select non-null values by pd.notnull
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp17 = woe_discrete_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])
plot_by_woe(df_temp17,90)

#cut them approximately the same width, not the same obs
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)

#homework
#dti
df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 50)
#index = the index after pd.cut
df_temp18 = woe_discrete_continuous(df_inputs_prepr_temp,'dti_factor',df_targets_prepr[df_inputs_prepr_temp.index] )
plot_by_woe(df_temp18,90)
pd.options.display.max_rows = 51 
df_temp18.tail(50)

#mths_since_last_record
#categorerized the na 
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]
# gives the dataframe without null
# sum(np.where(pd.isnull(df_inputs_prepr_temp['mths_since_last_record']), 1, 0)) 
df_inputs_prepr_temp['mths_since_last_record'] = pd.cut(df_inputs_prepr['mths_since_last_record'], 50)
df_temp19 = woe_discrete_continuous(df_inputs_prepr_temp,'mths_since_last_record',df_targets_prepr[df_inputs_prepr_temp.index] )
plot_by_woe(df_temp19,90)

df_temp19['n_obs']
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)

# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)

########################################################################################
######test dataset, change content of df_inputs_prepr to loan_data_inputs_test 
#########################################################################################
#woe_discrete

#save training prepressing data
loan_data_inputs_train = df_inputs_prepr



from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(df_c.drop('good_bad', axis = 1), df_c['good_bad'] ,test_size = 0.2, random_state =42)

df_inputs_prepr = loan_data_inputs_test

df_targets_prepr = loan_data_targets_test

dfx = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis = 1) 

#####group
#1
dfx.groupby(dfx.columns.values[0], as_index = False).count()
#2
dfc = dfx.groupby(dfx.columns.values[0], as_index = False)[dfx.columns.values[1]].count()



#mean
dfm = dfx.groupby(dfx.columns.values[0],  as_index = False)[dfx.columns.values[1]].mean()

dfc.columns = ['grade', 'c_gb']
dfm.columns = ['grade','prob_g']

df_new = pd.concat([dfc,dfm], axis = 1)

df_new = df_new.iloc[:,[0,1,3]]

#proportion
df_new['n_proportion'] = df_new['c_gb'] / df_new['c_gb'].sum()


## number of good and bad  in each category
# c_gb count of good and bad
df_new['n_good'] = df_new['c_gb'] * df_new['prob_g']
df_new['n_bad'] = df_new['c_gb'] *  (1 - df_new['prob_g']) 


#good in each category / good in all
df_new['prop_good'] = df_new['n_good']/ df_new['n_good'].sum()
df_new['prop_bad'] = df_new['n_bad'] / df_new['n_bad'].sum()

###WoE
df_new['WoE'] = np.log(df_new['prop_good'] / df_new['prop_bad']) 

df_new = df_new.sort_values(['WoE'])
df_new = df_new.reset_index(drop = True)

#info reveal the overall value
df_new['info_ratio'] = (df_new['prop_good'] - df_new['prop_bad']) * df_new['WoE']
df_new['info_ratio'] = df_new['info_ratio'].sum()

###automation
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis =1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis =1)
    df =df.iloc[:,[0, 1, 3]]
    #group by類別: count of good and bad
    #prop_good = good in every category =  (1 | category) /  sum in each
    df.columns = [df.columns.values[0],'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1- df['prop_good']) * df['n_obs']
    #prop_n_good : good in cerntain caterogies / overall good
    df['prop_n_good'] = df['n_good'] /  df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] =np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) *df['WoE']
    df['IV'] = df['IV'].sum()
    return df


df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

###visualization
#x: grade, y: debt
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 40):
    x = np.array(df_WoE.iloc[:,0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize = (10,4))
    plt.plot(x,y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels )
    #don't know how to use ax to rotate

plot_by_woe(df_temp )


###part1 preprocessing discrete variables: dummy variables

df_temp = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp

dl = pd.DataFrame(np.zeros(1))


dl['n_obs'] = df_temp.iloc[2,1] + sum(df_temp.iloc[[0,1,5],1])
dl['n_good'] =  df_temp.iloc[2,4] + sum(df_temp.iloc[[0,1,5],4])
dl['n_bad'] = df_temp.iloc[2,5] + sum(df_temp.iloc[[0,1,5],5])
dl['prop_n_good'] = df_temp.iloc[2,6] + sum(df_temp.iloc[[0,1,5],6])
dl['prop_n_bad'] = df_temp.iloc[2,7] + sum(df_temp.iloc[[0,1,5],7])
dl['WoE'] =np.log(dl['prop_n_good'] / dl['prop_n_bad'])
ndl = dl.iloc[0,1:]


#compile the column
df_temp.drop([0,1,5], axis = 0, inplace = True)
df_temp.iloc[0,1] = ndl.iloc[0]


df_temp.iloc[0,4] = ndl.iloc[1]
df_temp.iloc[0,5] = ndl.iloc[2]
df_temp.iloc[0,6] = ndl.iloc[3]
df_temp.iloc[0,7] = ndl.iloc[4]
df_temp.iloc[0,8] = ndl.iloc[5]

df_temp.iloc[0,2] = df_temp.iloc[0,4] / df_temp.iloc[0,1]
df_temp.iloc[0,3] = df_temp.iloc[0,5] / df_temp.iloc[0,1]

####woe and n_obs is correct merged#
###############





 
loan_data_inputs_test = df_inputs_prepr


loan_data_inputs_train.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/loan_data_inputs_train.csv')
loan_data_targets_train.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/loan_data_targets_test.csv')













