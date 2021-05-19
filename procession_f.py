#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 02:54:33 2021

@author: yumingchang
"""

import numpy as np
import pandas as pd

# set [0] columne to be index, but it will change the header, so apply header = None
l_data_inputs_train = pd.read_csv('/Users/yumingchang/Documents/python advance_ML/lending club/l_data_inputs_train.csv', index_col = 0)
l_data_targets_train = pd.read_csv('/Users/yumingchang/Documents/python advance_ML/lending club/l_data_targets_train.csv', index_col = 0 , header = None)
l_data_inputs_test = pd.read_csv('/Users/yumingchang/Documents/python advance_ML/lending club/l_data_inputs_test.csv', index_col = 0)
l_data_targets_test = pd.read_csv('/Users/yumingchang/Documents/python advance_ML/lending club/l_data_targets_test.csv', index_col = 0,header = None)


l_data_inputs_train.head()
# when preprocessing, we need to record the variables we  created.
inputs_train_with_ref_cat = l_data_inputs_train.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86',
]]






# Here we store the names of the reference category dummy variables in a list.
#use the lower weight of evident variable as a benchmark.
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

#dummy trap need to remove ref variable
inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
# From the dataframe with input variables, we drop the variables with variable names in the list with reference categories. 
inputs_train.head()


##########################
###model estimation
##########################
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd.options.display.max_rows = None

reg = LogisticRegression()
#input x and y
reg.fit(inputs_train, l_data_targets_train )
reg.intercept_
reg.coef_



#.value makes it a nd array??????????????
feature_name.shape = inputs_train.columns.values
summary_table = pd.DataFrame(columns = ['Feature name'], data =feature_name)
summary_table['coefficents'] = np.transpose(reg.coef_)

summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
reg.coef_.shape

#check variable significant
#find a way to check multi-variate p value,
#unit-variate = independent
from sklearn import linear_model
import scipy.stats as stat


####look at p-value for coefficient
class LogisticRegression_with_p_values:
    def __init__(self,*args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)
    def fit(self,X,y):
        self.model.fit(X,y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T, X)
        Cramer_rao = np.linalg.inv(F_ij)
        sigma_estimates =np.sqrt(np.diagonal(Cramer_rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values




reg = LogisticRegression_with_p_values()
reg.fit(inputs_train, l_data_targets_train)



summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table    
        
#add p value to the dataframe, but we don't get the p value for intercept
p_values = reg.p_values
p_values = np.append(np.nan, np.array(p_values))


summary_table['p_values'] = p_values
        
        
####let's only take significant variables.   
#### if we do so, it will be few variebles but those represent the rest area.  
        
#not significant:  addr_state:NM_VA     0.037453   2.428870e-01   
                # we could remove the whole categories rather than remove the miscelaniously one of them
                #keep all the dummy if some of them are stats significant
        
summary_table.loc[summary_table['p_values'] <= 0.05, :]     
#loc[row,col]

inputs_train_with_ref_cat = l_data_inputs_train.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86',
]]
        
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
inputs_train.head()    
reg = LogisticRegression_with_p_values()
reg.fit(inputs_train, l_data_targets_train)

pickle.dump(reg, open('/Users/yumingchang/Documents/python advance_ML/lending club/pd_model.sav', 'wb'))




#### validation test   
###out-of-sample


inputs_test_with_ref_cat = l_data_inputs_test.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86',
]]
        
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis = 1)
inputs_test.shape

inputs_test.head()      
        
y_hat_test = reg.model.predict(inputs_test)
y_hat_test
# p>= 50: 1
# p < 50: 0  

y_hat_test_proba = reg.model.predict_proba(inputs_test)

y_hat_test_proba
        
## array[pd, 1-pd]    #be good  # be bad.
        
y_hat_test_proba[:,1]

y_hat_test_proba = y_hat_test_proba[:,1]
        
l_data_targets_test_temp = l_data_targets_test     
l_data_targets_test_temp.reset_index(drop = True, inplace = True)
        
l_data_targets_test_temp



df_actual_predicted_probs = pd.concat([l_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], 
                                       axis = 1)


df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']
df_actual_predicted_probs.index = l_data_inputs_test.index

df_actual_predicted_probs.sort_index(inplace = True)


####accuracy

tr =0.9

df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
            rownames = ['Actual'], colnames =['Predicted'])
#10184 = a lot of FP will be giving

# n = 1 / all obs
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
            rownames = ['Actual'], colnames =['Predicted']) /df_actual_predicted_probs.shape[0]

#crosstab = compare table



#upper left divide bottom right
#overall accuracy
((pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], 
df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] 
+ (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], 
 df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1])


#bad are incorrectly classify as good as far more are far more important 



from sklearn.metrics import roc_curve, roc_auc_score

roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
#1 array fp rate
#2 array tp rate
#3 threshold

fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC Curve")

AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
#

#auc:
    #bad: 50- 60
    #poor: 60-70
    #fair: 70- 80
    #good: 80-90


#fpr, tpr,  decreasing threshold
roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

####gini and ks coef (cummulative %)
df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
df_actual_predicted_probs.head()
df_actual_predicted_probs = df_actual_predicted_probs.reset_index()


df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
#.cumsum() cumulative %
df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
#each population - cum_good, they grow simutaneosly
df_actual_predicted_probs['Cumulative N Bad'] = \
df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['loan_data_targets_test'].cumsum()

df_actual_predicted_probs.head()
df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] /(df_actual_predicted_probs.shape[0])

df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / df_actual_predicted_probs['loan_data_targets_test'].sum()

df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())

plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Bad'])
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.xlabel('cumulative % population')
plt.ylabel('cumulative % Bad')
plt.title('gini')

Gini = AUROC*2 - 1
#plt(y, x)
df_actual_predicted_probs['y_hat_test_proba']

#scorecard

summary_table


df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])
df_ref_categories['coefficients'] = 0
df_ref_categories['p_values'] =np.nan
df_ref_categories


df_scorecard = pd.concat([summary_table, df_ref_categories])

df_scorecard = df_scorecard.reset_index()


#**pd.str  Vectorized string, and only str to str vectorized.
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]


min_score = 300
max_score = 850 

min_sum_coef = df_scorecard.groupby('Original feature name')['coefficients'].min()

min_sum_coef = df_scorecard.groupby('Original feature name')['coefficients'].min().sum()


df_scorecard.groupby('Original feature name')['coefficients'].max()
max_sum_coef = df_scorecard.groupby('Original feature name')['coefficients'].max().sum()

temp_A = (max_score - min_score) / (max_sum_coef - min_sum_coef)

df_scorecard['Score - calculation'] = df_scorecard['coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)

#intecept = -103, because intercept is not a dummy, it is a coefficient. os the formula is not suitable for intercept

df_scorecard['Score - calculation'][0] = temp_A * (df_scorecard['coefficients'][0] - min_sum_coef) + min_score

#check minscore and max score close the desire score.
df_scorecard['Score - calculation'][0] = 313

df_scorecard['Score - Preliminary'] = df_scorecard['Score - calculation'].round()

#find lowest rows in each categories. 
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()

#find highest rows in each categories. 
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()


#fix the visualizeation of dataframe so i can find the largest difference
df_scorecard['Difference'] = (df_scorecard['Score - Preliminary'] - df_scorecard['Score - calculation'])

df_scorecard['score - final'] = df_scorecard['Score - Preliminary'] 

####calculation credit score
inputs_test_with_ref_cat_w_intercept.head()

inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat

#insert(0, column, values)
inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)

# make sure columns in test_ref_cat is arranged in the same order of scorecard.
inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
df_scorecard

scorecard_scores = df_scorecard['score - final']
inputs_test_with_ref_cat_w_intercept.shape
#value makes it an np array

#
scorecard_scores = scorecard_scores.values.reshape(102,1)
scorecard_scores.shape


inputs_test_with_ref_cat_w_intercept.shape

#multiply the metrix and sum: pd.dot()
#The dot method for Series computes the inner product,  eg. pd.dot()
#the matrix product   df.dot()

#category * scorecard and sum the result up. 
#################!!!!!!####################
y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)


####credit score to pd

sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)

y_hat_proba_from_score.head()

y_hat_test_proba[0:5]


####setting cut-offs


fpr, tpr, threshold = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
df_cutoffs = pd.concat([pd.DataFrame(threshold), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)
df_cutoffs.columns = ['threshold', 'fpr', 'tpr']



plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC Curve")


threshold.shape

#np.power(base, the exponents )
df_cutoffs['threshold'][0] = 1 - 1 / np.power(10, 16)


df_cutoffs['Score'] = ((np.log(df_cutoffs['threshold'] / (1 - df_cutoffs['threshold'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()
df_cutoffs['Score'][0] = max_score


df_cutoffs.tail()

def n_approved(p):
    return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p , 1, 0).sum()


#this list shows the approved case in different threshold.
df_cutoffs['N Approved'] = df_cutoffs['threshold'].apply(n_approved)

df_cutoffs['N Rejected'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]
df_cutoffs['Rejected Rate'] = df_cutoffs['N Rejected'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]


df_cutoffs.iloc[4970: 5000,]


#loc[row, col]
#accpt pd = 90%
#cutoff p >0.9
A_temp = df_cutoffs.loc[(df_cutoffs['threshold'] > 0.90)]
B_temp = A_temp[A_temp['threshold'] < 0.90005].loc[5314,:]
inputs_train_with_ref_cat.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/inputs_train_with_ref_cat.csv')
df_scorecard.to_csv('/Users/yumingchang/Documents/python advance_ML/lending club/df_scorecard.csv')

####homework


hw_temp = df_cutoffs[(df_cutoffs['threshold'] > 0.95)&(df_cutoffs['threshold'] < 1)]
## score = 645



#PSI
























































