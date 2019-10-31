'''
Author: Scott Gregoire, Ph.D.; Veronika Megler, Ph.D.
Created: 9/4/2019
'''

#Defining several convenience functions for analyzing the output of an HPO job and calculating the value of a loan portfolio. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sagemaker

def get_hpo_metrics(hpo_job_name):
    '''This function takes the name of a HPO job as input and returns a dataframe that contains the user-defined metrics calculated during the job run.'''
    
    df = pd.DataFrame(columns=['metric_name', 'value', 'training_job'])
    counter = 0
    for job in sagemaker.analytics.HyperparameterTuningJobAnalytics(hpo_job_name).training_job_summaries():

        counter += 1
        if counter%50==0: print(counter, ' Training Jobs loaded.')

        df_tmp = sagemaker.analytics.TrainingJobAnalytics(job['TrainingJobName']).dataframe()
        df_tmp = df_tmp.drop('timestamp', axis=1)
        df_tmp['training_job'] = job['TrainingJobName']
        df = df.append(df_tmp, ignore_index=True)
        
    return df.pivot(index='training_job', columns='metric_name', values='value')

def get_hyperparameters(hpo_job_name):
    '''This function takes the name of a HPO job as input and returns a dataframe that contains the values of the hyperparameters and the metric being optimized across training jobs.'''
    
    hpo_results = sagemaker.analytics.HyperparameterTuningJobAnalytics(hpo_job_name).training_job_summaries()
    
    df = pd.DataFrame([ {'TrainingJobName': result['TrainingJobName'], 
   'portfolio_value': float(result['FinalHyperParameterTuningJobObjectiveMetric']['Value']),
   'threshold': float(result['TunedHyperParameters']['threshold'])} for result in hpo_results])
    
    return df

def custom_pred_distro(positives, negatives, cutoff=0.5, title=None):
    '''This function generates distributions of predicted scores for actual positives and actual negatives. 
    
    Note that the cutoff argument only affects the coloring of the graphs. It does NOT affect any model
    results or predicted values.'''

    fig, axes = plt.subplots(2,1, figsize=(10,8))

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set(xlim=[0,1], xticks=np.arange(0, 1, step=0.1), xlabel='Model Score', ylabel='Count', title='Actual Negatives')
    axes[0].hist(negatives[negatives>cutoff], color='C1', label='False Positives')
    axes[0].hist(negatives[negatives<=cutoff], label='True Negatives')
    axes[0].legend()
    
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set(xlim=[0,1], xticks=np.arange(0, 1, step=0.1), xlabel='Model Score', ylabel='Count', title='Actual Positives')
    axes[1].hist(positives[positives>cutoff], label='True Positives')
    axes[1].hist(positives[positives<=cutoff], label='False Negatives')
    axes[1].legend()
    
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight='bold', x=0.52)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()
    
    return None

def calc_portfolio_value(x_test, y_test, y_prediction, y_prediction_discrete, threshold, fixed_cost=100, desired_return=0.05):
    '''
    Calculate value of a portfolio given model probability predictions.
    
    Assume Loss Given Default = portion of Gross approved amount NOT insured by SBA. 
    Assume lender sets zero-profit interest rate.
    Assume default occurs immediately after loan is extended, that is, bank receives no interest income.
    '''
    

    #NB: IN THIS CONTEXT, A POSITIVE REPRESENTS A DEFAULT, AND A NEGATIVE REPRESENTS A NON-DEFAULT.
    #break into positive and negative classes
    tp_mask = (y_test==1).reset_index(drop=True) & (y_prediction_discrete==1).reset_index(drop=True)
    fn_mask = (y_test==1).reset_index(drop=True) & (y_prediction_discrete==0).reset_index(drop=True)
    
    tn_mask = (y_test==0).reset_index(drop=True) & (y_prediction_discrete==0).reset_index(drop=True)
    fp_mask = (y_test==0).reset_index(drop=True) & (y_prediction_discrete==1).reset_index(drop=True)
    
    #calculate cost for each cell of confusion matrix
    x_test = x_test.reset_index(drop=True)
    
    SBA_Appv_percent = x_test['SBA_Appv']/x_test['GrAppv']
    LGD = 1 - SBA_Appv_percent
    
    tp_val = [- fixed_cost] * len(x_test[tp_mask].index) #predicted as default and defaulted. Rejected Loan.
    fn_val = -x_test[fn_mask]['GrAppv'] * LGD[fn_mask] - fixed_cost #predicted as non-default, but defaulted. Approved Loan. With SBA Insurance
    
    interest_rate = (desired_return + y_prediction * LGD)/(1 - y_prediction) #With SBA Insurance
    
    tn_val = interest_rate[tn_mask] * x_test[tn_mask]['GrAppv'] - fixed_cost# predicted as non-default, and didn't default. Approved Loan.
    fp_val = [- fixed_cost] * len(x_test[fp_mask].index) #predicted as default, but didn't default. Rejected Loan.
   
    #sum all values to calculate value of overall portfolio
    portfolio_value = np.sum(tp_val) + fn_val.sum() + tn_val.sum() + np.sum(fp_val)
    
    approved_loan_cnt, approved_loan_25, approved_loan_50, approved_loan_75 = x_test[y_prediction_discrete==0].describe().loc[['count', '25%', '50%', '75%'],'GrAppv']
    
    approved_loan_total = x_test.loc[y_prediction_discrete==0, 'GrAppv'].sum()
    
    rejected_loan_cnt, rejected_loan_25, rejected_loan_50, rejected_loan_75 = x_test[y_prediction_discrete==1].describe().loc[['count', '25%', '50%', '75%'],'GrAppv']
    
    rejected_loan_total = x_test.loc[y_prediction_discrete==1, 'GrAppv'].sum()
    
    approved_interest_rate_distro = interest_rate[y_prediction_discrete==0].describe()
     
    approved_interest_rate_cnt = approved_interest_rate_distro['count'] 
    approved_interest_rate_25 = approved_interest_rate_distro['25%'] 
    approved_interest_rate_50 = approved_interest_rate_distro['50%'] 
    approved_interest_rate_75 = approved_interest_rate_distro['75%'] 
    
    return (portfolio_value,
           approved_loan_total,
           approved_loan_cnt, 
           approved_loan_25, 
           approved_loan_50, 
           approved_loan_75,
           rejected_loan_total,
           rejected_loan_cnt, 
           rejected_loan_25, 
           rejected_loan_50, 
           rejected_loan_75,
           approved_interest_rate_cnt,
           approved_interest_rate_25,
           approved_interest_rate_50,
           approved_interest_rate_75)

def classification_report(positives, negatives, cutoff):
    '''This function draws a confusion matrix, using our cutoff.'''
 
    tp = (positives > cutoff).sum()
    fn = (positives <= cutoff).sum()

    tn = (negatives < cutoff).sum()
    fp = (negatives >= cutoff).sum()
    
    report = {}
    report['Accuracy'] = (tp + tn)/(tp + fn + tn + fp)
    report['Precision_1'] = tp/(tp + fp)
    report['Recall_1'] = tp/(tp + fn)
    report['Precision_0'] = tn/(tn + fn)
    report['Recall_0'] = tn/(tn + fp)
    
    return report 