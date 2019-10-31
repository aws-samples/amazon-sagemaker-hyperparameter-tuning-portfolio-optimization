'''
Author: Scott Gregoire, Ph.D.; Veronika Megler, Ph.D.
Created: 9/4/2019
'''

import os
import json
import pickle
import sys
import traceback
import time
from decimal import Decimal

import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load

import numpy as np
import boto3 as aws

# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = os.path.join(prefix, 'input/data')
model_path = os.path.join(prefix, 'model')
output_path = os.path.join(prefix, 'output')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

model_name = 'hpo_blog_trained_model.h5'

# This algorithm has input data called 'training' and 'test'. Since we run in
# File mode, the input files are copied to the directory specified here.
training_channel_name = 'train'
training_path = os.path.join(input_path, training_channel_name)
test_channel_name = 'test'
test_path = os.path.join(input_path, test_channel_name)
   
#portfolio_value
def calc_portfolio_value(x_test, y_test, y_prediction, y_prediction_discrete, threshold, fixed_cost=100, desired_return=0.05):
    '''
    Calculate value of a portfolio given model probability predictions.
    
    Assume Loss Given Default = portion of Gross approved amount NOT insured by SBA. 
    Assume lender sets zero-profit interest rate.
    Assume default occurs immediately after loan is extended, that is, bank receives no interest income.
    '''

    #NB: IN THIS CONTEXT, A POSITIVE REPRESENTS A DEFAULT, AND A NEGATIVE REPRESENTS A NON-DEFAULT.
    #break into positive and negative classes
    print('break into positive and negative classes')   
    tp_mask = (y_test==1).reset_index(drop=True) & (y_prediction_discrete==1).reset_index(drop=True)
    fn_mask = (y_test==1).reset_index(drop=True) & (y_prediction_discrete==0).reset_index(drop=True)
    
    tn_mask = (y_test==0).reset_index(drop=True) & (y_prediction_discrete==0).reset_index(drop=True)
    fp_mask = (y_test==0).reset_index(drop=True) & (y_prediction_discrete==1).reset_index(drop=True)
    
    #calculate cost for each cell of confusion matrix
    
    SBA_Appv_percent = x_test['SBA_Appv']/x_test['GrAppv']
    LGD = 1 - SBA_Appv_percent
    
    print('calculate cost for each cell of confusion matrix')
    tp_val = [- fixed_cost] * len(x_test[tp_mask].index) #predicted as default and defaulted. Rejected Loan.
    fn_val = -x_test[fn_mask]['GrAppv'] * LGD[fn_mask] - fixed_cost #predicted as non-default, but defaulted. Approved Loan. With SBA Insurance
    
    interest_rate = (desired_return + y_prediction * LGD)/(1 - y_prediction) #With SBA Insurance
    
    tn_val = interest_rate[tn_mask] * x_test[tn_mask]['GrAppv'] - fixed_cost# predicted as non-default, and didn't default. Approved Loan.
    fp_val = [- fixed_cost] * len(x_test[fp_mask].index) #predicted as default, but didn't default. Rejected Loan.
    
    #sum all values to calculate value of overall portfolio
    print('sum all values to calculate value of overall portfolio')
    portfolio_value = np.sum(tp_val) + fn_val.sum() + tn_val.sum() + np.sum(fp_val)
    
    print('calculate distributions') 
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
           
# The function to execute the training.
def train():
    print('Starting the training.')
    try:
        os.system('which python')
        os.system('pip freeze') 
        
        with open(param_path, 'r') as tc:
            hyperparameters = json.load(tc)
            
            threshold = float(hyperparameters['threshold'])
            model_pickle_bucket = hyperparameters['model_pickle_bucket']
            model_pickle_key = hyperparameters['model_pickle_key']
            
            print('threshold: ', threshold)
            print('model_pickle_bucket: ', model_pickle_bucket)
            print('model_pickle_key: ', model_pickle_key)
                
        #read in training data
        train_x = pd.read_csv(training_path + '/x_train.csv', squeeze=True)
        train_y = pd.read_csv(training_path + '/y_train.csv', squeeze=True)

        #scale training data 
        scaler = StandardScaler().fit(train_x) 
        x_train_scld = scaler.transform(train_x) 

        s3 = aws.resource('s3')    
        s3.Bucket(model_pickle_bucket).download_file(model_pickle_key, 'model.tar.gz')
        os.system('tar -xzvf model.tar.gz')
        model = load('random_forest_classifier.pkl')
        
        #calculate scoring metric
        #read in test data
        test_x = pd.read_csv(test_path + '/x_test.csv', squeeze=True)
        test_y = pd.read_csv(test_path + '/y_test.csv', squeeze=True)
        
        #scale test data
        test_x_scld = scaler.transform(test_x)
        
        #continuous and discrete predictions
        pred_y = model.predict_proba(test_x_scld)
        pred_y = pd.DataFrame(pred_y).loc[:,1]
        
        pred_y_discrete = (pred_y > threshold) + 0 #to change boolean to integer
                
        print('Calculating Metrics')
        
        #accuracy of model
        acc = model.score(test_x_scld, test_y)
        
        tn, fp, fn, tp = confusion_matrix(test_y, pred_y_discrete).ravel()
        
        (portfolio_value, approved_loan_total, approved_loan_cnt, approved_loan_25, approved_loan_50, approved_loan_75, rejected_loan_total, rejected_loan_cnt, rejected_loan_25, rejected_loan_50, rejected_loan_75, approved_interest_rate_cnt, approved_interest_rate_25, approved_interest_rate_50, approved_interest_rate_75) = calc_portfolio_value(x_test=test_x, y_test=test_y, y_prediction=pred_y, y_prediction_discrete=pred_y_discrete, fixed_cost=100, threshold=threshold)
        
        print('Printing Metrics')
        
        print('accuracy: {}'.format(acc))
        
        print('TN: {}'.format(tn))
        print('FP: {}'.format(fp))
        print('TP: {}'.format(tp))
        print('FN: {}'.format(fn))
        
        print('portfolio_value: {}'.format(portfolio_value))
        
        print('approved_loan_total: {}'.format(approved_loan_total))
        print('approved_loan_cnt: {}'.format(approved_loan_cnt))
        print('approved_loan_25: {}'.format(approved_loan_25)) 
        print('approved_loan_50: {}'.format(approved_loan_50)) 
        print('approved_loan_75: {}'.format(approved_loan_75)) 
        
        print('rejected_loan_total: {}'.format(rejected_loan_total))
        print('rejected_loan_cnt: {}'.format(rejected_loan_cnt))
        print('rejected_loan_25: {}'.format(rejected_loan_25)) 
        print('rejected_loan_50: {}'.format(rejected_loan_50)) 
        print('rejected_loan_75: {}'.format(rejected_loan_75))
        
        print('approved_interest_rate_cnt: {}'.format(approved_interest_rate_cnt))
        print('approved_interest_rate_25: {}'.format(approved_interest_rate_25)) 
        print('approved_interest_rate_50: {}'.format(approved_interest_rate_50)) 
        print('approved_interest_rate_75: {}'.format(approved_interest_rate_75))  
        
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)