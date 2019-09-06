'''
Author: Scott Gregoire, Ph.D.; Veronika Megler, Ph.D.
Email: sggregoi@amazon.com; meglerv@amazon.com
Created: 9/4/2019
'''

import argparse
import os
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import dump

# The function to execute the training.
def train(args):
    
    print('load training data.')
    x_train = pd.read_csv(os.path.join(args.train, 'x_train.csv'), squeeze=True)
    y_train = pd.read_csv(os.path.join(args.train, 'y_train.csv'), squeeze=True)
    
    print('scale training data.')
    scaler = StandardScaler().fit(x_train) 
    x_train_scld = scaler.transform(x_train) 

    print('estimate model')
    model = RandomForestClassifier(n_estimators=100, oob_score=True)
    model.fit(x_train_scld, y_train)

    print('persist model')
    dump(model, os.path.join(args.model_dir, 'random_forest_classifier.pkl'), protocol=2)

    print('load testing data.')
    x_test = pd.read_csv(os.path.join(args.test, 'x_test.csv'), squeeze=True)
    y_test = pd.read_csv(os.path.join(args.test, 'y_test.csv'), squeeze=True)

    x_test_scld = scaler.transform(x_test)

    print('calculate metrics')
    y_pred = model.predict_proba(x_test_scld)
    y_pred = pd.DataFrame(y_pred).loc[:,1]
    
    accuracy = model.score(x_test_scld, y_test)
    auc = roc_auc_score(y_test, y_pred)

    print('accuracy: ', accuracy)
    print('auc: ', auc)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args, _ = parser.parse_known_args()
    
    train(args)
