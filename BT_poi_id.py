#!/usr/bin/python

import sys
import pickle
import os
import pandas as pd
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
from time import time
from operator import itemgetter
from __future__ import print_function
from time import time
from operator import itemgetter
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_stock_value', 'from_poi_to_this_person_frac',
                 'from_this_person_to_poi_frac', 'bonus', 'salary','shared_receipt_with_poi',
                'restricted_stock_deferred']


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
for i in data_dict:
    if float(data_dict[i]['to_messages']) != 0.0 and data_dict[i]['to_messages'] != 'NaN':
        from_poi_to_this_person_frac = float(data_dict[i]['from_poi_to_this_person'])/float(data_dict[i]['to_messages'])
        data_dict[i]['from_poi_to_this_person_frac'] = from_poi_to_this_person_frac
    else:
        data_dict[i]['from_poi_to_this_person_frac'] = 0.0
        
    if float(data_dict[i]['from_messages']) != 0.0 and data_dict[i]['from_messages'] != 'NaN':
        from_this_person_to_poi_frac = float(data_dict[i]['from_this_person_to_poi'])/float(data_dict[i]['from_messages'])
        data_dict[i]['from_this_person_to_poi_frac'] = from_this_person_to_poi_frac
    else:
        data_dict[i]['from_this_person_to_poi_frac'] = 0.0
   # print data_dict[i]['to_messages']    

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


### Decision Tree
clf = DecisionTreeClassifier(criterion = 'entropy')
                          #   max_leaf_nodes= None, max_depth= 2, min_samples_leaf= 5)
    #criterion = 'entropy',min_samples_split = 2
    


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params


param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }


ts_gs = run_gridsearch(features, labels, clf, param_grid, cv=5)


test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)



### Adaboost
clf = AdaBoostClassifier()

test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)

'''### Decision tree with default parameters
clf = DecisionTreeClassifier()

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)'''


