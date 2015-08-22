#!/usr/bin/python

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import KFold

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )


## original list
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi', 
                 'director_fees', 'deferred_income', 'long_term_incentive','from_poi_to_this_person',
                 'from_poi_to_this_person_frac','from_this_person_to_poi_frac']

#print len(features_list)

## took out least important features
#features_list = ['poi', 'total_stock_value', 'expenses',
#                 'from_messages', 'other', 'director_fees']

from_poi_to_this_person_frac = 0.0
from_this_person_to_poi_frac = 0.0


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

#print len(data_dict["LAY KENNETH L"])

## took out least important features AND added in new modified features
#features_list = ['poi', 'total_stock_value', 'expenses', 'other', 'director_fees']




data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

#print len(features_train)

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print accuracy_score(pred,labels_test)


##ranking feature importance
clf_feat = clf.feature_importances_
print len(clf_feat)

counter = 0
feature_imp = []

print "Feature Importance:"
for i in clf_feat:
        counter += 1
      #  print features_list[counter-1]
        feature_imp.append((features_list[counter-1], i))        
feature_imp.append((features_list[counter], i))        
        
feature_imp_sorted = sorted(feature_imp, key=lambda tup: tup[1], reverse=True)       

print len(feature_imp_sorted)

for i in range(0,len(feature_imp_sorted)):
    print feature_imp_sorted[i]
