#!/usr/bin/python


import sys
import pickle

sys.path.append('../Users/bilal/Downloads/Machine Learning/ud120-projects/final_project/')
#print sys.path

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))


'''counter = 0'''
#for i in enron_data:
#      print i

  
# Total Datapoints


total_employees = float(len(enron_data))
print total_employees

print len(enron_data["METTS MARK"])
print enron_data["LAY KENNETH L"]

#for key, value in enron_data.iteritems() :
#    print key


# Total POI's in general
counter = 0
for i in enron_data:
    if enron_data[i]["poi"]:
        counter += 1
    else:
        pass
print counter  


# counting"NaN" total payments; percentage of people with "NaN" value
counter = 0
for i in enron_data:
    if enron_data[i]["total_payments"] == "NaN":
            counter += 1
    else:
        pass
print counter
print counter/total_employees

# counting POIs with "NaN" total payments
counter = 0
for i in enron_data:
    if enron_data[i]["poi"]:
        if enron_data[i]["total_payments"] == "NaN":
            counter += 1
    else:
        pass
print counter  




    


