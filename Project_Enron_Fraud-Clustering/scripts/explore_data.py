""" 
    The Enron dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
"""

# Imports
import pickle

# Count Number of POI's where 'total_payments' is 'NaN'
count = 0
enron_data = pickle.load(open("dataset.pkl", "rb"))
for person in enron_data:
	if enron_data[person]['poi'] == True:
		if enron_data[person]['total_payments'] == 'NaN':
			count += 1

# Count Number of POI's Total
count_poi = 0
with open("poi_names.txt") as file:
	url = file.readline()
	n_break = file.readline()
	poi = file.readlines()
	for person in poi:
		if '(y)' in person:
			count_poi += 1


# Output
print("Total Number of POI's")
print(len(poi))
print("\nNumber of POI's with Emails:")
print(count_poi)
print("\nNumber of POI's that don't have total payments:")
print(count)
print("\nTotal Payments for Specifific POI's:")
print("Jeffrey Skilling")
print(enron_data['SKILLING JEFFREY K']['total_payments'])
print("\nKen Lay")
print(enron_data['LAY KENNETH L']['total_payments'])
print("\nAndrew Fastow")
print(enron_data['FASTOW ANDREW S']['total_payments'])