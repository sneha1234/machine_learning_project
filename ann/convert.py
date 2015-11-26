#!/usr/bin/env python
import pandas as pd
import sys
import time, datetime
import numpy as np

column_names = ['Dates','Category','DayOfWeek','PdDistrict','X','Y'] # 'Address' not included for now
column_names_test = ['Dates','DayOfWeek','PdDistrict','X','Y'] # 'Address' not included for now
dofweek = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}

def writeData(filename, data):
	with open(filename, "w") as f:
		for i, row in enumerate(data):
			for j, d in enumerate(row):
				data[i][j] = str(data[i][j])

			f.write(",".join(data[i])+"\n")

def readData(filename, colnames= column_names):
	df = pd.read_csv(filename, quoting=1, usecols=colnames)
	return df

def convertData(df, test_data=False):
	pddistr = pd.unique(df.PdDistrict.ravel()).tolist()
	pddistr_key = dict(zip(pddistr, range(0, len(pddistr))))
	
	data = []

	if not test_data:
		catg = pd.unique(df.Category.ravel()).tolist()
		cat_key = dict(zip(catg, range(0, len(catg))))
	min_time = 0
	max_time = -1
	for i, row in df.iterrows():
		row_data = []
		# convert date to timestamp
		dt = datetime.datetime.strptime(row["Dates"], "%Y-%m-%d %H:%M:%S")
		ts = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)

		row_data.append(ts)

		# convert dofw
		row_data.append(dofweek[row["DayOfWeek"]])

		# Convert PdDistrict
		row_data.append(pddistr_key[row["PdDistrict"]])

		row_data.append(row["X"])
		row_data.append(row["Y"])
		
		if not test_data:
			# Convert Category
			row_data.append(cat_key[row["Category"]])
		data.append(row_data)
		
	'''
	#normalize
	sum_data = np.sum(data, axis=0)

	for i,row in enumerate(data):
		for j, d in enumerate(row):
			data[i][j] = d/float(sum_data[j])
	'''

	return data

if __name__ == '__main__':

	if(len(sys.argv) < 3):
		print "USAGE: python convert.py </path/to/train.csv> </path/to/test.csv>"
		sys.exit(0)

	# convert training and test and write to file
	df = readData(sys.argv[1])
	writeData("train_conv.txt", convertData(df))

	df = readData(sys.argv[2], column_names_test)
	writeData("test_conv.txt", convertData(df, True))

	df = None
	