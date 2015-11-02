'''
Functions for training data using naive bayes method
'''

import pandas as pd
import sys
from sets import Set
import random
import copy

column_names = ['Dates','Category','DayOfWeek','PdDistrict','Address','X','Y']

def readData(filename):
	df = pd.read_csv(filename, quoting=1, usecols=column_names)
	return df

# TODO optimize kmeans for improved runtimes
def kmeans(data, k, niter): # Data is a list of tuples
	assignments = [-1] * len(data)
	# First do random selection of k points
	centroids = random.sample(data, k)
	for r in range(niter):
		clusters = {}
		for i in range(k):
			clusters[i] = []

		clust_empty = copy.deepcopy(clusters)
		for i, point in enumerate(data):
			min_dist = None
			min_clust = None
			for j, centr in enumerate(centroids):
				dist = abs(point[0] - centr[0]) + abs(point[1] - centr[1])
				if(min_dist == None or dist<min_dist):
					min_dist = dist
					min_clust = j
			assignments[i] = min_clust
			clusters[min_clust].append(i)
		#print "iteration: ", r, "assignments: ", assignments
		i = 0
		for c in clusters:
			centroids[i] = [0,0]
			for j in clusters[c]:
				centroids[i][0] = centroids[i][0] + data[j][0]
				centroids[i][1] = centroids[i][1] + data[j][1]
			centroids[i][0] = centroids[i][0]/ float(len(clusters[c]))
			centroids[i][1] = centroids[i][1]/ float(len(clusters[c]))
			i = i+1

		clusters = clust_empty
	return assignments

def train(data):
	# Do some preprocessing
	#listdata = data.tolist()
	time = range(24)
	k = 20 # Number of clusters in location data
	attribute_values = {"Dates":Set(time)}
	loc_clusters = [] # Mapping of location data to different clusters
	for i in range(len(column_names)):
		if column_names[i] in ["Dates","Category","Y","Address"]:
			continue
		if column_names[i] == "X":
			x = data.loc[:,column_names[i]].values.tolist()
			y = data.loc[:,"Y"].values.tolist()
			loc = [[x[i], y[i] ]for i in range(len(data))]
			loc_clusters = kmeans(loc, k, 40)



			attribute_values["location"] = range(k)
		else:
			attribute_values[column_names[i]] = Set(data.loc[:,column_names[i]].values.tolist())

	# Next lets retrieve the classes and their counts
	N = len(data)
	c_count = {}
	c = data.loc[:,"Category"]
	for d in c:
		if( d in c_count):
			c_count[d] = c_count[d] + 1
		else:
			c_count[d] = 1

	




if __name__ == '__main__':
	df = readData("/Users/emmanuj/projects/crime_classification/data/train.csv")
	#print df.loc[:,"Category"]
	#train(df)
	'''
	data = [[-122.429602623594,37.7179038840514],
		[122.411836440259,37.730379007001204],
		[-122.423031175088,37.7854818747419],
		[-122.439037573428,37.776802154003896],
		[-122.365565425353,37.8096707013239],
		[-122.405832474482,37.7857446545609],
		[-122.40381672894699,37.7814638725237]]
	'''
	data = [
		[32.93177219,-117.35794043],
		[32.65197767,-117.1949272],
		[32.76729708,-117.29566966],
		[32.6256116,-116.90717166],
		[32.94487559,-116.98993715],
		[32.63983162,-116.87730954],
		[32.70968519,-116.88596449],
		[32.71965351,-116.99009538],
		[32.53969569,-117.17816327],
		[32.67525318,-117.44272539],
		[32.63746539,-117.26713284],
		[32.59385479,-117.2716769],
		[32.83985972,-116.97698771],
		[32.78680855,-117.01853394],
		[32.73530334,-117.29157678],
		[32.79851557,-117.40142425],
		[32.70086177,-117.28439117],
		[32.75931736,-116.93313824],
		[32.71845842,-117.29344568],
		[32.91665375,-117.04057364]
	]

	train(df)
	#print(kmeans(data, 3, 5))
