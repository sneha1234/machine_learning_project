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
			if not clusters[c]: continue
			centroids[i] = [0,0]
			for j in clusters[c]:
				centroids[i][0] = centroids[i][0] + data[j][0]
				centroids[i][1] = centroids[i][1] + data[j][1]
			centroids[i][0] = centroids[i][0]/ float(len(clusters[c]))
			centroids[i][1] = centroids[i][1]/ float(len(clusters[c]))
			i = i+1

		clusters = clust_empty
	return assignments
def flattenAttributes(data, assignments, k):
	values = []
	#lets process time
	for i in range(24):
		values.append("time_"+str(i))

	#next location
	for i in range(k):
		values.append("loc_"+str(i))

	#list of unique districts
	PdDistricts = pd.unique(data.PdDistrict.ravel())
	values += PdDistricts.tolist()

	#list of unique day of week
	dofw = pd.unique(data.DayOfWeek.ravel())
	values += dofw.tolist()

	return values

def train(data, loc_clusters, k):
	# Do some preprocessing
	#listdata = data.tolist()
	attr = flattenAttributes(data, loc_clusters, k)
	# Next lets retrieve the classes and their counts
	N = len(data)
	c_count = {}
	c = data.loc[:,"Category"]
	cat_id = {}
	i =0
	for d in c:
		if( d in c_count):
			c_count[d] = c_count[d] + 1
		else:
			c_count[d] = 1
			cat_id[d] = i
			i = i+1
	# initialize all probalities to 0
	probs = dict(zip(attr, [0]* len(attr)))
	train_data = []

	for i in range(len(cat_id)):
		train_data.append(copy.deepcopy(probs))

	#go through rows in data and compute count
	for index, row in data.iterrows():
		class_i = row["Category"]
		#time
		time_i =  "time_" + str(int(row["Dates"].split(" ")[1].split(":")[0])) # 2014-12-24 05:20:00 --> time_5 (hour part of the time)
		train_data[cat_id[class_i]][time_i]+=1

		#day of week
		train_data[cat_id[class_i]][row["DayOfWeek"]]+=1

		#PdDistrict
		train_data[cat_id[class_i]][row["PdDistrict"]]+=1

		#location
		loc_i =  "loc_" + str(loc_clusters[index])
		train_data[cat_id[class_i]][loc_i]+=1

	return [train_data, c_count]

def saveKmeans(loc_clusters, filename):
	with open(filename,"w") as f:
		for i in loc_clusters:
			f.write(str(i)+"\n")

def readKmeans(path):
	clust = []
	with open(filename,'r') as f:
		for line in f:
			clust.append(int(line.strip()))
	return clust

def saveTraingingData(train_data, class_data, path):
	t_file = path ==""?"train_data.txt":path+"/train_data.txt"
	c_file = path ==""?"cat_data.txt":path+"/cat_data.txt"
	with open(t_file,'w') as f:
		for class_i in train_data:
			l = []
			for key,val in class_i:
				l.append(key +":"+str(val))
			f.write(" ".join(l) +"\n")
	with open(c_file, 'w') as f:
		for key, val in class_data:
			f.write(key+":"+str(val)+"\n")

def loadTrainingData(filename):
	pass

def classify():


if __name__ == '__main__':
	filename = sys.argv[1]
	if(len(sys.argv) < 2):
		print "Usage: python nb.py <data_file>"
		sys.exit(0)

	data = readData(filename) #readData("/Users/emmanuj/projects/crime_classification/data/train.csv")

	print train(data, [], 10)
	sys.exit(0)
	x = data.loc[:,"X"].values.tolist()
	y = data.loc[:,"Y"].values.tolist()
	loc = [[x[i], y[i] ]for i in range(len(data))]
	k = 10
	loc_clusters = kmeans(loc, k, 5)


	clust_count = {}
	for i in range(k):
		clust_count[i] = 0

	for i in range(len(data)):
		clust_count[loc_clusters[i]] = clust_count[loc_clusters[i]] + 1
		#print x[i], y[i], loc_clusters[i]
	for key in clust_count:
		print key, clust_count[key]

	sys.exit(0)

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

	#train(df)
	print(kmeans(data, 2, 5))
