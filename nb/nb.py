'''
Functions for training data using naive bayes method
'''

import pandas as pd
import sys
from sets import Set
import random
import copy

column_names = ['Dates','Category','DayOfWeek','PdDistrict','Address','X','Y']
column_names_test = ['Dates','DayOfWeek','Address','PdDistrict','X','Y']

def readData(filename, colnames= column_names):
	df = pd.read_csv(filename, quoting=1, usecols=colnames)
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
	return [assignments, centroids]
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

	return [train_data, c_count, cat_id]

def saveKmeans(loc_clusters, centroids, path=""):
	assignments ="kmeans.txt"
	centr = "centroids.txt"
	if (path != ""):
		assignments = path+"/"+assignments
		centr = path + "/"+ centr

	with open(assignments,"w") as f:
		for i in loc_clusters:
			f.write(str(i)+"\n")

	with open(centr,"w") as f:
		for point in centroids:
			f.write(str(point[0]) +":"+ str(point[1])+"\n")


def readKmeans(path=""):
	assn ="kmeans.txt"
	centr = "centroids.txt"
	if (path != ""):
		assn = path+"/"+ assn
		centr = path +"/"+ assn

	clust = []
	with open(assn,'r') as f:
		for line in f:
			clust.append(int(line.strip()))

	centroids = []
	with open(centr, 'r') as f:
		for line in f:
			l = line.strip().split(":")
			centroids.append([float(l[0]), float(l[1])])

	return [clust, centroids]

def mapLocationToCluster(x, y, centroids):
	min_dist = None
	min_clust = None
	for i ,centr in enumerate(centroids):
		dist = abs(float(x) - centr[0]) + abs(float(y) - centr[1])
		if(min_dist == None or dist < min_dist):
			min_dist = dist
			min_clust = i
	return min_clust

def saveTrainingData(train_data, class_data, cat_id, path=""):
	t_file = "train_data.txt"
	c_file = "cat_data.txt"
	cat_file = "cat_id.txt"

	if (path !=""):
		t_file = path+"/"+ t_file
		c_file = path+"/"+c_file
		cat_file = path+"/"+cat_file

	with open(t_file,'w') as f:
		for class_i in train_data:
			#print class_i
			l = []
			for key,val in class_i.iteritems(): # [{},{},]
				l.append(key +":"+str(val))
			f.write(" ".join(l) +"\n")

	with open(c_file, 'w') as f:
		for key, val in class_data.iteritems():
			f.write(key+":"+str(val)+"\n")

	with open(cat_file, 'w') as f:
		for key, val in cat_id.iteritems():
			f.write(key+":"+str(val)+"\n")

def loadTrainingData(path=""):
	t_file = "train_data.txt"
	c_file = "cat_data.txt"
	cat_file = "cat_id.txt"

	if (path !=""):
		t_file = path+"/"+ t_file
		c_file = path+"/"+c_file
		cat_file = path+"/"+cat_file

	train_data = []
	class_data = {}
	cat_id = {}
	with open(t_file,'r') as f:
		for line in f:
			t_map = {}
			values = line.strip().split(" ")
			for val in values:
				v = val.strip().split(":")
				t_map[v[0]] = int(v[1])
			train_data.append(t_map)

	with open(c_file, 'r') as f:
		for line in f:
			val = line.strip().split(":")
			class_data[val[0]] = int(val[1])

	with open(cat_file, 'r') as f:
		for line in f:
			val = line.strip().split(":")
			cat_id[val[0]] = int(val[1])

	return [train_data, class_data, cat_id]

def classify(train_data, class_data, test_data, loc_clusters, cat_id, addr_clust, centroids, n):
	n = float(n) #number of training instance
	ad_skipped = 0
	results = []
	for index, row in test_data.iterrows():

		i = 0
		time_i =  "time_" + str(int(row["Dates"].split(" ")[1].split(":")[0])) # 2014-12-24 05:20:00 --> time_5 (hour part of the time)
		lat_long = str(row['X'])+":"+str(row['Y'])
		loc_i = "loc_"

		if(lat_long in loc_clusters):
			loc_i =  loc_i + str(loc_clusters[lat_long]) # Assuming Location is in the data. TODO: for later can map location to the centroids of clusters use the closest as the location.
		elif( row["Address"] in addr_clust):
			loc_i = loc_i + str(addr_clust[row["Address"]])
		else:
			loc_i = loc_i + str(mapLocationToCluster(float(row['X']), float(row['Y']), centroids))

		PdDistrict = row["PdDistrict"]
		dofw = row["DayOfWeek"]
		instance_class = {}
		for key, val in class_data.iteritems():
			val = float(val)
			p_class = float(val/n) * float(train_data[cat_id[key]][time_i]/val) * float(train_data[cat_id[key]][loc_i]/val) * float(train_data[cat_id[key]][PdDistrict]/val) * float(train_data[cat_id[key]][dofw]/val)
			instance_class[key] = p_class
			i += 1
		results.append(instance_class)
	return results

if __name__ == '__main__':

	if(len(sys.argv) < 2):
		print "Usage: python nb.py <data_file>"
		sys.exit(0)
	train_file = sys.argv[1] # trainingdata path
	test_file = sys.argv[2]

	print "Reading training data... "
	data = readData(train_file) #readData("/Users/emmanuj/projects/crime_classification/data/train.csv")

	x = data.loc[:,"X"].values.tolist()
	y = data.loc[:,"Y"].values.tolist()
	addr = data.loc[:,"Address"].values.tolist()
	loc = [[x[i], y[i] ]for i in range(len(data))]

	print "Computing kmeans... k = 100"
	k = 100
	#clust, centroids = kmeans(loc, k, 20)

	print "Writing kmeans data to file"
	#saveKmeans(clust, centroids)

	#print "Reading kmeans..."
	clust, centroids = readKmeans()

	print "Mapping locations to clusters... "
	#Map address and locations to their clusters
	loc_key = [str(x[i]) +":"+ str(y[i]) for i in range(len(data))]
	addr_clust = dict(zip(addr, clust))
	loc_clusters = dict(zip(loc_key, clust))

	print "Training... "
	t = train(data,clust, k)

	#print "Generating model ... "
	saveTrainingData(t[0],t[1], t[2])
	#print "Loading model ... "
	#t = loadTrainingData()

	print "Loading test data ... "
	test_d = readData(test_file, column_names_test)
	print "Classifying... "
	classify(t[0], t[1], test_d, loc_clusters,t[2], addr_clust, centroids, len(data))
