import numpy as np
import math
import operator
from collections import OrderedDict
Day1,Time1,DayOfWeek1,PdDistrict1,X1,Y1,Category=np.loadtxt('testdata2.csv',delimiter=',',unpack=True,dtype='str',skiprows=1, usecols = (0,1,2,3,4,5,6)) 
Id,Day2,Time2,DayOfWeek2,PdDistrict2,X2,Y2=np.loadtxt('newtestdata2.csv',delimiter=',',unpack=True,dtype='str',skiprows=1, usecols = (0,1,2,3,4,5,6)) 
x=0
distance=0
ldict={}
mainldict=OrderedDict()
for record in Id:
	y=0
	for day in Day1:
		#print 'x='+str(x)+' '+Day2[x]+' '+str(y)+' '+Day1[y]
		distance=(pow((float(Day2[x])-float(Day1[y])), 2)+pow((float(Time2[x])-float(Time1[y])), 2)+pow((float(DayOfWeek2[x])-float(DayOfWeek1[y])), 2)+pow((float(Y2[x])-float(Y1[y])), 2)+pow((float(X2[x])-float(X1[y])), 2))
		if PdDistrict2[x] is PdDistrict1[y]:
			distance=distance+1
		distance=math.sqrt(distance)
		ldict[y]=distance
		y+=1
	ldict=OrderedDict(sorted(ldict.items(),key=operator.itemgetter(1)))
	#ldict={}
	#for key in ldict:
		#print str(x)+' '+str(key)+' '+str(ldict[key])+'\n'
	#store only first k sorted categories
	i=0
	k=5 #for now
	for r in ldict:
		if i<k:
			if i==0:
				#print r
				mainldict[x]=[(r,Category[r])]
			else: 
				mainldict[x].append((r,Category[r]))
		i+=1
	x+=1
for key in mainldict:
	print str(key)+' '+str(mainldict[key])+'\n'
		