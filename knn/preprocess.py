import csv
import re
f = open("train.csv","rb")
csv_f = csv.reader(f)
new_data=[]
list2=[]
count=0
stop=[]
file=open("stopwords.txt","rb")
new_reader=csv.reader(file)
for line in new_reader:
	for item in line:
		stop.append(item)
#print stop
for row in csv_f:
    if count!=0 :
	new_data.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],count])
    count=count+1
f.close()
for listset in new_data:
	temp=listset[6].split()
	#print temp
	for item in temp:
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", item)
		if(item.lower() in stop or val is None):
			continue
		else:
			list2.append(item.lower())
set3=set(list2)
list2=list(set3)
print len(list2)
print list2 