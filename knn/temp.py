import numpy as np

def main():
 x=0
 if x < 5:
  Dates,Category,Descript,DayOfWeek,PdDistrict,Address,X,Y=np.loadtxt('train.csv',delimiter=',',unpack=True,dtype='str', usecols = (0,1,2,3,4,6,7,8))
  for eachDate in Dates:
   print eachDate
   saveLine=eachDate+'\n'
   saveFile=open('newCSV.csv','a')
   saveFile.write(saveLine)
   saveFile.close()
   x+=1
			
		
