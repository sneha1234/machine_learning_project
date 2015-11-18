import numpy as np
x=0
saveLine='Day,Time,DayOfWeek,PdDistrict,X,Y,Category'+'\n'
saveFile=open('testdata.csv','a')
saveFile.write(saveLine)
saveFile.close()
saveFile=open('testdata2.csv','a')
saveFile.write(saveLine)
saveFile.close()
dayL=[]
minL=[]
xL=[]
yL=[]
dict = {'1':31, '2':59, '3':90,'4':120,'5':151,'6':181,'7':212,'8':243,'9':273,'10':304,'11':334,'12':365};
Dates,Category,DayOfWeek,PdDistrict,X,Y=np.loadtxt('new.csv',delimiter=',',unpack=True,dtype='str',skiprows=1, usecols = (0,1,2,3,4,5)) 
for eachDate in Dates:
	newval=eachDate.split(' ')
	date=newval[0].split('/')
	time=newval[1].split(':')
	day=dict[date[0]]+int(date[1])
	ydiff=int(date[2])-2003
	if ydiff > 0:
		if date[2]==2004 or date[2]==2008 or date[2]==2012:
			day=day+(366*ydiff)
		else:
			day=day+(365*ydiff)
	minutes=int(time[0])*60 + int(time[1])
	saveLine=str(day)+','+str(minutes)+','+DayOfWeek[x]+','+PdDistrict[x]+','+X[x]+','+Y[x]+','+Category[x]+'\n' 
	saveFile=open('testdata.csv','a')
	saveFile.write(saveLine)
	saveFile.close()
	dayL.append(day)
	minL.append(minutes)
	xL.append(float(X[x]))
	yL.append(float(Y[x]))
	x+=1	
hday=max(dayL)
lday=min(dayL)
hmin=max(minL)
lmin=min(minL)
xmin=min(xL)
xmax=max(xL)
ymin=min(yL)
ymax=max(yL)
print 'max day: '+str(hday)+' min day:'+str(lday)+' max minutes:'+str(hmin)+' min minutes:'+str(lmin)
###########Normalization####################
Date,Time,DayOfWeek,PdDistrict,X,Y,Category=np.loadtxt('testdata.csv',delimiter=',',unpack=True,dtype='str',skiprows=1, usecols = (0,1,2,3,4,5,6)) 
y=0
for eachDay in Date:
	fday=((float(int(eachDay)))-lday)/(hday-lday)
	ftime=((float(int(Time[y])))-lmin)/(hmin-lmin)
	dayofweek=((float(DayOfWeek[y]))-1)/6
	xlat=((float(X[y]))-xmin)/(xmax-xmin)
	ylong=((float(Y[y]))-ymin)/(ymax-ymin)
	saveline2=str(fday)+','+str(ftime)+','+str(dayofweek)+','+PdDistrict[y]+','+str(xlat)+','+str(ylong)+','+Category[y]+'\n' 
	saveFile2=open('testdata2.csv','a')
	saveFile2.write(saveline2)
	saveFile2.close()
	y+=1