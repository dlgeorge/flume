"""
makedtopo.py

make moving topography representing moving headgate at flume

"""

import numpy as np
import pylab
import os
import pdb
import matplotlib.pyplot as pplt
from scipy import interpolate

pi = np.pi
nt = 200*2
nx = 200*2
ny = 440*2
doorz = 2.5
tstart = 0.0
tend =   0.85
tclose = 0.8
fname = os.path.join('topo','flumedoorsTXYZ_new.txt')
T = np.linspace(tstart,tend,nt)
X = np.linspace(0.0,1.0,nx)
Y = np.linspace(-0.1,2.1,ny)

#build gate angle
gate_angle_fname = os.path.join('flume_gateangle.txt')
TLR = np.loadtxt(gate_angle_fname)
TLRav = (TLR[:,0:3]+TLR[:,3:6] + TLR[:,6:9] + TLR[:,9:])/4.0

tgate = TLRav[1:,0]
thetaav = 0.5*(TLRav[1:,1] + TLRav[1:,2])
#append
tgate = np.hstack((tgate,[0.8,tend]))
thetaav = np.hstack((thetaav,[90.0,90.0]))

tp = 0.21
thp = (tp-0.2)**2 * (thetaav[0]/(tgate[0]-0.2)**2)
tgate = np.hstack((tp,tgate))
thetaav = np.hstack((thp,thetaav))

tp = 0.2
thp= 0.0
tgate = np.hstack((tp,tgate))
thetaav = np.hstack((thp,thetaav))

tgate = np.hstack(([tstart],tgate))
thetaav = np.hstack(([0.0],thetaav))



gate_angle_oft = interpolate.interp1d(tgate,thetaav,kind='quadratic',bounds_error=False,fill_value = 90.0)

#pdb.set_trace()
gate_angle = np.linspace(0.,90,nt)
#gate_angle = gate_angle_oft(T)
for tind in xrange(len(T)):
    if T[tind]>= 0.2:
        gate_angle[tind] = gate_angle_oft(T[tind])
    else:
        gate_angle[tind] = 0.0

pplt.plot(tgate,thetaav,'r.',T,gate_angle,'b')
pplt.show()

f = open(fname,'w')


def makedtopo():

    #ZofT = []
    tind = -1
    for t in T:
        tind = tind + 1
        theta = (np.pi/180.0)*gate_angle[tind]
        theta = max(0.0,theta)
        theta = min(0.5*np.pi,theta)
        theta2 = pi/2.0 - theta
        print t
        for y in reversed(Y):
            for x in X:
                z = fofdoor(x,y,t,theta,theta2)
                f.write("%s %s %s %s \n"%(t,x,y,z))
                #ZofT.append([t,x,y,z])
        #if t==tstart:
        #    Z  = np.vstack(ZofT)
        #    np.savetxt(fnametest,Z[:,1:])

    #ZofT = np.vstack(ZofT)
    #np.savetxt(fname,ZofT)
    f.close()

def fofdoor(x,y,t,theta,theta2):

    z = 0.0
    if(t<=tclose)&((x>=0.0)&(x<=1.0)&(y>=0)&(y<=2.0)):

        #left door
        xp = np.cos(theta)*x - np.sin(theta)*y
        yp = np.sin(theta)*x + np.cos(theta)*y
        if (xp>=0.0)&(xp<=0.1)&(yp>= 0.0)&(yp<=1.0):
            z = doorz

        #right door
        ybar = y -2.0
        xp = np.cos(theta2)*x - np.sin(theta2)*ybar
        yp = np.sin(theta2)*x + np.cos(theta2)*ybar
        if (xp>=0.0)&(xp<=1.0)&(yp>= 0.0)&(yp<=0.1):
            z = doorz

    return z




if __name__=='__main__':
    makedtopo()










