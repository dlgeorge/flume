"""
maketopo:

make some 'artificial' topo surrounding the flume area
this is for use with bed normal coordinates

"""

import numpy as np
import geotools.topotools as gt
import pylab
import os
import pdb

def zero(X,Y):

    Z = np.zeros(np.shape(X))

    return Z

def wallzero(X,Y):
    yind1 =  np.where((Y[:,0]>=-0.5)&(Y[:,0]<=0.0))[0]
    yind2 =  np.where((Y[:,0]>=2.0)&(Y[:,0]<=2.5))[0]
    xind  =  np.where((X[0,:]>=-15.0)&(X[0,:]<=90.0))[0]
    Z = np.zeros(np.shape(X))

    Z[np.ix_(yind1,xind)] = 1.5
    Z[np.ix_(yind2,xind)] = 1.5


    return Z

def flume_natrelease_failsurface(X,Y):

    deg2rad = np.pi/180.0
    depth = 0.65

    alpha2 = 23.2*deg2rad
    alpha3 = 19.0*deg2rad

    D3 = depth/np.tan(alpha2)
    D4 = depth/np.tan(alpha3)

    x0 = 0.0
    xm1 = -D3
    x1 = D4

    yind   = np.where((Y[:,0]<=2.0)&(Y[:,0]>=0.0))[0]
    x1ind  = np.where((X[0,:]>=x0)&(X[0,:]<x1))[0]
    xm1ind = np.where((X[0,:]<x0)&(X[0,:]>xm1))[0]

    #pdb.set_trace()

    Z=np.zeros(np.shape(X))
    Z[np.ix_(yind,x1ind)] = (x1-X[np.ix_(yind,x1ind)])*np.tan(alpha3)
    Z[np.ix_(yind,xm1ind)] = (X[np.ix_(yind,xm1ind)]-xm1)*np.tan(alpha2)


    return Z

def flume_natrelease_failsurface_4thOrder(X,Y):

    deg2rad = np.pi/180.0
    depth = 0.65
    alpha3 = 19.0*deg2rad
    sc = 0.65/5.0


    yind   = np.where((Y[:,0]<=2.0)&(Y[:,0]>=0.0))[0]

    D4 = depth/np.tan(alpha3)
    xp1 = D4
    x = [0.0,-2.0*sc,-5.5*sc,-14.0*sc,-15.0*sc]
    z = [depth,3.0*sc,1.4*sc,0.1*sc,0.0]

    xp1ind  = np.where((X[0,:]>=x[0])&(X[0,:]<xp1))[0] #ramp
    xparind = np.where((X[0,:]>x[-1])&(X[0,:]<x[0]))[0] # fail surface


    #pdb.set_trace()

    Z=np.zeros(np.shape(X))

    Z[np.ix_(yind,xp1ind)] = (xp1-X[np.ix_(yind,xp1ind)])*np.tan(alpha3)

    for i in xrange(len(x)):
        Li=np.ones(np.shape(X))

        for j in xrange(len(x)-1):
            Li = Li * (X-x[i-1-j])/(x[i]-x[i-1-j])

        Z[np.ix_(yind,xparind)] = Z[np.ix_(yind,xparind)] + z[i]*Li[np.ix_(yind,xparind)]

    return Z

def flume_natrelease_failsurface_4thOrder_paper(X,Y):

    deg2rad = np.pi/180.0
    depth = 0.65
    alpha3 = 19.0*deg2rad
    sc = 0.65/5.0


    yind   = np.where((Y[:,0]<=2.0)&(Y[:,0]>=0.0))[0]

    D4 = depth/np.tan(alpha3)
    xp1 = D4
    x = [0.0,-3.5*sc,-7.5*sc,-14.0*sc,-15.0*sc]
    #x = [0.0,-4.0*sc,-8.5*sc,-14.0*sc,-15.0*sc]
    z = [depth,3.0*sc,1.4*sc,0.1*sc,0.0]

    xp1ind  = np.where((X[0,:]>=x[0])&(X[0,:]<xp1))[0] #ramp
    xparind = np.where((X[0,:]>x[-1])&(X[0,:]<x[0]))[0] # fail surface


    #pdb.set_trace()

    Z=np.zeros(np.shape(X))

    Z[np.ix_(yind,xp1ind)] = (xp1-X[np.ix_(yind,xp1ind)])*np.tan(alpha3)

    for i in xrange(len(x)):
        Li=np.ones(np.shape(X))

        for j in xrange(len(x)-1):
            Li = Li * (X-x[i-1-j])/(x[i]-x[i-1-j])

        Z[np.ix_(yind,xparind)] = Z[np.ix_(yind,xparind)] + z[i]*Li[np.ix_(yind,xparind)]

    return Z


def flume_gaterelease_eta(X,Y):

    hopperlen = 4.7
    hmax = 1.9
    hoppertop = 3.3
    topangle = 17.0*np.pi/180.0
    flumeangle = 31.0*np.pi/180.0


    x0 = -hopperlen
    x2 = -hmax*np.cos(0.5*np.pi - flumeangle)

    x1 = x2 - hoppertop*np.cos(flumeangle-topangle)

    x3 = 0.0
    y2 = hmax*np.sin(0.5*np.pi - flumeangle)
    y1 = y2 - hoppertop*np.sin(flumeangle-topangle)
    slope0 = y1/(x1-x0)
    slope1 = (y2-y1)/(x2-x1)
    slope2 = -y2/(x3-x2)

    yind =  np.where((Y[:,0]<=2.0)&(Y[:,0]>=0.0))[0]
    x0ind = np.where((X[0,:]>=x0)&(X[0,:]<x1))[0]
    x1ind = np.where((X[0,:]>=x1)&(X[0,:]<x2))[0]
    x2ind = np.where((X[0,:]>=x2)&(X[0,:]<x3))[0]

    #pdb.set_trace()

    Z=np.zeros(np.shape(X))
    Z[np.ix_(yind,x0ind)] = (X[np.ix_(yind,x0ind)]-x0)*slope0
    Z[np.ix_(yind,x1ind)] =  y1+(X[np.ix_(yind,x1ind)]-x1)*slope1
    Z[np.ix_(yind,x2ind)] = -(x3-X[np.ix_(yind,x2ind)])*slope2

    return Z


def flume_natrelease_eta(X,Y):

    deg2rad = np.pi/180.0
    depth = 0.65

    alpha1 = 31.0*deg2rad
    alpha2 = 23.2*deg2rad
    alpha3 = 19.0*deg2rad

    D1 = depth/np.tan(alpha1)
    D2 = 3.0
    D3 = depth/np.tan(alpha2)
    D4 = depth/np.tan(alpha3)

    x0 = 0.0
    xm1 = -D3
    xm2 = -D3 - D2
    xm3 = -D3 - D2 - D1
    xpos = D4

    yind =  np.where((Y[:,0]<=2.2)&(Y[:,0]>=-0.2))[0]
    x1ind = np.where((X[0,:]>=xm3)&(X[0,:]<xm2))[0]
    x2ind = np.where((X[0,:]>=xm2)&(X[0,:]<xm1))[0]
    x3ind =  np.where((X[0,:]>=xm1)&(X[0,:]<=x0))[0]
    xposind = np.where((X[0,:]>x0)&(X[0,:]<xpos))[0]
    #pdb.set_trace()

    Z=np.zeros(np.shape(X))
    Z[np.ix_(yind,x1ind)] = (X[np.ix_(yind,x1ind)]-xm3)*np.tan(alpha1)
    Z[np.ix_(yind,x2ind)] = depth
    Z[np.ix_(yind,x3ind)] = depth
    Z[np.ix_(yind,xposind)] = (xpos -X[np.ix_(yind,xposind)])*np.tan(alpha3)

    return Z


#flat topo
outfile= 'ZeroTopo.tt2'
outfile = os.path.join('topo',outfile)
xlower = -10.0
xupper = 180
ylower = -10.0
yupper =  10.0
nxpoints = int((xupper-xlower)/0.1) + 1
nypoints = int((yupper-ylower)/0.1) + 1
gt.topo2writer(outfile,zero,xlower,xupper,ylower,yupper,nxpoints,nypoints)

#wall topo
outfile= 'Wall1Topo.tt2'
outfile = os.path.join('topo',outfile)
xlower = -15.0
xupper =  90.0
ylower =  -0.5
yupper =  0.0
nxpoints = int((xupper-xlower)/0.05) + 1
nypoints = int((yupper-ylower)/0.05) + 1
gt.topo2writer(outfile,wallzero,xlower,xupper,ylower,yupper,nxpoints,nypoints)

#wall topo
outfile= 'Wall2Topo.tt2'
outfile = os.path.join('topo',outfile)
xlower = -15.0
xupper =  90.0
ylower = 2.0
yupper = 2.5
nxpoints = int((xupper-xlower)/0.05) + 1
nypoints = int((yupper-ylower)/0.05) + 1
gt.topo2writer(outfile,wallzero,xlower,xupper,ylower,yupper,nxpoints,nypoints)



#initial surface
outfile= 'FlumeQinit.tt2'
outfile = os.path.join('topo',outfile)
xlower = -6.0
xupper =  1.0
ylower = -0.5
yupper =  2.5
nxpoints = int((xupper-xlower)/0.01) + 1
nypoints = int((yupper-ylower)/0.01) + 1
gt.topo2writer(outfile,flume_natrelease_eta,xlower,xupper,ylower,yupper,nxpoints,nypoints)

#failsurface topo
outfile= 'NatReleaseTopo.tt2'
outfile = os.path.join('topo',outfile)
xlower =  -4.0
xupper =  2.0
ylower =  0.0
yupper =  2.0
nxpoints = int((xupper-xlower)/0.01) + 1
nypoints = int((yupper-ylower)/0.01) + 1
gt.topo2writer(outfile,flume_natrelease_failsurface_4thOrder_paper,xlower,xupper,ylower,yupper,nxpoints,nypoints)