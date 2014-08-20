"""
makeaux:

make auxiliary input files

"""

import numpy as np
import geotools.topotools as gt
import pylab
import os
import pdb


def phi(X,Y):

    """
    bed friction angle
    """
    Z = np.ones(np.shape(X))
    Z = 38.0*Z
    deg2rad = np.pi/180.0
    Z = deg2rad*Z

    return Z

def flume_natrelease_phi(X,Y):

    """
    bed friction angle
    variable based on 4th order failsurface
    """

    deg2rad = np.pi/180.0
    depth = 0.65
    alpha3 = 19.0*deg2rad
    sc = 0.65/5.0

    D4 = depth/np.tan(alpha3)
    xp1 = D4
    x = [0.0,-2.0*sc,-5.5*sc,-14.0*sc,-15.0*sc]
    z = [0.65,3.0*sc,1.4*sc,0.1*sc,0.0]

    yind =  np.where((Y[:,0]<=20.0)&(Y[:,0]>=-20.0))[0]
    xp1ind  = np.where((X[0,:]>=x[0])&(X[0,:]<xp1))[0] #ramp
    xparind = np.where((X[0,:]>x[-1])&(X[0,:]<x[0]))[0] #fail surface


    Z = 37.5*np.ones(np.shape(X))
    Z[np.ix_(yind,xp1ind)] = 29.0

    Z = deg2rad*Z

    return Z

def flume_theta(X,Y):

    """
    angle theta in flume
    """
    deg2rad = np.pi/180.0
    flumelen = 90.0
    flumerad = 10.0
    theta1 = 31.0
    theta2 = 3.0

    D2 = flumelen + flumerad*(theta1 - theta2)*deg2rad

    #pdb.set_trace()

    yind =  np.where((Y[:,0]<=20.0)&(Y[:,0]>=-20.0))[0]
    x1ind = np.where(X[0,:]<=flumelen)[0]
    x2ind = np.where((X[0,:]>flumelen)&(X[0,:]<D2))[0]
    x3ind = np.where(X[0,:]>=D2)[0]

    Z=np.zeros(np.shape(X))
    Z[np.ix_(yind,x1ind)] = theta1
    Z[np.ix_(yind,x3ind)] = theta2
    Z[np.ix_(yind,x2ind)] = theta1 - (X[np.ix_(yind,x2ind)]-flumelen)/(deg2rad*flumerad)
    Z = deg2rad*Z

    return Z

def flume_theta_natrelease(X,Y):

    """
    angle theta in flume
    """
    deg2rad = np.pi/180.0
    flumelen = 90.0
    flumerad = 10.0
    theta1 = 31.0
    theta2 = 3.0

    D2 = flumelen + flumerad*(theta1 - theta2)*deg2rad

    #pdb.set_trace()

    yind =  np.where((Y[:,0]<=20.0)&(Y[:,0]>=-20.0))[0]
    x1ind = np.where(X[0,:]<=flumelen)[0]
    x2ind = np.where((X[0,:]>flumelen)&(X[0,:]<D2))[0]
    x3ind = np.where(X[0,:]>=D2)[0]

    Z=np.zeros(np.shape(X))
    Z[np.ix_(yind,x1ind)] = theta1
    Z[np.ix_(yind,x3ind)] = theta2
    Z[np.ix_(yind,x2ind)] = theta1 - (X[np.ix_(yind,x2ind)]-flumelen)/(deg2rad*flumerad)

    #set for natural release profile
    depth =  0.65
    alpha2 = 10.0*deg2rad
    alpha3 = 19.0*deg2rad

    D3 = depth/np.tan(alpha2)
    D4 = depth/np.tan(alpha3)

    x1 = D4
    x0 = 0.0
    xm1 = -D3

    x1ind = np.where((X[0,:]>x0)&(X[0,:]<=x1))[0]
    xm1ind = np.where((X[0,:]>=xm1)&(X[0,:]<=x0))[0]
    Z[np.ix_(yind,x1ind)] = 50.0
    Z[np.ix_(yind,xm1ind)] = 21.0


    Z = deg2rad*Z

    return Z



#phi file
outfile= 'FlumePhi.tt2'
outfile = os.path.join('aux',outfile)
xlower = -15.0
xupper = 160.0
ylower = -4.0
yupper =  6.0
nxpoints = int((xupper-xlower)/0.1) + 1
nypoints = int((yupper-ylower)/0.1) + 1
gt.topo2writer(outfile,phi,xlower,xupper,ylower,yupper,nxpoints,nypoints)


#theta file
outfile= 'FlumeTheta.tt2'
outfile = os.path.join('aux',outfile)
xlower = -15.0
xupper = 160.0
ylower = -4.0
yupper =  6.0
nxpoints = int((xupper-xlower)/0.1) + 1
nypoints = int((yupper-ylower)/0.1) + 1
gt.topo2writer(outfile,flume_theta,xlower,xupper,ylower,yupper,nxpoints,nypoints)



