"""
file: makegaugeplots.py

plot time series data from gauges and experiments

"""
import setrun
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import clawtools.gaugedata as cg
import numpy.ma as ma
import string
import pdb

tfinal = 16.0
setrundata = setrun.setrun()
m0 = setrundata.digdata.m0
mu  = setrundata.digdata.mu
rho_s = setrundata.digdata.rho_s
rho_f = setrundata.digdata.rho_f
grav = setrundata.geodata.gravity
kappita = setrundata.digdata.kappita
theta = 3.0*np.pi/180.
mcrit = setrundata.digdata.m_crit
delta = setrundata.digdata.delta
c1 = setrundata.digdata.c1
phi = 32.0
ramp = np.deg2rad(10.0)

def plotgauges():

    datafile = "_output/fort.gauge"
    setdatafile = "_output/setgauges.data"
    (allgaugedata,xgauges,gauge_nums) = getgaugedata(datafile,setdatafile)

    flumedata = getflumedata()

    gdata874 = cg.selectgauge(800,allgaugedata)

    t = gdata874['t']
    p = gdata874['q5']
    hu = gdata874['q2']
    h = gdata874['q1']
    hm = gdata874['q4']

    m = hm/h
    v = hu/h

    vf = flumedata['v']
    rf = flumedata['r']
    hf = flumedata['h']

    rho = rho_s*m + (1.0-m)*rho_f
    g = grav
    litho = g*rho*h
    eff = litho - p
    liq = eff/litho
    tan_phieff = liq*(np.tan(np.deg2rad(phi)))
    phieff = np.rad2deg(np.arctan(tan_phieff))
    alpha = 1./np.sqrt(2.)
    froude = v**2/(g*h)
    H = h/2 + (alpha*v + 0.5*g*h/v)**2/(grav*(1.0+ tan_phieff/np.tan(ramp)))

    #pdb.set_trace()
    plt.figure(1)
    plt.title('Velocity')
    plt.plot(t,v)
    plt.plot(vf[:,0],vf[:,1],'ro')

    plt.figure(2)
    plt.title('Depth')
    plt.plot(t,h)
    plt.plot(hf[:,0],hf[:,1],'ro')

    plt.figure(3)
    plt.title('Effective Friction Angle')
    plt.plot(t,phieff)

    plt.figure(4)
    plt.title('Runup prediction')
    plt.plot(t,H)
    plt.axis([0.,25.,0.0,5])

    plt.figure(5)
    plt.plot(t,froude)

    plt.show()

def getgaugedata(datafile="fort.gauge",setgaugefile="setgauges.data"):

    #-----get all gauge data--------------------------
    allgaugedata = cg.fortgaugeread(datafile,setgaugefile)

    #------find gauge locations and numbers-----------
    #setgaugefile='setgauges.data'
    fid=open(setgaugefile)
    inp='#'
    while inp == '#':
        inpl=fid.readline()
        inp=inpl[0]

    inp = fid.readline()
    mgauges=int(inp.split()[0])
    gaugelocs=[]
    linesread=0
    while linesread < mgauges :
        row=string.split(fid.readline())
        if row!=[]:
            gaugelocs.append(row)
            linesread=linesread+1
    fid.close()

    xgauges=[]
    gauge_nums=[]
    for gauge in xrange(mgauges):
        xgauges.append(allgaugedata[gauge]['x'])
        gauge_nums.append(allgaugedata[gauge]['gauge'])

    return allgaugedata,xgauges,gauge_nums

def getflumedata():
    """
    load the flume experimental data
    return a dictionary with variables
    """

    flumedir = 'flume'
    hinfile = os.path.join(flumedir,'h_1994_06_23.txt')
    rinfile = os.path.join(flumedir,'r_1994_06_23.txt')
    vinfile = os.path.join(flumedir,'v_1994_06_23.txt')

    datah = np.loadtxt(hinfile,skiprows=2)
    datar = np.loadtxt(rinfile,skiprows=2)
    datav = np.loadtxt(vinfile,skiprows=2)

    flumedata = {}
    flumedata['h'] = datah
    flumedata['r'] = datar
    flumedata['v'] = datav

    return flumedata





#---------------------------------------------------------
if __name__=='__main__':
    plotgauges()