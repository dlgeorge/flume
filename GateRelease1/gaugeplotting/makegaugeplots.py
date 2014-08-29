
"""
file: makegaugeplots.py

plot time series data from gauges and experiments
Note: this routine assumes that gauges are ordered from lowest x value to highest

"""
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import gaugedata as cg
import string
import pylab
import pdb
from numpy import ma
drytol = 1.e-3

mksize = 1
numfont = 18
myfont = 15
bigfont = 20
myfigsize=(8,4)
myfigsize2=(10,3)
mylinesize = 3
light_grey = (.5,.5,.5,.4)
rho_f = 1100.00
rho_s = 2700.00
theta = 31.0*np.pi/180.0
grav = 9.81
gmod = grav*np.cos(theta)
kappita = 1.e-10
mu = 0.005
alpha = 0.05

def makegaugeplots():

    #flume variables
    flume32 = SGM_data(32)
    flume66 = SGM_data(66)
    flume02 = SGM_data(2)

    (allgaugedata,xgauges,gauge_nums) = getgaugedata()

    # 32 meters
    t32 = flume32[:,0] - 0.2
    h32 = flume32[:,1]
    h32_high = h32 + flume32[:,2]
    h32_low = h32 - flume32[:,2]
    sigma32 = flume32[:,3]
    sigma32_high = sigma32 + flume32[:,4]
    sigma32_low = sigma32 - flume32[:,4]
    pbed32 = flume32[:,5]
    pbed32_high = pbed32 + flume32[:,6]
    pbed32_low = pbed32 - flume32[:,6]
    # 66 meters
    t66 = flume66[:,0] -0.2
    h66 = flume66[:,1]
    h66_high = h66 + flume66[:,2]
    h66_low = h66 - flume66[:,2]
    sigma66 = flume66[:,3]
    sigma66_high = sigma66 + flume66[:,4]
    sigma66_low =  sigma66 - flume66[:,4]
    pbed66 = flume66[:,5]
    pbed66_high = pbed66 + flume66[:,6]
    pbed66_low =  pbed66 - flume66[:,6]

    t02 = flume02[:,0] -0.2
    h02 = flume02[:,1]
    h02_high = h02 + flume02[:,2]
    h02_low  = h02 - flume02[:,2]

#plots----------------------------------------------

    #h at 32 meters
    gdata = cg.selectgauge(32,allgaugedata)
    t = gdata['t']
    h = gdata['q1']
    m = gdata['q4']/(gdata['q1']+1.e-16)
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta)
    plotcomparison(1,t32,h32_low,h32_high,t,h)
    plt.axis([0,20,-0.05,0.28])
    pylab.yticks([0.0,0.05,0.10,0.15,0.20,0.25],('0.0','5.0','10.0','15.0','20.0','25.0'),fontsize=numfont)
    pylab.xticks([.2,5,10,15,19.6],('','','','',''))

    #p at 32 meters
    gdata = cg.selectgauge(32,allgaugedata)
    plotcomparison(2,t32,pbed32_low,pbed32_high,gdata['t'],1.e-3*gdata['q5'])
    #plt.plot(t32,0.5*(sigma32_low + sigma32_high),'g')
    #hydroline, = plt.plot(t,1.e-3*rho_f*grav*np.cos(theta)*h,'r',linewidth=1)
    litholine, = plt.plot(t,1.e-3*sigma,'b',linewidth=1)
    plt.legend(('pore-fluid pressure','lithostatic pressure'))
    plt.axis([0,20,-1.2,4.2])
    pylab.xticks([.2,5,10,15,19.6],('0','5','10','15','20'),fontsize=numfont)
    pylab.yticks([-1,0.0,1.0,2.0,3.0,4.0],('','0.0','1.0','2.0','3.0','4.0'),fontsize=numfont)

    #sigma at 32 meters
    gdata = cg.selectgauge(32,allgaugedata)
    #plotcomparison(3,t32,sigma32_low,sigma32_high,gdata['t'],1.e-3*sigma)
    #plt.axis([0,20,-.5,4.2])


    #h at 66 meters
    gdata = cg.selectgauge(66,allgaugedata)
    plotcomparison(3,t66,h66_low,h66_high,gdata['t'],gdata['q1'])
    plt.axis([0,20,-0.08,0.3])
    t = gdata['t']
    h = gdata['q1']
    m = gdata['q4']/(gdata['q1']+1.e-16)
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta)
    plt.axis([0,20,-0.05,0.28])
    pylab.yticks([0.0,0.05,0.10,0.15,0.20,0.25],('','','','','',''))
    pylab.xticks([.2,5,10,15,19.6],('','','','',''))

    #p at 66 meters
    gdata = cg.selectgauge(66,allgaugedata)
    plotcomparison(4,t66,pbed66_low,pbed66_high,gdata['t'],1.e-3*gdata['q5'])
    #plt.plot(t,1.e-3*rho_f*grav*np.cos(theta)*h,'r')
    plt.plot(t,1.e-3*sigma,'b')
    plt.axis([0,20,-2.0,4.0])
    plt.legend(('pore-fluid pressure','lithostatic pressure'))
    plt.axis([0,20,-1.2,4.2])
    pylab.xticks([.2,5,10,15,19.6],('0','5','10','15','20'),fontsize=numfont)
    pylab.yticks([-1,0.0,1.0,2.0,3.0,4.0],('','','','','',''))

    #sigma at 66 meters
    gdata = cg.selectgauge(66,allgaugedata)
    h = gdata['q1']
    m = gdata['q4']/gdata['q1']
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta)
    #plotcomparison(6,t66,sigma66_low,sigma66_high,gdata['t'],1.e-3*sigma)
    #plt.axis([0,20,-.5,4.2])


    #h at 2 meters
    gdata = cg.selectgauge(2,datafile="fort.gauge")
    t_quad = gdata['t']
    h_quad = gdata['q1']

    l_exp = plotcomparison(0,t02,h02_low,h02_high,[],[])
    lines = plt.plot(t_quad,h_quad,'k',linewidth=mylinesize)

    plt.axis([0,10,-0.08,0.8])
    pylab.yticks([0.0,0.2,0.4,0.6,0.79],('0','20','40','60','80'),fontsize=numfont)
    pylab.xticks([.1, 2, 4, 6, 8, 9.9],('','','','','',''),fontsize=numfont)

    #p at 2 meters
    plt.figure(5,myfigsize2)
    gdata = cg.selectgauge(2,allgaugedata)
    t = gdata['t']
    p = 1.e-3*gdata['q5']
    h = gdata['q1']
    m = gdata['q4']/gdata['q1']
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta)
    plt.plot(t,p,'k',linewidth=mylinesize)
    plt.plot(t,1.e-3*sigma,'b')
    plt.plot(t,1.e-3*rho_f*grav*np.cos(theta)*h,'r')
    plt.legend(('pore-fluid pressure','lithostatic pressure','hydrostatic pressure'))
    plt.axis([0,10,-0.6,9.0])
    pylab.xticks([.1, 2, 4, 6, 8, 9.9],('0','2','4','6','8', '10'),fontsize=numfont)
    pylab.yticks([0,2,4,6,8],('0.0','2.0','4.0','6.0','8.0'),fontsize=numfont)

    #m at 32 meters
    plt.figure(6,myfigsize2)
    gdata = cg.selectgauge(32,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    hu = gdata['q2']
    u = ma.masked_where(h<=drytol, hu/h)
    shear = ma.masked_where(h<=drytol, 2.0*hu/h**2)
    m = ma.masked_where(h<=drytol,gdata['q4']/gdata['q1'])
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta) - p
    delta = 0.01
    S = (shear*0.005)/(2700.*(shear*delta)**2 + sigma)
    m_eqn = 0.64/(1.+np.sqrt(S))
    plt.plot(t,m,'b')
    plt.plot(t,m_eqn,'r')
    plt.legend(('m','m_eqn'))

    #m at 2 meters
    plt.figure(7,myfigsize2)
    gdata = cg.selectgauge(2,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    hu = gdata['q2']
    u = ma.masked_where(h<=drytol, hu/h)
    shear = ma.masked_where(h<=drytol, 2.0*hu/h**2)
    m = ma.masked_where(h<=drytol,gdata['q4']/gdata['q1'])
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta) - p
    delta = 0.01
    S = (shear*0.005)/(2700.*(shear*delta)**2 + sigma)
    m_eqn = 0.64/(1.+np.sqrt(S))
    plt.plot(t,m,'b')
    plt.plot(t,m_eqn,'r')
    plt.legend(('m','m_eqn'))

    #m at 66 meters
    plt.figure(8,myfigsize2)
    gdata = cg.selectgauge(66,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    hu = gdata['q2']
    u = ma.masked_where(h<=drytol, hu/h)
    shear = ma.masked_where(h<=drytol, 2.0*hu/h**2)
    m = ma.masked_where(h<=drytol,gdata['q4']/gdata['q1'])
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta) - p
    delta = 0.01
    S = (shear*0.005)/(2700.*(shear*delta)**2 + sigma)
    m_eqn = 0.64/(1.+np.sqrt(S))
    plt.plot(t,m,'b')
    plt.plot(t,m_eqn,'r')
    plt.legend(('m','m_eqn'))

    #p_eq at 32 meters
    plt.figure(9,myfigsize2)
    gdata = cg.selectgauge(2,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    hu = gdata['q2']
    u = ma.masked_where(h<=drytol, hu/h)
    vnorm = u
    shear = ma.masked_where(h<=drytol, 2.0*hu/h**2)
    m = ma.masked_where(h<=drytol,gdata['q4']/gdata['q1'])
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta) - p
    delta = 0.01
    S = (shear*mu)/(2700.*(shear*delta)**2 + sigma)
    m_eqn = 0.64/(1.+np.sqrt(S))
    compress = alpha/(m*(sigma +  1.e3))
    zeta = 3.0/(compress*h*2.0)  + (rho-rho_f)*rho_f*gmod/(4.0*rho)
    kperm = kappita*np.exp(-(m-0.60)/(0.04))
    krate=-zeta*2.0*kperm/(h*max(mu,1.e-16))
    p_hydro = h*rho_f*gmod
    tanpsi=(m-m_eqn)*np.tanh(shear/0.1)
    p_eq = p_hydro + 3.0*vnorm*tanpsi/(compress*h*krate)
    plt.plot(t,p,'k')
    #plt.plot(t,rho*gmod*h,'r')
    #plt.plot(t,rho_f*gmod*h,'b')
    plt.plot(t,p_eq,'r')
    #plt.legend(('p','p_eqn'))
    #show all plots


    #p at 80 meters
    plt.figure(10,myfigsize2)
    gdata = cg.selectgauge(80,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    m = gdata['q4']/gdata['q1']
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta)
    plt.plot(t,p,'k',linewidth=mylinesize)
    plt.plot(t,sigma,'b')
    plt.plot(t,rho_f*grav*np.cos(theta)*h,'r')
    plt.legend(('pore-fluid pressure','lithostatic pressure','hydrostatic pressure'))
    #plt.axis([0,10,-0.6,9.0])
    #pylab.xticks([.1, 2, 4, 6, 8, 9.9],('0','2','4','6','8', '10'),fontsize=numfont)
    #pylab.yticks([0,2,4,6,8],('0.0','2.0','4.0','6.0','8.0'),fontsize=numfont)

    #m at 80 meters
    plt.figure(11,myfigsize2)
    gdata = cg.selectgauge(80,allgaugedata)
    t = gdata['t']
    p = gdata['q5']
    h = gdata['q1']
    hu = gdata['q2']
    u = ma.masked_where(h<=drytol, hu/h)
    shear = ma.masked_where(h<=drytol, 2.0*hu/h**2)
    m = ma.masked_where(h<=drytol,gdata['q4']/gdata['q1'])
    rho = (1.0-m)*rho_f + m*rho_s
    sigma = h*rho*grav*np.cos(theta) - p
    delta = 0.01
    S = (shear*0.005)/(2700.*(shear*delta)**2 + sigma)
    m_eqn = 0.64/(1.+np.sqrt(S))
    plt.plot(t,m,'b')
    plt.plot(t,m_eqn,'r')
    plt.legend(('m','m_eqn'))

    pylab.show()


def plotcomparison(figno,tf,flumelow,flumehi,tg,gaugevar):
    """
    plot flume data as a shaded bar and geoclaw output
    """
    plt.figure(figno,figsize=myfigsize2)
    a=plt.plot(tg,gaugevar,linewidth=mylinesize,color='k')
    plt.fill_between(tf,flumehi,flumelow,edgecolor=light_grey,facecolor=light_grey)

    return a



def SGM_data(gauge):
    infile = 'SGM_'+str(gauge)+'m.txt'
    flumedata = np.loadtxt(infile)

    return flumedata

def getgaugedata():

    #-----get all gauge data--------------------------
    allgaugedata = cg.fortgaugeread()

    #------find gauge locations and numbers-----------
    setgaugefile='setgauges.data'
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
#---------------------------------------------------


if __name__=='__main__':
    makegaugeplots()