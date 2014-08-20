"""
file: makegaugeplots.py

plot time series data from gauges and experiments
Note: this routine assumes that gauges are ordered from lowest x value to highest

"""
import setrun
import matplotlib as mp
import matplotlib.pyplot as pplt
import os
import numpy as np
import clawtools.gaugedata as cg
import numpy.ma as ma
import string
import pdb

tfinal = 3 #6.75
setrundata = setrun.setrun()
m0 = setrundata.digdata.m0
mu  = setrundata.digdata.mu
rho_s = setrundata.digdata.rho_s
rho_f = setrundata.digdata.rho_f
theta = np.pi*31./180.
grav = setrundata.geodata.gravity
kappita = setrundata.digdata.kappita
theta = 31.0*np.pi/180.
mcrit = setrundata.digdata.m_crit
delta = setrundata.digdata.delta
c1 = setrundata.digdata.c1


gp=grav*np.cos(theta)
x0Lagrangian=[0.0975 -2.9, -2.3 + 0.65*np.tan(theta), -1.3 + 0.0512, -3.5 + 0.65*np.tan(theta),-2.3 + 0.10*np.tan(theta),-2.3 + 0.30*np.tan(theta),-2.3 + 0.50*np.tan(theta)]
T = np.linspace(0,tfinal,5000)
fntsize = 24
tickfntsize = 18
lnwidth= 1.8
mksize = 6


toffset =  2781.1
pinit_t = -.1


def plotLagrangian():
    fignumber = 1
    figuresize = (8,5)

    flumedata = getflumedata()
    (allgaugedata,xgauges,gauge_nums) = getgaugedata()
    for nx in xrange(len(x0Lagrangian)):
        x0 = x0Lagrangian[nx]
        Xoft = Lagrangian_Xoft(allgaugedata,xgauges,gauge_nums,x0,T)
        (Hoft,Uoft,Voft,Moft,Poft) = Lagrangian_Qoft(allgaugedata,xgauges,gauge_nums,Xoft,T)
        rho = (1.0-Moft)*rho_f + Moft*rho_s
        Litho = Hoft*gp*rho
        hydro = rho_f*gp*Hoft
        sigbed = Litho - Poft
        sigbed = sigbed.clip(min = 0.0)
        compress = 1./(sigbed + 1.e5)
        kperm = kappita*np.exp(-(Moft-0.6)/(0.04))
        Dilation_rate = -(2.0*kperm/(Hoft*mu))*(Poft - hydro)
        vnorm = np.sqrt(Uoft**2 + Voft**2)
        UoverD = np.ma.masked_where(Dilation_rate==0.0,vnorm/Dilation_rate)
        shear = 2.0*vnorm/Hoft
        sigbedc = rho_s*(delta*shear)**2
        Iv = mu*shear/(sigbed + sigbedc)
        meqn = mcrit/(1.0 + np.sqrt(Iv))
        dilatancy_angle = np.arctan(np.tanh(vnorm/0.1)*c1*(Moft - meqn))

        #pdb.set_trace()
        Lhead_litho = rho*Hoft/rho_f
        Lhead = Poft/(rho_f*gp)
        relP = Poft/Litho

        (Poft_10,Poft_30,Poft_50) = pressureprofile(Poft,Hoft)
        Lhead_10 = Poft_10/(rho_f*gp)
        Lhead_30 = Poft_30/(rho_f*gp)
        Lhead_50 = Poft_50/(rho_f*gp)


        Xoft = np.hstack((Xoft[0],Xoft))
        Lhead_10 = np.hstack((Lhead_10[0],Lhead_10))
        Lhead_30 = np.hstack((Lhead_30[0],Lhead_30))
        Lhead_50 = np.hstack((Lhead_50[0],Lhead_50))
        Lhead = np.hstack((Lhead[0],Lhead))
        Tp = np.hstack((-4,T))

        if nx==0:
            fignumber = 1

            pplt.figure(fignumber,figsize=figuresize)

            pplt.plot(Tp-pinit_t,Xoft,'c',linewidth=lnwidth)
            pplt.plot(flumedata['time_adjusted']-toffset,x0+flumedata['displacement_upper'],'c--',linewidth=lnwidth)
            pplt.axis([-0.05,2.05,-4.0,2.0],fontsize=fntsize)
            #pplt.xticks(np.arange(0,4),('0.0','1.0','2.0','3.0'),fontsize=tickfntsize)
            pplt.xticks(fontsize=tickfntsize)
            pplt.yticks(fontsize=tickfntsize)
            pplt.ylabel('distance from wall (m)',fontsize = fntsize)
            pplt.xlabel('time (s)',fontsize = fntsize)
            #pplt.gcf().subplots_adjust(bottom=0.15)

        elif nx==1:
            fignumber = 2
            pplt.figure(fignumber,figsize=figuresize)
            pplt.plot(Tp-pinit_t,Lhead*100,'r-',linewidth=lnwidth)
            #pplt.plot(T-pinit_t,Lhead_litho,'b.',markevery=50,markersize=mksize)
            pplt.plot(flumedata['time_adjusted']-toffset,flumedata['pressure_middle'],'r--',linewidth=lnwidth)

            pplt.axis([-0.05,2.05,-20.,140])
            #pplt.xticks(np.arange(0,4),('0.0','1.0','2.0','3.0'),fontsize=tickfntsize)
            pplt.xticks(fontsize=tickfntsize)
            pplt.yticks(fontsize=tickfntsize)
            pplt.ylabel('pressure head (cm)',fontsize = fntsize)
            pplt.xlabel('time (s)',fontsize = fntsize)
            #pplt.gcf().subplots_adjust(bottom=0.15)
            pplt.tight_layout()

        elif nx==2:
            fignumber = 1
            pplt.figure(fignumber,figsize=figuresize)

            pplt.plot(Tp-pinit_t,Xoft,'m',linewidth=lnwidth)
            pplt.plot(flumedata['time_adjusted']-toffset,x0+flumedata['displacement_lower'],'m--',linewidth=lnwidth)


            pplt.tight_layout()

        elif nx==3:
            """
            fignumber = 10
            pplt.figure(fignumber,figsize=figuresize)
            pplt.plot(T-pinit_t,Lhead,'r-',linewidth=lnwidth)
            pplt.plot(T-pinit_t,Lhead_litho,'b.',markevery=50,markersize=mksize)
            pplt.plot(flumedata['time_adjusted']-toffset,1.e-2*flumedata['pressure_upper'],'r--',linewidth=lnwidth)

            pplt.axis([-.3,3.2,-0.2,1.3])
            pplt.xticks(np.arange(0,4),('0.0','1.0','2.0','3.0'),fontsize=tickfntsize)
            pplt.yticks(fontsize=tickfntsize)
            pplt.ylabel('Pressure head (m)',fontsize = fntsize)
            pplt.xlabel('Time (s)',fontsize = fntsize)
            #pplt.gcf().subplots_adjust(bottom=0.15)
            pplt.tight_layout()
            """
        elif nx>3:
            fignumber =2
            pplt.figure(fignumber,figsize=figuresize)
            if nx == 5:
                pplt.plot(Tp-pinit_t,Lhead_30*100,'g-',linewidth=lnwidth)
                pplt.plot(flumedata['time_adjusted']-toffset,flumedata['pressure_middle_30'],'g--',linewidth=lnwidth)
            elif nx ==6:
                pplt.plot(Tp-pinit_t,Lhead_50*100,'b-',linewidth=lnwidth)
                pplt.plot(flumedata['time_adjusted']-toffset,flumedata['pressure_middle_50'],'b--',linewidth=lnwidth)

    pplt.show()

def plotEulerian():

    gdata1=cg.selectgauge(1,allgaugedata)
    t = gdata1['t']
    p = gdata1['q5']
    hu = gdata1['q2']
    h = gdata1['q1']
    hm = gdata1['q4']

    m = hm/h

    rho_s = 2700.
    rho_f = 1000.

    rho = rho_s*m + (1.0-m)*rho_f
    head = p/(rho_f*gp)

    pplt.figure(1)
    pplt.plot(t,head,t,h,t,hu)

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



def Lagrangian_Xoft(allgaugedata,xgauges,gauge_nums,x0,t):

    Xoft=np.array([])
    dt = t[1]-t[0]
    Xoft=np.hstack((Xoft,x0))

    for nt in xrange(len(t)-1):
        tn = t[nt]
        tnp= t[nt+1]
        x = Xoft[nt]
        #find appropriate gauges
        if xgauges[-1]>x:
            ind = np.where(xgauges>x)[0][0]
        else:
            ind = -1
        indm1 = max(ind-1,0)
        gaugeupper=gauge_nums[ind]
        gaugelower=gauge_nums[indm1]
        gdataupper = cg.selectgauge(gaugeupper,allgaugedata)
        gdatalower = cg.selectgauge(gaugelower,allgaugedata)
        #find position of gauges relative to x(tn)
        xlower=gdatalower['x']
        xupper=gdataupper['x']
        dx = xupper - xlower
        if dx > 0:
            chi = (x-xlower)/dx
        else:
            chi = 1.0
        tlower = gdatalower['t']
        tupper = gdataupper['t']
        #spatially intepolated velocity at tn and xn
        indlower = np.where(tlower>tn)[0][0]
        indupper = np.where(tupper<=tn)[-1][-1]
        uupper = gdataupper['q2'][indupper]/gdataupper['q1'][indupper]
        ulower= gdatalower['q2'][indlower]/gdatalower['q1'][indlower]
        if gdataupper['q1'][indupper]==0.0:
            uupper=0.0
            chi = 0.0
        if gdatalower['q1'][indlower]==0.0:
            ulower=0.0
            chi = 1.0
        u = chi*uupper + (1.0-chi)*ulower
        x = x + u*dt
        #spatially intepolated velocity at tn and xn
        #find appropriate gauges at tn + 0.5dt
        if xgauges[-1]>x:
            ind = np.where(xgauges>x)[0][0]
        else:
            ind = -1
        indm1 = max(ind-1,0)
        gaugeupper=gauge_nums[ind]
        gaugelower=gauge_nums[indm1]
        gdataupper = cg.selectgauge(gaugeupper,allgaugedata)
        gdatalower = cg.selectgauge(gaugelower,allgaugedata)
        #find position of gauges relative to x(tn + 0.5dt)
        xlower=gdatalower['x']
        xupper=gdataupper['x']
        dx = xupper - xlower
        if dx > 0:
            chi = (x-xlower)/dx
        else:
            chi = 1.0
        tlower = gdatalower['t']
        tupper = gdataupper['t']
        #spatially intepolated velocity at tn and xn
        indlower = np.where(tlower>tn+0.5*dt)[0][0]
        indupper = np.where(tupper<=tn+0.5*dt)[-1][-1]
        uupper = gdataupper['q2'][indupper]/gdataupper['q1'][indupper]
        ulower= gdatalower['q2'][indlower]/gdatalower['q1'][indlower]
        if gdataupper['q1'][indupper]==0.0:
            uupper=0.0
            chi = 0.0
        if gdatalower['q1'][indlower]==0.0:
            ulower=0.0
            chi = 1.0
        u2 = chi*uupper + (1.0-chi)*ulower
        x = Xoft[nt] + 0.5*(u + u2)*dt

        #pdb.set_trace()
        Xoft=np.hstack((Xoft,x))

    return Xoft

def Lagrangian_Qoft(allgaugedata,xgauges,gauge_nums,Xoft,t):
    Hoft = np.ones(np.shape(Xoft))
    Poft = np.ones(np.shape(Xoft))
    Moft = np.ones(np.shape(Xoft))
    Uoft = np.ones(np.shape(Xoft))
    Voft = np.ones(np.shape(Xoft))
    dt = t[1]-t[0]
    for nt in xrange(len(t)-1):
        tn = t[nt]
        tnp= t[nt+1]
        #find appropriate gauges
        x = Xoft[nt]
        if xgauges[-1]>x:
            ind = np.where(xgauges>x)[0][0]
        else:
            ind = -1
        indm1 = max(ind-1,0)
        gaugeupper=gauge_nums[ind]
        gaugelower=gauge_nums[indm1]
        gdataupper = cg.selectgauge(gaugeupper,allgaugedata)
        gdatalower = cg.selectgauge(gaugelower,allgaugedata)
        #find position of gauges relative to x(tn)
        xlower=gdatalower['x']
        xupper=gdataupper['x']
        dx = xupper - xlower
        if dx > 0:
            chi = (x-xlower)/dx
        else:
            chi = 1.0
        tlower = gdatalower['t']
        tupper = gdataupper['t']

        #spatially interpolated solution at tn
        indlower = np.where(tlower>tn)[0][0]
        indupper = np.where(tupper>tn)[0][0]

        #depth
        hupper = gdataupper['q1'][indupper]
        hlower= gdatalower['q1'][indlower]
        h = chi*hupper + (1.0-chi)*hlower
        Hoft[nt] = h

        #u--velocity
        uupper = gdataupper['q2'][indupper]/gdataupper['q1'][indupper]
        ulower=  gdatalower['q2'][indlower]/gdatalower['q1'][indlower]
        u = chi*uupper + (1.0-chi)*ulower
        Uoft[nt] = u

        #v--velocity
        vupper = gdataupper['q3'][indupper]/gdataupper['q1'][indupper]
        vlower=  gdatalower['q3'][indlower]/gdatalower['q1'][indlower]
        v = chi*vupper + (1.0-chi)*vlower
        Voft[nt] = v

        #solid volume frac
        mupper = gdataupper['q4'][indupper]/gdataupper['q1'][indupper]
        mlower=  gdatalower['q4'][indlower]/gdatalower['q1'][indlower]
        m = chi*mupper + (1.0-chi)*mlower
        Moft[nt] = m

        #pressure
        pupper = gdataupper['q5'][indupper]
        plower= gdatalower['q5'][indlower]
        p = chi*pupper + (1.0-chi)*plower
        Poft[nt] = p

    Hoft[-1] = Hoft[-2]
    Moft[-1] = Moft[-2]
    Uoft[-1] = Uoft[-2]
    Voft[-1] = Voft[-2]
    Poft[-1] = Poft[-2]

    return Hoft,Uoft,Voft,Moft,Poft

def getflumedata():
    """
    load the flume experimental data
    return a dictionary with variables
    """

    infile = '060999NatReleaseData.txt'
    data = np.loadtxt(infile,skiprows=1,usecols=(1,5,6,7,9,13))

    flumedata = {}
    flumedata['pressure_lower'] = data[:,0]
    flumedata['pressure_middle']=data[:,1]
    flumedata['pressure_middle_50'] = data[:,2]
    flumedata['pressure_middle_30'] = data[:,3]
    flumedata['pressure_upper'] = data[:,4]

    flumedata['time_adjusted'] = data[:,5]
    flumedata['time'] = data[:,5] - 2780.0

    infile = '060999extensometers.txt'
    data2 = np.loadtxt(infile,skiprows=1,usecols=(1,2,3))
    flumedata['displacement_upper'] = data2[:,0]-0.0975
    flumedata['displacement_lower'] = data2[:,1]-0.0512

    return flumedata

def pressureprofile(p,h):
    """
    determine pressure at h - 30 and h - 50 cm
    according to quadratic profile: equation 51
    """
    z_10 = (0.64 - 0.10)/0.64
    z_30 = (0.64 - 0.30)/0.64
    z_50 = (0.64 - 0.50)/0.64
    p_10 = p * (1.0 - z_10**2) - rho_f*gp*h*(z_10 - z_10**2)
    p_30 = p * (1.0 - z_30**2) - rho_f*gp*h*(z_30 - z_30**2)
    p_50 = p * (1.0 - z_50**2) - rho_f*gp*h*(z_50 - z_50**2)

    p_10 = p * (1.0 - z_10)
    p_30 = p * (1.0 - z_30)
    p_50 = p * (1.0 - z_50)


    return p_10,p_30,p_50



if __name__=='__main__':
    plotLagrangian()















