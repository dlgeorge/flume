"""
file: makescaleplot.py

make a small figures for legends in side-view plot

"""

import matplotlib as mp
import matplotlib.pyplot
import numpy as np

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-large')

plt = matplotlib.pyplot

x = np.linspace(0,10)
c = np.nan*x
m = x
r = x

figsize = (10,3)


plt.figure(1,figsize=figsize)
plt.plot(x,c,'c*',markersize=8,label = 'position of upper extensometer',markevery=100)
plt.plot(x,c,'m*',markersize=8,label = 'position of lower extensometer',markevery=100)
plt.plot(x,c,'rp',markersize=8,label = 'position of pressure piezometer',markevery=100)
plt.legend(numpoints=1,markerscale = 2,loc=3,prop=fontP)

frame=plt.gca()
frame.axes.get_xaxis().set_ticks([])
frame.axes.get_yaxis().set_ticks([])
frame.set_frame_on(False)

plt.figure(2,figsize=figsize)
plt.plot(x,c,'c',  label = 'simulated upper extensometer')
plt.plot(x,c,'c--',label = 'experimental upper extensometer')
plt.plot(x,c,'m',  label = 'simulated lower extensometer')
plt.plot(x,c,'m--',label = 'experimental lower extensometer')
plt.legend(numpoints=1,loc=3,prop=fontP)

frame=plt.gca()
frame.axes.get_xaxis().set_ticks([])
frame.axes.get_yaxis().set_ticks([])
frame.set_frame_on(False)

plt.figure(3,figsize=figsize)
plt.plot(x,c,'r--',label = 'measured pore-water pressure head')
plt.plot(x,c,'r',  label = 'simulated pore-water pressure head')
plt.plot(x,c,'b.', label = 'lithostatic pressure head')
plt.legend(numpoints=1,loc=3,prop=fontP)

frame=plt.gca()
frame.axes.get_xaxis().set_ticks([])
frame.axes.get_yaxis().set_ticks([])
frame.set_frame_on(False)

plt.show()

