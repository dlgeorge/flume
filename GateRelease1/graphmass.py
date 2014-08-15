"""
graphmass

graph changes in mass and volume
"""

import numpy as np
import geotools.topotools as gt
import pylab
import os
import pdb
import matplotlib.pyplot as mp

fname = '_output/total_mass.txt'
file = open(fname,'r')

data = file.readlines()
vec = np.ones((len(data),2))
row = 0
for line in data:

    lst = line.split()
    #pdb.set_trace()
    mass=float(lst[-1])
    time=float(lst[3][0:-1])

    vec[row,1] = mass
    vec[row,0] = time
    row = row + 1

np.savetxt('_output/total_mass_mod.tm',vec)

file.close()

fname = '_output/total_volume.txt'
file = open(fname,'r')

data = file.readlines()
vec2 = np.ones((len(data),2))
row = 0
for line in data:

    lst = line.split()
    #pdb.set_trace()
    mass=float(lst[-1])
    time=float(lst[3][0:-1])

    vec2[row,1] = mass
    vec2[row,0] = time
    row = row + 1

np.savetxt('_output/total_vol_mod.tm',vec2)

mp.plot(vec2[:,0],vec2[:,1],label='volume')
mp.plot(vec[:,0],vec[:,1],label='mass')

mp.xlabel(r'time')
mp.ylabel(r'percent change of original volume/mass')
#mp.axis([0,40.,-1.2,1.2])
mp.legend()
mp.show()

file.close()