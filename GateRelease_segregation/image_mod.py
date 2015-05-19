import os
import subprocess
import pdb

basename = 'GateRelease'
outfiles = 16
outdir = 'saveplots/planview/'

if not os.path.exists('saveplots'):
    os.system('mkdir saveplots')

if not os.path.exists('saveplots/planview'):
    os.system('mkdir saveplots/planview')

for i in range(outfiles + 1):

    fnum = '000'+str(i)
    fnum = fnum[-4:]
    fname = '_plots/frame'+fnum+'fig1.png'
    outname = outdir+basename+fnum+'.png'

    #pdb.set_trace()

    spcmd = ['convert','-trim','+repage', fname,  outname]
    p = subprocess.call(spcmd)



    spcmd = ['mogrify', '-rotate', '31.0', '-background', 'none' , outname]
    p = subprocess.call(spcmd)


imcommand = 'convert ' + outdir + basename + '*.png ' + outdir + basename + '.gif'
os.system(imcommand)



