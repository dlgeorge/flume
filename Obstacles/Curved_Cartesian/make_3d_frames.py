import os
import subprocess

basename = 'GateReleaseHopper'
outfiles = 20
outdir = 'saveplots/oblique_hopper/'

if not os.path.exists('saveplots'):
    os.system('mkdir saveplots')

if not os.path.exists('saveplots/oblique'):
    os.system('mkdir saveplots/oblique')

for i in range(outfiles + 1):

    fnum = '00000'+str(i)
    fnum = fnum[-5:]
    fname = '_output/frame'+fnum+'.png'
    outname = outdir+basename+fnum+'.png'

    #pdb.set_trace()

    spcmd = ['convert','-trim','+repage', fname,  outname]
    p = subprocess.call(spcmd)

    #os.system('rm ' + fname)




imcommand = 'convert ' + outdir + basename + '*.png ' + outdir + basename + '.gif'
os.system(imcommand)