import os
import subprocess

basename = 'NatReleaseLoose'
outfiles = 40
outdir = 'saveplots/oblique2/'

if not os.path.exists('saveplots'):
    os.system('mkdir saveplots')

if not os.path.exists('saveplots/oblique2'):
    os.system('mkdir saveplots/oblique2')

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