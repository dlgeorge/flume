
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

"""

from pyclaw.geotools import topotools
from pyclaw.data import Data
import matplotlib.pyplot as plt
import numpy as np


import pdb

eta_ind = 6
import local_dplot as ld
import dclaw.dplot as dd

#--------------------------
def setplot(plotdata):
#--------------------------

    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of pyclaw.plotters.data.ClawPlotData.
    Output: a modified version of plotdata.

    """

    plotdata.clearfigures()  # clear any old figures,axes,items data

    def fixup(current_data):
        t = current_data.t
        import pylab

        pylab.title('')
        xticktuple = ('75','80','85','90','95','100','105','110','115','120')
        pylab.xticks(np.linspace(75,120,10),xticktuple,fontsize=32)
        pylab.xlabel('Downslope distance from gate (m)',fontsize=32)
        pylab.yticks([],())
        #pylab.yticks([-5,-3,-1,1,3,5,7],('-6','-4','-2','0','2','4','6'),fontsize=18)
        pylab.axis('equal')
        #pylab.grid()
        #a = pplt.gca()
        #cgrid = a.grid
        #cgrid(which='major',axis='x',linewidth=0.25,color='0.75')
        #print lines
        #pdb.set_trace()
        #pplt.getp()
        plt.gcf().subplots_adjust(left=0.0,bottom=0.15,right=1.0,top=1.0,wspace = 0.0,hspace=0.0)
        #pylab.tight_layout(0.0,0.0)
        #pylab.xlim(74,122)
        #pylab.ylim(-4.0,6.0)
        #pylab.xlim(60,130)
        #pylab.ylim(-5.0,7.0)


    figkwargs = dict(figsize=(48*.3,11*.3/.85),dpi=1600)
    #-----------------------------------------
    # Figure for pcolor plot
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='pcolor', figno=0)
    plotfigure.show = True
    plotfigure.kwargs = figkwargs
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('pcolor')
    plotaxes.afteraxes = fixup
    plotaxes.title = ''
    plotaxes.xlimits=[80,130]
    

    # Debris
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = dd.particle_size
    plotitem.pcolor_cmap =  ld.white2red_colormap
    #plotitem.plot_var = dd.liquefaction_ratio
    #plotitem.pcolor_cmap = ld.oso_debris_colormap_liquefaction
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True
    plotitem.amr_gridlines_show = [0,0,0,0,0]
    plotitem.gridedges_show = 0
    plotitem.show = True


    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = ld.land
    plotitem.pcolor_cmap = ld.runoutpad_colormap
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 0.1
    plotitem.add_colorbar = False
    plotitem.amr_gridlines_show = [1,0,0,0,0]
    plotitem.kwargs = {'linewidths':0.001}
    plotitem.gridedges_show = 0
    plotitem.show = True


    # add contour lines of depth if desired
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = True
    plotitem.plot_var = ld.depth
    plotitem.contour_levels = np.linspace(0.0,0.18,10)
    plotitem.amr_contour_colors = ['k']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':1}
    plotitem.amr_contour_show = [0,1,1,1,0]
    plotitem.amr_gridlines_show =  [1,0,0,0,0,0]
    plotitem.gridedges_show = 0

        #-----------------------------------------
    # Figure for pcolor plot
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='pcolor2', figno=1)
    plotfigure.show = True
    plotfigure.kwargs = figkwargs
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('pcolor')
    plotaxes.afteraxes = fixup
    plotaxes.title = ''
    plotaxes.xlimits=[-6.0,120]
    

    # Debris
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = dd.particle_size
    plotitem.pcolor_cmap =  ld.white2red_colormap
    #plotitem.plot_var = dd.liquefaction_ratio
    #plotitem.pcolor_cmap = ld.oso_debris_colormap_liquefaction
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True
    plotitem.amr_gridlines_show = [0,0,0,0,0]
    plotitem.gridedges_show = 0
    plotitem.show = True


    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = ld.land
    plotitem.pcolor_cmap = ld.runoutpad_colormap
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 0.1
    plotitem.add_colorbar = False
    plotitem.amr_gridlines_show = [1,0,0,0,0]
    plotitem.kwargs = {'linewidths':0.001}
    plotitem.gridedges_show = 0
    plotitem.show = True


    # add contour lines of depth if desired
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = True
    plotitem.plot_var = ld.depth
    plotitem.contour_levels = np.linspace(0.0,0.18,10)
    plotitem.amr_contour_colors = ['k']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':1}
    plotitem.amr_contour_show = [0,1,1,1,0]
    plotitem.amr_gridlines_show =  [1,0,0,0,0,0]
    plotitem.gridedges_show = 0


    #-----------------------------------------

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'#range(10,21)#'all'#range(0,120,4)   # range(70,190,10)  # list of frames to print
    plotdata.print_gaugenos = 'all'            # list of gauges to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 1           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    return plotdata

