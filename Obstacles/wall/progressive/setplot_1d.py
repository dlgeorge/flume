
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

"""

from pyclaw.geotools import topotools
from pyclaw.data import Data
import matplotlib.pyplot as plt
import numpy as np



import local_dplot as ld
import dclaw.dplot as dd

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties

import numpy as np

mxlevel=3

#--------------------------
def setplot(plotdata):
#--------------------------

    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of pyclaw.plotters.data.ClawPlotData.
    Output: a modified version of plotdata.

    """


    from pyclaw.plotters import colormaps, geoplot
    from numpy import linspace

    plotdata.clearfigures()  # clear any old figures,axes,items data



    # To plot gauge locations on pcolor or contour plot, use this as
    # an afteraxis function:

    def addgauges(current_data):
        from pyclaw.plotters import gaugetools
        gaugetools.plot_gauge_locations(current_data.plotdata, \
             gaugenos='all', format_string='ko', add_labels=True)


    def fixup(current_data):
        t = current_data.t
        import pylab

        plt.title('')
        xticktuple = ('75')
        #pylab.xticks(np.linspace(83,97,7),xticktuple,fontsize=32)
        #pylab.xlabel('Downslope distance from gate (m)',fontsize=32)
        #plt.yticks([],())

        #------ time label bar ------------------------------------------------
        ts = (r't = %4.1f s' % t)
        #pylab.text()
        plt.text(85,4.5,ts,bbox={'facecolor':'white','alpha':1.0,'pad':10},fontsize=30)

        #pylab.yticks([-5,-3,-1,1,3,5,7],('-6','-4','-2','0','2','4','6'),fontsize=18)
        #pylab.axis('equal')
        #pylab.grid()
        #a = pplt.gca()
        #cgrid = a.grid
        #cgrid(which='major',axis='x',linewidth=0.25,color='0.75')
        #print lines
        #pdb.set_trace()
        #pplt.getp()
        #plt.gcf().subplots_adjust(left=0.0,bottom=0.15,right=1.0,top=1.0,wspace = 0.0,hspace=0.0)
        #plt.axis('equal')
        #plt.tight_layout(0.0,0.0)
        plt.axis('equal')
        plt.axis([78,102,-6,8])

        #pylab.xlim(60,130)
        #pylab.ylim(-5.0,7.0)




        #img = Image.open('scale.gif')
        #im = plt.imshow(img)

        #ax.annotate((r'10 km'))
        #pylab.xlabel('meters')
        #pylab.ylabel('meters')
        #pylab.axis('off')
        #pylab.axis('equal')
        #plt.tight_layout()

    def fixup1d(current_data):
        import pylab
        #addgauges(current_data)
        ax = plt.gca()
        t = current_data.t

        pylab.title('%5.3f seconds' % t, fontsize=20)
        #pylab.title(r'$m-m_{crit}=-0.02$',fontsize=40)
        pylab.title('')
        pylab.xticks(fontsize=15)
        pylab.yticks(fontsize=15)

        ts = (r't = %4.1f s' % t)
        #pylab.text()

        pylab.text(85.,1.8,ts,bbox={'facecolor':'white','alpha':1.0,'pad':10},fontsize=20)
        plt.grid()
        #pylab.axis('off')
        #plt.xlimits = [-2.0,3.0]
        #plt.ylimits = [80,95]
        #plt.axis('equal')
        #plt.tight_layout()

    def q_1d_fill(current_data):
        X = current_data.x
        Y = current_data.y
        a2dvar = current_data.var
        a2dvar2 = current_data.var2
        dy = current_data.dy

        #yind = np.where(np.abs(Y[0,:]-1.0)<=dy/2.0)[0]
        #xind = np.where(X[:,0]> -1.e10)[0]
        #import pdb;pdb.set_trace()
        if (current_data.grid.level==mxlevel):
            yind = np.where(np.abs(Y[0,:]-1.0)<=dy)[0]
            xind = np.where(X[:,0]> -1.e10)[0]
        else:
            yind = np.where(Y[0,:]>1.e10)[0]
            xind = np.where(X[:,0]>1.e10)[0]

        x = X[np.ix_(xind,yind)]
        a1dvar = a2dvar[np.ix_(xind,yind)]#-x*np.sin(31.*np.pi/180.0)
        a1dvar2 = a2dvar2[np.ix_(xind,yind)]#-x*np.sin(31.*np.pi/180.0)
        #pdb.set_trace() #<-----------------------------

        return x,a1dvar,a1dvar2

    def q_1d(current_data):
        X = current_data.x
        Y = current_data.y
        dy = current_data.dy
        a2dvar = current_data.var

        if (current_data.grid.level==mxlevel):
            yind = np.where(np.abs(Y[0,:]-1.0)<=dy)[0]
            xind = np.where(X[:,0]> -1.e10)[0]
        else:
            yind = np.where(Y[0,:]>1.e10)[0]
            xind = np.where(X[:,0]>1.e10)[0]

        x = X[np.ix_(xind,yind)]
        a1dvar = a2dvar[np.ix_(xind,yind)]#-x*np.sin(31.*np.pi/180.0)

        return x,a1dvar
    #-----------------------------------------
    # Figure for pcolor plot
    #-----------------------------------------
    figkwargs = dict(figsize=(8,6),dpi=1600)
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


    # Debris
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    #plotitem.plot_var = dd.particle_size
    #plotitem.pcolor_cmap =  ld.white2red_colormap
    plotitem.plot_var = dd.surface
    plotitem.pcolor_cmap = ld.white2red_colormap
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 1.6
    plotitem.add_colorbar = True
    plotitem.amr_gridlines_show = [1,0,0,0]
    plotitem.gridedges_show = 0
    plotitem.show = True


    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = ld.land
    plotitem.pcolor_cmap = ld.runoutpad_colormap
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 4.0
    plotitem.add_colorbar = False
    plotitem.amr_gridlines_show = [1,0,0,0]
    plotitem.kwargs = {'linewidths':0.001}
    plotitem.gridedges_show = 0
    plotitem.show = True


    # add contour lines of depth if desired
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = True
    plotitem.plot_var = ld.depth
    plotitem.contour_levels = np.linspace(0.0,1.5,31)
    plotitem.amr_contour_colors = ['k']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':1}
    plotitem.amr_contour_show = [0,0,0,0,0]
    plotitem.amr_contour_show[mxlevel-1] = 1
    plotitem.amr_gridlines_show =  [1,0,0,0,0,0]
    plotitem.gridedges_show = 0

    #-----------------------------------------
    # Figure for 1d plots
    #-----------------------------------------

    plotfigure = plotdata.new_plotfigure(name='Cross-section', figno=1)
    plotfigure.kwargs = {'figsize':(10,3.2),'frameon':False}
    #plotfigure.tight_layout = True


    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [80,102]
    plotaxes.ylimits = [-0.2,2.5]#'auto' #[-.1,2.0]
    plotaxes.kwargs = {'frameon':'False'}
    plotaxes.afteraxes = fixup1d

    # Set up for item on these axes: (plot topography as brown line)
    plotitem = plotaxes.new_plotitem(plot_type='1d_from_2d_data')
    plotitem.plot_var = ld.topo
    plotitem.map_2d_to_1d = q_1d
    #plotitem.amr_gridlines_show = [1,1,1]
    plotitem.color = 'brown'

    # Set up for item on these axes: (plot blue depth)
    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between_from_2d_data')
    plotitem.plot_var = ld.topo
    plotitem.plot_var2 = ld.surface
    plotitem.map_2d_to_1d = q_1d_fill
    #plotitem.amr_gridlines_show = [1,1,1]
    plotitem.color = 'blue'

    #-----------------------------------------
    # Figures for gauges
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Surface', figno=300, \
                    type='each_gauge')
    plotfigure.clf_each_gauge = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    #plotaxes.xlimits = [51.5e3,56.5e3]
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-.02,0.5]
    plotaxes.title = 'Surface'

    # Plot surface:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = 'b-'
    plotitem.show = True

    plotfigure = plotdata.new_plotfigure(name='Velocity', figno=301, \
                    type='each_gauge')
    plotfigure.clf_each_gauge = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    #plotaxes.xlimits = [51.5e3,56.5e3]
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-1.2,3.0]
    plotaxes.title = 'Discharge'

    # Plot surface:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 1
    plotitem.plotstyle = 'b-'
    plotitem.show = True
    #-----------------------------------------

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                   # print figures
    plotdata.print_format = 'png'               # file format
    plotdata.print_framenos = 'all'#range(60,201,10)   # list of frames to print
    plotdata.print_gaugenos = 'all'             # list of gauges to print
    plotdata.print_fignos = 'all'               # list of figures to print
    plotdata.html = True                        # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                       # create latex file of plots?
    plotdata.latex_figsperline = 1              # layout of plots
    plotdata.latex_framesperline = 1            # layout of plots
    plotdata.latex_makepdf = False              # also run pdflatex?

    return plotdata

