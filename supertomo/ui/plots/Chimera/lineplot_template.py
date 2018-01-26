# -----------------------------------------------------------------------------
# Make a matplotlib plot of density values along a ray from center of a map.
# Use the current view direction and update plot as models are rotated.
# For Yong Zi Tan for 3D FSC plotting.
#
# This script registers command "fscplot" which takes one argument, the density
# map for which the plot is made.  For example,
#
#    fscplot #3D FSC Map #Actual Density Map
#
# Created by Tom Goddard (Thanks!)
# Modified by Yong Zi Tan
#

def ray_values(v, direction):
	d = v.data
	center = [0.5*(s+1) for s in d.size]
	radius = 0.5*min([s*t for s,t in zip(d.size, d.step)])
	steps = max(d.size)
	from Matrix import norm
	dn = norm(direction)
	from numpy import array, arange, float32, outer
	dir = array(direction)/dn
	spacing = radius/dn
	radii = arange(0, steps, dtype = float32)*(radius/steps)
	ray_points = outer(radii, dir)
	values = v.interpolated_values(ray_points)
	return radii, values, radius

# -----------------------------------------------------------------------------
#
def plot(x, y, xlabel, ylabel, title, fig = None):
	import matplotlib.pyplot as plt
	global_x = #==global_x==#
	global_y = #==global_y==#
	if fig is None:
		fig = plt.figure()
		fig.plot = ax = fig.add_subplot(1,1,1)
	else:
		ax = fig.plot
		ax.clear()
	plt.subplots_adjust(top=0.85)
	ax.plot(x, y, linewidth=2.0)
	ax.plot(global_x, global_y, 'r', linewidth=1.0) # Plot global FSC
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_ylim(ymin = -0.2, ymax = 1.01)
	ax.set_title(title)
	ax.grid(True)
	fig.canvas.manager.show()
	return fig

# -----------------------------------------------------------------------------
#
def update_plot(fsc_map, fig = None):
	xf = fsc_map.openState.xform
	from chimera import Vector
	direction = xf.inverse().apply(Vector(0,0,-1)).data()
	preradii, values, radius = ray_values(fsc_map, direction)
	radii = []
	apix = #==apix==#
	resolution_list = []
	for i in range(len(preradii)):
		radii.append(preradii[i]/(radius*2*apix))
	for i in range(len(values)):
		if values[i] < 0.143:
			resolution_list.append(1/radii[i-1])
			break
	resolution = resolution_list[0]
	#title = '3D FSC plotted on axis %.3g,%.3g,%.3g.' % direction
	title = '3D FSC Plot.\nZ directional resolution (out-of-plane in blue) is %.2f.\nGlobal resolution (in red) is %.2f.' % (resolution, #==global_res==#)
	fig = plot(radii, values, xlabel = 'Spatial Resolution', ylabel = 'Correlation', title = title, fig = fig)
	color_map(resolution)
	return fig

# -----------------------------------------------------------------------------
#
def color_map(resolution):
	import chimera
	from chimera import runCommand
	maxres = #==maxres==#
	minres = #==minres==#
	a = (resolution-maxres)/(minres-maxres)
	r, g, b = 1-a, 0.0, a
	runCommand('color %0.2f,%0.2f,%0.2f,1.0 #1' % (r, g, b))

# -----------------------------------------------------------------------------
#
def fsc_plot(fscMap):
	fig = update_plot(fscMap)
	from chimera import triggers
	h = triggers.addHandler('OpenState', motion_cb, (fscMap, fig))

# -----------------------------------------------------------------------------
#

def motion_cb(trigger_name, mf, trigger_data):
	if 'transformation change' in trigger_data.reasons:
		fsc_map, fig = mf
		update_plot(fsc_map, fig)

# -----------------------------------------------------------------------------
#
def fscplot_cmd(cmdname, args):
	from Commands import volume_arg, parse_arguments
	req_args = [('fscMap', volume_arg)]
	kw = parse_arguments(cmdname, args, req_args)
	fsc_plot(**kw)

# -----------------------------------------------------------------------------
#
from Midas.midas_text import addCommand
addCommand('fscplot', fscplot_cmd)
