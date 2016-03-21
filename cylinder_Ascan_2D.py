import gprMax
import os

scene = gprMax.Scene()
oc = gprMax.GPRObjectCreator()

t = oc.create('title', 'A-scan from a metal cylinder buried in a dielectric half-space')
d = oc.create('domain', 0.240, 0.190, 0.002)
dis = oc.create('dx_dy_dz', 0.002, 0.002, 0.002)
tw = oc.create('time_window', 3e-9)
m = oc.create('material', 6, 0, 1, 0, 'half_space')
w = oc.create('waveform', 'ricker', 1, 1.5e9, 'my_ricker')
hd = oc.create('hertzian_dipole', 'z', 0.100, 0.170, 0, 'my_ricker', 0, 2e-9)
r = oc.create('rx', 0.140, 0.170, 0)
b = oc.create('box', 0, 0, 0, 0.240, 0.170, 0.002, 'half_space')
c = oc.create('cylinder', 0.120, 0.080, 0, 0.120, 0.080, 0.002, 0.010, 'pec')
g = oc.create('geometry_view', 0, 0, 0, 0.240, 0.190, 0.002, 0.002, 0.002, 0.002, 'cylinder_half_space', 'n')

scene.add(t, d, dis, tw, m, w, hd, r, b, c, g)

cwd = os.getcwd()
gprMax.run(scene=scene, input_directory=cwd, inputfile="cyclinder_Ascan_2D")
