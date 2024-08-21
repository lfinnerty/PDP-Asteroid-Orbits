import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def xy_to_rthet(x, y):
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan2(y, x)
	theta[theta<0]+=2*np.pi
	return r, theta

def rthet_to_xy(r,theta):
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return x, y

def dxy_to_drthet(dx,dy):
	pass

def xrthet_to_dxy(dr,dthet):
	pass

def make_orbit(thets, a, e, omega):
	return a*(1-e**2)/(1+e*np.cos(thets+omega))

def plot_ellipse(a, e, omega):
	thets=np.linspace(0,2*np.pi,200)
	rs = make_orbit(thets,a,e,omega)
	x, y = rthet_to_xy(rs,thets)
	### Fix me eventually will want to plot onto a common axes object thats passed in
	# ax.scatter(x,y)
	plt.scatter(x,y)
	# plt.show()

def fit_orb_params(xs, ys, dxs, dys):
	pass


if __name__ == '__main__':
	### True parameters
	a, e, omega = 1.0, 0.2, np.pi/4.
	print('True parameters:', a,e,omega)
	thetas = 2*np.pi*np.array([0.1, 0.3, 0.35, 0.45, 0.5])
	rs =  make_orbit(thetas, a, e, omega)
	rs = np.random.normal(loc=rs,scale=5e-2)
	drs = 0.05*np.ones_like(rs)

	opt = curve_fit(make_orbit, thetas, rs, sigma=drs)
	print('Fit parameters:',opt[0])

	x, y = rthet_to_xy(rs, thetas)
	plot_ellipse(*opt[0])
	plt.scatter(x,y,color='r')
	plt.scatter(0,0,color='k',s=100)
	plt.show()


	
	
