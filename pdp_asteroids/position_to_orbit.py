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

def solve_kepler(M, e):
    pass

def make_orbit(times, tperi, a, e, omega):
    period = np.sqrt(a**3)*365.25
    M = 2*np.pi*(times-tperi)/period
    E = solve_kepler(M, e)
    ### FIXME check if there's a sign issue for the arctangent
    nus = 2*np.arctangent(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    rs = a*(1-e*np.cos(nus))
    return rs, nus

def parallax_to_dist(t1, p1, e1, t2, p2, e2):
    theta_earth = 0.5*2*np.pi*(t2-t1)/365.25
    baseline = np.sin(theta_earth)

    parallax = ### fixme calculate the angle
    dist = baseline/parallax

    ### Calculate distance error

    return dist, dist_err

def ra_dist_to_r_theta(time, ra, rp):
    theta_earth = 2*np.pi/365.25 *(time-)#JD of march 22 2024)

    r = np.sqrt(1+rp**2 - 2*rp*np.cos(np.pi+theta_earth+ra))
    rerr = 

    theta= theta_earth + rp/r * np.sin(np.pi+theta_earth - ra)
    theta_err = 

    return r, rerr, theta, theta_err

def r_theta_to_ra_dist(time, r, theta):
    pass

def dist_to_parallax(ra, dist, t1, t2):
    pass

def inject_asteriod(ra, parallax, image):
    pass

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
    a, e, omega = 1.0, 0.93, np.pi/4.
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

