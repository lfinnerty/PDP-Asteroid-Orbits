import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time
import json
from pymultinest.solve import solve
import corner

def xy_to_rthet(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta[theta<0]+=2*np.pi
    return r, theta

def rthet_to_xy(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def solve_kepler(mobs, e, order=100):
    Eobs = np.copy(mobs)
    for i in range(1,order+1):
        Eobs = mobs + e*np.sin(Eobs)
    return Eobs

def make_orbit(times, phase0, a, e,omega):
    period = np.sqrt(a**3)*365.25
    # phase0 = (times-tperi)/period
    # phase0-=int(phase0[0])
    M = 2*np.pi*(phase0 + (times-times[0])/period)
    E = solve_kepler(M, e)
    ### FIXME check if there's a sign issue for the arctangent
    # nus = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    beta = e/(1+np.sqrt(1-e**2))
    nus = E + 2*np.arctan(beta*np.sin(E)/(1-beta*np.cos(E)))
    rs = a * (1-e**2)/(1+e*np.cos(nus))

    return rs, nus+omega

def parallax_to_dist(tp1, e1, p2, e2, dt):
    theta_earth = 2*np.pi*dt/365.25
    baseline = np.sin(theta_earth/2.) ### AU

    ra1, d1 = p1
    ra2, d2 = p2
    parallax = np.sqrt((ra1-ra2)**2 + (d1-d2)**2) ### Arcseconds
    dist = baseline/np.sin(parallax/206265)

    ### Calculate distance error
    # dist_err = 

    return dist, dist_err

def dist_to_parallax(time, r, theta, dt):
    theta_earth = 2*np.pi/365.25 *(time-2460392.400856)
    r_earth = 1
    xe, ye = rthet_to_xy(r_earth, theta_earth)
    xa, ya = rthet_to_xy(r, theta)
    distance = np.sqrt((xe-xa)**2+(ye-ya)**2)

    dtheta = dt*2*np.pi/365.25
    baseline = np.sin(dtheta/2) ### In AU
    parallax = np.arcsin(baseline/distance) ### In rad

    return parallax*206265 ### In AU

def ra_dist_to_r_theta(time, ra, rp):
    ### FIXME do this vector addition based, it's easier 
    heta_earth = 2*np.pi/365.25 *(time-2460392.400856)
    r_earth = 1
    xe, ye = rthet_to_xy(r_earth, theta_earth)

    dx, dy = rthet_to_xy(ra, rp)

    xtot = xe+dx
    ytot = ye+dy
    r, theta = xy_to_rthet(xtot, ytot)

    return r, rerr, theta, theta_err

def inject_asteroid(image, time, parallax):
    pass

def prior_transform(u):
    x = np.array(u)
    x[0] = u[0]

    ### a
    x[1] = u[1]*10

    ### e
    x[2] = u[2]

    ### omega
    x[3] = 2*np.pi*u[3]

    ### Tperi
    #period = np.sqrt(x[1]**3)*365.25
    # x[0] = period - period/2.
    

    return x

def loglike(x, times, rs, rerrs, thetas, thetaerrs):
    ### Obs, obs errs are sun-centered distances at specified times
    ### Angles are assumed to match
    fitr, fittheta = make_orbit(times, x[0], x[1], x[2], x[3])
    fitx, fity = rthet_to_xy(fitr, fittheta)
    obsx, obsy = rthet_to_xy(rs, thetas)

    return -0.5 * np.sum((fitx-obsx)**2/rerrs**2) - 0.5*np.sum((fity-obsy)**2/rerrs**2)
    # return -0.5*np.sum((fitr-rs)**2/rerrs**2 + (thetas - fittheta)**2/(thetaerrs)**2) 

class logl():
    def __init__(self, times, rs, rerrs, thetas, thetaerrs):
        self.times = times 
        self.rs = rs
        self.rerrs = rerrs
        self.thetas = thetas
        self.thetaerrs = thetaerrs
    def __call__(self, x):
        return loglike(x, self.times,self.rs, self.rerrs, self.thetas, self.thetaerrs)


if __name__ == '__main__':
    ### True parameters
    p0, a, e, omega = 0.2, 1.2, 0.83, np.pi/2.
    period = 365.25*np.sqrt(a**3)
    print(period)
    ### Given a list of dates, predict observables
    obsdates = ['2025-01-18', '2025-03-02', '2025-04-01']#, '2025-04-29', '2025-05-12', '2025-05-29', '2025-06-29']
    # obsdates = ['2025-01-'+str(i).zfill(2) for i in range(1,32,2)]
    #obsdates = obsdates + ['2025-02-'+str(i).zfill(2) for i in range(1,29,2)]

    jds = []
    for date in obsdates:
        jds.append(Time(date+'T23:59:59', format='isot', scale='utc').jd)
    jds = np.asarray(jds)
    
    rs, thetas = make_orbit(jds, p0, a, e, omega)

    ### From the rs, thetas of the orbits, make the observables
    # dists, ras = r_theta_to_ra_dist(jds, rs, thetas)


    ### Add errors to rs, thetas
    rs_err = np.random.normal(0*np.ones_like(rs),3e-2)
    rs_fit=rs+rs_err
    thetas_err = np.random.normal(0*np.ones_like(thetas),1e-4)
    thetas_fit = thetas+thetas_err


    prefix = 'fit_results/'
    loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
    result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=400, evidence_tolerance=0.5,
                    outputfiles_basename=prefix, verbose=False, resume=False)
    samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(thetas_fit, rs_fit)
    ptimes = np.linspace(0, 2*period,300)
    true_r, true_theta = make_orbit(ptimes, p0, a, e, omega)
    ax.plot(true_theta, true_r, color='k')
    for i in range(200):
        rs, thetas = make_orbit(ptimes, *samples[i])
        ax.plot(thetas,rs,color='r',alpha=0.05)
    ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g')
    plt.show()



