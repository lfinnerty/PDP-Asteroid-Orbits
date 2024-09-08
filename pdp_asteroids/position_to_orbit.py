import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time

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

def make_orbit(times, tperi, a, e,omega):
    period = np.sqrt(a**3)*365.25
    phase0 = (times-tperi)/period
    phase0-=int(phase0[0])
    M = 2*np.pi*phase0
    E = solve_kepler(M, e)
    ### FIXME check if there's a sign issue for the arctangent
    # nus = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    beta = e/(1+np.sqrt(1-e**2))
    nus = E + 2*np.arctan(beta*np.sin(E)/(1-beta*np.cos(E)))
    rs = a * (1-e**2)/(1+e*np.cos(nus))

    return rs, nus+omega

def parallax_to_dist(t1, p1, e1, t2, p2, e2):
    theta_earth = 0.5*2*np.pi*(t2-t1)/365.25
    baseline = np.sin(theta_earth)

    ra1, d1 = p1
    ra2, d2 = p2
    parallax = np.sqrt((ra1-ra2)**2 + (d1-d2)**2) ### fixme calculate the angle
    dist = baseline/parallax

    ### Calculate distance error
    # dist_err = 

    return dist, dist_err

def ra_dist_to_r_theta(time, ra, rp):
    theta_earth = 2*np.pi/365.25 *(time-2460392.400856)#JD of march 22 2024)

    r = np.sqrt(1+rp**2 - 2*rp*np.cos(np.pi+theta_earth+ra))
    # rerr = 

    theta= theta_earth + rp/r * np.sin(np.pi+theta_earth - ra)
    # theta_err = 

    return r, rerr, theta, theta_err

def r_theta_to_ra_dist(time, r, theta):
    pass

def dist_to_parallax(ra, dist, t1, t2):
    theta_earth = 0.5*2*np.pi*(t2-t1)/365.25
    baseline = np.sin(theta_earth)

    parallax = baseline/dist
    ### Break this up into a pair of positions



    p1 = (ra1,d1)
    p2 = (ra2,p2)
    return p1, p2


def inject_asteriod(ra, parallax, image):
    pass

def plot_ellipse(tperi, a, e, omega):
    period = np.sqrt(a**3)*365.25
    times = np.linspace(tperi,tperi+period,200)
    rs, nus = make_orbit(times, tperi,a,e,omega)
    x, y = rthet_to_xy(rs,nus)
    ### Fix me eventually will want to plot onto a common axes object thats passed in
    # ax.scatter(x,y)
    plt.scatter(x,y)
    # plt.show()

def fit_orb_params(xs, ys, dxs, dys):
    pass


if __name__ == '__main__':
    ### True parameters
    tperi, a, e, omega = 2460562.397130, 0.3, 0.6, -np.pi/2.
    print(365.25*np.sqrt(a**3))
    ### Given a list of dates, predict observables
    # obsdates = ['2025-01-18', '2025-03-02', '2025-04-29', '2025-05-12']
    obsdates = ['2025-01-'+str(i).zfill(2) for i in range(1,32)]
    obsdates = obsdates + ['2025-02-'+str(i).zfill(2) for i in range(1,29)]

    jds = []
    for date in obsdates:
        jds.append(Time(date+'T23:59:59', format='isot', scale='utc').jd)
    jds = np.asarray(jds)
    
    rs, thetas = make_orbit(jds, tperi, a, e, omega)
    
    ### For clean plotting
    ptheta = thetas%(2*np.pi) 
    arg = np.argsort(ptheta)
    prs = rs[arg]
    ptheta = ptheta[arg]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(ptheta,prs)
    plt.show()


    ### From the rs, thetas of the orbits, make the observables


    ### Add errors to the observables


    ### Fit to the observables+errors
