import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time
import json
import corner
from glob import glob
from astropy.io import fits

FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

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

def parallax_to_dist(p1, e1, p2, e2, dt):
    theta_earth = 2*np.pi*dt/365.25
    baseline = np.sin(theta_earth/2.) ### AU

    ra1, d1 = p1
    ra2, d2 = p2
    parallax = np.sqrt((ra1-ra2)**2 + (d1-d2)**2) ### Degrees
    dist = baseline/np.sin(parallax*np.pi/180.)

    ### Calculate distance error
    ### Note that the WCS has intrinsic error values we can use
    # if e1 is None:
    # e1 = [1e-8,1e-8]
    # if e2 is None:
    # e2 = [1e-8,1e-8]
    # dra = np.sqrt((e1[0]**2+e2[0]**2))
    # ddec = np.sqrt((e1[1]**2+e2[1]**2))

    # parallax_err =  np.sqrt((ra1-ra2)**2/parallax**2 * dra**2 + (d1-d2)**2/parallax**2 * ddec**2)
    # sinp_err = np.sqrt(np.cos(parallax*np.pi/180.)*parallax_err*np.pi/180.)
    ### Hardcoded because we're having problems
    dist_err = 0.05*dist # dist* sinp_err/np.sin(parallax*np.pi/180.)

    return dist, dist_err

def dist_to_parallax(time, r, theta, dt):
    theta_earth = 2*np.pi/365.25 *(time-2460392.400856)
    r_earth = 1
    xe, ye = rthet_to_xy(r_earth, theta_earth)
    xa, ya = rthet_to_xy(r, theta)
    distance = np.sqrt((xe-xa)**2+(ye-ya)**2)

    dtheta = dt*2*np.pi/365.25
    baseline = np.sin(dtheta/2) ### In AU
    print('distanc', distance)
    parallax = np.arcsin(baseline/distance) ### In rad

    sin_sun = r/distance * np.sin(theta-theta_earth)

    return parallax*206265, sin_sun ### In arcsec

def dist_to_r(time, theta, elong, dist, dist_err):
    ### FIXME do this vector addition based, it's easier 
    theta_earth = 2*np.pi/365.25 *(time-2460392.400856)
    r_earth = 1

    r = dist*elong/np.sin(theta-theta_earth)
    r_err = dist_err*elong/np.sin(theta-theta_earth)


    return r, r_err

def Gauss2D(x, y, amp, x0,y0,sigx,sigy,rot=0):
    a = np.cos(rot)**2/(2*sigx**2)+np.sin(rot)**2/(2*sigy**2)
    b = np.sin(2*rot)/(2*sigx**2)-np.sin(2*rot)/(2*sigy**2)
    c = np.sin(rot)**2/(2*sigx**2)+np.cos(rot)**2/(2*sigy**2)

    return amp*np.exp(-a*(x-x0)**2 - b*(x-x0)*(y-y0) - c*(y-y0)**2)


def inject_asteroid(hdulst, parallax, obsdate,obsdelta,  jd, theta, sin_sun,  fwhm, fluxlevel,noiselevel,output_str, output_dir: Path=FILE_DIR):
    ### Decide where to add inital PSF
    data = hdulst[0].data
    data[np.isnan(data)] = 3.
    ### Randomly flip the data around
    if np.random.uniform(0,1) > 0.5:
        data = data[::-1,:]
    if np.random.uniform(0,1) > 0.5:
        data = data[:,::-1]

    header = hdulst[0].header
    header.append(('obsdt',obsdelta,'Time in hours'),end=True)
    header.append(('jd', jd, 'Julian date'),end=True)
    header.append(('theta', theta, 'angle'),end=True)
    header.append(('elong', sin_sun, 'sin solar elongation'),end=True)
    header['CD2_1'] = 0.
    header['CD1_2'] = 0.
    # plt.imshow(data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))
    # plt.show()

    ### Get coordinate info in degrees
    ra0, dec0 = header['CRVAL1'], header['CRVAL2']
    dra, ddec = header['CD2_2'], header['CD1_1']
    x0, y0 = header['CRPIX2'], header['CRPIX1']

    ### Decide where to add the first PSF
    x1, y1 = np.random.randint(200, data.shape[1]-200), np.random.randint(200, data.shape[0]-200)
    ### Decide where to add the second PSF, making sure the offset gives the right parallax
    d2 = (parallax/3600.)**2
    ### Make sure we generate values in the image, away from edge effects
    # print(x1, y1, np.sqrt(d2)/dra)
    x2, y2 = -100, -100
    while x2 < 100 or y2 < 100 or x2 > 900 or y2 > 900:
        dx = np.random.uniform(low=-np.sqrt(d2),high=np.sqrt(d2))
        dy = np.sqrt(d2-dx**2)
        ### Randomize y offset sign
        if np.random.uniform(0,1) > 0.5: dy*=-1
        ### Convert to pixels and apply offset
        x2, y2 = x1+dx/dra, y1+dy/ddec
    # print(x1,x2, y1,y2)

    ### Now we actually add the PSFs to the images
    x, y = np.meshgrid(np.arange(data.shape[0]),np.arange(data.shape[1]))
    rot = np.random.uniform(0,2*np.pi)
    sigx, sigy = np.random.normal(fwhm,0.5), np.random.normal(fwhm,0.5)
    amp = np.nanpercentile(data,fluxlevel)
    psf1 = Gauss2D(x,y, amp, x1,y1,sigx,sigy,rot)
    psf2 = Gauss2D(x,y, amp, x2,y2,sigx,sigy,rot)

    ### Add PSFs to image
    im1 = data+psf1
    im2 = data+psf2
    ### Add photon noise
    im1 += np.random.normal(np.sqrt(np.abs(im1))) + np.random.normal(loc=0,scale=noiselevel,size=im1.shape)
    im2 += np.random.normal(np.sqrt(np.abs(im2))) + np.random.normal(loc=0,scale=noiselevel,size=im1.shape)

    # plt.imshow(im1-im2,vmin=-np.nanpercentile(data,10),vmax=np.nanpercentile(data,10))
    # plt.show()
    
    ### Write to disk
    out_dir = output_dir/'injected_images'/output_str
    out_dir.mkdir(exist_ok=True, parents=True)
    # out_dir.mkdir(exist_ok=True, parents=True)

    fname1 = out_dir/(obsdate+'_'+output_str+'_frame1.fits')
    fname2 = out_dir/(obsdate+'_'+output_str+'_frame2.fits')
    fits.writeto(fname1, data=im1, header=header, overwrite=True)
    fits.writeto(fname2, data=im2, header=header, overwrite=True)


    ### Return two image arrays
    return im1, im2, fname1, fname2

def prior_transform(u, phase0, a, e, omega):
    x = np.array(u)
    x[0] = u[0]*(phase0[1]-phase0[0])+phase0[0]

    ### a
    x[1] = u[1]*(a[1]-a[0])+a[0]

    ### e
    x[2] = u[2]*(e[1]-e[0])+e[0]

    ### omega
    x[3] = 2*np.pi*u[3]*(omega[1]-omega[0])+omega[0]

    ### Tperi
    #period = np.sqrt(x[1]**3)*365.25
    # x[0] = period - period/2.
    

    return x

class prior():
    def __init__(self, phase0,a,e,omega):
        self.phase0 = phase0
        self.a = a
        self.e = e
        self.omega = omega
    def __call__(self, x):
        return prior_transform(x, self.phase0,self.a, self.e, self.omega)

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


def run_fit(jds, rs_fit, rs_err, thetas_fit, thetas_err,sampler='dynesty', nlive=100,dlogz=0.5,bootstrap=0,phase0=[0,1],a=[0.1,10],e=[0,0.99],omega=[0,1]):
    prefix = '/content/fit_results/'
    if sampler=='dynesty':
        import dynesty
        loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
        prior_func = prior(phase0,a,e,omega)
        dsampler = dynesty.NestedSampler(loglike_func, prior_func, 4,
                                                 nlive=nlive,bootstrap=bootstrap)
        dsampler.run_nested(dlogz=dlogz)
        res = dsampler.results
        return res.samples_equal()
    else:
        import ultranest
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
        prior_func = prior(phase0,a,e,omega)
        sampler = ultranest.ReactiveNestedSampler(['phase0', 'a', 'e','omega'], loglike_func,prior_func,log_dir=prefix,resume='overwrite')
        result = sampler.run(min_num_live_points=nlive,dlogz=dlogz,min_ess=nlive,update_interval_volume_fraction=0.4,max_num_improvement_loops=1)
        return result['samples']
    # else:
    #     from pymultinest.solve import solve
    #     if not os.path.isdir(prefix):
    #         os.mkdir(prefix)
    #     loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
    #     result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=100, evidence_tolerance=0.5,
    #                     outputfiles_basename=prefix, verbose=False, resume=False)
    #     samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]
    #     return samples


# def run_fit_dynesty(jds, rs_fit, rs_err, thetas_fit, thetas_err):
#     loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
#     dsampler = dynesty.NestedSampler(loglike_func, prior_transform, 4,
#                                              nlive=400)
#     dsampler.run_nested(dlogz=0.5)
#     res = dsampler.results
#     return res.samples_equal()

def make_images(obsdate, jd, r, theta, delta,image_list, fwhm, fluxlevel,noiselevel,output_str, output_dir: Path=FILE_DIR):
    parallax, sin_sun = dist_to_parallax(jd, r, theta, delta)
    dtheta = delta*2*np.pi/365.25
    baseline = np.sin(dtheta/2)

    ### Pick an image
    nimages = len(image_list)
    idx = np.random.randint(0,nimages)
    hdulst = fits.open(image_list[idx])
    im1, im2, f1, f2 = inject_asteroid(hdulst, parallax, obsdate, delta, jd, theta, sin_sun, fwhm, fluxlevel,noiselevel,output_str, output_dir=output_dir)

    return im1, im2, f1, f2

def make_jds(dates):
    jds = []
    for date in dates:
        jds.append(Time(date+'T12:00:00', format='isot', scale='utc').jd)
    jds = np.asarray(jds)
    return jds


def plot_fit(rs_fit, thetas_fit, samples, truths=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ### Plot Earth

    ### Plot asteroid
    if truths is not None:
        a = truths[1]
        period = 365.25*np.sqrt(a**3)
        ptimes = np.linspace(0, 2*period,300)
        true_r, true_theta = make_orbit(ptimes, *truths)
        ax.plot(true_theta, true_r, color='k')
    
    ptimes = np.linspace(0, 2*period,300)
    for i in range(200):
        rs, thetas = make_orbit(ptimes, *samples[i])
        if i!=0:
            ax.plot(thetas,rs,color='r',alpha=0.05)
        else:
            ax.plot(thetas,rs,color='r',alpha=0.05,label='Possible asteroid orbits')
    ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g',label='Earth\'s orbit')
    ax.plot(np.linspace(0,2*np.pi,100),5.2*np.ones(100),color='m',label='Jupiter\'s orbit')
    
    ax.scatter(thetas_fit, rs_fit, label='Measured asteroid positions')
    ax.scatter(0,0,s=60,color='y',marker='*', label='Sun')

    ### Turn off axes ticks, set axis limits based on semi-major axis
    # ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    a_low, a_med, a_high = np.nanpercentile(samples[:,1], [0.5,0.5,0.95])
    rmax = 2*a_med+2*(a_high-a_med)
    if np.any(rs_fit < rmax):
        rmax = 2*np.max(rs_fit)
    ax.set_rmax(rmax)
    ax.legend()

    ### print orbital period
    period = np.round(np.sqrt(a_med**3),2)
    p_low = np.round(np.sqrt(a_low**3),2)
    p_high = np.round(np.sqrt(a_high**3),2)
    print(r'Measured orbital period = '+str(period)+' +'+str(p_high-period) + ' / -'+str(period-p_low)+' years')

    return fig


# if __name__ == '__main__':
#     ### Image filenames
#     image_list = glob('imagedata/*.fits')
#     nimages = len(image_list)
#     print(image_list)

#     ### True parameters
#     p0, a, e, omega = 0.2, 1.2, 0.83, np.pi/2.
#     period = 365.25*np.sqrt(a**3)

#     print('Period [days]:', period)
    

#     ### Given a list of dates, predict observables
#     obsdates = ['2025-01-18', '2025-03-02', '2025-04-01', '2025-04-29', '2025-05-12']
#     obsdelta = 4/24. ### In days

#     jds = []
#     for date in obsdates:
#         jds.append(Time(date+'T12:00:00', format='isot', scale='utc').jd)
#     jds = np.asarray(jds)
    
#     rs, thetas = make_orbit(jds, p0, a, e, omega)

#     for i in range(len(rs)):
#         # ax.scatter(2*np.pi/365.25 *(jds[i]-2460392.400856), 1)
#         # ax.scatter(thetas[i], rs[i])
#         parallax = dist_to_parallax(jds[i], rs[i], thetas[i], obsdelta)
#         dtheta = obsdelta*2*np.pi/365.25
#         baseline = np.sin(dtheta/2)
#         print(obsdates[i], 'Parallax:', parallax, 'Distance:', 206265*baseline/parallax)

#         ### Pick an image
#         idx = np.random.randint(0,nimages)
#         hdulst = fits.open(image_list[idx])
#         im1, im2 = inject_asteroid(hdulst, parallax, obsdates[i], obsdelta)



#     ## Add errors to rs, thetas
#     ### Note that students will give us values for r and its error
#     ### We will know theta already from the orbital parameters
#     rs_err = np.random.normal(0*np.ones_like(rs),3e-2)
#     rs_fit=rs+rs_err

#     ### This one we specify
#     thetas_err = np.random.normal(0*np.ones_like(thetas),1e-4)
#     thetas_fit = thetas+thetas_err


#     prefix = 'fit_results/'
#     loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
#     result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=400, evidence_tolerance=0.5,
#                     outputfiles_basename=prefix, verbose=False, resume=False)
#     samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]


#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.scatter(thetas_fit, rs_fit)
#     ptimes = np.linspace(0, 2*period,300)
#     true_r, true_theta = make_orbit(ptimes, p0, a, e, omega)
#     ax.plot(true_theta, true_r, color='k')
#     for i in range(200):
#         rs, thetas = make_orbit(ptimes, *samples[i])
#         ax.plot(thetas,rs,color='r',alpha=0.05)
#     ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g')
#     plt.show()



   
