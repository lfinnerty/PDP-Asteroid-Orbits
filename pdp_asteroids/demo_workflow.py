import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time
import json
from pymultinest.solve import solve
import corner
from glob import glob
from astropy.io import fits
from position_to_orbit import *
from image_switcher import FitsImageSwitcher as switch
from image_clicker import ImageClicker as click
# from IPython import embed
# embed()

FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
image_list = glob('imagedata/*.fits')
nimages = len(image_list)


def run_fit(jds, rs_fit, rs_err, thetas_fit, thetas_err):
    prefix = 'fit_results/'
    loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
    result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=400, evidence_tolerance=0.5,
                    outputfiles_basename=prefix, verbose=False, resume=False)
    samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]


def make_images(obsdate, jd, r, theta, delta):
    parallax = dist_to_parallax(jds, rs, theta, delta)
    dtheta = delta*2*np.pi/365.25
    baseline = np.sin(dtheta/2)

    ### Pick an image
    idx = np.random.randint(0,nimages)
    hdulst = fits.open(image_list[idx])
    im1, im2, f1, f2 = inject_asteroid(hdulst, parallax, obsdate, delta)

    return im1, im2, f1, f2

def make_jds(dates):
    jds = []
    for date in obsdates:
        jds.append(Time(date+'T12:00:00', format='isot', scale='utc').jd)
    jds = np.asarray(jds)
    return jds


def plot_fit(rs_fit, thetas_fit, samples, truths=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ### Plot Earth

    ### Plot asteroid
    if truth is not None:
        ptimes = np.linspace(0, 2*period,300)
        true_r, true_theta = make_orbit(ptimes, *truths)
        ax.plot(true_theta, true_r, color='k')
    ax.scatter(thetas_fit, rs_fit)
    ptimes = np.linspace(0, 2*period,300)
    for i in range(200):
        rs, thetas = make_orbit(ptimes, *samples[i])
        ax.plot(thetas,rs,color='r',alpha=0.05)
    ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g')
    
    return fig


if __name__ == '__main__':
    ### True parameters
    p0, a, e, omega = 0.2, 1.2, 0.83, np.pi/2.
    period = 365.25*np.sqrt(a**3)
    theta_err = 1e-4

    ### Initial date
    obsdates = ['2025-01-18']#, '2025-03-02', '2025-04-01', '2025-04-29', '2025-05-12']
    obsdeltas = [4/24.] ### In days

    fit_r, err_r = [],[]
    while True:
        ### Make the new images
        jds = make_jds(obsdates)
        rs, thetas = make_orbit(jds, p0, a, e, omega)
        _, _, fname1, fname2 = make_images(obsdates[-1], jds[-1], rs[-1], thetas[-1], obsdeltas[-1])


        ### Display the images, let them blink. Get values from image click
        ### Instantiate the image clicker
        accept = False
        while not accept:
            blinker = switch(fname1, fname2)
            blinker.display()

            img1 = click(fname1)
            img1.run_plot_context()
            img1.display()
            img2 = click(fname2)
            img2.run_plot_context()
            img2.display()
            acc = input('Accept positions? >> ')
            if acc in ['y', 'Y', 'yes']:
                accept = True

        dist, dist_err = parallax_to_dist(img1.get_coords(), img1.get_err(), img2.get_coords(), img2.get_err(), obsdeltas[-1])
        r, r_err = ra_dist_to_r_theta(jds[-1], thetas[-1], dist)

        fit_r.append(dist)
        err_r.append(r_err)

        ### Fit and plot the orbit
        samples = run_fit(jds, fit_r, err_r, thetas, 1e-4*np.ones_like(thetas))
        fig  = plot_figure_animation(rs_fit, thetas_fit, samples, truths=[p0,a,e,omega])
        plt.show(block=False)

        ### Ask for a new date, baseline
        newDate = False
        while not newDate:
            try:
                print('Current dates:', obsdates)
                new_date = input('Enter new date >> ')
                obsdates.append(new_date)
                jds = make_jds(obsdates)
                newDate = True
            except:
                print('Invalid date!')
        
        newDelta = False
        while not newDelta:
            try:
                print('Current time between frames, in hours:', obsdeltas[-1])
                new_delta = input('Enter new time between frames, in hours >> ')
                obsdelta = float(newDelta)/24.
                newDelta = True
            except:
                print('Invalid entry!')

        ### Add option for removing an old data point

        plt.close()

