import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits

FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def xy_to_rthet(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian (x, y) coordinates to polar (r, θ) coordinates.

    This function transforms 2D Cartesian coordinates into polar coordinates,
    where the angle θ is normalized to be within the range [0, 2π].

    Args:
        x (np.ndarray): Array of x-coordinates
        y (np.ndarray): Array of y-coordinates

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - r (np.ndarray): Array of radial distances from origin
            - theta (np.ndarray): Array of angles in radians, range [0, 2π]

    Note:
        - Input arrays must be broadcastable to the same shape
        - The angle is measured counterclockwise from the positive x-axis
        - Uses numpy's arctan2 for correct quadrant handling

    Example:
        >>> x = np.array([1, 1/np.sqrt(2), 0])
        >>> y = np.array([0, 1/np.sqrt(2), 1])
        >>> r, theta = xy_to_rthet(x, y)
        >>> print(r)  # array([1., 1., 1.])
        >>> print(theta)  # array([0., π/4, π/2])
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta[theta<0]+=2*np.pi
    return r, theta

def rthet_to_xy(r: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar (r, θ) coordinates to Cartesian (x, y) coordinates.

    This function transforms polar coordinates into 2D Cartesian coordinates
    using the standard transformation equations:
    x = r cos(θ)
    y = r sin(θ)

    Args:
        r (np.ndarray): Array of radial distances from origin
        theta (np.ndarray): Array of angles in radians

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x (np.ndarray): Array of x-coordinates
            - y (np.ndarray): Array of y-coordinates

    Note:
        - Input arrays must be broadcastable to the same shape
        - Assumes angles are in radians
        - The function works with both positive and negative angles

    Example:
        >>> r = np.array([1., 1., 1.])
        >>> theta = np.array([0., np.pi/4, np.pi/2])
        >>> x, y = rthet_to_xy(r, theta)
        >>> print(x)  # array([1., 0.707.., 0.])
        >>> print(y)  # array([0., 0.707.., 1.])
    """
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def solve_kepler(mobs: np.ndarray, e: float, order: int = 100) -> np.ndarray:
    """Solve Kepler's equation using fixed-point iteration.

    Solves the equation M = E - e*sin(E) for eccentric anomaly E,
    given mean anomaly M and eccentricity e. Uses the iterative method:
    E_{n+1} = M + e*sin(E_n)

    Args:
        mobs (np.ndarray): Array of mean anomalies M in radians
        e (float): Orbital eccentricity (0 ≤ e < 1)
        order (int, optional): Number of iterations for convergence. 
            Defaults to 100.

    Returns:
        np.ndarray: Array of eccentric anomalies E in radians

    Note:
        - Convergence is guaranteed for e < 1
        - The number of iterations needed for convergence increases with e
        - For nearly parabolic orbits (e ≈ 1), more iterations may be needed
        - Initial guess is E_0 = M

    Example:
        >>> M = np.array([0., np.pi/2, np.pi])
        >>> e = 0.1
        >>> E = solve_kepler(M, e)
        >>> np.all(np.abs(E - M - e*np.sin(E)) < 1e-6)  # True

    References (generated by Claude, hopefully not hallucinations):
        For details on Kepler's equation and solution methods, see:
        - Murray & Dermott "Solar System Dynamics", Chapter 2
        - Vallado "Fundamentals of Astrodynamics and Applications", Chapter 2
    """
    Eobs = np.copy(mobs)
    for _ in range(1,order+1):
        Eobs = mobs + e*np.sin(Eobs)
    return Eobs

def make_orbit(
    times: np.ndarray,
    phase0: float,
    a: float,
    e: float,
    omega: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate orbital positions at specified times for a Keplerian orbit.

    Computes the radial distances and orbital angles for an object in a
    Keplerian orbit around the Sun, given orbital elements and observation times.
    Uses the numerically stable tangent half-angle formula for computing
    true anomaly from eccentric anomaly.

    Args:
        times (np.ndarray): Array of observation times in Julian Days
        phase0 (float): Initial orbital phase in cycles
        a (float): Semi-major axis in AU
        e (float): Eccentricity (0 ≤ e < 1)
        omega (float): Argument of perihelion in radians

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - radii (np.ndarray): Array of radial distances in AU
            - angles (np.ndarray): Array of orbital angles in radians [0, 2π]

    Note:
        - Period is calculated using Kepler's Third Law
        - All angles are in radians
        - The orbital period assumes 365.25 days per year
        - Phase0 represents the orbital phase at the first observation time
        - The returned angle is the true anomaly plus omega, normalized to [0, 2π]
    """
    period = np.sqrt(a**3) * 365.25
    mean_anomaly = 2 * np.pi * (phase0 + (times - 0) / period)
    eccentric_anomaly = solve_kepler(mean_anomaly, e)
    
    # Convert eccentric anomaly to true anomaly using the tangent half-angle formula
    beta = e / (1 + np.sqrt(1 - e**2))
    true_anomaly = eccentric_anomaly + 2 * np.arctan(
        beta * np.sin(eccentric_anomaly) / (1 - beta * np.cos(eccentric_anomaly))
    )
    
    # Calculate radial distances using the orbit equation
    radii = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))

    # Add omega and normalize angles to [0, 2π]
    angles = (true_anomaly + omega) % (2 * np.pi)

    return radii, angles

def parallax_to_dist(
    coords1: tuple[float, float],
    err1: tuple[float, float],
    coords2: tuple[float, float],
    err2: tuple[float, float],
    dt: float,
    true_error_propagation: bool = False
) -> tuple[float, float]:
    """Calculate distance from parallax measurements between two observations.

    Computes the distance to an object using parallax measurements from two
    observations separated by a known time interval. Uses Earth's orbital motion
    as the baseline for parallax measurement.

    Args:
        coords1 (tuple[float, float]): First position (RA, Dec) in degrees
        err1 (tuple[float, float]): Uncertainties in first position (σRA, σDec)
        coords2 (tuple[float, float]): Second position (RA, Dec) in degrees
        err2 (tuple[float, float]): Uncertainties in second position (σRA, σDec)
        dt (float): Time difference between observations in days
        true_error_propagation (bool, optional): If True, uses full error
            propagation instead of simple percentage error. Currently not
            implemented. Defaults to False.

    Returns:
        tuple[float, float]: A tuple containing:
            - dist (float): Distance to object in AU
            - dist_err (float): Uncertainty in distance in AU

    Note:
        - Input coordinates must be in degrees
        - Uses Earth's orbital period of 365.25 days
        - When true_error_propagation=False, assumes 5% distance uncertainty
        - Baseline is calculated as sin(θ/2) where θ is Earth's orbital angle
          between observations

    Raises:
        NotImplementedError: If true_error_propagation=True

    Example:
        >>> coords1 = (180.0, 20.0)  # RA, Dec for first observation
        >>> coords2 = (180.1, 20.1)  # RA, Dec for second observation
        >>> err1 = err2 = (0.001, 0.001)  # Uncertainties
        >>> dist, dist_err = parallax_to_dist(coords1, err1, coords2, err2, 4.0)
    """
    
    # Calculate Earth's orbital angle and baseline
    theta_earth = 2 * np.pi * dt / 365.25
    baseline = np.sin(theta_earth / 2.)  # AU

    # Extract coordinates
    ra1, d1 = coords1
    ra2, d2 = coords2

    # Calculate parallax angle in degrees
    parallax = np.sqrt((ra1 - ra2)**2 + (d1 - d2)**2)
    
    # Convert distance using small angle approximation
    dist = baseline / np.sin(parallax * np.pi / 180.)

    if true_error_propagation:
        raise NotImplementedError("""
        Full error propagation is not yet implemented. The method would be:
        
        # Calculate combined coordinate uncertainties
        dra = np.sqrt((e1[0]**2 + e2[0]**2))
        ddec = np.sqrt((e1[1]**2 + e2[1]**2))
        
        # Propagate to parallax uncertainty
        parallax_err = np.sqrt(
            (ra1-ra2)**2/parallax**2 * dra**2 + 
            (d1-d2)**2/parallax**2 * ddec**2
        )
        
        # Convert to angular uncertainty
        sinp_err = np.sqrt(
            np.cos(parallax*np.pi/180.)*parallax_err*np.pi/180.
        )
        
        # Final distance uncertainty
        dist_err = dist * sinp_err/np.sin(parallax*np.pi/180.)
        """)
    else:
        # Use simple 3% distance uncertainty
        dist_err = 0.03 * dist

    return dist, dist_err

def dist_to_parallax(
    time: float,
    r: float,
    theta: float,
    dt: float
) -> tuple[float, float]:
    """Calculate expected parallax and solar elongation for an object's position.

    Given an object's position in polar coordinates and observation timing,
    computes the expected parallax angle and solar elongation. This is
    effectively the inverse of parallax_to_dist().

    Args:
        time (float): Observation time in Julian Days
        r (float): Object's distance from Sun in AU
        theta (float): Object's angular position in radians
        dt (float): Time difference between observations in days

    Returns:
        tuple[float, float]: A tuple containing:
            - parallax (float): Expected parallax angle in arcseconds
            - sin_sun (float): Sine of the solar elongation angle

    Note:
        - Uses JD 2460392.400856 as the reference epoch
        - Assumes Earth's orbit is circular with radius 1 AU
        - Converts output parallax from radians to arcseconds (x206265)
        - Solar elongation is returned as sin(angle) for efficiency
        - Earth's orbital period is assumed to be 365.25 days

    Example:
        >>> time = 2460393.5  # Julian Date
        >>> r = 2.5  # AU
        >>> theta = np.pi/4  # radians
        >>> dt = 4.0  # days
        >>> parallax, sin_sun = dist_to_parallax(time, r, theta, dt)
    """
    # Calculate Earth's position at observation time
    REFERENCE_JD = 2460392.400856  # Reference Julian Date
    RAD_TO_ARCSEC = 206265.0  # Conversion factor
    
    theta_earth = 2 * np.pi / 365.25 * (time - REFERENCE_JD)
    r_earth = 1  # Earth's orbital radius in AU
    
    # Convert positions to Cartesian coordinates
    xe, ye = rthet_to_xy(r_earth, theta_earth)  # Earth's position
    xa, ya = rthet_to_xy(r, theta)  # Asteroid's position
    
    # Calculate Earth-asteroid distance
    distance = np.sqrt((xe - xa)**2 + (ye - ya)**2)

    # Calculate parallax angle
    dtheta = dt * 2 * np.pi / 365.25
    baseline = np.sin(dtheta / 2)  # Baseline in AU
    parallax = np.arcsin(baseline / distance)  # in radians
    print(f"{distance=}, {parallax=} radians")

    # Calculate solar elongation
    sin_sun = r / distance * np.sin(theta - theta_earth)

    return parallax * RAD_TO_ARCSEC, sin_sun

def dist_to_r(
    time: float,
    theta: float,
    elong: float,
    dist: float,
    dist_err: float
) -> tuple[float, float]:
    """Convert geocentric distance to heliocentric distance.

    Calculates an object's distance from the Sun given its distance from Earth,
    angular position, and solar elongation. Uses the law of sines to relate
    Earth-object distance to Sun-object distance.

    Args:
        time (float): Observation time in Julian Days
        theta (float): Object's angular position in radians
        elong (float): Sine of the solar elongation angle
        dist (float): Distance from Earth to object in AU
        dist_err (float): Uncertainty in Earth-object distance in AU

    Returns:
        tuple[float, float]: A tuple containing:
            - r (float): Distance from Sun to object in AU
            - r_err (float): Uncertainty in Sun-object distance in AU

    Note:
        - Uses JD 2460392.400856 as the reference epoch
        - Assumes Earth's orbit is circular with radius 1 AU
        - Input elongation should be sine of the angle
        - Uses error propagation assuming error only in distance
        - Earth's orbital period is assumed to be 365.25 days
        
    Warning:
        Function can be unstable when sin(theta - theta_earth) is close to zero,
        i.e., when the object is near conjunction or opposition.

    Example:
        >>> time = 2460393.5  # Julian Date
        >>> theta = np.pi/4  # radians
        >>> elong = 0.7  # sine of elongation angle
        >>> dist = 2.0  # AU
        >>> dist_err = 0.1  # AU
        >>> r, r_err = dist_to_r(time, theta, elong, dist, dist_err)

    Todo:
        Consider reimplementing using vector addition for better numerical
        stability and clarity.
    """
    # Calculate Earth's angular position
    REFERENCE_JD = 2460392.400856
    theta_earth = 2 * np.pi / 365.25 * (time - REFERENCE_JD)

    # Calculate heliocentric distance using law of sines
    angle_diff = theta - theta_earth
    r = dist * elong / np.sin(angle_diff)
    
    # Propagate distance error
    r_err = dist_err * elong / np.sin(angle_diff)

    return r, r_err

def Gauss2D(
    x: np.ndarray,
    y: np.ndarray,
    amp: float,
    x0: float,
    y0: float,
    sigx: float,
    sigy: float,
    rot: float = 0
) -> np.ndarray:
    """Calculate a 2D Gaussian function with optional rotation.

    Implements a generalized 2D Gaussian function of the form:
    f(x,y) = amp * exp(-[a(x-x0)² + 2b(x-x0)(y-y0) + c(y-y0)²])
    where a, b, c are coefficients determined by the rotation and sigmas.

    Args:
        x (np.ndarray): x coordinates (meshgrid)
        y (np.ndarray): y coordinates (meshgrid)
        amp (float): Amplitude of the Gaussian
        x0 (float): Center x coordinate
        y0 (float): Center y coordinate
        sigx (float): Standard deviation in x direction
        sigy (float): Standard deviation in y direction
        rot (float, optional): Rotation angle in radians. Defaults to 0.

    Returns:
        np.ndarray: 2D array of Gaussian values

    Note:
        - The rotation is counterclockwise
        - x and y should be generated using np.meshgrid
        - The function handles arbitrary aspect ratios (sigx ≠ sigy)
        - For an unrotated Gaussian, set rot=0

    Example:
        >>> x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        >>> gauss = Gauss2D(x, y, amp=1.0, x0=0, y0=0, sigx=1, sigy=2, rot=np.pi/4)

    References:
        For the mathematical derivation, see:
        https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """
    # Calculate the coefficients for the rotated Gaussian
    a = np.cos(rot)**2 / (2 * sigx**2) + np.sin(rot)**2 / (2 * sigy**2)
    b = np.sin(2 * rot) / (2 * sigx**2) - np.sin(2 * rot) / (2 * sigy**2)
    c = np.sin(rot)**2 / (2 * sigx**2) + np.cos(rot)**2 / (2 * sigy**2)

    # Calculate the Gaussian using the quadratic form
    return amp * np.exp(-a * (x - x0)**2 - b * (x - x0) * (y - y0) - c * (y - y0)**2)


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
    # out_dir = output_dir/'injected_images'/output_str
    output_dir.mkdir(exist_ok=True, parents=True)
    # out_dir.mkdir(exist_ok=True, parents=True)

    fname1 = output_dir/(obsdate+'_'+output_str+'_frame1.fits')
    fname2 = output_dir/(obsdate+'_'+output_str+'_frame2.fits')
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


def run_fit(jds, rs_fit, rs_err, thetas_fit, thetas_err,sampler='dynesty', nlive=100,dlogz=0.5,bootstrap=0,phase0=[0,1],a=[0.4,40],e=[0,0.99],omega=[0,1]):
    prefix = '/content/fit_results/'
    if sampler=='dynesty':
        import dynesty
        loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
        prior_func = prior(phase0,a,e,omega)
        dsampler = dynesty.NestedSampler(loglike_func, prior_func, 4,
                                                 nlive=nlive,bootstrap=bootstrap)
        dsampler.run_nested(dlogz=dlogz,maxcall=200000)
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

def make_images(
        obsdate, 
        jd, 
        r, 
        theta, 
        delta,
        image_list,
        output_str,  
        fwhm = 3.5, 
        fluxlevel = 50,
        noiselevel = 20,
        output_dir: Path=FILE_DIR
        ):
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


def plot_fit(dates, rs_fit, thetas_fit, samples, truths=None, default_plot_period=10):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(10,10))
    ### Plot Earth

    ### Plot asteroid
    if truths is not None:
        a = truths[1]
        period = 365.25*np.sqrt(a**3)
        ptimes = np.linspace(0, 2*period,300)
        true_r, true_theta = make_orbit(ptimes, *truths)
        ax.plot(true_theta, true_r, color='k')
    else:
        period = default_plot_period*365.25 #years
    
    ptimes = np.linspace(0, 2*period,300)
    for i in range(200):
        rs, thetas = make_orbit(ptimes, *samples[i])
        if i!=0:
            ax.plot(thetas,rs,color='r',alpha=0.05)
        else:
            ax.plot(thetas,rs,color='r',alpha=0.05,label='Possible asteroid orbits')
    ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g',label='Earth\'s orbit')
    ax.plot(np.linspace(0,2*np.pi,100),5.2*np.ones(100),color='m',label='Jupiter\'s orbit')
    
    for i in range(len(rs_fit)):
        ax.scatter(thetas_fit[i], rs_fit[i], color='k',alpha=1.0,s=30,zorder=400)
        ax.text(thetas_fit[i]-1e-2,rs_fit[i]+1e-2,dates[i])
    ax.scatter(0,0,s=120,color='y',marker='*', label='Sun')

    ### Turn off axes ticks, set axis limits based on semi-major axis
    # ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    a_low, a_med, a_high = np.nanpercentile(samples[:,1], [0.05,0.5,0.95])
    rmax = 2*a_med+2*(a_high-a_med)
    if np.any(rs_fit < rmax) or rmax < 2.:
        rmax = 2*np.max(rs_fit)
        if rmax  < 2:
            rmax = 2.
    ax.set_rmax(rmax)
    ax.legend()

    ### print orbital period
    period = np.round(np.sqrt(a_med**3),2)
    errp_low = np.round(np.sqrt(a_med**3) - np.sqrt(a_low**3),2 )
    errp_high = np.round(np.sqrt(a_high**3) - np.sqrt(a_med**3),2)
    fig.text(0.1,0.8,r'Measured orbital period = '+str(period)+' +'+str(errp_low) + ' / -'+str(errp_high)+' years')

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



   
