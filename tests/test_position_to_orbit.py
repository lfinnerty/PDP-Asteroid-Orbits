import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits
import pdp_asteroids

import pdp_asteroids
from pdp_asteroids.position_to_orbit import (
    xy_to_rthet, 
    rthet_to_xy,
    solve_kepler,
    make_orbit,
    dist_to_parallax,
    Gauss2D,
    inject_asteroid,
    logl,
    prior_transform
)

def test_xy_to_rthet():
    x = np.array([1, 2**(-1/2), 0])
    y = np.array([0, 2**(-1/2), 1])
    r, theta = xy_to_rthet(x, y)
    
    expected_r = np.array([1, 1, 1])
    expected_theta = np.array([0, np.pi/4, np.pi/2])
    
    np.testing.assert_almost_equal(r, expected_r, decimal=6)
    np.testing.assert_almost_equal(theta, expected_theta, decimal=6)

def test_rthet_to_xy():
    r = np.array([1, 1, 1])
    theta = np.array([0, np.pi/4, np.pi/2])
    x, y = rthet_to_xy(r, theta)
    
    expected_x = np.array([1, 2**(-1/2), 0])
    expected_y = np.array([0, 2**(-1/2), 1])
    
    np.testing.assert_almost_equal(x, expected_x, decimal=6)
    np.testing.assert_almost_equal(y, expected_y, decimal=6)

def test_solve_kepler():
    M = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    e = 0.1
    E = solve_kepler(M, e)
    assert np.all(np.abs(E - M - e*np.sin(E)) < 1e-6)

def test_make_orbit():
    times = np.array([0, 100, 200])
    phase0, a, e, omega = 0.1, 1.5, 0.3, np.pi/4
    rs, nus = make_orbit(times, phase0, a, e, omega)
    assert len(rs) == 3
    assert len(nus) == 3

def test_dist_to_parallax():
    time, r, theta, dt = 2460000, 2, np.pi/3, 1
    parallax = dist_to_parallax(time, r, theta, dt)
    assert isinstance(parallax, float)

def test_Gauss2D():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    amp, x0, y0, sigx, sigy = 1, 1, 1, 1, 1
    result = Gauss2D(x, y, amp, x0, y0, sigx, sigy)
    assert result.shape == (3,)


from pymultinest.solve import solve

@pytest.fixture
def setup_test_data():
    # Set up test parameters
    p0, a, e, omega = 0.2, 1.2, 0.83, np.pi/2.
    period = 365.25 * np.sqrt(a**3)
    
    obsdates = ['2025-01-18', '2025-03-02', '2025-04-01', '2025-04-29', '2025-05-12']
    obsdelta = 4/24.  # In days
    
    jds = [Time(date+'T12:00:00', format='isot', scale='utc').jd for date in obsdates]
    jds = np.array(jds)
    
    rs, thetas = make_orbit(jds, p0, a, e, omega)
    
    return p0, a, e, omega, period, obsdates, obsdelta, jds, rs, thetas


def test_injection(setup_test_data):
    p0, a, e, omega, period, obsdates, obsdelta, jds, rs, thetas = setup_test_data

    ### Image filenames
    root_dir = Path(pdp_asteroids.__file__).parent
    image_list = [str(s) for s in list((root_dir/'imagedata').glob('*.fits'))]
    nimages = len(image_list)

    ### True parameters
    p0, a, e, omega = 0.2, 1.2, 0.83, np.pi/2.
    period = 365.25*np.sqrt(a**3)
        

    ### Given a list of dates, predict observables
    obsdates = ['2025-01-18', '2025-03-02', '2025-04-01', '2025-04-29', '2025-05-12']
    obsdelta = 4/24. ### In days

    jds = []
    for date in obsdates:
        jds.append(Time(date+'T12:00:00', format='isot', scale='utc').jd)
    jds = np.asarray(jds)
    
    rs, thetas = make_orbit(jds, p0, a, e, omega)

    for i in range(len(rs)):
        # ax.scatter(2*np.pi/365.25 *(jds[i]-2460392.400856), 1)
        # ax.scatter(thetas[i], rs[i])
        parallax = dist_to_parallax(jds[i], rs[i], thetas[i], obsdelta)
        dtheta = obsdelta*2*np.pi/365.25
        baseline = np.sin(dtheta/2)

        ### Pick an image
        idx = np.random.randint(0,nimages)
        hdulst = fits.open(image_list[idx])
        im1, im2 = inject_asteroid(hdulst, parallax, obsdates[i], obsdelta)

def test_fit_result(setup_test_data):
    p0, a, e, omega, period, obsdates, obsdelta, jds, rs, thetas = setup_test_data
    root_dir = Path(pdp_asteroids.__file__).parent

    ## Add errors to rs, thetas
    ### Note that students will give us values for r and its error
    ### We will know theta already from the orbital parameters
    rs_err = np.random.normal(0*np.ones_like(rs),3e-2)
    rs_fit=rs+rs_err

    ### This one we specify
    thetas_err = np.random.normal(0*np.ones_like(thetas),1e-4)
    thetas_fit = thetas+thetas_err


    prefix = root_dir / 'fit_results/'
    prefix.mkdir(exist_ok=True)
    prefix = str(prefix) + '/'
    loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
    result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=400, evidence_tolerance=0.5,
                    outputfiles_basename=prefix, verbose=False, resume=False)
    samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]



#    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#    ax.scatter(thetas_fit, rs_fit)
#    ptimes = np.linspace(0, 2*period,300)
#    true_r, true_theta = make_orbit(ptimes, p0, a, e, omega)
#    ax.plot(true_theta, true_r, color='k')
#    for i in range(200):
#        rs, thetas = make_orbit(ptimes, *samples[i])
#        ax.plot(thetas,rs,color='r',alpha=0.05)
#    ax.plot(np.linspace(0,2*np.pi,100),np.ones(100),color='g')
#    plt.show()







#def test_integration(setup_test_data):
#    p0, a, e, omega, period, obsdates, obsdelta, jds, rs, thetas = setup_test_data
#
#    root_dir = Path(pdp_asteroids.__file__).parent
#    
#    # Get image files
#    image_files = list((root_dir/'imagedata').glob('*.fits'))
#    assert len(image_files) > 0, "No image files found in pdp_asteroids/imagedata/"
#
#    # Create a temporary directory for injected images
#    injected_dir = root_dir / "injected_images"
#    injected_dir.mkdir(exist_ok=True)
#    
#    # Inject asteroids into images
#    for i, obsdate in enumerate(obsdates):
#        parallax = dist_to_parallax(jds[i], rs[i], thetas[i], obsdelta)
#        idx = np.random.randint(0, len(image_files))
#        print(image_files[idx], idx)
#        with fits.open(image_files[idx]) as hdulst:
#            im1, im2 = inject_asteroid(hdulst, parallax, obsdate, obsdelta, output_dir=Path('./tests/'))
#        
#        # Check that injected images were created
#        assert (injected_dir / f'{obsdate}_frame1.fits').exists()
#        assert (injected_dir / f'{obsdate}_frame2.fits').exists()
#    
#    # Add errors to rs, thetas
#    rs_err = np.random.normal(0, 3e-2, size=rs.shape)
#    rs_fit = rs + rs_err
#    
#    thetas_err = np.random.normal(0, 1e-4, size=thetas.shape)
#    thetas_fit = thetas + thetas_err
#    
#    # Run MultiNest fit
#    prefix = str(root_dir / 'fit_results/')
#    loglike_func = logl(jds, rs_fit, rs_err, thetas_fit, thetas_err)
#    result = solve(loglike_func, prior_transform, n_dims=4, n_live_points=400, 
#                   evidence_tolerance=0.5, outputfiles_basename=prefix, 
#                   verbose=False, resume=False)
#    
#    # Check that fit results were created
#    assert (tmp_path / 'fit_results/post_equal_weights.dat').exists()
#    
#    # Load and check fit results
#    samples = np.genfromtxt(prefix+'post_equal_weights.dat')[:,:-1]
#    assert samples.shape[1] == 4  # Should have 4 parameters
#    
##    # Plot results (optional, comment out if you don't want to generate plots during testing)
##    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
##    ax.scatter(thetas_fit, rs_fit)
##    ptimes = np.linspace(0, 2*period, 300)
##    true_r, true_theta = make_orbit(ptimes, p0, a, e, omega)
##    ax.plot(true_theta, true_r, color='k')
##    for i in range(min(200, len(samples))):
##        rs, thetas = make_orbit(ptimes, *samples[i])
##        ax.plot(thetas, rs, color='r', alpha=0.05)
##    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100), color='g')
##    plt.savefig(str(tmp_path / 'orbit_fit.png'))
##    plt.close()
#
#    # You could add more specific assertions here to check the quality of the fit
#    # For example, checking if the median of the posterior is close to the true values
#    median_fit = np.median(samples, axis=0)
#    assert np.allclose([p0, a, e, omega], median_fit, rtol=0.1, atol=0.1)
