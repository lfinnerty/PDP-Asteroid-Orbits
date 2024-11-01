import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits

from pdp_asteroids.position_to_orbit import (
    xy_to_rthet,
    rthet_to_xy,
    solve_kepler,
    make_orbit,
    dist_to_parallax,
    Gauss2D,
    inject_asteroid,
    Prior,
    run_fit
)

# Get path to test data relative to this test file
TEST_DIR = Path(__file__).parent
TEST_DATA = TEST_DIR / 'testdata.fits'

@pytest.fixture
def test_fits_data():
    """Load the test FITS file"""
    return fits.open(TEST_DATA)

@pytest.fixture
def orbit_params():
    """Basic orbital parameters for testing"""
    return {
        'phase0': 0.2,
        'a': 1.2,
        'e': 0.83,
        'omega': np.pi/2,
    }

@pytest.fixture
def observation_times():
    """Sample observation dates and times"""
    dates = ['2025-01-18', '2025-03-02', '2025-04-01']
    times = [Time(date+'T12:00:00', format='isot', scale='utc').jd 
            for date in dates]
    return np.array(times)

def test_xy_to_rthet():
    """Test conversion from cartesian to polar coordinates"""
    x = np.array([1, 2**(-1/2), 0])
    y = np.array([0, 2**(-1/2), 1])
    r, theta = xy_to_rthet(x, y)
    
    expected_r = np.array([1, 1, 1])
    expected_theta = np.array([0, np.pi/4, np.pi/2])
    
    np.testing.assert_almost_equal(r, expected_r, decimal=6)
    np.testing.assert_almost_equal(theta, expected_theta, decimal=6)

def test_rthet_to_xy():
    """Test conversion from polar to cartesian coordinates"""
    r = np.array([1, 1, 1])
    theta = np.array([0, np.pi/4, np.pi/2])
    x, y = rthet_to_xy(r, theta)
    
    expected_x = np.array([1, 2**(-1/2), 0])
    expected_y = np.array([0, 2**(-1/2), 1])
    
    np.testing.assert_almost_equal(x, expected_x, decimal=6)
    np.testing.assert_almost_equal(y, expected_y, decimal=6)

def test_solve_kepler():
    """Test Kepler equation solver"""
    M = np.array([0, np.pi/2, np.pi])
    e = 0.1
    E = solve_kepler(M, e)
    # Check if solution satisfies Kepler's equation
    assert np.all(np.abs(E - M - e*np.sin(E)) < 1e-6)

def test_make_orbit(orbit_params, observation_times):
    """Test orbit generation"""
    rs, nus = make_orbit(
        observation_times, 
        orbit_params['phase0'],
        orbit_params['a'],
        orbit_params['e'],
        orbit_params['omega']
    )
    assert len(rs) == len(observation_times)
    assert len(nus) == len(observation_times)
    # Check physical constraints
    assert np.all(rs > 0)  # Positive distances
    assert np.all(nus >= 0) & np.all(nus <= 2*np.pi)  # Valid angles

def test_dist_to_parallax():
    """Test parallax calculation"""
    time = 2460000
    r = 2.0
    theta = np.pi/3
    dt = 1.0
    
    parallax, sin_sun = dist_to_parallax(time, r, theta, dt)
    
    # Basic sanity checks
    assert isinstance(parallax, float)
    assert isinstance(sin_sun, float)
    assert -1 <= sin_sun <= 1  # Valid sine value
    assert parallax > 0  # Positive parallax

@pytest.mark.parametrize("output_str", ["test_injection"])
def test_inject_asteroid(test_fits_data, orbit_params, observation_times, tmp_path, output_str):
    """Test asteroid injection into images using real test data"""
    # Injection parameters
    jd = observation_times[0]
    theta = np.pi/4
    sin_sun = 0.5
    fwhm = 3.0
    fluxlevel = 95
    noiselevel = 0.1
    
    # Test injection
    im1, im2, fname1, fname2 = inject_asteroid(
        test_fits_data,
        parallax=10.0,  # arcsec
        obsdate="2025-01-18",
        obsdelta=1.0,
        jd=jd,
        theta=theta,
        sin_sun=sin_sun,
        fwhm=fwhm,
        fluxlevel=fluxlevel,
        noiselevel=noiselevel,
        output_str=output_str,
        output_dir=tmp_path
    )
    
    # Check outputs
    test_shape = test_fits_data[0].data.shape
    assert isinstance(im1, np.ndarray)
    assert isinstance(im2, np.ndarray)
    assert im1.shape == test_shape
    assert im2.shape == test_shape
    assert Path(fname1).exists()
    assert Path(fname2).exists()
    
    # Check that injected images have reasonable values
    assert not np.any(np.isnan(im1))
    assert not np.any(np.isnan(im2))
    assert np.median(im1) > 0
    assert np.median(im2) > 0

def test_orbit_fitting(orbit_params, observation_times):
    """Test orbit fitting functionality"""
    # Generate synthetic data
    true_rs, true_thetas = make_orbit(
        observation_times,
        orbit_params['phase0'],
        orbit_params['a'],
        orbit_params['e'],
        orbit_params['omega']
    )
    
    # Add some noise
    rs_err = 0.03 * np.ones_like(true_rs)
    rs_fit = true_rs + np.random.normal(0, 0.03, size=len(true_rs))
    thetas_err = 1e-4 * np.ones_like(true_thetas)
    thetas_fit = true_thetas + np.random.normal(0, 1e-4, size=len(true_thetas))
    
    # Run the fit
    samples = run_fit(
        observation_times, 
        rs_fit, 
        rs_err, 
        thetas_fit, 
        thetas_err,
        sampler='dynesty',
        nlive=100,
        dlogz=0.5,
        bootstrap=0,
        phase0=[0,1],
        a=[0.1,10],
        e=[0,0.99],
        omega=[0,1]
    )
    
    # Check output shape and basic constraints
    assert samples.shape[1] == 4  # Four parameters
    assert np.all(samples[:, 1] > 0)  # Semi-major axis > 0
    assert np.all((samples[:, 2] >= 0) & (samples[:, 2] < 1))  # 0 ≤ e < 1