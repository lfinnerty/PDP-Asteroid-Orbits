import numpy as np
import pytest

from pdp_asteroids.position_to_orbit import xy_to_rthet, rthet_to_xy, make_orbit, plot_ellipse

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

def test_make_orbit():
    thets = np.array([0, np.pi/2, np.pi])
    a, e, omega = 1.0, 0.2, np.pi/4.
    orbit = make_orbit(thets, a, e, omega)
    
    expected_orbit = a*(1 - e**2) / (1 + e * np.cos(thets + omega))
    
    np.testing.assert_almost_equal(orbit, expected_orbit, decimal=6)

def test_plot_ellipse():
    # Basic test to ensure the function runs without errors
    a, e, omega = 1.0, 0.2, np.pi/4.
    plot_ellipse(a, e, omega)

def test_full_orbit():
    # Integration test to check if the flow of functions works
    a, e, omega = 1.0, 0.2, np.pi/4.
    thetas = 2*np.pi*np.array([0.1, 0.3, 0.35, 0.45, 0.5])
    rs = make_orbit(thetas, a, e, omega)
    
    # Perturb orbit slightly for test
    rs_noisy = np.random.normal(loc=rs, scale=5e-2)
    
    # Convert to x, y and back to r, theta
    x, y = rthet_to_xy(rs_noisy, thetas)
    r_reconstructed, theta_reconstructed = xy_to_rthet(x, y)
    
    # Assert round-trip conversion retains reasonable accuracy
    np.testing.assert_almost_equal(rs_noisy, r_reconstructed, decimal=3)
    np.testing.assert_almost_equal(thetas, theta_reconstructed, decimal=3)