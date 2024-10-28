import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from dataclasses import asdict

from pdp_asteroids.investigation_manager import OrbitInvestigation, ObservationData


@pytest.fixture
def mock_observation_dir(tmp_path):
    """Create a mock observation directory structure."""
    base = tmp_path / "observations_2024"
    group = base / "test"
    group.mkdir(parents=True)
    
    # Create proper FITS files with minimal valid structure
    data = np.zeros((10, 10))
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -32
    header['NAXIS'] = 2
    header['NAXIS1'] = 10
    header['NAXIS2'] = 10
    header['EXTEND'] = True
    header['JD'] = 2460000.5
    header['obsdt'] = 0.25
    header['theta'] = 1.0
    header['elong'] = 0.5
    
    for frame in [1, 2]:
        fname = group / f"2024-01-01_test_frame{frame}.fits"
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(fname, overwrite=True)
    
    return str(base)


@pytest.fixture
def investigation(mock_observation_dir):
    """Create an investigation instance with mocked data directory."""
    return OrbitInvestigation(base_path=mock_observation_dir)


def test_initialization(investigation):
    """Test proper initialization."""
    assert investigation.current_date is None
    assert investigation.current_header is None
    assert investigation.current_selector is None


def test_context_manager(investigation, tmp_path):
    """Test context manager functionality."""
    with investigation as inv:
        assert isinstance(inv, OrbitInvestigation)


def test_load_observation(investigation):
    """Test loading an observation."""
    investigation.load_observation("2024-01-01")
    assert investigation.current_date == "2024-01-01"
    assert investigation.current_header['JD'] == 2460000.5
    assert investigation.current_header['obsdt'] == 0.25


@patch('pdp_asteroids.investigation_manager.DualImageClickSelector')
def test_mark_asteroid(mock_selector_class, investigation):
    """Test asteroid marking interface."""
    mock_selector = Mock()
    mock_selector_class.return_value = mock_selector
    
    investigation.load_observation("2024-01-01")
    investigation.mark_asteroid()
    
    assert investigation.current_selector == mock_selector
    mock_selector.display.assert_called_once()


def test_list_available_dates(investigation):
    """Test listing available observation dates."""
    dates = investigation.list_available_dates()
    assert "2024-01-01" in dates


def test_examine_images(investigation):
    """Test image examination interface."""
    investigation.load_observation("2024-01-01")
    with patch('pdp_asteroids.investigation_manager.FitsImageSwitcher') as mock_switcher:
        mock_instance = Mock()
        mock_switcher.return_value = mock_instance
        investigation.examine_images()
        mock_instance.display.assert_called_once()


def test_process_measurements_without_selector(investigation):
    """Test processing measurements without click data."""
    investigation.load_observation("2024-01-01")
    with pytest.raises(ValueError, match="No measurements to process"):
        investigation.process_measurements()


@patch('pdp_asteroids.investigation_manager.dist_to_r')
@patch('pdp_asteroids.investigation_manager.parallax_to_dist')
def test_process_measurements(mock_parallax_to_dist, mock_dist_to_r, investigation):
    """Test processing valid measurements."""
    investigation.load_observation("2024-01-01")
    
    # Create mock selector with coordinates
    mock_selector = Mock()
    mock_selector.get_coords.return_value = (
        SkyCoord(10.0 * u.deg, 20.0 * u.deg),
        SkyCoord(10.1 * u.deg, 20.1 * u.deg)
    )
    mock_selector.get_errors.return_value = (
        np.array([0.01, 0.01]) * u.deg,
        np.array([0.01, 0.01]) * u.deg
    )
    investigation.current_selector = mock_selector
    
    # Mock return values
    mock_parallax_to_dist.return_value = (2.0, 0.1)
    mock_dist_to_r.return_value = (3.0, 0.2)
    
    investigation.process_measurements()
    
    # Verify data was stored
    assert investigation.current_date in investigation.data
    obs_data = investigation.data[investigation.current_date]
    assert obs_data.r == 3.0
    assert obs_data.r_err == 0.2


@patch('pdp_asteroids.investigation_manager.run_fit')
@patch('pdp_asteroids.investigation_manager.plot_fit')
def test_fit_orbit(mock_plot_fit, mock_run_fit, investigation):
    """Test orbit fitting."""
    # Add some mock observation data using the actual dataclass
    investigation.data["2024-01-01"] = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=1.0,
        theta_err=1e-4
    )
    
    mock_run_fit.return_value = np.array([[1, 2, 3, 4]])
    mock_plot_fit.return_value = Mock()
    
    fig = investigation.fit_orbit()
    assert mock_run_fit.called
    assert mock_plot_fit.called


def test_fit_orbit_no_data(investigation):
    """Test orbit fitting with no data."""
    with pytest.raises(ValueError, match="No processed observations"):
        investigation.fit_orbit()


def test_save_load_data(investigation):
    """Test saving and loading investigation data."""
    # Add some data using the actual dataclass
    investigation.data["2024-01-01"] = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=1.0,
        theta_err=1e-4
    )
    
    # Save and reload
    investigation._save_data()
    new_investigation = OrbitInvestigation(
        base_path=investigation.base_path
    )
    
    assert "2024-01-01" in new_investigation.data
    loaded_data = new_investigation.data["2024-01-01"]
    assert loaded_data.jd == 2460000.5
    assert loaded_data.r == 2.0
    assert loaded_data.r_err == 0.1
    assert loaded_data.theta == 1.0
    assert loaded_data.theta_err == 1e-4