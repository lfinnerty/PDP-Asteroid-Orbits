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



@pytest.fixture
def mock_hf_manager(mocker):
    """Mock HuggingFace manager for testing."""
    mock_manager = mocker.Mock()
    mock_manager.pull_changes.return_value = Path("/test/path")
    return mock_manager



def test_get_new_observations_no_token(investigation):
    """Test attempt to get new observations without HF token."""
    investigation.hf_manager = None
    
    with pytest.raises(ValueError, match="HuggingFace token not provided"):
        investigation.get_new_observations()

def test_get_new_observations_no_changes(investigation, mock_hf_manager, tmp_path):
    """Test when no new observations are available."""
    investigation.hf_manager = mock_hf_manager
    investigation.base_path = tmp_path
    
    # Create test observation file
    group_path = tmp_path / investigation.group
    group_path.mkdir()
    (group_path / "2024-01-01_test_frame1.fits").touch()
    
    # Mock pull_changes to not add any new files
    mock_hf_manager.pull_changes.return_value = tmp_path
    
    new_dates = investigation.get_new_observations()
    assert len(new_dates) == 0

def test_get_new_observations_success(investigation, mock_hf_manager, tmp_path):
   """Test successful retrieval of new observations."""
   # Setup test environment
   investigation.hf_manager = mock_hf_manager
   investigation.base_path = tmp_path
   
   # Create initial observation file
   group_path = tmp_path / "observations_2024" / investigation.group
   group_path.mkdir(exist_ok=True)
   (group_path / f"2024-01-01_{investigation.group}_frame1.fits").touch()
   
   def mock_pull(*args, **kwargs):
       (group_path / f"2024-01-02_{investigation.group}_frame1.fits").touch()
       return tmp_path
       
   mock_hf_manager.pull_changes.side_effect = mock_pull
   
   # Test getting new observations
   new_dates = investigation.get_new_observations()
   assert "2024-01-02" in new_dates

def test_get_new_observations_pull_failure(investigation, mock_hf_manager):
    """Test handling of pull operation failure."""
    investigation.hf_manager = mock_hf_manager
    mock_hf_manager.pull_changes.side_effect = Exception("Pull failed")
    
    new_dates = investigation.get_new_observations()
    assert len(new_dates) == 0

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



def test_fit_orbit_no_data(investigation):
    """Test orbit fitting with no data."""
    with pytest.raises(ValueError, match="No processed observations"):
        investigation.fit_orbit()


@patch('pdp_asteroids.investigation_manager.run_fit')
@patch('pdp_asteroids.investigation_manager.plot_fit')
def test_fit_orbit(mock_plot_fit, mock_run_fit, investigation):
    """Test orbit fitting and plotting."""
    # Add some mock observation data using the actual dataclass
    investigation.data["2024-01-01"] = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=1.0,
        theta_err=1e-4
    )
    
    # Configure mocks
    mock_run_fit.return_value = np.array([[1, 2, 3, 4]])
    mock_plot_fit.return_value = Mock()
    
    # Test orbit fitting
    samples = investigation.fit_orbit()
    assert mock_run_fit.called
    assert not mock_plot_fit.called  # Should not be called during fit_orbit
    
    # Test orbit plotting
    figures = investigation.plot_orbit()
    assert mock_plot_fit.called
    assert len(figures) == 1  # Should have one figure for our one fit


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
    investigation._save_measurements()
    
    # Create new investigation instance
    new_investigation = OrbitInvestigation(
        base_path=investigation.base_path
    )
    
    # Verify data was loaded
    assert "2024-01-01" in new_investigation.data
    loaded_data = new_investigation.data["2024-01-01"]
    assert loaded_data.jd == 2460000.5
    assert loaded_data.r == 2.0
    assert loaded_data.r_err == 0.1
    assert loaded_data.theta == 1.0
    assert loaded_data.theta_err == 1e-4


# Add a new test for orbit fits saving/loading
def test_save_load_orbit_fits(investigation):
    """Test saving and loading orbit fits data."""
    # Add some mock observation data
    investigation.data["2024-01-01"] = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=1.0,
        theta_err=1e-4
    )
    
    # Add some orbit fits
    with patch('pdp_asteroids.investigation_manager.run_fit') as mock_run_fit:
        mock_run_fit.return_value = np.array([[1, 2, 3, 4]])
        investigation.fit_orbit()
        investigation.fit_orbit()  # Create a second fit
    
    # Save and reload
    investigation._save_orbits()
    
    # Create new investigation instance
    new_investigation = OrbitInvestigation(
        base_path=investigation.base_path
    )
    
    # Verify orbit fits were loaded
    assert len(new_investigation.orbit_fits) == 2
    assert 0 in new_investigation.orbit_fits
    assert 1 in new_investigation.orbit_fits
    np.testing.assert_array_equal(
        new_investigation.orbit_fits[0],
        np.array([[1, 2, 3, 4]])
    )