import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
from astropy.io import fits
import pickle
from pdp_asteroids.investigation_manager import OrbitInvestigation, ObservationData


@pytest.fixture
def mock_fits_header():
    """Create a mock FITS header with required fields."""
    header = {
        'JD': 2460000.5,
        'obsdt': 0.25,  # 6 hours
        'theta': np.pi/4,
        'elong': 0.5
    }
    return header


@pytest.fixture
def mock_observation_path(tmp_path):
    """Create a temporary directory with mock observation files."""
    group_path = tmp_path / "test"
    group_path.mkdir()
    
    # Create mock FITS files
    dates = ['2024-01-01', '2024-01-02']
    for date in dates:
        for frame in [1, 2]:
            filename = f"{date}_test_frame{frame}.fits"
            mock_file = group_path / filename
            # Create empty FITS file with header
            hdu = fits.PrimaryHDU()
            hdu.header['JD'] = 2460000.5
            hdu.header['obsdt'] = 0.25
            hdu.header['theta'] = np.pi/4
            hdu.header['elong'] = 0.5
            hdu.writeto(mock_file)
    
    return tmp_path


@pytest.fixture
def investigation(mock_observation_path):
    """Create an OrbitInvestigation instance with mock data."""
    return OrbitInvestigation(base_path=str(mock_observation_path), group="test")


def test_init(investigation):
    """Test initialization of OrbitInvestigation."""
    assert investigation.group == "test"
    assert investigation.data == {}
    assert investigation.current_date is None
    assert investigation.current_header is None


def test_list_available_dates(investigation):
    """Test listing available observation dates."""
    dates = investigation.list_available_dates()
    assert len(dates) == 2
    assert "2024-01-01" in dates
    assert "2024-01-02" in dates


def test_load_observation(investigation):
    """Test loading an observation date."""
    investigation.load_observation("2024-01-01")
    assert investigation.current_date == "2024-01-01"
    assert investigation.current_header is not None
    assert investigation.current_header['JD'] == 2460000.5


def test_load_invalid_observation(investigation):
    """Test loading a non-existent observation date."""
    with pytest.raises(FileNotFoundError):
        investigation.load_observation("2099-01-01")


@patch('pdp_asteroids.investigation_manager.FitsImageSwitcher')
def test_examine_images(mock_switcher, investigation):
    """Test image examination functionality."""
    investigation.load_observation("2024-01-01")
    investigation.examine_images()
    mock_switcher.assert_called_once()
    mock_switcher.return_value.display.assert_called_once()


@patch('pdp_asteroids.investigation_manager.DualImageClicker')
def test_mark_asteroid(mock_clicker, investigation):
    """Test asteroid marking functionality."""
    investigation.load_observation("2024-01-01")
    investigation.mark_asteroid()
    mock_clicker.assert_called_once()
    mock_clicker.return_value.run_plot_context.assert_called_once()
    mock_clicker.return_value.display.assert_called_once()


def test_process_measurements_without_clicks(investigation):
    """Test processing measurements without click data."""
    investigation.load_observation("2024-01-01")
    with pytest.raises(ValueError, match="Missing required data"):
        investigation.process_measurements()


@patch('pdp_asteroids.investigation_manager.parallax_to_dist')
@patch('pdp_asteroids.investigation_manager.dist_to_r')
def test_process_measurements(mock_dist_to_r, mock_parallax_to_dist, investigation):
    """Test processing valid measurements."""
    investigation.load_observation("2024-01-01")
    
    # Mock click data
    coords1 = np.array([10.0, 20.0])
    coords2 = np.array([10.1, 20.1])
    err1 = np.array([0.01, 0.01])
    err2 = np.array([0.01, 0.01])
    investigation._current_clicks = ((coords1, coords2), (err1, err2))
    
    # Mock return values
    mock_parallax_to_dist.return_value = (2.0, 0.1)
    mock_dist_to_r.return_value = (3.0, 0.2)
    
    investigation.process_measurements()
    
    assert investigation.current_date in investigation.data
    obs_data = investigation.data[investigation.current_date]
    assert isinstance(obs_data, ObservationData)
    assert obs_data.r == 3.0
    assert obs_data.r_err == 0.2


def test_save_and_load_data(investigation, tmp_path):
    """Test saving and loading investigation data."""
    # Add some test data
    test_data = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=np.pi/4,
        theta_err=1e-4
    )
    investigation.data["2024-01-01"] = test_data
    
    # Save data
    investigation._save_data()
    
    # Create new investigation instance
    new_investigation = OrbitInvestigation(
        base_path=str(tmp_path),
        group="test"
    )
    
    # Check if data was loaded correctly
    assert "2024-01-01" in new_investigation.data
    loaded_data = new_investigation.data["2024-01-01"]
    assert loaded_data.jd == test_data.jd
    assert loaded_data.r == test_data.r
    assert loaded_data.r_err == test_data.r_err
    assert loaded_data.theta == test_data.theta


@patch('pdp_asteroids.investigation_manager.run_fit')
@patch('pdp_asteroids.investigation_manager.plot_fit')
def test_fit_orbit(mock_plot_fit, mock_run_fit, investigation):
    """Test orbit fitting functionality."""
    # Add test data
    investigation.data["2024-01-01"] = ObservationData(
        jd=2460000.5,
        r=2.0,
        r_err=0.1,
        theta=np.pi/4
    )
    
    mock_run_fit.return_value = np.array([[0.1, 2.0, 0.1, np.pi/4]])
    mock_plot_fit.return_value = MagicMock()
    
    fig = investigation.fit_orbit()
    
    mock_run_fit.assert_called_once()
    mock_plot_fit.assert_called_once()
    assert fig is not None


def test_fit_orbit_no_data(investigation):
    """Test orbit fitting with no processed observations."""
    with pytest.raises(ValueError, match="No processed observations available"):
        investigation.fit_orbit()


def test_context_manager(investigation):
    """Test context manager functionality."""
    with investigation as inv:
        inv.data["2024-01-01"] = ObservationData(
            jd=2460000.5,
            r=2.0,
            r_err=0.1,
            theta=np.pi/4
        )
    
    # Check if data was saved
    save_file = investigation.group_path / "save.p"
    assert save_file.exists()
    
    with open(save_file, 'rb') as f:
        loaded_data = pickle.load(f)
        assert "2024-01-01" in loaded_data


if __name__ == "__main__":
    pytest.main(["-v"])