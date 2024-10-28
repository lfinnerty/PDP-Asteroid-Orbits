from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from pdp_asteroids.image_clicker import DualImageClickSelector


@pytest.fixture
def mock_fits_files(tmp_path):
    """Create mock FITS files with WCS information."""
    # Create test files
    file1 = tmp_path / "test1.fits"
    file2 = tmp_path / "test2.fits"
    
    # Create sample data and WCS
    data = np.random.rand(100, 100)
    header = fits.Header()
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 45.0  # RA center
    header['CRVAL2'] = 30.0  # Dec center
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CDELT1'] = -0.001
    header['CDELT2'] = 0.001
    
    # Write files
    fits.writeto(file1, data, header)
    fits.writeto(file2, data, header)
    
    return str(file1), str(file2)


@pytest.fixture
def mock_axis_coords():
    """Create a mock for axis coordinate formatting."""
    coords = Mock()
    coord0 = Mock()
    coord1 = Mock()
    coords.__getitem__ = lambda self, idx: coord0 if idx == 0 else coord1
    return coords


@pytest.fixture
def selector(mock_fits_files, mock_axis_coords, monkeypatch):
    """Create a DualImageClickSelector instance with mocked display."""
    # Create a more complete mock figure
    mock_fig = Mock()
    mock_ax1 = Mock()
    mock_ax2 = Mock()
    
    # Create mock artist for plot returns
    mock_artist = Mock()
    mock_transform = Mock()
    
    # Configure the axes with required attributes
    for ax in [mock_ax1, mock_ax2]:
        ax.coords = mock_axis_coords
        ax.imshow = Mock(return_value=Mock())
        ax.grid = Mock()
        ax.set_title = Mock()
        # Make plot return a list containing a mock artist
        ax.plot = Mock(return_value=[mock_artist])
        # Add get_transform method
        ax.get_transform = Mock(return_value=mock_transform)
    
    mock_ax1.wcs = WCS(fits.open(mock_fits_files[0])[0].header)
    mock_ax2.wcs = WCS(fits.open(mock_fits_files[1])[0].header)
    
    # Set up the figure mock
    mock_fig.add_subplot.side_effect = [mock_ax1, mock_ax2]
    mock_fig.__getitem__ = lambda x: mock_fig
    mock_fig.canvas = Mock()
    mock_fig.canvas.draw_idle = Mock()
    
    monkeypatch.setattr(plt, 'figure', Mock(return_value=mock_fig))
    
    selector = DualImageClickSelector(mock_fits_files[0], mock_fits_files[1])
    selector.ax1 = mock_ax1
    selector.ax2 = mock_ax2
    
    return selector


def test_initialization(selector):
    """Test proper initialization of the selector."""
    assert selector.coords1 is None
    assert selector.coords2 is None
    assert selector.err1 is None
    assert selector.err2 is None
    assert selector.wcs1 is not None
    assert selector.wcs2 is not None


def test_load_fits(selector, mock_fits_files):
    """Test FITS file loading and processing."""
    file1, _ = mock_fits_files
    data, wcs = selector._load_fits(file1, 0.1)
    
    assert isinstance(data, np.ndarray)
    assert isinstance(wcs, WCS)
    assert data.shape == (100, 100)
    assert not np.any(np.isnan(data))
    # Allow for small numerical errors in saturation
    assert data.max() <= 0.105  # 5% tolerance on saturation factor


def test_format_coords(selector):
    """Test coordinate formatting."""
    coord = SkyCoord(45 * u.deg, 30 * u.deg)
    formatted = selector._format_coords(coord)
    assert 'RA: 45.00°' in formatted
    assert 'Dec: 30.00°' in formatted


def test_get_coords(selector):
    """Test coordinate getter."""
    coords = selector.get_coords()
    assert coords == (None, None)
    
    # Simulate a click by setting coordinates
    selector.coords1 = SkyCoord(45 * u.deg, 30 * u.deg)
    coords = selector.get_coords()
    assert coords[0] is not None
    assert coords[1] is None


def test_get_errors(selector):
    """Test error getter."""
    errors = selector.get_errors()
    assert errors == (None, None)
    
    # Simulate error setting
    selector.err1 = np.array([0.001, 0.001]) * u.degree
    errors = selector.get_errors()
    assert errors[0] is not None
    assert errors[1] is None


def test_clear(selector):
    """Test clearing of selections."""
    # Set some mock data
    selector.coords1 = SkyCoord(45 * u.deg, 30 * u.deg)
    selector.coords2 = SkyCoord(46 * u.deg, 31 * u.deg)
    selector.err1 = np.array([0.001, 0.001]) * u.degree
    selector.err2 = np.array([0.001, 0.001]) * u.degree
    
    # Add some mock artists
    mock_artist1 = Mock()
    mock_artist2 = Mock()
    selector.current_artists1 = [mock_artist1]
    selector.current_artists2 = [mock_artist2]
    
    selector.clear()
    
    # Verify artists were removed
    mock_artist1.remove.assert_called_once()
    mock_artist2.remove.assert_called_once()
    
    assert selector.coords1 is None
    assert selector.coords2 is None
    assert selector.err1 is None
    assert selector.err2 is None
    assert selector.coord_label1.value == 'Click left image to select position'
    assert selector.coord_label2.value == 'Click right image to select position'
    assert len(selector.current_artists1) == 0
    assert len(selector.current_artists2) == 0


@pytest.mark.parametrize('click_axes, expected_label', [
    ('ax1', 'coords1'),
    ('ax2', 'coords2')
])
def test_click_handling(selector, click_axes, expected_label):
    """Test click event handling."""
    # Create mock click event
    event = Mock()
    event.xdata = 50
    event.ydata = 50
    event.inaxes = getattr(selector, click_axes)
    
    # Trigger click handler
    selector._on_click(event)
    
    # Check that coordinates were set
    assert getattr(selector, expected_label) is not None
