import pytest
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from unittest.mock import patch
from pdp_asteroids.image_switcher import FitsImageSwitcher  # Update this import

# Set Plotly configuration to prevent auto-opening of browser windows
import plotly.io as pio
pio.renderers.default = 'svg'

@pytest.fixture
def mock_fits_files(tmp_path):
    # Create two mock FITS files
    file1 = tmp_path / "test1.fits"
    file2 = tmp_path / "test2.fits"
    
    data1 = np.random.rand(100, 100)
    data2 = np.random.rand(100, 100)
    
    hdu1 = fits.PrimaryHDU(data1)
    hdu2 = fits.PrimaryHDU(data2)
    
    hdu1.writeto(file1)
    hdu2.writeto(file2)
    
    return str(file1), str(file2)

@pytest.fixture
def mock_display():
    with patch('IPython.display.display') as mock:
        yield mock

def test_fits_image_switcher_creation(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    assert isinstance(switcher.fig, go.Figure)
    assert len(switcher.fig.data) == 2
    assert switcher.fig.data[0].visible == True
    assert switcher.fig.data[1].visible == False

def test_fits_image_switcher_downsampling(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2, downsample_factor=2)
    
    # Check if downsampling worked
    assert switcher.fig.data[0].z.shape == (50, 50)
    assert switcher.fig.data[1].z.shape == (50, 50)

def test_fits_image_switcher_normalization(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)

    # Check if normalization worked
    assert np.all(switcher.fig.data[0].z >= 0)
    assert np.all(switcher.fig.data[0].z <= 0.15)  # Max value should be 0.1 due to clipping

def test_fits_image_switcher_layout(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    assert switcher.fig.layout.title.text == "Image 1"
    assert switcher.fig.layout.height == 800
    assert switcher.fig.layout.width == 800
    assert switcher.fig.layout.yaxis.scaleanchor == "x"
    assert switcher.fig.layout.yaxis.scaleratio == 1
    assert switcher.fig.layout.yaxis.showticklabels == False
    assert switcher.fig.layout.xaxis.showticklabels == False

def test_fits_image_switcher_buttons(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    assert len(switcher.fig.layout.updatemenus) == 1
    buttons = switcher.fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert buttons[0].label == "Show Image 1"
    assert buttons[1].label == "Show Image 2"

@pytest.mark.parametrize("button_index, expected_visibility", [(0, [True, False]), (1, [False, True])])
def test_fits_image_switcher_button_functionality(mock_fits_files, button_index, expected_visibility):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    button = switcher.fig.layout.updatemenus[0].buttons[button_index]
    assert button.args[0]["visible"] == expected_visibility

def test_fits_image_switcher_zsmooth(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    assert switcher.fig.data[0].zsmooth == 'best'
    assert switcher.fig.data[1].zsmooth == 'best'

def test_fits_image_switcher_y_axis_reversal(mock_fits_files):
    file1, file2 = mock_fits_files
    switcher = FitsImageSwitcher(file1, file2)
    
    assert switcher.fig.data[0].y[0] > switcher.fig.data[0].y[-1]
    assert switcher.fig.data[1].y[0] > switcher.fig.data[1].y[-1]

