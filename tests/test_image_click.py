from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pdp_asteroids
from pdp_asteroids.image_clicker import ImageClicker
from dash import Dash
from dash.testing.application_runners import import_app


@pytest.fixture
def fits_file():
    root_dir = Path(pdp_asteroids.__file__).parent
    fits_file = root_dir / 'injected_images/2025-03-02_frame2.fits'
    return str(fits_file)

def test_image_clicker_initialization(fits_file):
    clicker = ImageClicker(fits_file)
    assert clicker.file_name == fits_file
    assert clicker.coords is None

def test_get_coords(fits_file):
    clicker = ImageClicker(fits_file)
    assert clicker.get_coords() is None

def test_run_plot_context(fits_file):
    clicker = ImageClicker(fits_file)
    clicker.run_plot_context()
    
    assert clicker.fig is not None
    assert clicker.app is not None
    assert isinstance(clicker.app, Dash)

def test_load_and_process_fits(fits_file):
    clicker = ImageClicker(fits_file)
    data = clicker.load_and_process_fits()
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2  # Should be a 2D array

def test_create_figure(fits_file):
    clicker = ImageClicker(fits_file)
    data = clicker.load_and_process_fits()
    fig = clicker.create_figure(data)
    assert fig is not None
    assert len(fig.data) == 1  # Should have one trace (the heatmap)

def test_add_hover_template(fits_file):
    clicker = ImageClicker(fits_file)
    clicker.run_plot_context()
    clicker.add_hover_template()
    
    assert clicker.fig.data[0].hovertemplate is not None
    assert clicker.fig.data[0].customdata is not None

if __name__ == "__main__":
    pytest.main()
