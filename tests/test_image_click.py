from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pdp_asteroids
from pdp_asteroids.image_clicker import ImageClicker

@pytest.fixture
def fits_file():
    root_dir = Path(pdp_asteroids.__file__).parent
    fits_file = root_dir / 'injected_images/2025-03-02_frame2.fits'
    return str(fits_file)

def test_image_clicker_initialization(fits_file):
    clicker = ImageClicker(fits_file)
    assert clicker.file_name == fits_file
    assert clicker.coords == []

def test_get_coords(fits_file):
    clicker = ImageClicker(fits_file)
    assert clicker.get_coords() == []

def test_run_plot_context(fits_file):
    clicker = ImageClicker(fits_file)
    clicker.run_plot_context()
    
    assert clicker.fig is not None
    assert clicker.ax is not None
    assert clicker.cid is not None

def test_onclick(fits_file):
    clicker = ImageClicker(fits_file)
    clicker.run_plot_context()

    # Create a more complete mock MouseEvent
    canvas = clicker.fig.canvas
    x, y = 3.0, 4.0  # Example pixel coordinates as floats
    event = MouseEvent('button_press_event', canvas, x, y, button=MouseButton.LEFT)
    event.xdata = x  # Explicitly set xdata
    event.ydata = y  # Explicitly set ydata
    event.inaxes = clicker.ax  # Set the inaxes attribute

    # Call onclick method
    clicker.fig.canvas.callbacks.process('button_press_event', event)

    assert len(clicker.coords) == 1
    expected_world_coords = clicker.wcs.all_pix2world(x, y, 0)
    assert np.allclose(clicker.coords[0], expected_world_coords)


def test_refresh_plot(fits_file):
    clicker = ImageClicker(fits_file)
    clicker.run_plot_context()
    
    # Add some coordinates
    clicker.coords = [(1, 1),]
    
    # Simulate refresh button click
    clicker.refresh_button.click()
    
    assert clicker.coords == []

if __name__ == "__main__":
    pytest.main()
