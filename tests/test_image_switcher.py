import pytest
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from unittest.mock import patch
from pdp_asteroids.image_switcher import fits_image_switcher  # Replace 'your_module' with the actual module name

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
def mock_show():
    with patch('plotly.graph_objects.Figure.show') as mock:
        yield mock

def test_fits_image_switcher_creation(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].visible == True
    assert fig.data[1].visible == False
    mock_show.assert_called_once()

def test_fits_image_switcher_downsampling(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2, downsample_factor=2)
    
    # Check if downsampling worked
    assert fig.data[0].z.shape == (50, 50)
    assert fig.data[1].z.shape == (50, 50)

def test_fits_image_switcher_normalization(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)

    # Check if normalization worked
    assert np.all(fig.data[0].z >= 0)
    assert np.allclose(fig.data[0].z, np.clip(fig.data[0].z, 0, 0.1), atol=1e-2)


def test_fits_image_switcher_layout(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    assert fig.layout.title.text == "Image 1"
    assert fig.layout.height == 600
    assert fig.layout.width == 800
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
    assert fig.layout.yaxis.showticklabels == False
    assert fig.layout.xaxis.showticklabels == False

def test_fits_image_switcher_buttons(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    assert len(fig.layout.updatemenus) == 1
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert buttons[0].label == "Show Image 1"
    assert buttons[1].label == "Show Image 2"

@pytest.mark.parametrize("button_index, expected_visibility", [(0, [True, False]), (1, [False, True])])
def test_fits_image_switcher_button_functionality(mock_fits_files, mock_show, button_index, expected_visibility):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    button = fig.layout.updatemenus[0].buttons[button_index]
    assert button.args[0]["visible"] == expected_visibility

def test_fits_image_switcher_colorscale(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)

    # Check if it's a reversed grayscale colorscale
    colorscale = fig.data[0].colorscale
    assert len(colorscale) > 1, f"Colorscale should have multiple entries, but has {len(colorscale)}"

    # Print colorscale for debugging
    print("Colorscale:")
    for i, (pos, color) in enumerate(colorscale):
        print(f"  {i}: {pos} - {color}")

    # Check first and last colors
    assert colorscale[0][1].startswith('rgb('), f"First color should be in RGB format, but is {colorscale[0][1]}"
    assert colorscale[-1][1].startswith('rgb('), f"Last color should be in RGB format, but is {colorscale[-1][1]}"

    # Function to safely parse RGB values
    def parse_rgb(color_str):
        if not color_str.startswith('rgb(') or not color_str.endswith(')'):
            return None
        try:
            return [int(x) for x in color_str[4:-1].split(',')]
        except ValueError:
            return None

    # Check if it's monotonically increasing in brightness
    brightnesses = []
    for _, color in colorscale:
        rgb = parse_rgb(color)
        if rgb is None:
            print(f"Warning: Unable to parse color: {color}")
            continue
        brightnesses.append(sum(rgb))

    assert len(brightnesses) > 1, "Not enough valid colors in colorscale"
    assert all(b1 <= b2 for b1, b2 in zip(brightnesses, brightnesses[1:])), \
        "Brightness is not monotonically increasing"

    # Check if it's a grayscale
    for _, color in colorscale:
        rgb = parse_rgb(color)
        if rgb is None:
            continue
        assert len(set(rgb)) == 1, f"Color {color} is not grayscale"

    # Optionally, check if it's reversed (black to white)
    first_rgb = parse_rgb(colorscale[0][1])
    last_rgb = parse_rgb(colorscale[-1][1])
    if first_rgb and last_rgb:
        assert sum(first_rgb) < sum(last_rgb), "Colorscale is not from dark to light"

def test_fits_image_switcher_zsmooth(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    assert fig.data[0].zsmooth == "best"
    assert fig.data[1].zsmooth == "best"

def test_fits_image_switcher_y_axis_reversal(mock_fits_files, mock_show):
    file1, file2 = mock_fits_files
    fig = fits_image_switcher(file1, file2)
    
    assert fig.data[0].y[0] > fig.data[0].y[-1]
    assert fig.data[1].y[0] > fig.data[1].y[-1]
