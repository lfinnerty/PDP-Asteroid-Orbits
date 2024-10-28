import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ipympl.backend_nbagg import Canvas
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from IPython.display import display
import ipywidgets as widgets
from typing import Optional, Tuple

plt.ioff()
plt.switch_backend('module://ipympl.backend_nbagg')

class DualImageClickSelector:
    def __init__(
        self,
        file1: str,
        file2: str,
        saturation_factor1: float = 0.1,
        saturation_factor2: float = 0.1,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """Initialize the dual image selector."""
        self.file1 = file1
        self.file2 = file2
        self.saturation_factors = (saturation_factor1, saturation_factor2)
        
        # State variables
        self.coords1: Optional[SkyCoord] = None
        self.coords2: Optional[SkyCoord] = None
        self.err1: Optional[np.ndarray] = None
        self.err2: Optional[np.ndarray] = None
        self.wcs1: Optional[WCS] = None
        self.wcs2: Optional[WCS] = None
        self.current_artists1 = []
        self.current_artists2 = []
        
        # Create labels for coordinates
        self.coord_label1 = widgets.Label(value='Click left image to select position')
        self.coord_label2 = widgets.Label(value='Click right image to select position')
        
        # Create output widget for displaying debug info
        self.debug_output = widgets.Output()
        
        # Load the data first
        self.data1, self.wcs1 = self._load_fits(file1, saturation_factor1)
        self.data2, self.wcs2 = self._load_fits(file2, saturation_factor2)
        
        # Create the figure using plt to ensure proper backend setup
        self.fig = plt.figure(figsize=figsize)
        self.ax1 = self.fig.add_subplot(121, projection=self.wcs1)
        self.ax2 = self.fig.add_subplot(122, projection=self.wcs2)
        
        # Setup the plots
        self._setup_plots()
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _format_coords(self, skycoord: SkyCoord) -> str:
        """Format RA/Dec coordinates in decimal degrees."""
        ra = skycoord.ra.deg
        dec = skycoord.dec.deg
        return f'RA: {ra:.2f}°\nDec: {dec:.2f}°'

    def _draw_cross(self, ax, x, y, current_artists):
        """Draw a cross marker at the specified pixel position."""
        size = 20  # pixels
        
        # Clear existing markers
        for artist in current_artists:
            artist.remove()
        current_artists.clear()
        
        # Draw the cross in pixel coordinates
        current_artists.extend([
            ax.plot([x-size, x+size], [y, y], 'r-', linewidth=2, transform=ax.get_transform('pixel'))[0],
            ax.plot([x, x], [y-size, y+size], 'r-', linewidth=2, transform=ax.get_transform('pixel'))[0]
        ])

    def _load_fits(self, filename: str, saturation_factor: float) -> Tuple[np.ndarray, WCS]:
        """Load and process FITS image data."""
        with fits.open(filename) as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)
            
            # Basic normalization
            vmin, vmax = np.percentile(data, [1, 99])
            norm_data = (data - vmin) / (vmax - vmin)
            norm_data = np.clip(norm_data, 0, norm_data.max() * saturation_factor)
            
            return norm_data, wcs
    
    def _setup_plots(self):
        """Configure both subplot displays."""
        # Display images
        self.im1 = self.ax1.imshow(self.data1, cmap='Greys_r', origin='lower', 
                                  interpolation='nearest')
        self.im2 = self.ax2.imshow(self.data2, cmap='Greys_r', origin='lower',
                                  interpolation='nearest')
        
        # Configure axes
        for ax, title in [(self.ax1, 'Image 1'), (self.ax2, 'Image 2')]:
            ax.grid(True, color='lightgray', ls=':', alpha=0.5)
            ax.set_title(title)
            # Use decimal degrees for axis labels with 2 decimal places
            ax.coords[0].set_major_formatter('d.dd')
            ax.coords[1].set_major_formatter('d.dd')
        
        self.fig.tight_layout(pad=3.0)
    
    def _on_click(self, event):
        """Handle click events for both plots."""
        with self.debug_output:
            if event.inaxes not in (self.ax1, self.ax2):
                print("Click outside axes")
                return
            
            # Determine which plot was clicked
            if event.inaxes == self.ax1:
                ax = self.ax1
                wcs = self.wcs1
                coord_label = self.coord_label1
                coords_attr = 'coords1'
                err_attr = 'err1'
                current_artists = self.current_artists1
            else:
                ax = self.ax2
                wcs = self.wcs2
                coord_label = self.coord_label2
                coords_attr = 'coords2'
                err_attr = 'err2'
                current_artists = self.current_artists2
            
            # Convert click position to sky coordinates
            skycoord = ax.wcs.pixel_to_world(event.xdata, event.ydata)
            
            # Store coordinates
            setattr(self, coords_attr, skycoord)
            
            # Calculate position uncertainty (in degrees)
            pixel_scale = np.abs(wcs.pixel_scale_matrix.diagonal())
            err_deg = 3.5 * pixel_scale
            setattr(self, err_attr, err_deg * u.degree)
            
            # Draw the cross marker at pixel coordinates
            self._draw_cross(ax, event.xdata, event.ydata, current_artists)
            
            # Update coordinate display
            err = getattr(self, err_attr)
            coord_str = self._format_coords(skycoord)
            coord_label.value = (
                f'{coord_str}\n'
                f'Uncertainty: {err[0]:.2f}°/{err[1]:.2f}°'
            )
            
            self.fig.canvas.draw_idle()
    
    def get_coords(self) -> Tuple[Optional[SkyCoord], Optional[SkyCoord]]:
        """Get selected coordinates from both images."""
        return self.coords1, self.coords2
    
    def get_errors(self) -> Tuple[Optional[u.Quantity], Optional[u.Quantity]]:
        """Get error estimates from both images."""
        return self.err1, self.err2
    
    def display(self):
        """Display the dual image interface."""
        with self.debug_output:
            self.debug_output.clear_output()
        
        display(widgets.VBox([
            widgets.HBox([self.coord_label1, self.coord_label2]),
            self.fig.canvas,
            self.debug_output
        ]))
    
    def clear(self):
        """Clear selections in both images."""
        for artists in (self.current_artists1, self.current_artists2):
            for artist in artists:
                artist.remove()
            artists.clear()
        
        self.coords1 = self.coords2 = None
        self.err1 = self.err2 = None
        self.coord_label1.value = 'Click left image to select position'
        self.coord_label2.value = 'Click right image to select position'
        self.fig.canvas.draw_idle()