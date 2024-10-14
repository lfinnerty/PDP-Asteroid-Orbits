from typing import Tuple
import numpy as np
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

class ImageClicker:
    def __init__(self, file_name: str):
        self.coords = []
        self.file_name = file_name
        self.refresh_button = widgets.Button(description="Refresh (clear click)")
        self.output = widgets.Output()
        self.fig = None
        self.ax = None
        self.cid = None
        self.wcs = None

    def get_coords(self) -> Tuple[float, float]:
        return self.coords

    def run_plot_context(self) -> None:
        def initialize_image() -> None:
            with fits.open(self.file_name) as hdul:
                image_data = hdul[0].data
                self.wcs = WCS(hdul[0].header)

            norm = simple_norm(image_data, 'linear', percent=99)
            self.ax.imshow(image_data, cmap='gray', norm=norm)
            self.fig.canvas.draw()

        def onclick(event: MouseEvent) -> None:
            if event.inaxes != self.ax:
                return

            ix, iy = event.xdata, event.ydata
            self.coords.append(np.array(self.wcs.all_pix2world(ix, iy, 0)))

            # Add a circle with crosshair on the image
            self.ax.scatter(ix, iy, color='red', s=100, marker='o', facecolor='none')
            self.ax.scatter(ix, iy, color='red', s=100, marker='+', lw=0.5)
            self.fig.canvas.draw()

            # Convert pixel coordinates to RA and DEC
            if self.wcs:
                ra, dec = self.wcs.all_pix2world(ix, iy, 0)
                print(f"Clicked at RA: {ra:.5f}, Dec: {dec:.5f}")

            # Stop the event handler after one point is clicked
            if len(self.coords) >= 1:
                self.fig.canvas.mpl_disconnect(self.cid)

        display(self.refresh_button)
        display(self.output)

        # Display the initial figure
        with self.output:
            clear_output(wait=True)
            self.fig, self.ax = plt.subplots()
            # Connect the click event to the handler
            self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

            def refresh_plot(b: widgets.Button) -> None:
                self.coords = []
                with self.output:
                    self.ax.clear()
                    self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
                    initialize_image()

            self.refresh_button.on_click(refresh_plot)
            initialize_image()
