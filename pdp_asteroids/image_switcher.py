import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.io import fits
import numpy as np
from IPython.display import display

class FitsImageSwitcher:
    def __init__(self, file1, file2, downsample_factor=4):
        self.file1 = file1
        self.file2 = file2
        self.downsample_factor = downsample_factor
        self.fig = None
        self.create_figure()

    def load_and_process_fits(self, file):
        with fits.open(file) as hdul:
            data = hdul[0].data
        
        # Downsample the image
        data = data[::self.downsample_factor, ::self.downsample_factor]
        
        # Normalize the data
        vmin, vmax = np.percentile(data, [1, 99])
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, norm_data.max()*0.03)
        
        return norm_data

    def create_imshow_like_heatmap(self, data, title):
        return go.Heatmap(
            z=data,
            colorscale='Greys_r',
            showscale=False,
            name=title,
            zsmooth='best',
            y=list(range(data.shape[0]))[::-1],
        )

    def create_figure(self):
        # Load and process FITS files
        data1 = self.load_and_process_fits(self.file1)
        data2 = self.load_and_process_fits(self.file2)
        
        # Create heatmap traces for both images
        trace1 = self.create_imshow_like_heatmap(data1, "Image 1")
        trace2 = self.create_imshow_like_heatmap(data2, "Image 2")
        
        # Create the figure
        self.fig = make_subplots(rows=1, cols=1)
        self.fig.add_trace(trace1)
        self.fig.add_trace(trace2)
        
        # Update layout to maintain aspect ratio and remove axis labels
        self.fig.update_layout(
            title_text="Image 1",
            height=800,
            width=800,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                showticklabels=False,
            ),
            xaxis=dict(
                showticklabels=False,
            ),
        )
        
        # Set initial visibility
        self.fig.data[0].visible = True
        self.fig.data[1].visible = False
        
        # Create and add buttons
        buttons = [
            dict(label="Show Image 1",
                 method="update",
                 args=[{"visible": [True, False]},
                       {"title": "Image 1"}]),
            dict(label="Show Image 2",
                 method="update",
                 args=[{"visible": [False, True]},
                       {"title": "Image 2"}])
        ]
        
        self.fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=buttons
            )]
        )

    def display(self):
        display(self.fig)
