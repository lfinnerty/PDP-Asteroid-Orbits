import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.io import fits
import numpy as np
from ipywidgets import widgets
from IPython.display import display

def fits_image_switcher(file1, file2, downsample_factor=4):
    def load_and_process_fits(file):
        with fits.open(file) as hdul:
            data = hdul[0].data
        
        # Downsample the image
        data = data[::downsample_factor, ::downsample_factor]
        
        # Normalize the data
        vmin, vmax = np.percentile(data, [1, 99])
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, norm_data.max()*0.1)
        
        return norm_data

    # Load and process FITS files
    data1 = load_and_process_fits(file1)
    data2 = load_and_process_fits(file2)
    
    # Create a function to create a heatmap trace that behaves more like imshow
    def create_imshow_like_heatmap(data, title):
        return go.Heatmap(
            z=data,
            colorscale='Greys_r',  # Default colormap in matplotlib
            showscale=False,
            name=title,
            zsmooth='best',  # For smoother interpolation, similar to imshow
            # hoverinfo='none',  # Disable hover text for a cleaner look
            # Reverse the y-axis to match imshow's top-left origin
            y=list(range(data.shape[0]))[::-1],
        )
    
    # Create heatmap traces for both images
    trace1 = create_imshow_like_heatmap(data1, "Image 1")
    trace2 = create_imshow_like_heatmap(data2, "Image 2")
    
    # Create the figure
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    
    # Update layout to maintain aspect ratio and remove axis labels
    fig.update_layout(
        title_text="Image 1",
        height=600,
        width=800,
        yaxis=dict(
            scaleanchor="x",  # This ensures that the aspect ratio is maintained
            scaleratio=1,
            showticklabels=False,  # Hide tick labels
        ),
        xaxis=dict(
            showticklabels=False,  # Hide tick labels
        ),
    )
    
    # Set initial visibility
    fig.data[0].visible = True
    fig.data[1].visible = False
    
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
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.2,
            buttons=buttons
        )]
    )
    
    # Display the figure
    fig.show()
    return fig
