from typing import Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.io import fits
from astropy.wcs import WCS
import dash
from dash import dcc, html, clientside_callback, Input, Output, State

class ImageClicker:
    def __init__(self, file_name: str, downsample_factor: int = 1,saturation_factor: float = 0.1):
        self.coords : Optional[np.ndarray] = None
        self.err : Optional[np.ndarray] = None
        self.file_name = file_name
        self.downsample_factor = downsample_factor
        self.saturation_factor = saturation_factor
        self.fig = None
        self.wcs = None
        self.app = None

    def get_coords(self) -> Optional[np.ndarray]:

        return self.coords

    def get_err(self) -> Optional[np.ndarray]:
        return self.err

    def load_and_process_fits(self):
        with fits.open(self.file_name) as hdul:
            data = hdul[0].data
            self.wcs = WCS(hdul[0].header)
        
        # Downsample the image
        data = data[::self.downsample_factor, ::self.downsample_factor]
        
        # Normalize the data
        vmin, vmax = np.percentile(data, [1, 99])
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, norm_data.max()*self.saturation_factor)
        
        return norm_data

    def create_figure(self, data):
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Heatmap(
            z=data,
            colorscale='Greys_r',
            showscale=False,
            hoverinfo='none'
        ))

        fig.update_layout(
            height=600,
            width=600,
            title_text="Click on the image to mark a point",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        # Reverse y-axis to match astronomical image convention
        fig.update_yaxes(autorange="reversed")

        return fig

    def run_plot_context(self) -> None:
        app = dash.Dash(__name__, use_pages=False)
    
        data = self.load_and_process_fits()
        self.fig = self.create_figure(data)

        app.layout = html.Div([
            dcc.Graph(id='image-plot', figure=self.fig),
            html.Div(id='click-data'),
        ])

        clientside_callback(
            """
            function(clickData, figure) {
                if (clickData === null) return dash_clientside.no_update;
                
                const newClick = {
                    x: clickData.points[0].x,
                    y: clickData.points[0].y
                };
                
                // Create a new trace for the clicked point
                const newTrace = {
                    x: [newClick.x],
                    y: [newClick.y],
                    mode: 'markers',
                    marker: {symbol: 'cross', size: 10, color: 'red'},
                    showlegend: false,
                    type: 'scatter'
                };
                
                // Return the figure with only the heatmap and the new marker
                return {
                    ...figure,
                    data: [figure.data[0], newTrace]
                };
            }
            """,
            Output('image-plot', 'figure'),
            Input('image-plot', 'clickData'),
            State('image-plot', 'figure')
        )

        @app.callback(
        Output('click-data', 'children'),
        Input('image-plot', 'clickData'),
        )
        def update_click_data(clickData):
            if clickData:
                x, y = clickData['points'][0]['x'], clickData['points'][0]['y']
                ra, dec = self.wcs.all_pix2world(x * self.downsample_factor, y * self.downsample_factor, 0)
                self.coords = np.array([ra, dec])
                self.err = 3.5*np.diag(self.wcs.wcs)
                return f"Clicked at RA: {ra:.5f} +/- {self.err[0]:.7f}, Dec: {dec:.5f} +/- {self.err[1]:.7f}"

            return "Click on the image to mark a point"

        self.app = app

    def add_hover_template(self):
        if self.fig is not None and self.wcs is not None:
            ny, nx = self.fig.data[0].z.shape
            y, x = np.mgrid[0:ny, 0:nx]
            ra, dec = self.wcs.all_pix2world(x * self.downsample_factor, y * self.downsample_factor, 0)

            hovertemplate = 'RA: %{customdata[0]:.5f}<br>Dec: %{customdata[1]:.5f}'
            self.fig.data[0].customdata = np.dstack((ra, dec))
            self.fig.data[0].hovertemplate = hovertemplate
            self.fig.data[0].hoverinfo = 'text'

            if self.app is not None:
                self.app.layout.children[0].figure = self.fig

    def display(self):
        if self.app is not None:
            self.app.run_server(mode='inline')



class DualImageClicker:
    def __init__(
        self,
        file1: str,
        file2: str,
        downsample_factor: int = 1,
        saturation_factor1: float = 0.1,
        saturation_factor2: float = 0.1
    ):
        """Initialize a dual image clicking interface.
        
        Args:
            file1: Path to first FITS file
            file2: Path to second FITS file
            downsample_factor: Factor by which to downsample images
            saturation_factor1: Saturation scaling for first image
            saturation_factor2: Saturation scaling for second image
        """
        self.file1 = file1
        self.file2 = file2
        self.downsample_factor = downsample_factor
        self.saturation_factor1 = saturation_factor1
        self.saturation_factor2 = saturation_factor2
        
        # Initialize state variables
        self.coords1: Optional[np.ndarray] = None
        self.coords2: Optional[np.ndarray] = None
        self.err1: Optional[np.ndarray] = None
        self.err2: Optional[np.ndarray] = None
        self.wcs1: Optional[WCS] = None
        self.wcs2: Optional[WCS] = None
        self.fig = None
        self.app = None

    def get_coords(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the clicked coordinates for both images.
        
        Returns:
            Tuple containing coordinates for image 1 and image 2
        """
        return self.coords1, self.coords2

    def get_errors(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the error estimates for both images.
        
        Returns:
            Tuple containing error estimates for image 1 and image 2
        """
        return self.err1, self.err2

    def load_and_process_fits(self, filename: str, saturation_factor: float) -> Tuple[np.ndarray, WCS]:
        """Load and process a FITS file.
        
        Args:
            filename: Path to FITS file
            saturation_factor: Saturation scaling factor
            
        Returns:
            Tuple of (processed image data, WCS object)
        """
        with fits.open(filename) as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)
        
        # Downsample the image
        data = data[::self.downsample_factor, ::self.downsample_factor]
        
        # Normalize and clip the data
        vmin, vmax = np.percentile(data, [1, 99])
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, norm_data.max() * saturation_factor)
        
        return norm_data, wcs

    def create_figure(self, data1: np.ndarray, data2: np.ndarray) -> go.Figure:
        """Create a side-by-side figure with both images and initial marker traces.
        
        Args:
            data1: Processed data for first image
            data2: Processed data for second image
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Image 1', 'Image 2'),
            horizontal_spacing=0.05
        )
        
        # Add heatmap traces
        for idx, data in enumerate([data1, data2], 1):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    colorscale='Greys_r',
                    showscale=False,
                    hoverinfo='none'
                ),
                row=1, col=idx
            )
            
        # Add initial marker traces (invisible)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(symbol='cross', size=10, color='red'),
                showlegend=False,
                visible=False,
                xaxis='x1',
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(symbol='cross', size=10, color='red'),
                showlegend=False,
                visible=False,
                xaxis='x2',
                yaxis='y2'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            width=1200,
            title_text="Click on each image to mark points",
            showlegend=False
        )

        # Set equal aspect ratio for both subplots
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            autorange="reversed"
        )

        return fig

    def add_hover_templates(self):
        """Add hover templates showing RA/Dec coordinates."""
        if self.fig is None or self.wcs1 is None or self.wcs2 is None:
            return

        for idx, wcs in enumerate([self.wcs1, self.wcs2]):
            data = self.fig.data[idx]
            ny, nx = data.z.shape
            y, x = np.mgrid[0:ny, 0:nx]
            ra, dec = wcs.all_pix2world(
                x * self.downsample_factor,
                y * self.downsample_factor,
                0
            )
            
            data.customdata = np.dstack((ra, dec))
            data.hovertemplate = 'RA: %{customdata[0]:.5f}<br>Dec: %{customdata[1]:.5f}'
            data.hoverinfo = 'text'

    def run_plot_context(self):
        """Set up and run the Dash application context."""
        app = dash.Dash(__name__)
        
        # Load and process both images
        data1, self.wcs1 = self.load_and_process_fits(self.file1, self.saturation_factor1)
        data2, self.wcs2 = self.load_and_process_fits(self.file2, self.saturation_factor2)
        
        # Create the figure with initial marker traces
        self.fig = self.create_figure(data1, data2)
        self.add_hover_templates()
        
        app.layout = html.Div([
            dcc.Graph(id='dual-image-plot', figure=self.fig),
            html.Div(id='click-data-1'),
            html.Div(id='click-data-2'),
        ])

        @app.callback(
            [Output('dual-image-plot', 'figure'),
             Output('click-data-1', 'children'),
             Output('click-data-2', 'children')],
            Input('dual-image-plot', 'clickData'),
            State('dual-image-plot', 'figure')
        )
        def update_clicks(clickData, figure):
            if not clickData:
                return figure, "Click on Image 1", "Click on Image 2"

            # Determine which subplot was clicked
            curve_number = clickData['points'][0]['curveNumber']
            x = clickData['points'][0]['x']
            y = clickData['points'][0]['y']
            
            # The first two traces are heatmaps, then marker1, then marker2
            marker1_trace = 2
            marker2_trace = 3
            
            if curve_number == 0:  # First image clicked
                # Update marker position and make visible
                figure['data'][marker1_trace]['x'] = [x]
                figure['data'][marker1_trace]['y'] = [y]
                figure['data'][marker1_trace]['visible'] = True
                
                # Update coordinates
                ra, dec = self.wcs1.all_pix2world(
                    x * self.downsample_factor,
                    y * self.downsample_factor,
                    0
                )
                self.coords1 = np.array([ra, dec])
                self.err1 = 3.5 * np.diag(self.wcs1.pixel_scale_matrix)
                msg1 = f"Image 1: RA={ra:.5f}±{self.err1[0]:.7f}, Dec={dec:.5f}±{self.err1[1]:.7f}"
                msg2 = "Click on Image 2" if self.coords2 is None else \
                       f"Image 2: RA={self.coords2[0]:.5f}±{self.err2[0]:.7f}, Dec={self.coords2[1]:.5f}±{self.err2[1]:.7f}"
            
            else:  # Second image clicked
                # Update marker position and make visible
                figure['data'][marker2_trace]['x'] = [x]
                figure['data'][marker2_trace]['y'] = [y]
                figure['data'][marker2_trace]['visible'] = True
                
                # Update coordinates
                ra, dec = self.wcs2.all_pix2world(
                    x * self.downsample_factor,
                    y * self.downsample_factor,
                    0
                )
                self.coords2 = np.array([ra, dec])
                self.err2 = 3.5 * np.diag(self.wcs2.pixel_scale_matrix)
                msg2 = f"Image 2: RA={ra:.5f}±{self.err2[0]:.7f}, Dec={dec:.5f}±{self.err2[1]:.7f}"
                msg1 = "Click on Image 1" if self.coords1 is None else \
                       f"Image 1: RA={self.coords1[0]:.5f}±{self.err1[0]:.7f}, Dec={self.coords1[1]:.5f}±{self.err1[1]:.7f}"

            return figure, msg1, msg2

        self.app = app

    def display(self):
        """Display the interactive plot."""
        if self.app is not None:
            self.app.run_server(mode='inline')