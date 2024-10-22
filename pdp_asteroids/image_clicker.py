from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.io import fits
from astropy.wcs import WCS
import dash
from dash import dcc, html, clientside_callback
from dash.dependencies import Input, Output, State

class ImageClicker:
    def __init__(self, file_name: str, downsample_factor: int = 1,saturation_factor: float = 0.1):
        self.coords : Optional[np.ndarray] = None
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
                self.err = [3.5*np.array(self.wcs.cdelt)]
                return f"Clicked at RA: {ra:.5f}, Dec: {dec:.5f}"

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
