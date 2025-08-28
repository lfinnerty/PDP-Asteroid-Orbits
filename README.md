# PDP Asteroid Orbits

Software tools and Jupyter notebooks for running asteroid orbit labs with the PDP 2024 Hartnell team. This package provides functionality for simulating asteroid observations, analyzing astronomical images, and fitting orbital parameters.

## Quick Links

- [Example Student Investigation Notebook](https://colab.research.google.com/drive/1aFKznQHYxGdu3Qqyb_sl_BuehrleEUx-?usp=sharing)
- [Example Facilitator Image Generation Notebook](https://colab.research.google.com/drive/1LTrjRjsOBlfpTeiKi8TaiWbeBhOCeynH?usp=sharing)

## Installation

To install this package, first create a clean Python environment. 
If you use conda, this can be accomplished for a "pdp" environment by typing:

```sh
conda create -n "pdp" python=3.9
conda activate pdp
```

Then, clone this repository and install it and its dependencies:

```sh
git clone https://github.com/lfinnerty/PDP-Asteroid-Orbits.git
cd PDP-Asteroid-Orbits
pip install -e .
```

Run the tests to verify your installation:

```sh
pytest
```

## Core Functionality

### Image Analysis Tools

#### DualImageClickSelector (`image_clicker.py`)
- Interactive tool for marking asteroid positions in paired FITS images
- Supports WCS coordinate transformations
- Provides position uncertainty estimates
- Real-time coordinate display and visual markers

#### FitsImageSwitcher (`image_switcher.py`)
- Tool for comparing two FITS images using an interactive blinker interface
- Supports image normalization and downsampling
- Interactive buttons for switching between images
- Maintains aspect ratio and scaling

### Orbit Analysis

#### OrbitInvestigation (`investigation_manager.py`)
- Main class for managing student orbit investigations
- Handles data loading, measurement processing, and orbit fitting
- Supports saving/loading investigation state
- Integrates with HuggingFace for data synchronization

#### Position and Orbit Calculations (`position_to_orbit.py`)
- Functions for converting between coordinate systems
- Kepler equation solver
- Orbit fitting using nested sampling
- Synthetic image generation with realistic asteroid motion

### Remote Data Management

#### HuggingFaceManager (`hf_utils.py`)
- Manages interactions with HuggingFace datasets
- Supports pulling new observations
- Handles secure token storage
- Provides backup and restore functionality

#### TokenHandler (`token_handler.py`)
- Secure storage and retrieval of API tokens
- Uses Fernet symmetric encryption
- Maintains separate key and token storage

## Workflow Overview

1. **Image Generation (Facilitators)**
   - Use the facilitator notebook to generate synthetic asteroid observations
   - Set orbital parameters and observation dates
   - Generate FITS image pairs with realistic asteroid motion
   - Upload to HuggingFace dataset

2. **Student Investigation**
   - Students use the investigation notebook to:
     - Load observation pairs
     - Mark asteroid positions
     - Calculate distances using parallax
     - Fit orbital parameters
     - Visualize results

3. **Data Management**
   - Investigation progress automatically saved
   - Results synchronized with HuggingFace
   - Support for multiple student groups

## Repository Structure

```
PDP-Asteroid-Orbits/
├── pdp_asteroids/
│   ├── image_clicker.py     # Interactive image analysis tools
│   ├── image_switcher.py    # Image comparison utilities
│   ├── investigation_manager.py  # Investigation workflow management
│   ├── position_to_orbit.py  # Orbital calculations and fitting
│   ├── hf_utils.py          # HuggingFace integration
│   └── token_handler.py     # Security utilities
├── tests/                   # Test suite
├── notebooks/               # Example notebooks
└── pyproject.toml          # Package configuration
```

## Dependencies

Key dependencies include:
- astropy: For astronomical calculations and FITS handling
- matplotlib: For visualization and interactive plotting
- numpy: For numerical computations
- plotly: For interactive visualizations
- dynesty/ultranest: For orbital parameter fitting
- huggingface_hub: For data synchronization
- cryptography: For secure token handling

See `pyproject.toml` for a complete list of dependencies.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite, ensure it passes. Add tests covering your new code.
5. Submit a pull request (but we aren't actively maintaining this repo, so you might need to draw our attention to it!)

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Authors

- Luke Finnerty (lfinnert@umich.edu)
- Evan Anders (evanhanders@gmail.com)
