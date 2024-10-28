from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u

from .image_switcher import FitsImageSwitcher
from .image_clicker import DualImageClickSelector
from .position_to_orbit import (
    parallax_to_dist, 
    dist_to_r,
    run_fit,
    plot_fit
)


@dataclass
class ObservationData:
    """Container for observation data and measurements."""
    jd: float
    r: float
    r_err: float
    theta: float
    theta_err: float = 1e-4


class OrbitInvestigation:
    """Context manager for student orbit investigation workflow."""
    
    def __init__(self, base_path: str = "/content/observations_2024/", group: str = "test"):
        """Initialize the investigation manager.
        
        Args:
            base_path: Path to observation data directory
            group: Student group identifier
        """
        self.base_path = Path(base_path)
        self.group = group
        self.group_path = self.base_path / group
        self.data: Dict[str, ObservationData] = {}
        
        # Current state
        self.current_date: Optional[str] = None
        self.current_header: Optional[dict] = None
        self.current_selector: Optional[DualImageClickSelector] = None
        
        # Load existing data if available
        self._load_saved_data()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_data()
    
    def list_available_dates(self) -> List[str]:
        """List all available observation dates for the group."""
        dates = set()
        print("Available observation dates:")
        for file in self.group_path.glob("*.fits"):
            if "_frame" in file.name:
                date = file.name.split("_")[0]
                dates.add(date)
        dates_list = sorted(list(dates))
        for date in dates_list:
            status = "✓" if date in self.data else " "
            print(f"[{status}] {date}")
        return dates_list
    
    def load_observation(self, date: str) -> None:
        """Load a specific observation date for analysis."""
        self.current_date = date
        self.current_selector = None
        
        fname1 = self.group_path / f"{date}_{self.group}_frame1.fits"
        self.current_header = fits.getheader(fname1, 0)
        
        print(f"\nLoaded observation from {date}")
        print(f"Julian Date: {self.current_header['JD']:.6f}")
        print(f"Time between frames: {self.current_header['obsdt']*24:.1f} hours")
        if date in self.data:
            print("\nNote: This date already has measurements.")
    
    def examine_images(self, saturation: float = 0.1) -> None:
        """Display image blinker for current observation."""
        if not self.current_date:
            raise ValueError("No observation date loaded. Use load_observation() first.")
            
        fname1 = self.group_path / f"{self.current_date}_{self.group}_frame1.fits"
        fname2 = self.group_path / f"{self.current_date}_{self.group}_frame2.fits"
        
        switcher = FitsImageSwitcher(str(fname1), str(fname2), 
                                   saturation_factor=saturation)
        switcher.display()
        
        print("\nUse the buttons above to switch between images.")
        print("Look for an object that changes position between frames.")
    
    def mark_asteroid(self, saturation1: float = 0.1, saturation2: float = 0.1) -> None:
        """Display both images side by side for marking the asteroid."""
        if not self.current_date:
            raise ValueError("No observation date loaded. Use load_observation() first.")
            
        fname1 = self.group_path / f"{self.current_date}_{self.group}_frame1.fits"
        fname2 = self.group_path / f"{self.current_date}_{self.group}_frame2.fits"
        
        self.current_selector = DualImageClickSelector(
            str(fname1), 
            str(fname2),
            saturation_factor1=saturation1,
            saturation_factor2=saturation2
        )
        self.current_selector.display()
        
        print("\nClick on the asteroid in each image.")
        print("The cross markers show your selected positions.")
        print("Click again to update a position if needed.")
    
    def process_measurements(self) -> None:
        """Process the current image measurements and save results."""
        if not self.current_selector:
            raise ValueError("No measurements to process. Mark asteroid positions first.")
        
        coords1, coords2 = self.current_selector.get_coords()
        errs1, errs2 = self.current_selector.get_errors()
        
        if any(x is None for x in [coords1, coords2, errs1, errs2]):
            raise ValueError("Incomplete measurements. Please mark both positions.")
        
        # Convert coordinates to format expected by parallax_to_dist
        p1 = (coords1.ra.deg, coords1.dec.deg)
        p2 = (coords2.ra.deg, coords2.dec.deg)
        e1 = errs1.to(u.deg).value
        e2 = errs2.to(u.deg).value
        
        dist, dist_err = parallax_to_dist(p1, e1, p2, e2, self.current_header['obsdt'])
        
        r, r_err = dist_to_r(
            self.current_header['jd'],
            self.current_header['theta'],
            self.current_header['elong'],
            dist, dist_err
        )
        
        self.data[self.current_date] = ObservationData(
            jd=self.current_header['jd'],
            r=r,
            r_err=r_err,
            theta=self.current_header['theta']
        )
        
        print(f"\nProcessed measurements for {self.current_date}:")
        print(f"Distance from Earth: {dist:.3f} ± {dist_err:.3f} AU")
        print(f"Distance from Sun: {r:.3f} ± {r_err:.3f} AU")
        
        self._save_data()
    
    def fit_orbit(self, sampler: str = 'dynesty') -> plt.Figure:
        """Fit an orbit to all processed observations."""
        if not self.data:
            raise ValueError("No processed observations available")
            
        # Prepare arrays for fitting
        dates = sorted(self.data.keys())
        jds = np.array([self.data[d].jd for d in dates])
        rs = np.array([self.data[d].r for d in dates])
        rerrs = np.array([self.data[d].r_err for d in dates])
        thetas = np.array([self.data[d].theta for d in dates])
        terrs = np.array([self.data[d].theta_err for d in dates])
        
        print(f"Fitting orbit using {len(dates)} observations:")
        for date in dates:
            print(f"  • {date}")
        
        samples = run_fit(jds, rs, rerrs, thetas, terrs, sampler=sampler)
        fig = plot_fit(rs, thetas, samples)
        
        return fig
    
    def _load_saved_data(self) -> None:
        """Load previously saved data if available."""
        save_file = self.group_path / "save.p"
        if save_file.exists():
            with open(save_file, 'rb') as f:
                self.data = pickle.load(f)
    
    def _save_data(self) -> None:
        """Save current data to disk."""
        save_file = self.group_path / "save.p"
        with open(save_file, 'wb') as f:
            pickle.dump(self.data, f)