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
from .hf_utils import hf_manager, HuggingFaceManager

@dataclass
class ObservationData:
    """Container for observation data and measurements."""
    jd: float
    r: float
    r_err: float
    theta: float
    theta_err: float = 1e-4


class OrbitInvestigation:
    """Context manager for student orbit investigation workflow.
    
    This class manages the state and workflow for investigating asteroid orbits,
    including loading images, capturing clicks, calculating distances, and fitting
    orbits.
    
    Attributes:
        base_path (Path): Base path to observation data
        group (str): Student group identifier
        data (Dict[str, ObservationData]): Dictionary of observation data keyed by date
        current_date (Optional[str]): Currently selected observation date
        current_header (Optional[dict]): FITS header for current observation
    """
    
    def __init__(
        self, 
        base_path: str = "/content/observations_2024/", 
        group: str = "test",
        hf_token: Optional[str] = None,
        repo_id: str = "hartnellpdp/observations_2024"
    ):
        """Initialize the investigation manager.
        
        Args:
            base_path (str): Path to observation data directory
            group (str): Student group identifier
            hf_token (Optional[str]): HuggingFace access token
            repo_id (str): HuggingFace repository ID
        """
        self.base_path = Path(base_path)
        self.group = group
        self.group_path = self.base_path / group
        self.data: Dict[str, ObservationData] = {}
        self.repo_id = repo_id
        
        # Initialize HuggingFace manager if token provided
        if hf_token is not None:
            self.hf_manager = HuggingFaceManager(hf_token)
        else:
            self.hf_manager = hf_manager
        
        # Current state
        self.current_date: Optional[str] = None
        self.current_header: Optional[dict] = None
        self._current_clicks: Optional[Tuple] = None
        self.current_selector: Optional[FitsImageSwitcher] = None
        
        # Load existing data if available
        self._load_saved_data()
    
    def get_new_observations(self, backup: bool = True) -> List[str]:
        """Pull latest observations from HuggingFace repository.
        
        This method updates the local observation files by pulling changes
        from the remote repository. It returns a list of new observation
        dates that weren't previously available.
        
        Args:
            backup (bool): If True, creates a backup before pulling changes
            
        Returns:
            List[str]: New observation dates that were added
            
        Raises:
            ValueError: If HuggingFace manager not initialized or pull fails
        """
        if not self.hf_manager:
            raise ValueError(
                "HuggingFace token not provided. Initialize with hf_token "
                "to enable remote updates."
            )
        
        # Get current list of dates
        existing_dates = set(self.list_available_dates())
        
        try:
            # Pull latest changes
            self.hf_manager.pull_changes(
                repo_id=self.repo_id,
                local_dir=str(self.base_path),
                backup=backup
            )
            
            # Get updated list of dates
            new_dates = set(self.list_available_dates())
            added_dates = sorted(list(new_dates - existing_dates))
            
            if added_dates:
                print("\nNew observations available for dates:")
                for date in added_dates:
                    print(f"  • {date}")
            else:
                print("\nNo new observations available.")
                
            return added_dates
            
        except Exception as e:
            print(f"Failed to update observations: {str(e)}")
            print("Continuing with existing observation data.")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save data automatically."""
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
    
    def fit_orbit(self, nlive=100,dlogz=0.5,bootstrap=0,phase0=[0,1],a=[0.1,10],e=[0,0.99],omega=[0,1], sampler='dynesty') -> plt.Figure:
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
        
        samples = run_fit(jds, rs, rerrs, thetas, terrs, nlive=nlive,dlogz=dlogz,bootstrap=0,phase0=phase0,a=a,e=e,omega=omega, sampler=sampler)
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