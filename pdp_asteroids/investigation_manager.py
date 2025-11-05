from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
    plot_fit,
    plot_fit_animation
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
    including loading images, capturing clicks, calculating distances, fitting
    orbits, and synchronizing data with remote storage.
    
    Attributes:
        base_path (Path): Base path to observation data
        group (str): Student group identifier
        data (Dict[str, ObservationData]): Dictionary of observation data keyed by date
        orbit_fits (Dict[int, np.ndarray]): Dictionary of orbit fit samples
        current_date (Optional[str]): Currently selected observation date
        current_header (Optional[dict]): FITS header for current observation
        measurements_file (Path): Path to measurements pickle file
        orbits_file (Path): Path to orbit fits pickle file
    """
    # base_path: str = "/content/observations_2024/", 
    def __init__(
        self, 
        base_path: str = "/content/observations_2025/", 
        group: str = "test",
        hf_token: Optional[str] = None,
        repo_id: str = "hartnellpdp/observations_2025"
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
        self.measurements_file = self.group_path / "measurements.p"
        self.orbits_file = self.group_path / "orbits.p"
        self.data: Dict[str, ObservationData] = {}
        self.orbit_fits: Dict[int, np.ndarray] = {}
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
        self.current_selector: Optional[DualImageClickSelector] = None
        
        # Load existing data if available
        self._load_measurements()
        self._load_orbits()
    
    def process_measurements(self) -> None:
        """Process the current image measurements and save results.
        
        This method calculates distances from the marked positions, saves the
        results locally, and syncs them to the remote repository.
        
        Raises:
            ValueError: If measurements are incomplete or processing fails
        """
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
        
        # Save and sync data
        self._save_measurements()
        # self._sync_measurements_to_remote()
    
    def fit_orbit(self, **kwargs) -> np.ndarray:
        """Fit an orbit to all processed observations.
        
        This method performs the orbit fitting and stores the results both
        locally and in the remote repository.
        
        Args:
            **kwargs: Additional arguments passed to run_fit()
            
        Returns:
            np.ndarray: Fitted orbit samples
            
        Raises:
            ValueError: If no processed observations available
        """
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
        
        # Run the fit
        samples = run_fit(jds, rs, rerrs, thetas, terrs, **kwargs)
        
        # Store results with new index
        new_index = max(self.orbit_fits.keys(), default=-1) + 1
        self.orbit_fits[new_index] = samples
        
        # Save and sync fit results
        self._save_orbits()
        # self._sync_orbits_to_remote()
        
        return samples
    
    def plot_orbit(self, save_dir: Optional[str] = None) -> List[plt.Figure]:
        """Plot all stored orbit fits.
        
        Args:
            save_dir: Optional directory to save plot files
            
        Returns:
            List[plt.Figure]: List of generated figures
            
        Raises:
            ValueError: If no orbit fits available
        """
        if not self.orbit_fits:
            raise ValueError("No orbit fits available")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        
        # Prepare data arrays
        dates = sorted(self.data.keys())
        rs = np.array([self.data[d].r for d in dates])
        thetas = np.array([self.data[d].theta for d in dates])
        
        figures = []
        for fit_index, samples in self.orbit_fits.items():
            fig = plot_fit(dates, rs, thetas, samples,fit_index+1)
            
            if save_dir:
                fig.savefig(save_path / f"orbit_fit_{fit_index+1}.png")
            
            figures.append(fig)
        
        return figures


    def plot_orbit_animation(self, num: Optional[int] = -1,
                              save_dir: Optional[str] = None) -> List[plt.Figure]:
        """Plot stored orbit fits.
        
        Args:
            num: Index of orbit fit to plot, defaults to -1 (last orbit plotted)
            save_dir: Optional directory to save plot files
            
        Returns:
            List[plt.Figure]: List of generated figures
            
        Raises:
            ValueError: If no orbit fits available
        """
        if not self.orbit_fits:
            raise ValueError("No orbit fits available")
        
        # if save_dir:
        #     save_path = Path(save_dir)
        #     save_path.mkdir(exist_ok=True, parents=True)
        if not save_dir:
            save_path = self.group_path
        
        # Prepare data arrays
        dates = sorted(self.data.keys())
        rs = np.array([self.data[d].r for d in dates])
        thetas = np.array([self.data[d].theta for d in dates])
        
        # animations = []
        nfits = len(self.orbit_fits.items())
        ### If -1 is given (default), just does the last animation
        ### This is intended to let the students go back to a previous fit
        ### and see how things have changed if they want to
        ### 1 index for the student's clarity
        if num ==-1: 
            num = nfits-1
        else:
            num = num-1
        for fit_index, samples in self.orbit_fits.items():
            if fit_index == num:
                ani = plot_fit_animation(dates, rs, thetas, samples, fit_index+1)
                ani.save(save_path / f"orbit_fit_{fit_index+1}.gif")
        
        return ani
    
    def clear_investigation(self, confirm: bool = True) -> None:
        """Clear local investigation data and reset the investigation state.
        
        This function removes local data files and resets the investigation state.
        Note that this does not remove files from the remote HuggingFace repository - 
        those files will persist and may be re-downloaded during future pull operations.
        
        Args:
            confirm: If True, requires user confirmation before deleting
            
        Raises:
            ValueError: If confirmation is denied
        """
        if confirm:
            response = input(
                "This will clear your local investigation data and state. "
                "Note: Remote files in HuggingFace will persist. "
                "Type 'yes' to confirm: "
            )
            if response.lower() != 'yes':
                raise ValueError("Operation cancelled")
        
        # Clear local data files
        if self.measurements_file.exists():
            self.measurements_file.unlink()
        if self.orbits_file.exists():
            self.orbits_file.unlink()
        
        # Reset instance state
        self.data.clear()
        self.orbit_fits.clear()
        print(
            "Local investigation data cleared successfully.\n"
            "Note: Data files in the HuggingFace repository still exist and "
            "may be re-downloaded on the next pull operation."
        )
        
        # Create empty files to prevent re-downloading old data
        if self.hf_manager:
            try:
                # Create empty pickle files
                with open(self.measurements_file, 'wb') as f:
                    pickle.dump({}, f)
                with open(self.orbits_file, 'wb') as f:
                    pickle.dump({}, f)
                
                # Push empty files to override remote data
                self.hf_manager.push_to_hf(
                    folder=self.group_path,
                    repo_id=self.repo_id,
                    commit_message=f"Reset investigation data for {self.group}"
                )
            except Exception as e:
                print(
                    f"Warning: Failed to push empty files to remote: {str(e)}\n"
                    "Old data may be retrieved on next pull."
                )
    
    def _save_measurements(self) -> None:
        """Save observation data to local pickle file."""
        with open(self.measurements_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def _save_orbits(self) -> None:
        """Save orbit fits to local pickle file."""
        with open(self.orbits_file, 'wb') as f:
            pickle.dump(self.orbit_fits, f)
    
    def _load_measurements(self) -> None:
        """Load previously saved observation data if available."""
        if self.measurements_file.exists():
            with open(self.measurements_file, 'rb') as f:
                self.data = pickle.load(f)
    
    def _load_orbits(self) -> None:
        """Load previously saved orbit fits if available."""
        if self.orbits_file.exists():
            with open(self.orbits_file, 'rb') as f:
                self.orbit_fits = pickle.load(f)
    
    def _sync_measurements_to_remote(self) -> None:
        """Sync observation data to remote repository."""
        if self.hf_manager:
            try:
                self.hf_manager.push_to_hf(
                    folder=self.group_path,
                    repo_id=self.repo_id,
                    commit_message=f"Update measurements for {self.group}"
                )
            except Exception as e:
                print(f"Warning: Failed to sync measurements: {str(e)}")
    
    def _sync_orbits_to_remote(self) -> None:
        """Sync orbit fits to remote repository."""
        if self.hf_manager:
            try:
                self.hf_manager.push_to_hf(
                    folder=self.group_path,
                    repo_id=self.repo_id,
                    commit_message=f"Update orbit fits for {self.group}"
                )
            except Exception as e:
                print(f"Warning: Failed to sync orbit fits: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save data automatically."""
        self._save_measurements()
        self._save_orbits()
    
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
    
    def list_available_dates(self) -> List[str]:
        """List all available observation dates for the group.
        
        Returns:
            List[str]: List of available observation dates
        """
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
        """Load a specific observation date for analysis.
        
        Args:
            date: Observation date to load
            
        Raises:
            ValueError: If the observation files don't exist
        """
        self.current_date = date
        self.current_selector = None
        
        fname1 = self.group_path / f"{date}_{self.group}_frame1.fits"
        if not fname1.exists():
            raise ValueError(f"No observation found for date {date}")
            
        self.current_header = fits.getheader(fname1, 0)
        
        print(f"\nLoaded observation from {date}")
        print(f"Julian Date: {self.current_header['JD']:.6f}")
        print(f"Time between frames: {self.current_header['obsdt']*24:.1f} hours")
        if date in self.data:
            print("\nNote: This date already has measurements.")
    
    def examine_images(self, saturation: float = 0.1) -> None:
        """Display image blinker for current observation.
        
        Args:
            saturation: Saturation level for image display
            
        Raises:
            ValueError: If no observation is loaded
        """
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
        """Display both images side by side for marking the asteroid.
        
        Args:
            saturation1: Saturation level for first image
            saturation2: Saturation level for second image
            
        Raises:
            ValueError: If no observation is loaded
        """
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