import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, snapshot_download, upload_folder
from .token_handler import TokenHandler

# Default repositories and paths
DEFAULT_REPO_ID = "hartnellpdp/observations_2024"
DEFAULT_WRITE_FOLDER = Path("/content/new_observations/")

class HuggingFaceManager:
    """Manages interactions with HuggingFace repositories for PDP Asteroids project.
    
    This class provides utilities for cloning and pushing to HuggingFace datasets,
    specifically designed for the PDP Asteroids observation workflow.
    
    Attributes:
        access_token (str): HuggingFace access token for authentication
        api (HfApi): HuggingFace API instance
    """
    
    def __init__(self, access_token: Optional[str] = None):
        """Initialize the HuggingFace manager.
        
        Args:
            access_token: Optional HuggingFace access token. If not provided,
                         will attempt to load from secure storage.
        """
        self.token_handler = TokenHandler()
        
        if access_token is None:
            access_token = self.token_handler.get_token('huggingface')
            if access_token is None:
                raise ValueError(
                    "No access token provided and none found in secure storage. "
                    "Use store_token() to save a token first."
                )
        
        self.access_token = access_token
        self.api = HfApi(token=access_token)
    
    @classmethod
    def store_token(cls, token: str) -> None:
        """Securely store a HuggingFace API token.
        
        Args:
            token: The API token to store
        """
        handler = TokenHandler()
        handler.store_token('huggingface', token)

    def clone_repo(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        local_dir: Optional[str] = None,
    ) -> Path:
        """Clone a HuggingFace repository using an access token.
        
        Args:
            repo_id: Repository ID (e.g., 'username/repo-name')
            local_dir: Local directory to clone into. Defaults to /content/{repo-name}
            
        Returns:
            Path to the cloned repository directory
            
        Raises:
            ValueError: If repo_id is invalid or connection fails
        """
        try:
            # If no local_dir specified, create one based on repo name
            if local_dir is None:
                local_dir = f"/content/{repo_id.split('/')[-1]}"
            local_path = Path(local_dir)

            # Clean up any existing directory
            if local_path.exists():
                self._clean_directory(local_path)

            # Clone the repository using HF API
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                repo_type="dataset",
                token=self.access_token
            )

            print(f"Successfully cloned {repo_id} to {local_path}")
            return local_path

        except Exception as e:
            raise ValueError(f"Failed to clone repository: {str(e)}")

    def push_to_hf(
        self,
        folder: Path = DEFAULT_WRITE_FOLDER,
        repo_id: str = DEFAULT_REPO_ID,
        commit_message: str = "Update from facilitator notebook"
    ) -> None:
        """Add files and push changes to a HuggingFace repository.
        
        Args:
            folder: Path to folder containing files to push
            repo_id: Name of dataset repository
            commit_message: Git commit message
            
        Raises:
            ValueError: If folder doesn't exist or push fails
        """
        if not folder.exists():
            raise ValueError(f"Specified folder does not exist: {folder}")

        try:
            upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                token=self.access_token,
                commit_message=commit_message,
                repo_type="dataset"
            )
            print(f"Successfully pushed {folder} to {repo_id}")

        except Exception as e:
            raise ValueError(f"Failed to push to repository: {str(e)}")

    @staticmethod
    def _clean_directory(path: Path) -> None:
        """Recursively remove a directory and its contents.
        
        Args:
            path: Directory path to clean
        """
        import shutil
        shutil.rmtree(path)


# Create default instance only if token exists
token_handler = TokenHandler()
if token := token_handler.get_token('huggingface'):
    hf_manager = HuggingFaceManager(token)
else:
    hf_manager = None