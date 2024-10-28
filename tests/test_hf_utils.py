import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from pdp_asteroids.hf_utils import HuggingFaceManager


@pytest.fixture
def mock_hf_api():
    with patch('pdp_asteroids.hf_utils.HfApi') as mock_api:
        yield mock_api

@pytest.fixture
def mock_snapshot_download():
    with patch('pdp_asteroids.hf_utils.snapshot_download') as mock_download:
        yield mock_download

@pytest.fixture
def mock_upload_folder():
    with patch('pdp_asteroids.hf_utils.upload_folder') as mock_upload:
        yield mock_upload

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_repo"
    test_dir.mkdir()
    return test_dir

def test_init():
    """Test HuggingFaceManager initialization."""
    manager = HuggingFaceManager(access_token="test_token")
    assert manager.access_token == "test_token"

def test_clone_repo_success(mock_hf_api, mock_snapshot_download, temp_dir):
    """Test successful repository cloning."""
    manager = HuggingFaceManager()
    
    # Configure mock
    mock_snapshot_download.return_value = str(temp_dir)
    
    # Test cloning
    result = manager.clone_repo(
        repo_id="test/repo",
        local_dir=str(temp_dir)
    )
    
    assert result == temp_dir
    mock_snapshot_download.assert_called_once()

def test_clone_repo_failure(mock_hf_api, mock_snapshot_download):
    """Test repository cloning failure."""
    manager = HuggingFaceManager()
    
    # Configure mock to raise an exception
    mock_snapshot_download.side_effect = Exception("Connection failed")
    
    with pytest.raises(ValueError, match="Failed to clone repository"):
        manager.clone_repo(repo_id="test/repo")

def test_push_to_hf_success(mock_hf_api, mock_upload_folder, temp_dir):
    """Test successful push to repository."""
    manager = HuggingFaceManager()
    
    # Create test file in temporary directory
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    
    # Test pushing
    manager.push_to_hf(
        folder=temp_dir,
        repo_id="test/repo",
        commit_message="Test commit"
    )
    
    mock_upload_folder.assert_called_once_with(
        folder_path=str(temp_dir),
        repo_id="test/repo",
        token=manager.access_token,
        commit_message="Test commit",
        repo_type="dataset"
    )

def test_push_to_hf_nonexistent_folder(mock_hf_api):
    """Test pushing from a non-existent folder."""
    manager = HuggingFaceManager()
    
    with pytest.raises(ValueError, match="Specified folder does not exist"):
        manager.push_to_hf(
            folder=Path("/nonexistent/folder"),
            repo_id="test/repo"
        )

def test_push_to_hf_upload_failure(mock_hf_api, mock_upload_folder, temp_dir):
    """Test handling of upload failure."""
    manager = HuggingFaceManager()
    
    # Configure mock to raise an exception
    mock_upload_folder.side_effect = Exception("Upload failed")
    
    with pytest.raises(ValueError, match="Failed to push to repository"):
        manager.push_to_hf(
            folder=temp_dir,
            repo_id="test/repo"
        )

def test_clean_directory(temp_dir):
    """Test directory cleaning functionality."""
    # Create some test files
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    
    sub_dir = temp_dir / "subdir"
    sub_dir.mkdir()
    sub_file = sub_dir / "subfile.txt"
    sub_file.write_text("sub content")
    
    # Clean directory
    manager = HuggingFaceManager()
    manager._clean_directory(temp_dir)
    
    # Verify directory was removed
    assert not temp_dir.exists()