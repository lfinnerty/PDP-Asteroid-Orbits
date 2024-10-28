from pathlib import Path
import base64
from cryptography.fernet import Fernet
import json
from typing import Optional

REPO_PATH = Path(__file__).parent 

class TokenHandler:
    """Handles secure storage and retrieval of API tokens.
    
    This class provides methods to securely store and retrieve API tokens
    using Fernet symmetric encryption. The encryption key is stored separately
    from the encrypted tokens.
    """
    
    def __init__(self, key_file: str = str(REPO_PATH/".token.key"), token_file: str = str(REPO_PATH/".tokens")):
        """Initialize the token handler.
        
        Args:
            key_file: Path to store the encryption key
            token_file: Path to store the encrypted tokens
        """
        self.key_file = Path(key_file)
        self.token_file = Path(token_file)
        self._key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
        
    def _initialize_key(self) -> None:
        """Create or load the encryption key."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                self._key = base64.urlsafe_b64decode(f.read())
        else:
            self._key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(base64.urlsafe_b64encode(self._key))
        
        self._fernet = Fernet(self._key)
    
    def store_token(self, name: str, token: str) -> None:
        """Securely store an API token.
        
        Args:
            name: Identifier for the token
            token: The API token to store
        """
        if self._fernet is None:
            self._initialize_key()
            
        # Load existing tokens if any
        tokens = {}
        if self.token_file.exists():
            with open(self.token_file, 'rb') as f:
                encrypted_data = f.read()
                decrypted_data = self._fernet.decrypt(encrypted_data)
                tokens = json.loads(decrypted_data)
        
        # Add or update token
        tokens[name] = token
        
        # Encrypt and save
        encrypted_data = self._fernet.encrypt(json.dumps(tokens).encode())
        with open(self.token_file, 'wb') as f:
            f.write(encrypted_data)
    
    def get_token(self, name: str) -> Optional[str]:
        """Retrieve a stored API token.
        
        Args:
            name: Identifier for the token to retrieve
            
        Returns:
            The decrypted token if found, None otherwise
        """
        if not self.token_file.exists():
            return None
            
        if self._fernet is None:
            self._initialize_key()
            
        try:
            with open(self.token_file, 'rb') as f:
                encrypted_data = f.read()
                decrypted_data = self._fernet.decrypt(encrypted_data)
                tokens = json.loads(decrypted_data)
                return tokens.get(name)
        except Exception:
            return None