"""Base class for recommendation models with common interface.

This abstract base class provides a unified interface for all recommendation
models, eliminating code duplication for common methods like predict,
get_user_embeddings, and get_item_embeddings.

All models should inherit from this class to ensure consistent interface.
"""

from __future__ import annotations

import abc
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class BaseRecommender(abc.ABC):
    """Abstract base class for recommendation models.
    
    Provides common interface and shared functionality for all recommendation
    models in the codebase. Eliminates code duplication for embedding
    access and prediction methods.
    """

    def __init__(self, num_users: int, num_items: int, 
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        """Initialize base recommender.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items  
            embedding_dim: Dimensionality of embeddings
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device
        
        # These will be populated after training
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None

    @abc.abstractmethod
    def fit(self, ratings, user_to_idx, item_to_idx, **kwargs) -> dict[str, float]:
        """Train the model on rating data.
        
        Args:
            ratings: DataFrame with user-item ratings
            user_to_idx: Mapping from user IDs to internal indices
            item_to_idx: Mapping from item IDs to internal indices
            **kwargs: Model-specific training parameters
            
        Returns:
            Dictionary with training metrics (e.g., loss, time)
        """
        pass

    def predict(self, user_idx: int,
                item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict scores for items for a given user.
        
        Args:
            user_idx: Internal user index
            item_indices: Specific item indices to score (None = all items)
            
        Returns:
            Array of prediction scores
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model not trained yet - no embeddings available")
        
        u = self.user_embeddings[user_idx]
        items = self.item_embeddings if item_indices is None \
            else self.item_embeddings[item_indices]
        
        # Default: dot product similarity
        return items @ u

    def get_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings.
        
        Returns:
            Array of user embeddings
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet - no embeddings available")
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings.
        
        Returns:
            Array of item embeddings
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.item_embeddings is None:
            raise ValueError("Model not trained yet - no embeddings available")
        return self.item_embeddings

    def _validate_embeddings(self) -> None:
        """Validate that embeddings have been extracted and have correct shape.
        
        Raises:
            ValueError: If embeddings are None or have incorrect shape
        """
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model not trained yet - no embeddings available")
        
        if self.user_embeddings.shape[0] != self.num_users:
            raise ValueError(f"User embeddings shape {self.user_embeddings.shape} "
                           f"doesn't match num_users {self.num_users}")
        
        if self.item_embeddings.shape[0] != self.num_items:
            raise ValueError(f"Item embeddings shape {self.item_embeddings.shape} "
                           f"doesn't match num_items {self.num_items}")

    def _set_embeddings(self, user_emb: np.ndarray, item_emb: np.ndarray) -> None:
        """Set embeddings after training with validation.
        
        Args:
            user_emb: User embeddings array
            item_emb: Item embeddings array
        """
        self.user_embeddings = user_emb
        self.item_embeddings = item_emb
        self._validate_embeddings()


class BasePyTorchRecommender(BaseRecommender):
    """Base class for PyTorch-based recommendation models.
    
    Adds PyTorch-specific utilities and common training patterns.
    """
    
    def __init__(self, num_users: int, num_items: int, 
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        """Initialize PyTorch recommender."""
        super().__init__(num_users, num_items, embedding_dim, device)
        
        # Import torch here to avoid import issues for non-PyTorch models
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError("PyTorch is required for PyTorchRecommender")

    def _get_device(self) -> str:
        """Get the appropriate device for training."""
        if self.device == "cuda" and self.torch.cuda.is_available():
            return "cuda"
        elif self.device == "mps" and self.torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _extract_embeddings_from_model(self, model, user_attr: str = "user_embedding", 
                                     item_attr: str = "item_embedding") -> None:
        """Extract embeddings from PyTorch model attributes.
        
        Args:
            model: Trained PyTorch model
            user_attr: Name of user embedding attribute
            item_attr: Name of item embedding attribute
        """
        model.eval()
        with self.torch.no_grad():
            user_emb = getattr(model, user_attr).weight.cpu().numpy()
            item_emb = getattr(model, item_attr).weight.cpu().numpy()
            self._set_embeddings(user_emb, item_emb)

    def _extract_embeddings_from_forward(self, model, user_ids, item_ids,
                                      user_repr_attr: str = "user_repr",
                                      item_repr_attr: str = "item_repr") -> None:
        """Extract embeddings by forward pass through model.
        
        Args:
            model: Trained PyTorch model
            user_ids: Tensor of all user IDs
            item_ids: Tensor of all item IDs  
            user_repr_attr: Method name for user representation
            item_repr_attr: Method name for item representation
        """
        device = self._get_device()
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        
        model.eval()
        with self.torch.no_grad():
            user_emb = getattr(model, user_repr_attr)(user_ids).cpu().numpy()
            item_emb = getattr(model, item_repr_attr)(item_ids).cpu().numpy()
            self._set_embeddings(user_emb, item_emb)


class BaseLibraryRecommender(BaseRecommender):
    """Base class for models using external libraries (Cornac, scikit-surprise, etc.).
    
    Handles library-specific initialization and embedding extraction patterns.
    """
    
    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        """Initialize library-based recommender."""
        super().__init__(num_users, num_items, embedding_dim, device)
        self.model = None  # Will be set by subclass

    def _extract_library_embeddings(self, user_getter, item_getter) -> None:
        """Extract embeddings from library model using getter functions.
        
        Args:
            user_getter: Function to get user embeddings
            item_getter: Function to get item embeddings
        """
        user_emb = user_getter()
        item_emb = item_getter()
        self._set_embeddings(user_emb, item_emb)
