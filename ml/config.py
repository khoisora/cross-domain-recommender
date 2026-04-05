"""ML-specific configuration and hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from backend.config.settings import get_settings


@dataclass
class MLConfig:
    """Consolidated ML configuration derived from app settings."""

    # Device
    device: str = "cpu"
    seed: int = 42

    # Paths
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Matrix Factorization
    mf_embedding_dim: int = 64
    mf_lr: float = 0.01
    mf_epochs: int = 80
    mf_reg_lambda: float = 0.01

    # NCF / NeuMF
    ncf_gmf_dim: int = 32
    ncf_mlp_dim: int = 32
    ncf_epochs: int = 20
    ncf_lr: float = 0.001
    ncf_num_negatives: int = 4

    # LightGCN
    lgcn_embedding_dim: int = 64
    lgcn_n_layers: int = 3
    lgcn_epochs: int = 300
    lgcn_lr: float = 0.001
    lgcn_reg_lambda: float = 1e-4

    # CMF
    cmf_embedding_dim: int = 64
    cmf_epochs: int = 80
    cmf_lr: float = 0.005

    
    # Bi-TGCF
    bi_tgcf_embedding_dim: int = 64
    bi_tgcf_n_layers: int = 2
    bi_tgcf_epochs: int = 200
    bi_tgcf_lr: float = 0.001
    bi_tgcf_align_lambda: float = 0.1

    @classmethod
    def from_settings(cls) -> MLConfig:
        """Build MLConfig from application settings."""
        s = get_settings()
        return cls(
            device=s.ml_device,
            seed=s.ml_seed,
            artifacts_dir=s.artifacts_path,
            data_dir=s.data_path,
            embedding_model=s.embedding_model,
            embedding_dim=s.embedding_dim,
        )

    @property
    def checkpoint_dir(self) -> Path:
        p = self.artifacts_dir / "checkpoints"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_dir(self) -> Path:
        p = self.artifacts_dir / "indexes"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def embedding_cache_dir(self) -> Path:
        p = self.artifacts_dir / "embeddings"
        p.mkdir(parents=True, exist_ok=True)
        return p
