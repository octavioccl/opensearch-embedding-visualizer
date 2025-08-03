"""Configuration management for OpenSearch Vector Visualizer."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class ReductionMethod(str, Enum):
    """Supported dimensionality reduction methods."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


class OpenSearchConfig(BaseModel):
    """OpenSearch connection configuration."""
    host: str = Field(default="localhost", description="OpenSearch host")
    port: int = Field(default=9201, description="OpenSearch port")
    username: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")
    use_ssl: bool = Field(default=False, description="Use SSL connection")
    verify_certs: bool = Field(default=False, description="Verify SSL certificates")
    ca_certs: Optional[str] = Field(default=None, description="Path to CA certificates")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class IndexConfig(BaseModel):
    """Index configuration for fetching embeddings."""
    name: Optional[str] = Field(default=None, description="Index name")
    embedding_field: str = Field(default="embedding", description="Field containing embeddings")
    name_field: str = Field(default="name", description="Field containing document names/IDs")
    max_docs: int = Field(default=1000, description="Maximum documents to fetch")
    
    @validator('max_docs')
    def validate_max_docs(cls, v):
        if v <= 0:
            raise ValueError('max_docs must be positive')
        return v


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    method: ReductionMethod = Field(default=ReductionMethod.TSNE, description="Dimensionality reduction method")
    n_components: int = Field(default=3, description="Number of dimensions (2 or 3)")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    
    # t-SNE specific parameters
    perplexity: float = Field(default=30.0, description="t-SNE perplexity parameter")
    learning_rate: float = Field(default=200.0, description="t-SNE learning rate")
    max_iter: int = Field(default=1000, description="t-SNE maximum number of iterations")
    
    # UMAP specific parameters
    n_neighbors: int = Field(default=15, description="UMAP n_neighbors parameter")
    min_dist: float = Field(default=0.1, description="UMAP min_dist parameter")
    metric: str = Field(default="cosine", description="UMAP distance metric")
    
    # PCA specific parameters
    whiten: bool = Field(default=False, description="PCA whitening")
    
    @validator('n_components')
    def validate_n_components(cls, v):
        if v not in [2, 3]:
            raise ValueError('n_components must be 2 or 3')
        return v
        
    @validator('perplexity')
    def validate_perplexity(cls, v):
        if v <= 0:
            raise ValueError('perplexity must be positive')
        return v


class FilterConfig(BaseModel):
    """Configuration for filtering data."""
    query: Optional[Dict[str, Any]] = Field(default=None, description="OpenSearch query")
    fields_to_include: Optional[List[str]] = Field(default=None, description="Fields to include in results")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="asc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('sort_order must be "asc" or "desc"')
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    
    # UI settings
    page_title: str = Field(default="OpenSearch Vector Visualizer", description="Page title")
    page_icon: str = Field(default="🔍", description="Page icon")
    layout: str = Field(default="wide", description="Page layout")
    
    model_config = {
        "extra": "forbid",  # Don't allow extra fields
        "validate_assignment": True  # Validate on assignment
    }