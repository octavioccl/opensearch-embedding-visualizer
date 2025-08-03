"""Data models and types for OpenSearch Vector Visualizer."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DocumentData(BaseModel):
    """Represents a document with embedding and metadata."""
    id: str = Field(description="Document ID")
    name: str = Field(description="Document name/title")
    embedding: List[float] = Field(description="Document embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "arbitrary_types_allowed": True
    }


@dataclass
class VisualizationResult:
    """Result of dimensionality reduction visualization."""
    points: np.ndarray  # Reduced dimensional points
    labels: List[str]   # Document names/labels
    metadata: pd.DataFrame  # Full metadata DataFrame
    method: str         # Reduction method used
    n_components: int   # Number of dimensions
    explained_variance_ratio: Optional[np.ndarray] = None  # For PCA
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for visualization."""
        df = pd.DataFrame({
            'name': self.labels,
            'x': self.points[:, 0],
            'y': self.points[:, 1]
        })
        
        if self.n_components == 3:
            df['z'] = self.points[:, 2]
        else:
            df['z'] = 0
            
        # Add metadata columns
        for col in self.metadata.columns:
            if col not in df.columns:
                df[col] = self.metadata[col].values
                
        return df


class FilterResult(BaseModel):
    """Result of applying filters to data."""
    total_docs: int = Field(description="Total documents before filtering")
    filtered_docs: int = Field(description="Documents after filtering")
    filter_summary: str = Field(description="Human-readable filter summary")
    execution_time: float = Field(description="Filter execution time in seconds")


class ConnectionStatus(BaseModel):
    """OpenSearch connection status."""
    connected: bool = Field(description="Whether connection is successful")
    cluster_name: Optional[str] = Field(default=None, description="Cluster name")
    version: Optional[str] = Field(default=None, description="OpenSearch version")
    error_message: Optional[str] = Field(default=None, description="Error message if connection failed")


class IndexInfo(BaseModel):
    """Information about an OpenSearch index."""
    name: str = Field(description="Index name")
    doc_count: int = Field(description="Number of documents")
    size: str = Field(description="Index size")
    health: str = Field(description="Index health status")
    has_embedding_field: bool = Field(default=False, description="Whether embedding field exists")
    has_name_field: bool = Field(default=False, description="Whether name field exists")
    embedding_dimensions: Optional[int] = Field(default=None, description="Embedding vector dimensions")
    
    
class SearchQuery(BaseModel):
    """OpenSearch query configuration."""
    query: Dict[str, Any] = Field(default_factory=lambda: {"match_all": {}}, description="OpenSearch query")
    size: int = Field(default=1000, description="Number of results to return")
    source_includes: Optional[List[str]] = Field(default=None, description="Fields to include")
    source_excludes: Optional[List[str]] = Field(default=None, description="Fields to exclude")
    sort: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sort configuration")
    
    def to_opensearch_body(self) -> Dict[str, Any]:
        """Convert to OpenSearch request body."""
        body = {
            "query": self.query,
            "size": self.size
        }
        
        if self.source_includes or self.source_excludes:
            body["_source"] = {}
            if self.source_includes:
                body["_source"]["includes"] = self.source_includes
            if self.source_excludes:
                body["_source"]["excludes"] = self.source_excludes
                
        if self.sort:
            body["sort"] = self.sort
            
        return body


# Type aliases for better readability
EmbeddingArray = np.ndarray
MetadataDict = Dict[str, Any]
QueryDict = Dict[str, Any]