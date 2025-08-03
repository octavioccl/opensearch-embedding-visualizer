"""Core visualization engine for embedding dimensionality reduction."""

import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
from loguru import logger

from .config import VisualizationConfig, ReductionMethod
from .models import DocumentData, VisualizationResult, EmbeddingArray


class EmbeddingVisualizer:
    """Core visualization engine for embedding dimensionality reduction."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize visualizer with configuration."""
        self.config = config
        
    def reduce_dimensions(self, documents: List[DocumentData]) -> VisualizationResult:
        """Reduce embedding dimensions and prepare visualization data."""
        start_time = time.time()
        
        # Extract embeddings and create metadata DataFrame
        embeddings = np.array([doc.embedding for doc in documents])
        labels = [doc.name for doc in documents]
        
        # Create metadata DataFrame
        metadata_dict = {'name': labels, 'id': [doc.id for doc in documents]}
        for doc in documents:
            for key, value in doc.metadata.items():
                if key not in metadata_dict:
                    metadata_dict[key] = []
                metadata_dict[key].append(value)
        
        # Fill missing values
        max_len = len(documents)
        for key, values in metadata_dict.items():
            if len(values) < max_len:
                values.extend([None] * (max_len - len(values)))
        
        metadata_df = pd.DataFrame(metadata_dict)
        
        logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]}D to {self.config.n_components}D using {self.config.method}")
        
        # Apply dimensionality reduction
        if self.config.method == ReductionMethod.PCA:
            reduced_embeddings, explained_variance = self._apply_pca(embeddings)
        elif self.config.method == ReductionMethod.TSNE:
            reduced_embeddings, explained_variance = self._apply_tsne(embeddings), None
        elif self.config.method == ReductionMethod.UMAP:
            reduced_embeddings, explained_variance = self._apply_umap(embeddings), None
        else:
            raise ValueError(f"Unsupported reduction method: {self.config.method}")
        
        execution_time = time.time() - start_time
        logger.info(f"Dimensionality reduction completed in {execution_time:.2f}s")
        
        return VisualizationResult(
            points=reduced_embeddings,
            labels=labels,
            metadata=metadata_df,
            method=self.config.method.value,
            n_components=self.config.n_components,
            explained_variance_ratio=explained_variance
        )
    
    def _apply_pca(self, embeddings: EmbeddingArray) -> Tuple[EmbeddingArray, np.ndarray]:
        """Apply PCA dimensionality reduction."""
        pca = PCA(
            n_components=self.config.n_components,
            whiten=self.config.whiten,
            random_state=self.config.random_state
        )
        
        reduced = pca.fit_transform(embeddings)
        return reduced, pca.explained_variance_ratio_
    
    def _apply_tsne(self, embeddings: EmbeddingArray) -> EmbeddingArray:
        """Apply t-SNE dimensionality reduction."""
        # Pre-reduce with PCA if embeddings are high-dimensional
        if embeddings.shape[1] > 50:
            logger.info("Pre-reducing with PCA for t-SNE efficiency")
            pca = PCA(n_components=50, random_state=self.config.random_state)
            embeddings = pca.fit_transform(embeddings)
        
        # Adjust perplexity if necessary
        perplexity = min(self.config.perplexity, embeddings.shape[0] - 1)
        if perplexity != self.config.perplexity:
            logger.warning(f"Adjusted perplexity from {self.config.perplexity} to {perplexity} due to dataset size")
        
        tsne = TSNE(
            n_components=self.config.n_components,
            perplexity=perplexity,
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            init='pca',
            verbose=0
        )
        
        return tsne.fit_transform(embeddings)
    
    def _apply_umap(self, embeddings: EmbeddingArray) -> EmbeddingArray:
        """Apply UMAP dimensionality reduction."""
        # Adjust n_neighbors if necessary
        n_neighbors = min(self.config.n_neighbors, embeddings.shape[0] - 1)
        if n_neighbors != self.config.n_neighbors:
            logger.warning(f"Adjusted n_neighbors from {self.config.n_neighbors} to {n_neighbors} due to dataset size")
        
        reducer = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
            verbose=False
        )
        
        return reducer.fit_transform(embeddings)
    
    def create_plot(self, result: VisualizationResult, 
                   color_by: Optional[str] = None,
                   size_by: Optional[str] = None,
                   custom_colors: Optional[Dict[str, str]] = None) -> go.Figure:
        """Create interactive plot from visualization result."""
        df = result.to_dataframe()
        
        # Prepare plot parameters
        plot_params = {
            'hover_name': 'name',
            'title': f'{result.method.upper()} Visualization ({result.n_components}D)',
            'labels': {'x': 'Dimension 1', 'y': 'Dimension 2'}
        }
        
        if result.n_components == 3:
            plot_params['labels']['z'] = 'Dimension 3'
        
        # Add color mapping
        if color_by and color_by in df.columns:
            plot_params['color'] = color_by
            if custom_colors:
                plot_params['color_discrete_map'] = custom_colors
            
            # Use categorical color scale for string columns
            if df[color_by].dtype == 'object':
                plot_params['color_discrete_sequence'] = px.colors.qualitative.Set3
        
        # Add size mapping
        if size_by and size_by in df.columns and pd.api.types.is_numeric_dtype(df[size_by]):
            plot_params['size'] = size_by
            plot_params['size_max'] = 15
        
        # Create hover template
        hover_fields = ['name']
        if color_by and color_by not in hover_fields:
            hover_fields.append(color_by)
        if size_by and size_by not in hover_fields:
            hover_fields.append(size_by)
        
        # Add additional metadata fields to hover
        for col in df.columns:
            if col not in ['x', 'y', 'z', 'name'] and col not in hover_fields and len(hover_fields) < 8:
                hover_fields.append(col)
        
        plot_params['hover_data'] = hover_fields
        
        # Create plot
        if result.n_components == 3:
            fig = px.scatter_3d(df, x='x', y='y', z='z', **plot_params)
            
            # Update 3D plot layout
            fig.update_layout(
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3',
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=50)
            )
        else:
            fig = px.scatter(df, x='x', y='y', **plot_params)
            fig.update_layout(height=700)
        
        # Common layout updates
        fig.update_layout(
            title={
                'text': plot_params['title'],
                'x': 0.5,
                'xanchor': 'center'
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white' if self._is_dark_theme() else 'black')
        )
        
        # Add explained variance information for PCA
        if result.method == 'pca' and result.explained_variance_ratio is not None:
            variance_text = f"Explained variance: {result.explained_variance_ratio.sum():.1%}"
            if result.n_components >= 2:
                variance_text += f" (PC1: {result.explained_variance_ratio[0]:.1%}, PC2: {result.explained_variance_ratio[1]:.1%}"
                if result.n_components == 3:
                    variance_text += f", PC3: {result.explained_variance_ratio[2]:.1%}"
                variance_text += ")"
            
            fig.add_annotation(
                text=variance_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        
        return fig
    
    def create_comparison_plot(self, results: List[VisualizationResult], 
                              color_by: Optional[str] = None) -> go.Figure:
        """Create comparison plot with multiple reduction methods."""
        from plotly.subplots import make_subplots
        
        n_methods = len(results)
        if n_methods == 1:
            return self.create_plot(results[0], color_by)
        
        # Create subplots
        if n_methods == 2:
            subplot_layout = (1, 2)
        elif n_methods <= 4:
            subplot_layout = (2, 2)
        else:
            subplot_layout = (3, 3)
        
        fig = make_subplots(
            rows=subplot_layout[0],
            cols=subplot_layout[1],
            subplot_titles=[f"{r.method.upper()}" for r in results],
            specs=[[{"type": "scatter"}] * subplot_layout[1]] * subplot_layout[0]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, result in enumerate(results):
            df = result.to_dataframe()
            row = i // subplot_layout[1] + 1
            col = i % subplot_layout[1] + 1
            
            # Prepare color data
            if color_by and color_by in df.columns:
                # Create color mapping for categorical data
                if df[color_by].dtype == 'object':
                    unique_values = df[color_by].unique()
                    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
                    marker_colors = [color_map[val] for val in df[color_by]]
                else:
                    marker_colors = df[color_by]
            else:
                marker_colors = colors[0]
            
            fig.add_trace(
                go.Scatter(
                    x=df['x'],
                    y=df['y'],
                    mode='markers',
                    marker=dict(
                        color=marker_colors,
                        size=5,
                        opacity=0.7
                    ),
                    text=df['name'],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Dimensionality Reduction Comparison",
            height=600 if subplot_layout[0] == 1 else 800,
            showlegend=False
        )
        
        return fig
    
    def _is_dark_theme(self) -> bool:
        """Check if dark theme is enabled (placeholder for theme detection)."""
        # This could be enhanced to detect actual theme
        return True
    
    def get_embedding_statistics(self, embeddings: EmbeddingArray) -> Dict[str, Any]:
        """Calculate statistics about the embeddings."""
        return {
            'n_samples': embeddings.shape[0],
            'n_dimensions': embeddings.shape[1],
            'mean_norm': np.linalg.norm(embeddings, axis=1).mean(),
            'std_norm': np.linalg.norm(embeddings, axis=1).std(),
            'sparsity': (embeddings == 0).sum() / embeddings.size,
            'mean_value': embeddings.mean(),
            'std_value': embeddings.std(),
            'min_value': embeddings.min(),
            'max_value': embeddings.max()
        }