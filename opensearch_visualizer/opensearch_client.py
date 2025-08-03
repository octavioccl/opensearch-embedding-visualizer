"""Enhanced OpenSearch client with filtering and error handling."""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from opensearchpy import OpenSearch
from loguru import logger

from .config import OpenSearchConfig, IndexConfig, FilterConfig
from .models import (
    DocumentData, ConnectionStatus, IndexInfo, SearchQuery, 
    FilterResult, EmbeddingArray, MetadataDict
)


class OpenSearchClient:
    """Enhanced OpenSearch client with filtering and visualization support."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize OpenSearch client."""
        self.config = config
        self.client = None
        self._connection_status = None
        
    def connect(self) -> ConnectionStatus:
        """Connect to OpenSearch cluster."""
        try:
            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)
                
            self.client = OpenSearch(
                hosts=[{'host': self.config.host, 'port': self.config.port}],
                http_auth=auth,
                use_ssl=self.config.use_ssl,
                verify_certs=self.config.verify_certs,
                ca_certs=self.config.ca_certs,
                ssl_show_warn=False,
                timeout=self.config.timeout
            )
            
            # Test connection
            info = self.client.info()
            cluster_name = info.get('cluster_name')
            version = info.get('version', {}).get('number')
            
            self._connection_status = ConnectionStatus(
                connected=True,
                cluster_name=cluster_name,
                version=version
            )
            
            logger.info(f"Connected to OpenSearch cluster: {cluster_name} (v{version})")
            return self._connection_status
            
        except Exception as e:
            error_msg = f"Failed to connect to OpenSearch: {str(e)}"
            logger.error(error_msg)
            self._connection_status = ConnectionStatus(
                connected=False,
                error_message=error_msg
            )
            return self._connection_status
    
    def get_connection_status(self) -> Optional[ConnectionStatus]:
        """Get current connection status."""
        return self._connection_status
    
    def list_indices(self) -> List[str]:
        """List all available indices."""
        if not self.client:
            raise RuntimeError("Not connected to OpenSearch")
            
        try:
            indices = self.client.indices.get_alias("*")
            return [idx for idx in indices.keys() if not idx.startswith('.')]
        except Exception as e:
            logger.error(f"Failed to list indices: {str(e)}")
            raise
    
    def get_index_info(self, index_name: str, embedding_field: str = "embedding", 
                      name_field: str = "name") -> IndexInfo:
        """Get information about an index."""
        if not self.client:
            raise RuntimeError("Not connected to OpenSearch")
            
        try:
            # Get index stats
            stats = self.client.indices.stats(index=index_name)
            index_stats = stats['indices'][index_name]
            
            doc_count = index_stats['total']['docs']['count']
            size = self._format_bytes(index_stats['total']['store']['size_in_bytes'])
            
            # Get index health
            health = self.client.cluster.health(index=index_name)['status']
            
            # Check field existence and get embedding dimensions
            mapping = self.client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings'].get('properties', {})
            
            has_embedding_field = embedding_field in properties
            has_name_field = name_field in properties
            embedding_dimensions = None
            
            if has_embedding_field:
                # Try to get embedding dimensions from a sample document
                try:
                    sample = self.client.search(
                        index=index_name,
                        body={
                            "query": {"exists": {"field": embedding_field}},
                            "size": 1,
                            "_source": [embedding_field]
                        }
                    )
                    
                    if sample['hits']['hits']:
                        embedding = sample['hits']['hits'][0]['_source'][embedding_field]
                        if isinstance(embedding, list):
                            embedding_dimensions = len(embedding)
                        elif isinstance(embedding, bytes):
                            # Handle binary embeddings (e.g., int8)
                            embedding_dimensions = len(embedding)
                            
                except Exception as e:
                    logger.warning(f"Could not determine embedding dimensions: {str(e)}")
            
            return IndexInfo(
                name=index_name,
                doc_count=doc_count,
                size=size,
                health=health,
                has_embedding_field=has_embedding_field,
                has_name_field=has_name_field,
                embedding_dimensions=embedding_dimensions
            )
            
        except Exception as e:
            logger.error(f"Failed to get index info for {index_name}: {str(e)}")
            raise
    
    def fetch_embeddings(self, index_config: IndexConfig, 
                        filter_config: Optional[FilterConfig] = None) -> Tuple[List[DocumentData], FilterResult]:
        """Fetch embeddings with optional filtering."""
        if not self.client:
            raise RuntimeError("Not connected to OpenSearch")
        
        if not index_config.name:
            raise ValueError("Index name is required for fetching embeddings")
            
        start_time = time.time()
        
        # Build search query
        query = SearchQuery(
            query=filter_config.query if filter_config and filter_config.query else {"match_all": {}},
            size=index_config.max_docs,
            source_includes=[index_config.embedding_field, index_config.name_field]
        )
        
        # Add additional fields if specified
        if filter_config and filter_config.fields_to_include:
            query.source_includes.extend(filter_config.fields_to_include)
            query.source_includes = list(set(query.source_includes))  # Remove duplicates
        
        # Add sorting if specified
        if filter_config and filter_config.sort_by:
            query.sort = [{filter_config.sort_by: {"order": filter_config.sort_order}}]
        
        try:
            # Get total count first
            count_response = self.client.count(
                index=index_config.name,
                body={"query": query.query}
            )
            total_docs = count_response['count']
            
            # Execute search
            response = self.client.search(
                index=index_config.name,
                body=query.to_opensearch_body()
            )
            
            documents = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # Extract embedding
                if index_config.embedding_field not in source:
                    continue
                    
                embedding = source[index_config.embedding_field]
                if isinstance(embedding, bytes):
                    # Handle binary embeddings (e.g., int8)
                    embedding = np.frombuffer(embedding, dtype=np.int8).astype(np.float32).tolist()
                elif not isinstance(embedding, list):
                    continue
                
                # Extract name
                name = source.get(index_config.name_field, hit['_id'])
                
                # Extract metadata
                metadata = {k: v for k, v in source.items() 
                           if k not in [index_config.embedding_field, index_config.name_field]}
                
                documents.append(DocumentData(
                    id=hit['_id'],
                    name=str(name),
                    embedding=embedding,
                    metadata=metadata
                ))
            
            execution_time = time.time() - start_time
            
            # Create filter summary
            filter_summary = self._create_filter_summary(filter_config, total_docs, len(documents))
            
            filter_result = FilterResult(
                total_docs=total_docs,
                filtered_docs=len(documents),
                filter_summary=filter_summary,
                execution_time=execution_time
            )
            
            logger.info(f"Fetched {len(documents)} documents from {index_config.name} in {execution_time:.2f}s")
            return documents, filter_result
            
        except Exception as e:
            logger.error(f"Failed to fetch embeddings: {str(e)}")
            raise
    
    def build_filter_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build OpenSearch query from user-friendly filters."""
        if not filters:
            return {"match_all": {}}
        
        must_clauses = []
        
        for field, value in filters.items():
            if isinstance(value, str):
                # Text search
                if '*' in value or '?' in value:
                    # Wildcard query
                    must_clauses.append({"wildcard": {field: value}})
                else:
                    # Match query
                    must_clauses.append({"match": {field: value}})
            elif isinstance(value, (int, float)):
                # Exact term match for numbers
                must_clauses.append({"term": {field: value}})
            elif isinstance(value, dict):
                # Range query
                if 'gte' in value or 'lte' in value or 'gt' in value or 'lt' in value:
                    must_clauses.append({"range": {field: value}})
            elif isinstance(value, list):
                # Terms query (OR)
                must_clauses.append({"terms": {field: value}})
        
        if len(must_clauses) == 1:
            return must_clauses[0]
        elif len(must_clauses) > 1:
            return {"bool": {"must": must_clauses}}
        else:
            return {"match_all": {}}
    
    def get_field_values(self, index_name: str, field_name: str, max_values: int = 100) -> List[str]:
        """Get unique values for a field (for filter dropdowns)."""
        if not self.client:
            raise RuntimeError("Not connected to OpenSearch")
            
        try:
            response = self.client.search(
                index=index_name,
                body={
                    "aggs": {
                        "unique_values": {
                            "terms": {
                                "field": f"{field_name}.keyword",
                                "size": max_values
                            }
                        }
                    },
                    "size": 0
                }
            )
            
            buckets = response.get('aggregations', {}).get('unique_values', {}).get('buckets', [])
            return [bucket['key'] for bucket in buckets]
            
        except Exception as e:
            logger.warning(f"Failed to get field values for {field_name}: {str(e)}")
            return []
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}TB"
    
    def _create_filter_summary(self, filter_config: Optional[FilterConfig], 
                              total_docs: int, filtered_docs: int) -> str:
        """Create human-readable filter summary."""
        if not filter_config or not filter_config.query or filter_config.query == {"match_all": {}}:
            return f"No filters applied. Showing {filtered_docs} of {total_docs} documents."
        
        filter_parts = []
        
        # Add query information
        if filter_config.query:
            filter_parts.append("Custom query applied")
        
        # Add sorting information
        if filter_config.sort_by:
            filter_parts.append(f"Sorted by {filter_config.sort_by} ({filter_config.sort_order})")
        
        filter_desc = ", ".join(filter_parts) if filter_parts else "Filters applied"
        return f"{filter_desc}. Showing {filtered_docs} of {total_docs} documents."