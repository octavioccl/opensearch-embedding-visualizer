#!/usr/bin/env python3
"""
Generate sample embedding data for testing the OpenSearch Vector Visualizer.

This script creates sample documents with embeddings and loads them into OpenSearch
for testing purposes.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from opensearchpy import OpenSearch
from faker import Faker
import random

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

fake = Faker()


def generate_embeddings(n_samples: int = 1000, n_dims: int = 1024, 
                       n_clusters: int = 5, dtype: str = 'float32') -> List[Dict[str, Any]]:
    """Generate sample embeddings with realistic clustering."""
    np.random.seed(42)
    
    # Create cluster centers
    cluster_centers = np.random.normal(0, 1, size=(n_clusters, n_dims))
    
    documents = []
    categories = ['Technology', 'Science', 'Business', 'Health', 'Education']
    
    for i in range(n_samples):
        # Assign to a cluster
        cluster_id = np.random.choice(n_clusters)
        
        # Generate embedding around cluster center
        noise = np.random.normal(0, 0.3, size=n_dims)
        embedding = cluster_centers[cluster_id] + noise
        
        # Convert to appropriate dtype
        if dtype == 'int8':
            # Scale to int8 range
            embedding = (embedding * 100).astype(np.int8)
        else:
            embedding = embedding.astype(np.float32)
        
        # Generate realistic metadata
        doc = {
            '_id': f'doc_{i:06d}',
            'name': f'Document {i:04d}',
            'title': fake.sentence(nb_words=6),
            'content': fake.text(max_nb_chars=500),
            'category': categories[cluster_id],
            'author': fake.name(),
            'timestamp': fake.date_time_between(start_date='-2y', end_date='now').isoformat(),
            'score': round(random.uniform(0.1, 1.0), 3),
            'word_count': random.randint(50, 2000),
            'language': random.choice(['en', 'es', 'fr', 'de', 'it']),
            'tags': fake.words(nb=random.randint(1, 5)),
            'embedding': embedding.tolist(),
            'cluster_id': cluster_id
        }
        
        documents.append(doc)
    
    return documents


def create_index_mapping(embedding_dim: int, dtype: str = 'float32') -> Dict[str, Any]:
    """Create OpenSearch index mapping for embeddings."""
    embedding_type = "knn_vector" if dtype == 'float32' else "binary"
    
    mapping = {
        "mappings": {
            "properties": {
                "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "category": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "timestamp": {"type": "date"},
                "score": {"type": "float"},
                "word_count": {"type": "integer"},
                "language": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "tags": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "cluster_id": {"type": "integer"},
                "embedding": {
                    "type": embedding_type,
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil" if dtype == 'float32' else "hamming",
                        "engine": "faiss" if dtype == 'float32' else "lucene"
                    }
                }
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        }
    }
    
    return mapping


def load_data_to_opensearch(documents: List[Dict[str, Any]], 
                           host: str = 'localhost', 
                           port: int = 9200,
                           index_name: str = 'sample_embeddings',
                           username: str = None,
                           password: str = None):
    """Load sample data into OpenSearch."""
    # Create client
    auth = (username, password) if username and password else None
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    # Test connection
    try:
        info = client.info()
        print(f"Connected to OpenSearch: {info['cluster_name']} (v{info['version']['number']})")
    except Exception as e:
        print(f"Failed to connect to OpenSearch: {e}")
        return False
    
    # Delete index if it exists
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)
    
    # Create index with mapping
    embedding_dim = len(documents[0]['embedding'])
    dtype = 'int8' if isinstance(documents[0]['embedding'][0], (int, np.integer)) else 'float32'
    
    mapping = create_index_mapping(embedding_dim, dtype)
    
    print(f"Creating index: {index_name}")
    client.indices.create(index=index_name, body=mapping)
    
    # Bulk insert documents
    print(f"Inserting {len(documents)} documents...")
    
    from opensearchpy.helpers import bulk
    
    def doc_generator():
        for doc in documents:
            yield {
                "_index": index_name,
                "_id": doc['_id'],
                "_source": {k: v for k, v in doc.items() if k != '_id'}
            }
    
    success, failed = bulk(client, doc_generator(), chunk_size=100)
    print(f"Successfully inserted: {success}, Failed: {len(failed) if failed else 0}")
    
    # Refresh index
    client.indices.refresh(index=index_name)
    
    # Verify insertion
    count = client.count(index=index_name)['count']
    print(f"Total documents in index: {count}")
    
    return True


def export_to_file(documents: List[Dict[str, Any]], filename: str):
    """Export documents to JSON file."""
    with open(filename, 'w') as f:
        json.dump(documents, f, indent=2, default=str)
    print(f"Exported {len(documents)} documents to {filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate sample embedding data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--dtype', choices=['float32', 'int8'], default='float32', help='Embedding data type')
    parser.add_argument('--host', default='localhost', help='OpenSearch host')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port')
    parser.add_argument('--index', default='sample_embeddings', help='Index name')
    parser.add_argument('--username', help='OpenSearch username')
    parser.add_argument('--password', help='OpenSearch password')
    parser.add_argument('--export-file', help='Export to JSON file instead of OpenSearch')
    parser.add_argument('--no-opensearch', action='store_true', help='Skip OpenSearch upload')
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples} samples with {args.dims} dimensions...")
    documents = generate_embeddings(
        n_samples=args.samples,
        n_dims=args.dims, 
        n_clusters=args.clusters,
        dtype=args.dtype
    )
    
    if args.export_file:
        export_to_file(documents, args.export_file)
    
    if not args.no_opensearch:
        success = load_data_to_opensearch(
            documents,
            host=args.host,
            port=args.port,
            index_name=args.index,
            username=args.username,
            password=args.password
        )
        
        if success:
            print(f"\n✅ Sample data successfully loaded!")
            print(f"🔍 You can now use the visualizer with:")
            print(f"   - Index: {args.index}")
            print(f"   - Embedding field: embedding")
            print(f"   - Name field: name")
            print(f"   - Categories available: Technology, Science, Business, Health, Education")
        else:
            print("❌ Failed to load data to OpenSearch")


if __name__ == '__main__':
    main()