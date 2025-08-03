# OpenSearch Vector Visualizer

An advanced embedding visualization tool for OpenSearch with 3D visualization, filtering capabilities, and multiple dimensionality reduction techniques.

![OpenSearch Vector Visualizer](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🌟 Features

- **Interactive 3D Visualization**: Explore embeddings with zoom, rotate, and pan controls
- **Multiple Reduction Methods**: t-SNE, UMAP, and PCA for different analytical perspectives
- **Advanced Filtering**: Complex queries with text, range, categorical, and date filters
- **Real-time Exploration**: Interactive filtering and visualization updates
- **Flexible Configuration**: Support for various embedding formats (int8, float32)
- **Export Capabilities**: Save configurations and results for reproducibility
- **Performance Optimized**: Efficient handling of large embedding datasets
- **User-Friendly Interface**: Intuitive Streamlit-based web interface

## 📋 Requirements

### OpenSearch Index Requirements

Your OpenSearch index should contain:
- **Embedding field**: High-dimensional vectors (int8 or float32)
- **Name/ID field**: Document identifiers or names
- **Optional**: Additional metadata fields for filtering and coloring

### System Requirements

- Python 3.9+
- OpenSearch cluster (local or remote)
- 4GB+ RAM recommended for large datasets

## 🚀 Installation

### Using Poetry (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/opensearch-vector-visualizer.git
   cd opensearch-vector-visualizer
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

### Using pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/opensearch-vector-visualizer.git
   cd opensearch-vector-visualizer
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## 🎯 Quick Start

### 1. Start the Application

```bash
# Using Poetry
poetry run streamlit run opensearch_visualizer/main.py

# Or using the script command
poetry run opensearch-visualizer

# Using pip installation
streamlit run opensearch_visualizer/main.py
```

### 2. Configure Connection

1. Open your browser to `http://localhost:8501`
2. In the sidebar, configure your OpenSearch connection:
   - **Host**: Your OpenSearch host (e.g., `localhost`)
   - **Port**: Your OpenSearch port (e.g., `9200`)
   - **Authentication**: Username/password if required
   - **SSL Settings**: Configure if using HTTPS

### 3. Select Index and Fields

1. Choose your index from the dropdown
2. Configure field mappings:
   - **Embedding Field**: Field containing vector embeddings
   - **Name Field**: Field containing document names/IDs
   - **Max Documents**: Number of documents to visualize

### 4. Apply Filters (Optional)

1. Navigate to the "Filters" tab
2. Use preset filters or create custom ones:
   - **Text Search**: Search within specific fields
   - **Range Filters**: Numeric range filtering
   - **Categorical Filters**: Select specific categories
   - **Date Filters**: Time-based filtering
   - **Custom Queries**: Raw OpenSearch queries

### 5. Visualize Your Data

1. Choose dimensionality reduction method:
   - **t-SNE**: Best for local structure preservation
   - **UMAP**: Balanced local and global structure
   - **PCA**: Linear reduction, fastest
2. Set visualization parameters
3. Click "Load & Visualize Data"

## 📊 Dimensionality Reduction Methods

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Best for**: Discovering local clusters and patterns
- **Parameters**:
  - `Perplexity`: Balance between local and global structure (5-50)
  - `Learning Rate`: Optimization step size (10-1000)
  - `Iterations`: Number of optimization steps (250-2000)

### UMAP (Uniform Manifold Approximation and Projection)
- **Best for**: Preserving both local and global structure
- **Parameters**:
  - `N Neighbors`: Size of local neighborhood (2-50)
  - `Min Distance`: Minimum distance between points (0.0-1.0)
  - `Metric`: Distance metric (cosine, euclidean, manhattan)

### PCA (Principal Component Analysis)
- **Best for**: Understanding data variance and quick exploration
- **Parameters**:
  - `Whiten`: Remove correlations between components

## 🔍 Advanced Filtering

### Filter Types

1. **Text Search Filters**:
   ```json
   {"content": "machine learning"}
   {"title": "*neural*"}  // Wildcard search
   ```

2. **Range Filters**:
   ```json
   {"score": {"gte": 0.7, "lte": 1.0}}
   {"timestamp": {"gte": "2023-01-01"}}
   ```

3. **Categorical Filters**:
   ```json
   {"category.keyword": ["technology", "science"]}
   {"language.keyword": "en"}
   ```

4. **Custom OpenSearch Queries**:
   ```json
   {
     "bool": {
       "must": [
         {"match": {"content": "embedding"}},
         {"range": {"score": {"gte": 0.8}}}
       ]
     }
   }
   ```

### Filter Presets

The application includes several preset filters:
- **Recent Documents**: Documents from the last 30 days
- **High Confidence**: Documents with confidence score > 0.8
- **English Documents**: Documents in English language
- **Long Documents**: Documents with more than 1000 characters

## 🛠️ Configuration

### Environment Variables

Create a `.env` file for default configurations:

```env
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin
DEFAULT_INDEX=embeddings
DEFAULT_EMBEDDING_FIELD=embedding
DEFAULT_NAME_FIELD=name
```

### Configuration Files

Export and import JSON configuration files for reproducible setups:

```python
# Export current configuration
config = app.config.json(indent=2)

# Import configuration
from opensearch_visualizer.config import AppConfig
config = AppConfig.parse_file("config.json")
```

## 📈 Performance Tips

### For Large Datasets

1. **Use Filtering**: Apply filters to reduce data size before visualization
2. **Limit Documents**: Set appropriate `max_docs` limit (1000-5000 for smooth interaction)
3. **Pre-reduce with PCA**: For t-SNE, automatic PCA pre-reduction is applied for >50D embeddings
4. **Choose UMAP**: Generally faster than t-SNE for large datasets

### Memory Optimization

- Monitor memory usage with large embedding datasets
- Consider processing data in batches for very large indices
- Use appropriate data types (int8 vs float32) for embeddings

## 🔧 Development

### Running Tests

```bash
poetry run pytest tests/
```

### Code Quality

```bash
# Format code
poetry run black opensearch_visualizer/

# Sort imports
poetry run isort opensearch_visualizer/

# Type checking
poetry run mypy opensearch_visualizer/

# Linting
poetry run flake8 opensearch_visualizer/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests and quality checks
5. Submit a pull request

## 📚 Examples

### Basic Usage Example

```python
from opensearch_visualizer.config import AppConfig, OpenSearchConfig
from opensearch_visualizer.opensearch_client import OpenSearchClient
from opensearch_visualizer.visualizer import EmbeddingVisualizer

# Configure connection
config = AppConfig()
config.opensearch.host = "localhost"
config.opensearch.port = 9200

# Connect and visualize
client = OpenSearchClient(config.opensearch)
client.connect()

documents, _ = client.fetch_embeddings(config.index)
visualizer = EmbeddingVisualizer(config.visualization)
result = visualizer.reduce_dimensions(documents)
plot = visualizer.create_plot(result)
```

### Advanced Filtering Example

```python
from opensearch_visualizer.filters import FilterBuilder

# Build complex filter
filter_builder = FilterBuilder(client, "my_index")
filters = {
    "content": "*machine learning*",
    "score": {"gte": 0.8},
    "category.keyword": ["AI", "ML"]
}

query = filter_builder.build_opensearch_query(filters)
```

## 🤝 Inspiration and References

This project draws inspiration from several excellent embedding visualization tools:

- [**gpt-intuition**](https://github.com/epec254/gpt-intuition): Streamlit-based embedding visualization
- [**Vectory**](https://github.com/pentoai/vectory): Embedding evaluation toolkit
- [**FiftyOne**](https://docs.voxel51.com/): Computer vision dataset management and visualization
- [**TensorFlow Projector**](https://projector.tensorflow.org/): Google's embedding projector
- [**Nomic Atlas**](https://docs.nomic.ai/atlas/): Large-scale embedding visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenSearch team for the excellent search engine
- Plotly team for interactive visualization capabilities
- Streamlit team for the amazing web app framework
- scikit-learn, UMAP, and other open-source libraries

## 📞 Support

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/opensearch-vector-visualizer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/opensearch-vector-visualizer/discussions)

---

**Built with ❤️ for the OpenSearch and ML community**