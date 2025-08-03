# OpenSearch Vector Visualizer - Quick Start Guide

Get up and running with the OpenSearch Vector Visualizer in just a few minutes!

## 🚀 Quick Demo Setup

### Option 1: Automated Demo Setup (Recommended)

Run our automated setup script to get everything working immediately:

```bash
# Make sure you have Docker and Poetry installed
./scripts/setup_demo.sh
```

This will:
- Start OpenSearch in Docker
- Install Python dependencies
- Generate sample embedding datasets
- Provide connection instructions

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Start OpenSearch** (if you don't have one running):
   ```bash
   docker-compose up -d opensearch
   ```

3. **Generate sample data**:
   ```bash
   poetry run python scripts/generate_sample_data.py --samples 1000
   ```

4. **Start the visualizer**:
   ```bash
   poetry run streamlit run opensearch_visualizer/main.py
   ```

## 🎯 Using Your Own Data

If you already have embeddings in OpenSearch:

1. **Start the visualizer**:
   ```bash
   poetry run streamlit run opensearch_visualizer/main.py
   ```

2. **Configure connection** in the sidebar:
   - Host: Your OpenSearch host
   - Port: Your OpenSearch port  
   - Authentication if needed

3. **Select your index** and configure fields:
   - Embedding field: Field containing your vectors
   - Name field: Field with document identifiers

4. **Apply filters** (optional):
   - Use the Filters tab to narrow down your data

5. **Visualize**:
   - Choose reduction method (t-SNE, UMAP, or PCA)
   - Click "Load & Visualize Data"

## 📊 Sample Datasets

The demo setup creates these datasets:

| Dataset | Documents | Dimensions | Type | Description |
|---------|-----------|------------|------|-------------|
| `demo_embeddings` | 2000 | 512 | float32 | Main demo dataset with realistic clusters |
| `demo_int8_embeddings` | 1000 | 768 | int8 | Demonstrates int8 embedding support |
| `demo_2d_embeddings` | 500 | 2 | float32 | Quick testing with 2D data |

## 🔧 Configuration Tips

### Best Performance Settings

- **For large datasets (>10k docs)**: Use UMAP, limit to 5000 docs
- **For exploration**: Start with t-SNE at perplexity 30
- **For overview**: Use PCA first, then dive deeper with t-SNE/UMAP

### Filtering Examples

```json
// Text search
{"content": "*machine learning*"}

// Category filter  
{"category.keyword": ["Technology", "Science"]}

// Score range
{"score": {"gte": 0.8}}

// Date range
{"timestamp": {"gte": "2023-01-01", "lte": "2023-12-31"}}

// Complex query
{
  "bool": {
    "must": [
      {"match": {"content": "AI"}},
      {"range": {"score": {"gte": 0.7}}}
    ]
  }
}
```

## 🎨 Visualization Tips

1. **Color by category** to see semantic clusters
2. **Size by confidence/score** to identify high-quality data
3. **Use 2D for quick overview**, 3D for detailed exploration
4. **Try different reduction methods** - each reveals different patterns
5. **Apply filters** to focus on specific data subsets

## 📈 Dimensionality Reduction Guide

### t-SNE
- **Best for**: Local clustering patterns
- **Perplexity**: 5-50 (30 is good default)
- **Use when**: You want to see tight clusters

### UMAP  
- **Best for**: Balanced local/global structure
- **N-neighbors**: 2-50 (15 is good default)
- **Use when**: You want preserved overall structure

### PCA
- **Best for**: Quick overview and variance analysis
- **Use when**: You want linear relationships or quick exploration

## 🛠️ Troubleshooting

### Connection Issues
- Check OpenSearch is running: `curl http://localhost:9200`
- Verify credentials if using authentication
- Check firewall/network settings

### Performance Issues
- Reduce max documents (try 1000-5000)
- Apply filters to reduce dataset size
- Use PCA for quick exploration
- Close other browser tabs

### Visualization Issues
- Try different reduction methods
- Adjust method-specific parameters
- Check for data quality issues
- Ensure embedding field contains vectors

## 🔗 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [configuration options](opensearch_visualizer/config.py)
- Check out the [example scripts](scripts/)
- Customize filters for your use case

## 💡 Tips for Real-World Usage

1. **Start small**: Test with a subset of your data first
2. **Understand your embeddings**: Check dimensionality and data types
3. **Use metadata**: Rich metadata makes filtering and coloring more useful
4. **Iterate**: Try different methods and parameters to find what works
5. **Save configurations**: Export settings for reproducible analysis

---

**Need help?** Check the [troubleshooting section](README.md#troubleshooting) or open an issue!