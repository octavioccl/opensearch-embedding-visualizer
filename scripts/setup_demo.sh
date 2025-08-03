#!/bin/bash

# OpenSearch Vector Visualizer Demo Setup Script
# This script sets up a complete demo environment with OpenSearch and sample data

set -e

echo "🚀 Setting up OpenSearch Vector Visualizer Demo Environment"
echo "============================================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is required but not installed. Please install Poetry first."
    echo "   Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📁 Working directory: $PROJECT_ROOT"

# Install dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Start OpenSearch with Docker Compose
echo "🐳 Starting OpenSearch with Docker Compose..."
docker-compose up -d opensearch

# Wait for OpenSearch to be ready
echo "⏳ Waiting for OpenSearch to start..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo "✅ OpenSearch is ready!"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - waiting 5 seconds..."
    sleep 5
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ OpenSearch failed to start within expected time"
    echo "   Check logs with: docker-compose logs opensearch"
    exit 1
fi

# Generate and load sample data
echo "📊 Generating sample embedding data..."
poetry run python scripts/generate_sample_data.py \
    --samples 2000 \
    --dims 512 \
    --clusters 8 \
    --dtype float32 \
    --index demo_embeddings \
    --host localhost \
    --port 9200

# Create additional demo indices with different configurations
echo "📊 Creating additional demo datasets..."

# Int8 embeddings dataset
poetry run python scripts/generate_sample_data.py \
    --samples 1000 \
    --dims 768 \
    --clusters 5 \
    --dtype int8 \
    --index demo_int8_embeddings \
    --host localhost \
    --port 9200

# Small 2D dataset for quick testing
poetry run python scripts/generate_sample_data.py \
    --samples 500 \
    --dims 2 \
    --clusters 3 \
    --dtype float32 \
    --index demo_2d_embeddings \
    --host localhost \
    --port 9200

echo ""
echo "✅ Demo environment setup complete!"
echo ""
echo "📋 Available Demo Datasets:"
echo "  1. demo_embeddings (2000 docs, 512D, float32)"
echo "  2. demo_int8_embeddings (1000 docs, 768D, int8)"
echo "  3. demo_2d_embeddings (500 docs, 2D, float32)"
echo ""
echo "🔍 OpenSearch Dashboard: http://localhost:5601"
echo "🔗 OpenSearch API: http://localhost:9200"
echo ""
echo "🚀 Start the visualizer with:"
echo "   poetry run streamlit run opensearch_visualizer/main.py"
echo "   or"
echo "   poetry run python scripts/run_visualizer.py"
echo ""
echo "⚙️  Connection settings for the visualizer:"
echo "   - Host: localhost"
echo "   - Port: 9200"
echo "   - Index: demo_embeddings (or others)"
echo "   - Embedding field: embedding"
echo "   - Name field: name"
echo ""
echo "🛑 To stop the demo environment:"
echo "   docker-compose down"