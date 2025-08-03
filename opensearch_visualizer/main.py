"""
Main Streamlit application for OpenSearch Vector Visualizer.

An advanced embedding visualization tool with filtering capabilities,
multiple dimensionality reduction techniques, and interactive exploration.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
import time
import json

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opensearch_visualizer.config import (
    AppConfig, OpenSearchConfig, IndexConfig, 
    VisualizationConfig, FilterConfig, ReductionMethod
)
from opensearch_visualizer.opensearch_client import OpenSearchClient
from opensearch_visualizer.visualizer import EmbeddingVisualizer
from opensearch_visualizer.filters import FilterBuilder, FilterPresets, FilterSummary
from opensearch_visualizer.models import DocumentData, VisualizationResult


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

# Configure Streamlit page
st.set_page_config(
    page_title="OpenSearch Vector Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


class VectorVisualizerApp:
    """Main application class for the vector visualizer."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = AppConfig()
        
        # Initialize session state first
        self._init_session_state()
        
        # Get client from session state if it exists
        self.client: Optional[OpenSearchClient] = st.session_state.get('opensearch_client', None)
        self.visualizer: Optional[EmbeddingVisualizer] = None
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'connected' not in st.session_state:
            st.session_state.connected = False
        if 'opensearch_client' not in st.session_state:
            st.session_state.opensearch_client = None
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'visualization_results' not in st.session_state:
            st.session_state.visualization_results = {}
        if 'filter_config' not in st.session_state:
            st.session_state.filter_config = FilterConfig()
    
    def run(self):
        """Run the main application."""
        # Header
        st.title("🔍 OpenSearch Vector Visualizer")
        st.markdown("""
        **Advanced embedding visualization tool** with filtering capabilities, 
        multiple dimensionality reduction techniques, and interactive exploration.
        """)
        
        # Sidebar configuration
        with st.sidebar:
            self._render_sidebar()
        
        # Main content
        if st.session_state.connected and self.client:
            self._render_main_content()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self):
        """Render sidebar configuration."""
        st.header("⚙️ Configuration")
        
        # Connection settings
        self._render_connection_settings()
        
        if st.session_state.connected:
            # Index settings
            self._render_index_settings()
            
            # Visualization settings
            self._render_visualization_settings()
    
    def _render_connection_settings(self):
        """Render OpenSearch connection settings."""
        st.subheader("🔌 OpenSearch Connection")
        
        # Connection status indicator
        if self.client and st.session_state.connected:
            try:
                connection_status = self.client.get_connection_status()
                if connection_status and connection_status.connected:
                    st.success(f"✅ Connected to {connection_status.cluster_name}")
                else:
                    st.error("❌ Connection lost")
            except:
                st.error("❌ Connection error")
        else:
            st.info("⏳ Not connected")
        
        with st.form("connection_form"):
            host = st.text_input("Host", value=self.config.opensearch.host)
            port = st.number_input("Port", value=self.config.opensearch.port, min_value=1, max_value=65535)
            
            # Authentication
            with st.expander("🔐 Authentication"):
                username = st.text_input("Username", value=self.config.opensearch.username or "")
                password = st.text_input("Password", type="password", value=self.config.opensearch.password or "")
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                use_ssl = st.checkbox("Use SSL", value=self.config.opensearch.use_ssl)
                verify_certs = st.checkbox("Verify Certificates", value=self.config.opensearch.verify_certs)
                timeout = st.number_input("Timeout (seconds)", value=self.config.opensearch.timeout, min_value=1)
            
            connect_button = st.form_submit_button("🚀 Connect", type="primary")
            
            if connect_button:
                self._connect_to_opensearch(host, port, username, password, use_ssl, verify_certs, timeout)
    
    def _render_index_settings(self):
        """Render index configuration settings."""
        st.subheader("📊 Index Configuration")
        
        # Check if client is connected
        if not self.client:
            st.warning("⚠️ Please configure and connect to OpenSearch first.")
            return
            
        # Double-check connection status
        try:
            connection_status = self.client.get_connection_status()
            if not connection_status or not connection_status.connected:
                st.warning("⚠️ OpenSearch connection lost. Please reconnect.")
                # Clear session state if connection is lost
                st.session_state.opensearch_client = None
                st.session_state.connected = False
                return
                
            # Test the connection with a simple call
            self.client.client.cluster.health()
            
        except Exception as e:
            st.warning(f"⚠️ Connection test failed: {str(e)}. Please reconnect.")
            # Clear session state if connection test fails
            st.session_state.opensearch_client = None
            st.session_state.connected = False
            self.client = None
            return
        
        # List available indices
        try:
            indices = self.client.list_indices()
            
            selected_index = st.selectbox(
                "Select Index",
                options=indices,
                key="selected_index"
            )
            
            if selected_index:
                # Get index info
                try:
                    index_info = self.client.get_index_info(selected_index)
                    
                    # Display index information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documents", f"{index_info.doc_count:,}")
                        st.metric("Health", index_info.health)
                    with col2:
                        st.metric("Size", index_info.size)
                        if index_info.embedding_dimensions:
                            st.metric("Embedding Dims", index_info.embedding_dimensions)
                    
                    # Field configuration - Make this prominent
                    st.subheader("🏷️ Field Configuration")
                    
                    # Get available fields from index mapping
                    try:
                        mapping = self.client.client.indices.get_mapping(index=selected_index)
                        properties = mapping[selected_index]['mappings'].get('properties', {})
                        available_fields = list(properties.keys())
                        
                        # Suggest likely embedding and name fields
                        embedding_candidates = [f for f in available_fields if any(keyword in f.lower() 
                                              for keyword in ['embedding', 'vector', 'emb', 'vec'])]
                        name_candidates = [f for f in available_fields if any(keyword in f.lower() 
                                         for keyword in ['name', 'title', 'id', 'doc_id', 'document'])]
                        
                        st.info(f"📋 Available fields: {', '.join(available_fields[:10])}" + 
                               (f" and {len(available_fields)-10} more..." if len(available_fields) > 10 else ""))
                        
                    except Exception as e:
                        available_fields = []
                        embedding_candidates = []
                        name_candidates = []
                        st.warning(f"Could not retrieve field information: {str(e)}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Embedding field selection
                        if embedding_candidates:
                            default_embedding = embedding_candidates[0]
                            st.success(f"💡 Suggested embedding field: {default_embedding}")
                        else:
                            default_embedding = "embedding"
                            
                        if available_fields:
                            embedding_field = st.selectbox(
                                "Embedding Field", 
                                options=available_fields,
                                index=available_fields.index(default_embedding) if default_embedding in available_fields else 0,
                                help="Field containing vector embeddings"
                            )
                        else:
                            embedding_field = st.text_input(
                                "Embedding Field", 
                                value=default_embedding,
                                help="Field containing vector embeddings"
                            )
                    
                    with col2:
                        # Name field selection
                        if name_candidates:
                            default_name = name_candidates[0]
                            st.success(f"💡 Suggested name field: {default_name}")
                        else:
                            default_name = "name"
                            
                        if available_fields:
                            name_field = st.selectbox(
                                "Name/ID Field", 
                                options=available_fields,
                                index=available_fields.index(default_name) if default_name in available_fields else 0,
                                help="Field containing document names or IDs"
                            )
                        else:
                            name_field = st.text_input(
                                "Name/ID Field", 
                                value=default_name,
                                help="Field containing document names or IDs"
                            )
                    
                    # Max documents setting
                    max_docs = st.number_input(
                        "Max Documents to Visualize", 
                        value=1000, 
                        min_value=10, 
                        max_value=10000,
                        step=100,
                        help="Maximum documents to fetch for visualization (recommended: 1000-5000 for best performance)"
                    )
                    
                    # Update config
                    self.config.index.name = selected_index
                    self.config.index.embedding_field = embedding_field
                    self.config.index.name_field = name_field
                    self.config.index.max_docs = max_docs
                    
                    # Test configuration button
                    st.subheader("🧪 Test Configuration")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        test_button = st.button("🔍 Test Fields", type="secondary", use_container_width=True)
                    
                    with col2:
                        if test_button:
                            self._test_field_configuration(selected_index, embedding_field, name_field)
                    
                    # Configuration summary
                    st.success(f"✅ **Ready to visualize!**  \n"
                              f"📊 Index: `{selected_index}`  \n"
                              f"🔢 Embedding Field: `{embedding_field}`  \n"
                              f"🏷️ Name Field: `{name_field}`  \n"
                              f"📈 Max Documents: `{max_docs:,}`")
                        
                except Exception as e:
                    st.error(f"Failed to get index info: {str(e)}")
                    
        except Exception as e:
            st.error(f"Failed to list indices: {str(e)}")
    
    def _render_visualization_settings(self):
        """Render visualization configuration settings."""
        st.subheader("📈 Visualization Settings")
        
        # Reduction method
        method = st.selectbox(
            "Dimensionality Reduction",
            options=[m.value for m in ReductionMethod],
            index=1,  # Default to t-SNE
            format_func=lambda x: x.upper(),
            help="Choose the dimensionality reduction technique"
        )
        
        # Dimensions
        n_components = st.selectbox(
            "Dimensions", 
            options=[2, 3], 
            index=1,
            help="2D or 3D visualization"
        )
        
        # Method-specific parameters
        if method == ReductionMethod.TSNE.value:
            with st.expander("t-SNE Parameters"):
                perplexity = st.slider("Perplexity", 5, 50, 30, help="Balance local vs global structure")
                learning_rate = st.slider("Learning Rate", 10, 1000, 200, help="Step size for optimization")
                max_iter = st.slider("Max Iterations", 250, 2000, 1000, help="Maximum number of optimization steps")
        
        elif method == ReductionMethod.UMAP.value:
            with st.expander("UMAP Parameters"):
                n_neighbors = st.slider("N Neighbors", 2, 50, 15, help="Size of local neighborhood")
                min_dist = st.slider("Min Distance", 0.0, 1.0, 0.1, help="Minimum distance between points")
                metric = st.selectbox("Metric", ["cosine", "euclidean", "manhattan"], help="Distance metric")
        
        elif method == ReductionMethod.PCA.value:
            with st.expander("PCA Parameters"):
                whiten = st.checkbox("Whiten", help="Remove correlations between components")
        
        # Update visualization config
        self.config.visualization.method = ReductionMethod(method)
        self.config.visualization.n_components = n_components
        
        if method == ReductionMethod.TSNE.value:
            self.config.visualization.perplexity = perplexity
            self.config.visualization.learning_rate = learning_rate
            self.config.visualization.max_iter = max_iter
        elif method == ReductionMethod.UMAP.value:
            self.config.visualization.n_neighbors = n_neighbors
            self.config.visualization.min_dist = min_dist
            self.config.visualization.metric = metric
        elif method == ReductionMethod.PCA.value:
            self.config.visualization.whiten = whiten
        
        # Initialize visualizer
        self.visualizer = EmbeddingVisualizer(self.config.visualization)
    
    def _render_main_content(self):
        """Render main content area."""
        # Create tabs for different functionality
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Visualization", "🔍 Filters", "📊 Statistics", "⚙️ Settings"])
        
        with tab1:
            self._render_visualization_tab()
        
        with tab2:
            self._render_filters_tab()
        
        with tab3:
            self._render_statistics_tab()
        
        with tab4:
            self._render_settings_tab()
    
    def _render_visualization_tab(self):
        """Render main visualization tab."""
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("🚀 Actions")
            
            if st.button("📥 Load & Visualize Data", type="primary", use_container_width=True):
                self._load_and_visualize_data()
            
            # Visualization options
            if st.session_state.current_data:
                st.subheader("🎨 Display Options")
                
                # Get available columns for coloring
                available_columns = list(st.session_state.current_data[0].metadata.keys()) + ['None']
                
                color_by = st.selectbox(
                    "Color by",
                    options=available_columns,
                    key="color_by"
                )
                
                size_by = st.selectbox(
                    "Size by", 
                    options=available_columns,
                    key="size_by"
                )
                
                # Update visualization if options changed
                if color_by != 'None' or size_by != 'None':
                    self._update_visualization(
                        color_by if color_by != 'None' else None,
                        size_by if size_by != 'None' else None
                    )
        
        with col1:
            # Display visualization
            if 'current_plot' in st.session_state and st.session_state.current_plot:
                st.plotly_chart(st.session_state.current_plot, use_container_width=True)
            else:
                self._render_placeholder()
    
    def _test_field_configuration(self, index_name: str, embedding_field: str, name_field: str):
        """Test the selected field configuration."""
        try:
            with st.spinner("Testing field configuration..."):
                # Test query to check fields exist and get sample data
                test_query = {
                    "query": {"exists": {"field": embedding_field}},
                    "size": 1,
                    "_source": [embedding_field, name_field]
                }
                
                response = self.client.client.search(
                    index=index_name,
                    body=test_query
                )
                
                if response['hits']['total']['value'] == 0:
                    st.error(f"❌ No documents found with field '{embedding_field}'")
                    return
                
                # Check the sample document
                sample_doc = response['hits']['hits'][0]['_source']
                
                # Test embedding field
                if embedding_field not in sample_doc:
                    st.error(f"❌ Embedding field '{embedding_field}' not found in document")
                    return
                
                embedding_data = sample_doc[embedding_field]
                if not isinstance(embedding_data, (list, bytes)):
                    st.error(f"❌ Embedding field '{embedding_field}' does not contain vector data")
                    return
                
                if isinstance(embedding_data, list):
                    embedding_type = "float32"
                    embedding_dim = len(embedding_data)
                elif isinstance(embedding_data, bytes):
                    embedding_type = "int8"
                    embedding_dim = len(embedding_data)
                else:
                    st.error(f"❌ Unsupported embedding format in field '{embedding_field}'")
                    return
                
                # Test name field
                name_data = sample_doc.get(name_field, "Not found")
                
                # Show test results
                st.success("✅ **Field Configuration Test Results:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Embedding Field", embedding_field)
                    st.metric("Vector Type", embedding_type)
                    st.metric("Vector Dimensions", f"{embedding_dim:,}")
                
                with col2:
                    st.metric("Name Field", name_field)
                    st.metric("Sample Name", str(name_data)[:50] + ("..." if len(str(name_data)) > 50 else ""))
                    st.metric("Total Documents", f"{response['hits']['total']['value']:,}")
                
                st.info("💡 Configuration looks good! You can now proceed to visualization.")
                
        except Exception as e:
            st.error(f"❌ Field test failed: {str(e)}")
            st.info("💡 Try adjusting your field names or check the index mapping.")
    
    def _render_filters_tab(self):
        """Render filters configuration tab."""
        if not self.client or not self.config.index.name:
            st.warning("Please configure connection and index first.")
            return
        
        # Filter presets
        preset_query = FilterPresets.render_preset_selector()
        if preset_query:
            st.session_state.filter_config.query = preset_query
        
        # Advanced filter builder
        filter_builder = FilterBuilder(self.client, self.config.index.name)
        user_filters = filter_builder.render_filter_ui()
        
        if user_filters:
            # Convert user filters to OpenSearch query
            opensearch_query = filter_builder.build_opensearch_query(user_filters)
            st.session_state.filter_config.query = opensearch_query
            
            # Show query preview
            with st.expander("🔍 Query Preview"):
                st.json(opensearch_query)
        
        # Clear filters button
        if st.button("🗑️ Clear All Filters", type="secondary"):
            # Reset filter config
            st.session_state.filter_config = FilterConfig()
            
            # Clear all filter-related session state keys
            filter_keys_to_clear = [
                "text_search_field",
                "text_search_value", 
                "text_search_type",
                "range_filters",
                "categorical_field",
                "date_field", 
                "start_date",
                "end_date",
                "custom_query_text",
                "filter_preset"
            ]
            
            for key in filter_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Also clear current data to force reload without filters
            if 'current_data' in st.session_state:
                st.session_state.current_data = None
                
            # Clear cached visualization results
            if 'visualization_results' in st.session_state:
                st.session_state.visualization_results = {}
                
            if 'current_plot' in st.session_state:
                st.session_state.current_plot = None
            
            st.success("✅ All filters cleared! Data will be reloaded without filters.")
            st.rerun()
    
    def _render_statistics_tab(self):
        """Render statistics and analysis tab."""
        if not st.session_state.current_data:
            st.info("No data loaded. Please load data first.")
            return
        
        # Data overview
        st.subheader("📊 Data Overview")
        
        documents = st.session_state.current_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(documents))
        with col2:
            embeddings = np.array([doc.embedding for doc in documents])
            st.metric("Embedding Dimensions", embeddings.shape[1])
        with col3:
            unique_names = len(set(doc.name for doc in documents))
            st.metric("Unique Names", unique_names)
        with col4:
            avg_metadata = np.mean([len(doc.metadata) for doc in documents])
            st.metric("Avg Metadata Fields", f"{avg_metadata:.1f}")
        
        # Embedding statistics
        if self.visualizer:
            embedding_stats = self.visualizer.get_embedding_statistics(embeddings)
            
            st.subheader("🔢 Embedding Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Norm", f"{embedding_stats['mean_norm']:.3f}")
                st.metric("Sparsity", f"{embedding_stats['sparsity']:.3%}")
                st.metric("Min Value", f"{embedding_stats['min_value']:.3f}")
            
            with col2:
                st.metric("Std Norm", f"{embedding_stats['std_norm']:.3f}")
                st.metric("Mean Value", f"{embedding_stats['mean_value']:.3f}")
                st.metric("Max Value", f"{embedding_stats['max_value']:.3f}")
        
        # Metadata analysis
        st.subheader("📋 Metadata Analysis")
        
        # Collect all metadata fields
        all_metadata = {}
        for doc in documents:
            for key, value in doc.metadata.items():
                if key not in all_metadata:
                    all_metadata[key] = []
                all_metadata[key].append(value)
        
        if all_metadata:
            metadata_df = pd.DataFrame({
                'Field': list(all_metadata.keys()),
                'Unique Values': [len(set(str(v) for v in values if v is not None)) for values in all_metadata.values()],
                'Null Count': [sum(1 for v in values if v is None) for values in all_metadata.values()],
                'Data Type': [type(next((v for v in values if v is not None), None)).__name__ for values in all_metadata.values()]
            })
            
            st.dataframe(metadata_df, use_container_width=True)
        else:
            st.info("No metadata fields found.")
    
    def _render_settings_tab(self):
        """Render settings and configuration tab."""
        st.subheader("⚙️ Application Settings")
        
        # Export/Import configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Configuration**")
            config_json = self.config.model_dump_json(indent=2)
            st.download_button(
                "📥 Download Config",
                data=config_json,
                file_name="opensearch_visualizer_config.json",
                mime="application/json"
            )
        
        with col2:
            st.write("**Import Configuration**")
            uploaded_file = st.file_uploader("Upload Config", type="json")
            if uploaded_file:
                try:
                    config_data = json.loads(uploaded_file.read())
                    self.config = AppConfig.model_validate(config_data)
                    st.success("✅ Configuration imported successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to import configuration: {str(e)}")
        
        # Reset application
        st.write("**Reset Application**")
        if st.button("🔄 Reset All Settings", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    def _render_welcome_screen(self):
        """Render welcome screen when not connected."""
        st.info("👆 Configure OpenSearch connection in the sidebar to get started!")
        
        # Feature overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌟 Features
            
            - **Interactive 3D Visualization**: Explore embeddings with zoom, rotate, and pan
            - **Multiple Reduction Methods**: t-SNE, UMAP, and PCA for different perspectives  
            - **Advanced Filtering**: Complex queries with text, range, and categorical filters
            - **Real-time Exploration**: Interactive filtering and visualization updates
            - **Export Capabilities**: Save configurations and results
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Requirements
            
            Your OpenSearch index should contain:
            - **Embedding field**: High-dimensional vectors (int8 or float32)
            - **Name/ID field**: Document identifiers or names  
            - **Optional**: Additional metadata fields for filtering and coloring
            
            ### 🚀 Getting Started
            
            1. Configure OpenSearch connection in sidebar
            2. Select index and field mappings
            3. Apply filters if needed
            4. Load and visualize your embeddings!
            """)
    
    def _render_placeholder(self):
        """Render placeholder for empty visualization."""
        st.markdown("""
        <div style="text-align: center; padding: 100px; color: #888;">
            <h3>🎯 Ready to Explore Your Embeddings</h3>
            <p>Configure your settings and click "Load & Visualize Data" to get started!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _connect_to_opensearch(self, host: str, port: int, username: str, password: str, 
                              use_ssl: bool, verify_certs: bool, timeout: int):
        """Connect to OpenSearch cluster."""
        with st.spinner("Connecting to OpenSearch..."):
            try:
                # Update config
                self.config.opensearch.host = host
                self.config.opensearch.port = port
                self.config.opensearch.username = username if username else None
                self.config.opensearch.password = password if password else None
                self.config.opensearch.use_ssl = use_ssl
                self.config.opensearch.verify_certs = verify_certs
                self.config.opensearch.timeout = timeout
                
                # Create client and connect
                self.client = OpenSearchClient(self.config.opensearch)
                status = self.client.connect()
                
                if status.connected:
                    # Store client in session state for persistence across reruns
                    st.session_state.opensearch_client = self.client
                    st.session_state.connected = True
                    st.success(f"✅ Connected to {status.cluster_name} (v{status.version})")
                else:
                    # Clear session state on failed connection
                    st.session_state.opensearch_client = None
                    st.session_state.connected = False
                    st.error(f"❌ {status.error_message}")
                    
            except Exception as e:
                # Clear session state on exception
                st.session_state.opensearch_client = None
                st.session_state.connected = False
                st.error(f"❌ Connection failed: {str(e)}")
    
    def _load_and_visualize_data(self):
        """Load data and create visualization."""
        if not self.client or not self.visualizer:
            st.error("Please configure connection and visualization settings first.")
            return
        
        with st.spinner("Loading and processing data..."):
            try:
                # Load data with filters
                documents, filter_result = self.client.fetch_embeddings(
                    self.config.index,
                    st.session_state.filter_config
                )
                
                if not documents:
                    st.warning("No documents found with current filters.")
                    return
                
                # Store data
                st.session_state.current_data = documents
                
                # Display filter summary
                FilterSummary.render_summary(filter_result, st.session_state.filter_config.query or {})
                
                # Create visualization
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                
                viz_result = self.visualizer.reduce_dimensions(documents)
                progress_bar.progress(75)
                
                # Create plot
                plot = self.visualizer.create_plot(viz_result)
                progress_bar.progress(100)
                
                # Store results
                st.session_state.current_plot = plot
                st.session_state.visualization_results[self.config.visualization.method.value] = viz_result
                
                st.success(f"✅ Processed {len(documents)} embeddings using {self.config.visualization.method.value.upper()}")
                
            except Exception as e:
                st.error(f"❌ Failed to load and visualize data: {str(e)}")
                logger.error(f"Visualization error: {str(e)}")
    
    def _update_visualization(self, color_by: Optional[str] = None, size_by: Optional[str] = None):
        """Update visualization with new display options."""
        if not self.visualizer or 'current_plot' not in st.session_state:
            return
        
        method = self.config.visualization.method.value
        if method in st.session_state.visualization_results:
            viz_result = st.session_state.visualization_results[method]
            plot = self.visualizer.create_plot(viz_result, color_by=color_by, size_by=size_by)
            st.session_state.current_plot = plot


def main():
    """Main entry point for the application."""
    try:
        app = VectorVisualizerApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()