"""Advanced filtering functionality for OpenSearch queries."""

from typing import Any, Dict, List, Optional, Union
import streamlit as st
from datetime import datetime, date
import pandas as pd

from .models import SearchQuery, DocumentData
from .opensearch_client import OpenSearchClient


class FilterBuilder:
    """Builder for creating complex OpenSearch filters through UI."""
    
    def __init__(self, client: OpenSearchClient, index_name: str):
        """Initialize filter builder."""
        self.client = client
        self.index_name = index_name
        self._available_fields = None
        
    def get_available_fields(self) -> List[str]:
        """Get list of available fields in the index."""
        if self._available_fields is None:
            try:
                # Get index mapping to find available fields
                mapping = self.client.client.indices.get_mapping(index=self.index_name)
                properties = mapping[self.index_name]['mappings'].get('properties', {})
                self._available_fields = list(properties.keys())
            except Exception:
                self._available_fields = []
        
        return self._available_fields
    
    def render_filter_ui(self) -> Dict[str, Any]:
        """Render filter UI components and return filter configuration."""
        st.subheader("🔍 Advanced Filters")
        
        filters = {}
        
        # Text search filters
        with st.expander("📝 Text Search Filters", expanded=True):
            text_filters = self._render_text_filters()
            filters.update(text_filters)
        
        # Range filters  
        with st.expander("📊 Range Filters"):
            range_filters = self._render_range_filters()
            filters.update(range_filters)
        
        # Categorical filters
        with st.expander("🏷️ Categorical Filters"):
            categorical_filters = self._render_categorical_filters()
            filters.update(categorical_filters)
        
        # Date filters
        with st.expander("📅 Date Filters"):
            date_filters = self._render_date_filters()
            filters.update(date_filters)
        
        # Custom query
        with st.expander("⚙️ Custom OpenSearch Query"):
            custom_query = self._render_custom_query()
            if custom_query:
                return {"custom_query": custom_query}
        
        return filters
    
    def _render_text_filters(self) -> Dict[str, Any]:
        """Render text search filter controls."""
        filters = {}
        
        # Text fields for search
        available_fields = self.get_available_fields()
        text_fields = [f for f in available_fields if not f.endswith('.keyword')]
        
        if text_fields:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_field = st.selectbox(
                    "Field to search",
                    options=[""] + text_fields,
                    key="text_search_field"
                )
            
            with col2:
                search_text = st.text_input(
                    "Search text",
                    placeholder="Enter search term...",
                    key="text_search_value"
                )
            
            if selected_field and search_text:
                search_type = st.radio(
                    "Search type",
                    options=["Match", "Wildcard", "Exact"],
                    horizontal=True,
                    key="text_search_type"
                )
                
                if search_type == "Match":
                    filters[selected_field] = search_text
                elif search_type == "Wildcard":
                    filters[selected_field] = f"*{search_text}*"
                elif search_type == "Exact":
                    filters[f"{selected_field}.keyword"] = search_text
        
        return filters
    
    def _render_range_filters(self) -> Dict[str, Any]:
        """Render range filter controls."""
        filters = {}
        
        available_fields = self.get_available_fields()
        
        # Add range filter
        if st.button("➕ Add Range Filter", key="add_range_filter"):
            if "range_filters" not in st.session_state:
                st.session_state.range_filters = []
            st.session_state.range_filters.append(len(st.session_state.range_filters))
        
        if "range_filters" in st.session_state:
            for i in st.session_state.range_filters:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        field = st.selectbox(
                            "Field",
                            options=[""] + available_fields,
                            key=f"range_field_{i}"
                        )
                    
                    with col2:
                        min_val = st.number_input(
                            "Min value",
                            value=None,
                            key=f"range_min_{i}"
                        )
                    
                    with col3:
                        max_val = st.number_input(
                            "Max value", 
                            value=None,
                            key=f"range_max_{i}"
                        )
                    
                    with col4:
                        if st.button("🗑️", key=f"remove_range_{i}"):
                            st.session_state.range_filters.remove(i)
                            st.rerun()
                    
                    if field and (min_val is not None or max_val is not None):
                        range_query = {}
                        if min_val is not None:
                            range_query["gte"] = min_val
                        if max_val is not None:
                            range_query["lte"] = max_val
                        filters[field] = range_query
        
        return filters
    
    def _render_categorical_filters(self) -> Dict[str, Any]:
        """Render categorical filter controls."""
        filters = {}
        
        available_fields = self.get_available_fields()
        
        if available_fields:
            selected_field = st.selectbox(
                "Categorical field",
                options=[""] + available_fields,
                key="categorical_field"
            )
            
            if selected_field:
                # Get unique values for this field
                unique_values = self.client.get_field_values(self.index_name, selected_field)
                
                if unique_values:
                    selected_values = st.multiselect(
                        f"Select values for {selected_field}",
                        options=unique_values,
                        key=f"categorical_values_{selected_field}"
                    )
                    
                    if selected_values:
                        filters[f"{selected_field}.keyword"] = selected_values
        
        return filters
    
    def _render_date_filters(self) -> Dict[str, Any]:
        """Render date filter controls."""
        filters = {}
        
        available_fields = self.get_available_fields()
        date_fields = [f for f in available_fields if 'date' in f.lower() or 'time' in f.lower()]
        
        if date_fields:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_field = st.selectbox(
                    "Date field",
                    options=[""] + date_fields,
                    key="date_field"
                )
            
            with col2:
                start_date = st.date_input(
                    "Start date",
                    value=None,
                    key="start_date"
                )
            
            with col3:
                end_date = st.date_input(
                    "End date",
                    value=None,
                    key="end_date"
                )
            
            if date_field and (start_date or end_date):
                date_range = {}
                if start_date:
                    date_range["gte"] = start_date.isoformat()
                if end_date:
                    date_range["lte"] = end_date.isoformat()
                filters[date_field] = date_range
        
        return filters
    
    def _render_custom_query(self) -> Optional[Dict[str, Any]]:
        """Render custom OpenSearch query input."""
        st.write("Enter a custom OpenSearch query in JSON format:")
        
        query_text = st.text_area(
            "Custom Query",
            placeholder='{"match": {"field_name": "search_term"}}',
            height=150,
            key="custom_query_text"
        )
        
        if query_text.strip():
            try:
                import json
                custom_query = json.loads(query_text)
                st.success("✅ Valid JSON query")
                return custom_query
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {str(e)}")
        
        return None
    
    def build_opensearch_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build OpenSearch query from filters."""
        if "custom_query" in filters:
            return filters["custom_query"]
        
        return self.client.build_filter_query(filters)


class FilterPresets:
    """Predefined filter presets for common use cases."""
    
    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get available filter presets."""
        return {
            "Recent Documents": {
                "description": "Documents from the last 30 days",
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": "now-30d/d"
                        }
                    }
                }
            },
            "High Confidence": {
                "description": "Documents with confidence score > 0.8",
                "query": {
                    "range": {
                        "confidence": {
                            "gte": 0.8
                        }
                    }
                }
            },
            "English Documents": {
                "description": "Documents in English language",
                "query": {
                    "term": {
                        "language.keyword": "en"
                    }
                }
            },
            "Long Documents": {
                "description": "Documents with more than 1000 characters",
                "query": {
                    "range": {
                        "text_length": {
                            "gte": 1000
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def render_preset_selector() -> Optional[Dict[str, Any]]:
        """Render preset selector UI."""
        st.subheader("📋 Filter Presets")
        
        presets = FilterPresets.get_presets()
        
        selected_preset = st.selectbox(
            "Choose a preset filter",
            options=["None"] + list(presets.keys()),
            key="filter_preset"
        )
        
        if selected_preset != "None":
            preset_config = presets[selected_preset]
            st.info(f"📝 {preset_config['description']}")
            
            if st.button(f"Apply {selected_preset} Preset"):
                return preset_config["query"]
        
        return None


class FilterSummary:
    """Display filter summary and statistics."""
    
    @staticmethod
    def render_summary(filter_result, active_filters: Dict[str, Any]):
        """Render filter summary."""
        if not filter_result:
            return
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents", 
                f"{filter_result.total_docs:,}"
            )
        
        with col2:
            st.metric(
                "Filtered Documents", 
                f"{filter_result.filtered_docs:,}",
                delta=f"{filter_result.filtered_docs - filter_result.total_docs:,}"
            )
        
        with col3:
            percentage = (filter_result.filtered_docs / filter_result.total_docs * 100) if filter_result.total_docs > 0 else 0
            st.metric(
                "Retention Rate",
                f"{percentage:.1f}%"
            )
        
        with col4:
            st.metric(
                "Execution Time",
                f"{filter_result.execution_time:.2f}s"
            )
        
        # Filter description
        st.info(f"📋 {filter_result.filter_summary}")
        
        # Active filters details
        if active_filters:
            with st.expander("🔍 Active Filters Details"):
                for field, value in active_filters.items():
                    st.write(f"**{field}:** {value}")


def create_sample_filters() -> Dict[str, Dict[str, Any]]:
    """Create sample filters for demo purposes."""
    return {
        "sample_text_filter": {
            "description": "Search for documents containing 'machine learning'",
            "filters": {"content": "machine learning"}
        },
        "sample_range_filter": {
            "description": "Documents with score between 0.7 and 1.0",
            "filters": {"score": {"gte": 0.7, "lte": 1.0}}
        },
        "sample_category_filter": {
            "description": "Documents in 'technology' category",
            "filters": {"category.keyword": ["technology"]}
        }
    }