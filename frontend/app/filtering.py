import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Dict, Set

# Backend API base URL - adjust as needed
API_BASE_URL = "http://192.168.18.35:8000"

def get_available_filters():
    """Fetch available filters from the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/getFilters")
        if response.status_code == 200:
            return response.json()
            print(f"")
        else:
            st.error(f"Failed to fetch filters: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return []

def apply_filters(selected_filters):
    """Apply selected filters using the backend API."""
    try:
        params = {"methods": selected_filters}
        response = requests.get(f"{API_BASE_URL}/applyFilters", params=params)
        if response.status_code == 200:
            result = response.json()
            # Debug: print the response structure
            print("API Response structure:")
            print(f"Keys in response: {result.keys()}")
            print(f"Original count: {result.get('original_count', 'N/A')}")
            print(f"Filtered count: {result.get('filtered_count', 'N/A')}")
            print(f"Relationships type: {type(result.get('filtered_relationships', []))}")
            print(f"Sample relationships: {result.get('filtered_relationships', [])[:3]}")
            return result
        else:
            st.error(f"Failed to apply filters: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

def get_primary_keys():
    """Fetch primary keys from the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/getPrimaryKeys")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch primary keys: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to backend for primary keys: {e}")
        return {}

def parse_relationships_from_api(relationships):
    """Parse relationships from API response to format expected by visualizer."""
    parsed = []
    for rel in relationships:
        if isinstance(rel, (list, tuple)) and len(rel) == 2:
            # Relationships are already in the format (reference_attr, dependent_attr)
            parsed.append((str(rel[0]), str(rel[1])))
        elif isinstance(rel, dict):
            # Fallback for dictionary format
            ref_attr = f"{rel.get('ref_table', '')}.{rel.get('ref_column', '')}"
            dep_attr = f"{rel.get('dep_table', '')}.{rel.get('dep_column', '')}"
            parsed.append((ref_attr, dep_attr))
    return parsed

def extract_table_attributes(relationships: List[Tuple[str, str]], tables: List[str], primary_keys: Dict = None) -> Dict[str, Dict[str, Set[str]]]:
    """Extract table attributes and their types (reference/dependent/primary) from relationships."""
    table_attrs = {}
    
    # Initialize all tables
    for table in tables:
        table_attrs[table] = {
            'primary_keys': set(), 
            'references': set(), 
            'dependents': set(),
            'other_attributes': set()
        }
    
    # Add primary keys if available
    if primary_keys:
        for table_name, pk_info in primary_keys.items():
            if table_name in table_attrs and pk_info:
                # pk_info is typically (full_name, score), extract column name
                if isinstance(pk_info, (list, tuple)) and len(pk_info) >= 1:
                    pk_full_name = pk_info[0]
                    if '.' in pk_full_name:
                        _, pk_column = pk_full_name.split('.', 1)
                        table_attrs[table_name]['primary_keys'].add(pk_column)
    
    # Process relationships
    for ref_attr, dep_attr in relationships:
        if '.' in str(ref_attr) and '.' in str(dep_attr):
            ref_table, ref_col = str(ref_attr).split('.', 1)
            dep_table, dep_col = str(dep_attr).split('.', 1)
            
            # Initialize tables if not exists
            if ref_table not in table_attrs:
                table_attrs[ref_table] = {
                    'primary_keys': set(), 
                    'references': set(), 
                    'dependents': set(),
                    'other_attributes': set()
                }
            if dep_table not in table_attrs:
                table_attrs[dep_table] = {
                    'primary_keys': set(), 
                    'references': set(), 
                    'dependents': set(),
                    'other_attributes': set()
                }
            
            # Add attributes
            table_attrs[ref_table]['references'].add(ref_col)
            table_attrs[dep_table]['dependents'].add(dep_col)
    
    return table_attrs

def calculate_table_positions(table_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """Calculate positions for tables in a grid layout."""
    positions = {}
    num_tables = len(table_names)
    
    if num_tables == 0:
        return positions
    
    # Calculate grid dimensions
    cols = min(3, num_tables)
    rows = (num_tables + cols - 1) // cols
    
    # Calculate spacing
    x_spacing = 8 / max(1, cols - 1) if cols > 1 else 0
    y_spacing = 6 / max(1, rows - 1) if rows > 1 else 0
    
    # Position tables
    for i, table_name in enumerate(table_names):
        row = i // cols
        col = i % cols
        
        x = 1 + col * x_spacing
        y = 6.5 - row * y_spacing
        
        positions[table_name] = (x, y)
    
    return positions

def draw_table(ax, table_name: str, attributes: Dict[str, Set[str]], x: float, y: float):
    """Draw a table with its attributes."""
    
    # Create a unique set of all attributes with their types
    all_attrs = set()
    attr_types = {}
    
    # Add primary keys first (highest priority)
    for attr in attributes.get('primary_keys', set()):
        all_attrs.add(attr)
        attr_types[attr] = 'primary_key'
    
    # Add references (if not already primary key)
    for attr in attributes.get('references', set()):
        all_attrs.add(attr)
        if attr not in attr_types:
            attr_types[attr] = 'reference'
    
    # Add dependents (if not already primary key or reference)
    for attr in attributes.get('dependents', set()):
        all_attrs.add(attr)
        if attr not in attr_types:
            attr_types[attr] = 'dependent'
    
    # Add other attributes
    for attr in attributes.get('other_attributes', set()):
        all_attrs.add(attr)
        if attr not in attr_types:
            attr_types[attr] = 'other'
    
    # Convert to sorted list
    all_attrs = sorted(list(all_attrs))
    
    if not all_attrs:
        all_attrs = ['(no attributes)']
        attr_types['(no attributes)'] = 'other'
    
    # Calculate table dimensions
    table_width = 2.5
    header_height = 0.3
    attr_height = 0.25
    table_height = header_height + len(all_attrs) * attr_height
    
    # Draw table background
    table_box = FancyBboxPatch(
        (x - table_width/2, y - table_height/2),
        table_width, table_height,
        boxstyle="round,pad=0.02",
        facecolor='lightblue',
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(table_box)
    
    # Draw table header
    header_box = FancyBboxPatch(
        (x - table_width/2, y + table_height/2 - header_height),
        table_width, header_height,
        boxstyle="round,pad=0.02",
        facecolor='darkblue',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(header_box)
    
    # Add table name
    ax.text(x, y + table_height/2 - header_height/2, table_name,
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Add attributes and track their positions
    attr_positions = {}
    for i, attr in enumerate(all_attrs):
        attr_y = y + table_height/2 - header_height - (i + 0.5) * attr_height
        
        # Color code and prefix attributes based on type
        attr_type = attr_types.get(attr, 'other')
        
        if attr_type == 'primary_key':
            color = 'gold'
            prefix = '🔑 '
        elif attr_type == 'reference':
            color = 'red'
            prefix = '→ '
        elif attr_type == 'dependent':
            color = 'blue'
            prefix = '← '
        else:
            color = 'black'
            prefix = ''
        
        ax.text(x - table_width/2 + 0.1, attr_y, f"{prefix}{attr}",
                ha='left', va='center', fontsize=8, color=color, weight='bold' if attr_type == 'primary_key' else 'normal')
        
        # Store attribute position for drawing relationships
        attr_positions[attr] = (x, attr_y)
    
    return attr_positions

def draw_relationships(ax, relationships: List[Tuple[str, str]], attr_positions: Dict[str, Dict[str, Tuple[float, float]]]):
    """Draw arrows between related attributes."""
    
    for ref_attr, dep_attr in relationships:
        if '.' in str(ref_attr) and '.' in str(dep_attr):
            ref_table, ref_col = str(ref_attr).split('.', 1)
            dep_table, dep_col = str(dep_attr).split('.', 1)
            
            # Check if both attributes exist in positions
            if (ref_table in attr_positions and ref_col in attr_positions[ref_table] and
                dep_table in attr_positions and dep_col in attr_positions[dep_table]):
                
                # Get positions
                ref_pos = attr_positions[ref_table][ref_col]
                dep_pos = attr_positions[dep_table][dep_col]
                
                # Draw arrow
                arrow = patches.FancyArrowPatch(
                    ref_pos, dep_pos,
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle="->",
                    mutation_scale=15,
                    color='green',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(arrow)

def visualize_filtered_relationships(relationships: List[Tuple[str, str]], table_names: List[str], primary_keys: Dict = None) -> plt.Figure:
    """Create a visualization of filtered database relationships."""
    # Extract table attributes
    table_attrs = extract_table_attributes(relationships, table_names, primary_keys)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Calculate positions for tables
    table_positions = calculate_table_positions(list(table_attrs.keys()))
    
    # Draw tables
    attr_positions = {}
    
    for table_name, (x, y) in table_positions.items():
        attr_pos = draw_table(ax, table_name, table_attrs[table_name], x, y)
        attr_positions[table_name] = attr_pos
    
    # Draw relationships
    draw_relationships(ax, relationships, attr_positions)
    
    plt.title("Filtered Database Relationship Diagram", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_filter_visualizer():
    """Creates a Streamlit app for filtering and visualizing database relationships."""
    st.title("Database Relationship Filter & Visualizer")
    st.markdown("Filter and visualize database relationships using various data quality checks.")
    
    # Fetch available filters
    with st.spinner("Loading available filters..."):
        available_filters = get_available_filters()
    
    if not available_filters:
        st.error("Unable to load filters. Please ensure the backend API is running.")
        return
    
    # Display filter information
    st.header("Available Filters")
    
    # Create filter descriptions
    filter_descriptions = {
        "primary_key_check": "Validates primary key constraints and relationships",
        "null_check": "Checks for null value constraints in key relationships", 
        "auto_incremental_pk_check": "Validates auto-incremental primary key patterns",
        "multi_dependence_check": "Identifies and validates multi-column dependencies",
        "dependent_referencing_check": "Validates referential integrity constraints"
    }
    
    # Multi-select for filters
    selected_filters = st.multiselect(
        "Select filters to apply:",
        options=available_filters,
        default=[],
        help="Choose one or more filters to apply to the database relationships"
    )
    
    # Show filter descriptions
    if selected_filters:
        st.subheader("Selected Filter Descriptions")
        for filter_name in selected_filters:
            description = filter_descriptions.get(filter_name, "No description available")
            st.markdown(f"**{filter_name}:** {description}")
    
    # Apply filters button
    if st.button("Apply Filters & Visualize", type="primary", disabled=len(selected_filters) == 0):
        if selected_filters:
            with st.spinner("Applying filters and generating visualization..."):
                result = apply_filters(selected_filters)
                
                if result and result.get('success'):
                    # Display filter results summary
                    st.success("Filters applied successfully!")
                    
                    # Show filtering statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Initial Relationships", 
                            result.get('original_count', 0)
                        )
                    
                    with col2:
                        st.metric(
                            "Final Relationships", 
                            result.get('filtered_count', 0),
                            delta=result.get('filtered_count', 0) - result.get('original_count', 0)
                        )
                    
                    with col3:
                        reduction_pct = 0
                        if result.get('original_count', 0) > 0:
                            reduction_pct = ((result.get('original_count', 0) - result.get('filtered_count', 0)) / 
                                           result.get('original_count', 0)) * 100
                        st.metric("Reduction %", f"{reduction_pct:.1f}%")
                    
                    # Show detailed filter results
                    st.subheader("Filter Results Detail")
                    filter_results = result.get('filter_results', {})
                    
                    for filter_name, filter_data in filter_results.items():
                        initial = filter_data.get('before_count', 0)
                        final = filter_data.get('after_count', 0)
                        removed = filter_data.get('filtered_out', 0)
                        st.markdown(f"**{filter_name}**: Removed {removed} relationships ({initial} → {final})")
                    
                    # Visualize the filtered relationships
                    relationships = result.get('filtered_relationships', [])
                    table_names = result.get('table_names', [])
                    primary_keys = result.get('primary_keys', {})
                    
                    if relationships and table_names:
                        st.subheader("Relationship Visualization")
                        
                        # Parse relationships based on your API format
                        parsed_relationships = parse_relationships_from_api(relationships)
                        
                        # Generate visualization
                        fig = visualize_filtered_relationships(parsed_relationships, table_names, primary_keys)
                        st.pyplot(fig)
                        
                        # Show legend and statistics
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Legend")
                            st.markdown("� **Gold text**: Primary keys")
                            st.markdown("�🔴 **Red text**: Reference attributes (→)")
                            st.markdown("🔵 **Blue text**: Dependent attributes (←)")
                            st.markdown("🟢 **Green arrows**: Relationships")
                        
                        with col2:
                            st.subheader("Statistics")
                            st.markdown(f"**Tables**: {len(table_names)}")
                            st.markdown(f"**Relationships**: {len(parsed_relationships)}")
                            st.markdown(f"**Filters Applied**: {len(selected_filters)}")
                            if primary_keys:
                                st.markdown(f"**Primary Keys Found**: {len(primary_keys)}")
                    else:
                        st.info("No relationships found after applying filters.")
                        
                elif result:
                    st.error(f"Failed to apply filters: {result.get('error', 'Unknown error')}")
                    if 'available_filters' in result:
                        st.info(f"Available filters: {', '.join(result['available_filters'])}")
        else:
            st.warning("Please select at least one filter to apply.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Database Filter Visualizer",
        page_icon="🔍",
        layout="wide"
    )
    
    create_filter_visualizer()

if __name__ == "__main__":
    main()