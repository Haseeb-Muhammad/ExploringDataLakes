import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Dict, Set

# Backend API base URL - adjust as needed
API_BASE_URL = "http://192.168.18.35:8000"

def get_available_similarity_methods():
    """Fetch available similarity methods from the backend API."""
    try:
        response = requests.get(f"{API_BASE_URL}/getSimilarityMethods")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch similarity methods: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return []

def apply_similarity(method, threshold):
    """Apply similarity filtering using the backend API."""
    try:
        params = {"method": method, "threshold": threshold}
        response = requests.get(f"{API_BASE_URL}/applySimilarity", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to apply similarity: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

def parse_relationships_from_api(relationships):
    """Parse relationships from API response to format expected by visualizer."""
    parsed = []
    for rel in relationships:
        if isinstance(rel, dict):
            # Assuming the relationship format from your API
            ref_attr = f"{rel.get('ref_table', '')}.{rel.get('ref_column', '')}"
            dep_attr = f"{rel.get('dep_table', '')}.{rel.get('dep_column', '')}"
            parsed.append((ref_attr, dep_attr))
        elif isinstance(rel, (list, tuple)) and len(rel) == 2:
            parsed.append((rel[0], rel[1]))
    return parsed

def extract_table_attributes(relationships: List[Tuple[str, str]], tables: List[str]) -> Dict[str, Dict[str, Set[str]]]:
    """Extract table attributes and their types (reference/dependent) from relationships."""
    table_attrs = {}
    
    # Initialize all tables
    for table in tables:
        table_attrs[table] = {'references': set(), 'dependents': set()}
    
    # Process relationships
    for ref_attr, dep_attr in relationships:
        if '.' in str(ref_attr) and '.' in str(dep_attr):
            ref_table, ref_col = str(ref_attr).split('.', 1)
            dep_table, dep_col = str(dep_attr).split('.', 1)
            
            # Initialize tables if not exists
            if ref_table not in table_attrs:
                table_attrs[ref_table] = {'references': set(), 'dependents': set()}
            if dep_table not in table_attrs:
                table_attrs[dep_table] = {'references': set(), 'dependents': set()}
            
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
    
    # Combine all attributes
    all_attrs = list(attributes['references']) + list(attributes['dependents'])
    if not all_attrs:
        all_attrs = ['(no attributes)']
    
    # Calculate table dimensions
    table_width = 2.0
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
        
        # Color code attributes
        color = 'black'
        prefix = ''
        if attr in attributes['references']:
            color = 'red'
            prefix = '→ '
        elif attr in attributes['dependents']:
            color = 'blue'
            prefix = '← '
        
        ax.text(x - table_width/2 + 0.1, attr_y, f"{prefix}{attr}",
                ha='left', va='center', fontsize=8, color=color)
        
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

def visualize_similarity_relationships(relationships: List[Tuple[str, str]], table_names: List[str]) -> plt.Figure:
    """Create a visualization of similarity-filtered database relationships."""
    # Extract table attributes
    table_attrs = extract_table_attributes(relationships, table_names)
    
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
    
    plt.title("Similarity-Based Database Relationship Diagram", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_similarity_visualizer():
    """Creates a Streamlit app for similarity-based filtering and visualization of database relationships."""
    st.title("Database Relationship Similarity Calculator & Visualizer")
    st.markdown("Filter database relationships based on column name similarity using various similarity methods.")
    
    # Fetch available similarity methods
    with st.spinner("Loading available similarity methods..."):
        available_methods = get_available_similarity_methods()
    
    if not available_methods:
        st.error("Unable to load similarity methods. Please ensure the backend API is running.")
        return
    
    # Display method information
    st.header("Similarity Methods")
    
    # Create method descriptions
    method_descriptions = {
        "fastText": "Uses pre-trained FastText embeddings to calculate semantic similarity between column names",
        "sentenceTransformer": "Uses Sentence Transformer models to encode column names into dense vectors for similarity comparison",
        "TF_IDF": "Uses Term Frequency-Inverse Document Frequency vectorization with character n-grams to compare text similarity",
        "partial_ratio": "Uses fuzzy string matching (Levenshtein distance) to find the best matching substring similarity"
    }
    
    # Method selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_method = st.selectbox(
            "Select similarity method:",
            options=available_methods,
            help="Choose the method to calculate similarity between column names"
        )
    
    with col2:
        threshold = st.slider(
            "Similarity threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Relationships with similarity above this threshold will be kept"
        )
    
    # Show method description
    if selected_method:
        st.subheader("Selected Method Description")
        description = method_descriptions.get(selected_method, "No description available")
        st.markdown(f"**{selected_method}:** {description}")
    
    # Apply similarity button
    if st.button("Apply Similarity & Visualize", type="primary", disabled=not selected_method):
        if selected_method:
            with st.spinner("Applying similarity filtering and generating visualization..."):
                result = apply_similarity(selected_method, threshold)
                
                if result and result.get('success'):
                    # Display similarity results summary
                    st.success("Similarity filtering applied successfully!")
                    
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
                    
                    # Show similarity statistics
                    similarity_stats = result.get('similarity_stats', {})
                    if similarity_stats:
                        st.subheader("Similarity Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average Similarity", f"{similarity_stats.get('average', 0):.3f}")
                        
                        with col2:
                            st.metric("Maximum Similarity", f"{similarity_stats.get('maximum', 0):.3f}")
                        
                        with col3:
                            st.metric("Minimum Similarity", f"{similarity_stats.get('minimum', 0):.3f}")
                        
                        with col4:
                            st.metric("Above Threshold", similarity_stats.get('above_threshold', 0))
                    
                    # Show method and threshold details
                    st.subheader("Applied Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Method**: {result.get('method', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Threshold**: {result.get('threshold', 0):.2f}")
                    
                    # Visualize the similarity-filtered relationships
                    relationships = result.get('filtered_relationships', [])
                    table_names = result.get('table_names', [])
                    
                    if relationships and table_names:
                        st.subheader("Relationship Visualization")
                        
                        # Parse relationships based on your API format
                        parsed_relationships = parse_relationships_from_api(relationships)
                        
                        # Generate visualization
                        fig = visualize_similarity_relationships(parsed_relationships, table_names)
                        st.pyplot(fig)
                        
                        # Show legend and statistics
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Legend")
                            st.markdown("🔴 **Red text**: Reference attributes (→)")
                            st.markdown("🔵 **Blue text**: Dependent attributes (←)")
                            st.markdown("🟢 **Green arrows**: Relationships")
                        
                        with col2:
                            st.subheader("Statistics")
                            st.markdown(f"**Tables**: {len(table_names)}")
                            st.markdown(f"**Relationships**: {len(parsed_relationships)}")
                            st.markdown(f"**Method**: {selected_method}")
                            st.markdown(f"**Threshold**: {threshold:.2f}")
                        
                        # Show detailed similarity scores (optional, in an expander)
                        similarity_scores = result.get('similarity_scores', [])
                        if similarity_scores:
                            with st.expander("View Detailed Similarity Scores"):
                                st.subheader("Similarity Scores for All Relationships")
                                
                                # Convert to DataFrame for better display
                                import pandas as pd
                                df_scores = pd.DataFrame(similarity_scores)
                                df_scores['similarity'] = df_scores['similarity'].round(3)
                                df_scores = df_scores.sort_values('similarity', ascending=False)
                                
                                # Color code based on threshold
                                def highlight_threshold(val):
                                    color = 'lightgreen' if val > threshold else 'lightcoral'
                                    return f'background-color: {color}'
                                
                                styled_df = df_scores.style.applymap(highlight_threshold, subset=['similarity'])
                                st.dataframe(styled_df, use_container_width=True)
                                
                                st.markdown(f"**Green**: Above threshold ({threshold:.2f}) - **Red**: Below threshold")
                    else:
                        st.info("No relationships found after applying similarity filtering.")
                        
                elif result:
                    st.error(f"Failed to apply similarity: {result.get('error', 'Unknown error')}")
                    if 'available_methods' in result:
                        st.info(f"Available methods: {', '.join(result['available_methods'])}")
        else:
            st.warning("Please select a similarity method.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Database Similarity Visualizer",
        page_icon="🔗",
        layout="wide"
    )
    
    create_similarity_visualizer()

if __name__ == "__main__":
    main()
