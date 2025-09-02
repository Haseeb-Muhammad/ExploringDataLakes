import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import List, Tuple, Dict, Set

def visualize_database_relationships(relationships: List[Tuple[str, str]], table_names: List[str]) -> plt.Figure:
    """
    Create a database relationship visualization.
    
    Args:
        relationships: List of tuples representing relationships as (reference_table.reference_attr, dependent_table.dependent_attr)
        table_names: List of table names to include in visualization (including isolated tables)
    
    Returns:
        matplotlib.pyplot.Figure: The generated ERD diagram
    
    Example:
        relationships = [
            ("customers.customer_id", "orders.customer_id"),
            ("orders.order_id", "order_items.order_id"),
            ("products.product_id", "order_items.product_id")
        ]
        table_names = ["customers", "orders", "order_items", "products", "suppliers"]
        fig = visualize_database_relationships(relationships, table_names)
    """
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
    table_boxes = {}
    attr_positions = {}
    
    for table_name, (x, y) in table_positions.items():
        box, attr_pos = draw_table(ax, table_name, table_attrs[table_name], x, y)
        table_boxes[table_name] = box
        attr_positions[table_name] = attr_pos
    
    # Draw relationships
    draw_relationships(ax, relationships, attr_positions)
    
    plt.title("Database Relationship Diagram", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_database_visualizer():
    """
    Creates a Streamlit app for visualizing database relationships.
    """
    st.title("Database Relationship Visualizer")
    
    # Sample data for testing
    sample_relationships = [
        ("customers.customer_id", "orders.customer_id"),
        ("orders.order_id", "order_items.order_id"),
        ("products.product_id", "order_items.product_id"),
        ("categories.category_id", "products.category_id")
    ]
    
    sample_tables = ["customers", "orders", "order_items", "products", "categories", "suppliers"]
    
    # Input sections in main area
    st.header("Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relationships")
        relationships_text = st.text_area(
            "Enter relationships (one per line): table.attribute -> table.attribute",
            value="\n".join([f"{rel[0]} -> {rel[1]}" for rel in sample_relationships]),
            height=200,
            key="relationships"
        )
    
    with col2:
        st.subheader("Tables")
        tables_text = st.text_area(
            "Enter table names (one per line):",
            value="\n".join(sample_tables),
            height=200,
            key="tables"
        )
    
    # Parse inputs
    relationships = parse_relationships(relationships_text)
    tables = parse_tables(tables_text)
    
    # Generate visualization
    if st.button("Generate Visualization", type="primary"):
        if relationships or tables:
            with st.spinner("Generating visualization..."):
                fig = visualize_database_relationships(relationships, tables)
                st.pyplot(fig)
                
                # Show legend and instructions below the chart
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Legend")
                    st.markdown("🔴 **Red text**: Reference attributes (→)")
                    st.markdown("🔵 **Blue text**: Dependent attributes (←)")
                    st.markdown("🟢 **Green arrows**: Relationships")
                
                with col2:
                    st.subheader("Instructions")
                    st.markdown("""
                    1. Enter relationships in format: `table.column -> table.column`
                    2. Enter all table names (including isolated tables)
                    3. Click 'Generate Visualization'
                    
                    **Example:**
                    ```
                    customers.customer_id -> orders.customer_id
                    orders.order_id -> order_items.order_id
                    ```
                    """)
        else:
            st.warning("Please provide either relationships or table names.")

# ...existing code...

def extract_table_attributes(relationships: List[Tuple[str, str]], tables: List[str]) -> Dict[str, Dict[str, Set[str]]]:
    """
    Extract table attributes and their types (reference/dependent) from relationships.
    
    Returns:
        Dict with structure: {table_name: {'references': set(), 'dependents': set()}}
    """
    table_attrs = {}
    
    # Initialize all tables
    for table in tables:
        table_attrs[table] = {'references': set(), 'dependents': set()}
    
    # Process relationships
    for ref_attr, dep_attr in relationships:
        if '.' in ref_attr and '.' in dep_attr:
            ref_table, ref_col = ref_attr.split('.', 1)
            dep_table, dep_col = dep_attr.split('.', 1)
            
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

def draw_table(ax, table_name: str, attributes: Dict[str, Set[str]], x: float, y: float) -> Tuple[patches.Rectangle, Dict[str, Tuple[float, float]]]:
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
    
    return table_box, attr_positions

def draw_relationships(ax, relationships: List[Tuple[str, str]], attr_positions: Dict[str, Dict[str, Tuple[float, float]]]):
    """Draw arrows between related attributes."""
    
    for ref_attr, dep_attr in relationships:
        if '.' in ref_attr and '.' in dep_attr:
            ref_table, ref_col = ref_attr.split('.', 1)
            dep_table, dep_col = dep_attr.split('.', 1)
            
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

def parse_relationships(text: str) -> List[Tuple[str, str]]:
    """Parse relationship text into list of tuples."""
    relationships = []
    for line in text.strip().split('\n'):
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                ref = parts[0].strip()
                dep = parts[1].strip()
                relationships.append((ref, dep))
    return relationships

def parse_tables(text: str) -> List[str]:
    """Parse table names from text."""
    return [table.strip() for table in text.strip().split('\n') if table.strip()]

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Database Relationship Visualizer",
        page_icon="🗄️",
        layout="wide"
    )
    
    create_database_visualizer()

if __name__ == "__main__":
    main()