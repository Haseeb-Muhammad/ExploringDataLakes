import streamlit as st
import requests
import json
from typing import Dict, Any
import pandas as pd
from util import *
from erd_visualizer import visualize_database_relationships
# Configure the page
st.set_page_config(
    page_title="Data Lake Explorer",
    page_icon="🏞️",
    layout="wide"
)


def display_database_description(description: Dict[str, Any]):
    """Display the database description in a formatted way"""
    if not description or "tables" not in description:
        st.warning("No database description available.")
        return
    
    st.subheader("📋 Database Description")
    
    for table_name, table_info in description["tables"].items():
        with st.expander(f"Table: {table_name}", expanded=False):
            st.write(f"**Description:** {table_info.get('note', 'No description available')}")
            
            if "columns" in table_info and table_info["columns"]:
                st.write("**Columns:**")
                col_data = []
                for col_name, col_info in table_info["columns"].items():
                    col_data.append({
                        "Column Name": col_name,
                        "Type": col_info.get("type", "Unknown"),
                        "Description": col_info.get("note", "No description available")
                    })
                
                df = pd.DataFrame(col_data)
                st.dataframe(df, use_container_width=True)

def welcome_page():
    """Welcome page with file upload functionality"""
    st.title("🏞️ Data Lake Explorer")
    st.markdown("Welcome to the Data Lake Explorer! Upload your database files and ground truth to get started.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Database Files")
        database_files = st.file_uploader(
            "Choose database files (.csv)",
            type=['csv'],
            accept_multiple_files=True,
            key="db_files"
        )
        
        if database_files:
            if st.button("Upload Database Files", key="upload_db"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file in enumerate(database_files):
                    result = upload_database_file(file)
                    if result:
                        st.success(f"✅ Uploaded: {file.name}")
                        success_count += 1
                    else:
                        st.error(f"❌ Failed to upload: {file.name}")
                    
                    progress_bar.progress((i + 1) / len(database_files))
                
                st.info(f"Uploaded {success_count} out of {len(database_files)} files successfully.")
    
    with col2:
        st.subheader("🎯 Upload Ground Truth File")
        ground_truth_file = st.file_uploader(
            "Choose ground truth file (.json)",
            type=['json'],
            key="gt_file"
        )
        
        if ground_truth_file:
            if st.button("Upload Ground Truth", key="upload_gt"):
                result = upload_ground_truth_file(ground_truth_file)
                if result:
                    st.success(f"✅ Ground truth uploaded successfully!")
                    st.json(result)
                    # Force a rerun to display the database description
                    st.rerun()
    
    # Action buttons
    st.markdown("---")
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        if st.button("🔄 Reset Database", key="reset_db"):
            result = reset_database()
            if result:
                st.success("Database reset successfully!")
                st.session_state.show_description = False
                st.rerun()
            else:
                st.error("Failed to reset database.")
    
    with button_col2:
        if st.button("🔍 Go to Clustering", key="go_clustering"):
            st.session_state.page = "clustering"
            st.rerun()
    
    with button_col3:
        if st.button("Show database description",key="show_description"):
            mock_description = get_database_description()
            display_database_description(mock_description)
    

def clustering_page():
    """Clustering page with algorithm selection and results"""
    st.title("🔍 Clustering Analysis")
    st.markdown("Perform clustering analysis on your database tables using various algorithms.")
    
    # Back button
    if st.button("← Back to Welcome", key="back_welcome"):
        st.session_state.page = "welcome"
        st.rerun()
    
    st.markdown("---")
    
    # Clustering method selection
    st.subheader("⚙️ Select Clustering Method")
    
    clustering_methods = {
        "hdbScan": "HDBSCAN - Hierarchical Density-Based Spatial Clustering",
        # Add more methods here as they become available
    }
    
    selected_method = st.selectbox(
        "Choose a clustering algorithm:",
        options=list(clustering_methods.keys()),
        format_func=lambda x: clustering_methods[x],
        key="clustering_method"
    )
    
    # Clustering execution
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🚀 Run Clustering", key="run_clustering"):
            st.session_state.clustering_running = True
    
    # Display clustering results
    if st.session_state.get('clustering_running', False):
        with st.spinner(f"Running {clustering_methods[selected_method]}..."):
            result = perform_clustering(selected_method)
            
            if result:
                st.success("✅ Clustering completed successfully!")
                
                st.subheader("📊 Clustering Results")
                
                # Display results in a structured format
                if isinstance(result, dict):
                    for level, clusters in result.items():
                        st.write(f"**Level {level}:**")
                        
                        if isinstance(clusters, dict):
                            for cluster_no, table_names in clusters.items():
                                with st.expander(f"Cluster {cluster_no} ({len(table_names)} tables)"):
                                    for table_name in table_names:
                                        st.write(f"• {table_name}")
                        else:
                            st.write(clusters)
                        
                        st.markdown("---")
                else:
                    st.json(result)
                
                # Option to download results
                result_json = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=result_json,
                    file_name=f"clustering_results_{selected_method}.json",
                    mime="application/json"
                )
            else:
                st.error("❌ Clustering failed. Please check your data and try again.")
            
            st.session_state.clustering_running = False

def main():
    """Main application function"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page_options = {
        "welcome": "🏠 Welcome",
        "clustering": "🔍 Clustering"
    }
    
    selected_page = st.sidebar.radio(
        "Go to:",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        index=list(page_options.keys()).index(st.session_state.page)
    )
    
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    # Display appropriate page
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "clustering":
        clustering_page()

if __name__ == "__main__":
    main()