import streamlit as st
import requests
import pandas as pd
import json
import io
from typing import List, Dict, Any
import time

# Backend URL
BACKEND_URL = "http://backend:8000"  # For Docker Compose
# BACKEND_URL = "http://localhost:8000"  # For local development

# Initialize session state
if 'database_files' not in st.session_state:
    st.session_state.database_files = []
if 'ground_truth_file' not in st.session_state:
    st.session_state.ground_truth_file = None
if 'upload_completed' not in st.session_state:
    st.session_state.upload_completed = False

st.set_page_config(
    page_title="Database Upload Interface",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Database Files Upload Interface")
st.markdown("---")

# Create two columns for layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.header("📊 Database Files (CSV)")
    st.write("Upload multiple CSV files representing your database tables")
    
    # Database files upload
    database_files = st.file_uploader(
        "Select Database CSV Files",
        type=['csv'],
        accept_multiple_files=True,
        key="database_uploader",
        help="Upload all CSV files that represent your database tables"
    )
    
    # Store files in session state
    if database_files:
        st.session_state.database_files = database_files
    
    # Display uploaded database files
    if st.session_state.database_files:
        st.success(f"✅ {len(st.session_state.database_files)} database files selected")
        
        with st.expander("📋 View Database Files Details", expanded=False):
            total_size = 0
            for i, file in enumerate(st.session_state.database_files):
                total_size += file.size
                
                with st.container():
                    st.write(f"**{i+1}. {file.name}**")
                    
                    # Try to preview the CSV
                    try:
                        file.seek(0)
                        df = pd.read_csv(file)
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Rows", len(df))
                            st.metric("Columns", len(df.columns))
                        
                        with col_b:
                            st.metric("Size", f"{file.size:,} bytes")
                            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum():,} bytes")
                        
                        # Show column names
                        st.write("**Columns:**", ", ".join(df.columns.tolist()))
                        
                        # Show sample data
                        if st.checkbox(f"Preview data for {file.name}", key=f"preview_{i}"):
                            st.dataframe(df.head(), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")
                    
                    st.markdown("---")
            
            st.info(f"**Total:** {len(st.session_state.database_files)} files, {total_size:,} bytes")

with col2:
    st.header("🎯 Ground Truth File (JSON)")
    st.write("Upload the ground truth file for validation/testing")
    
    # Ground truth file upload
    ground_truth_file = st.file_uploader(
        "Select Ground Truth JSON File",
        type=['json'],
        key="ground_truth_uploader",
        help="Upload the JSON file containing ground truth data"
    )
    
    # Store ground truth file in session state
    if ground_truth_file:
        st.session_state.ground_truth_file = ground_truth_file
    
    # Display ground truth file details
    if st.session_state.ground_truth_file:
        st.success("✅ Ground truth file selected")
        
        with st.expander("📋 View Ground Truth Details", expanded=False):
            file = st.session_state.ground_truth_file
            st.write(f"**Filename:** {file.name}")
            st.write(f"**Size:** {file.size:,} bytes")
            
            # Try to preview JSON
            try:
                file.seek(0)
                json_data = json.load(file)
                
                st.write(f"**Type:** {type(json_data).__name__}")
                
                if isinstance(json_data, list):
                    st.write(f"**Array Length:** {len(json_data)}")
                    if len(json_data) > 0:
                        st.write(f"**Sample Structure:** {type(json_data[0]).__name__}")
                        if isinstance(json_data[0], dict):
                            st.write(f"**Keys:** {', '.join(json_data[0].keys())}")
                
                elif isinstance(json_data, dict):
                    st.write(f"**Keys:** {', '.join(json_data.keys())}")
                
                # Show preview
                if st.checkbox("Preview JSON content", key="preview_json"):
                    if len(str(json_data)) < 2000:
                        st.json(json_data)
                    else:
                        st.json(str(json_data)[:2000] + "... (truncated)")
                        
            except Exception as e:
                st.error(f"Error reading JSON: {str(e)}")

with col3:
    st.header("🚀 Actions")
    
    # Upload button
    upload_button = st.button(
        "📤 Upload All Files",
        type="primary",
        disabled=not (st.session_state.database_files and st.session_state.ground_truth_file),
        help="Upload all database files and ground truth file to backend",
        use_container_width=True
    )
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.database_files = []
        st.session_state.ground_truth_file = None
        st.session_state.upload_completed = False
        st.rerun()
    
    # Status indicators
    st.markdown("### 📊 Status")
    
    if st.session_state.database_files:
        st.success(f"Database: {len(st.session_state.database_files)} files")
    else:
        st.warning("Database: No files")
    
    if st.session_state.ground_truth_file:
        st.success("Ground Truth: Ready")
    else:
        st.warning("Ground Truth: Missing")
    
    if st.session_state.upload_completed:
        st.success("✅ Upload Complete")

# Handle upload
if upload_button:
    if not st.session_state.database_files:
        st.error("❌ Please upload at least one database CSV file")
    elif not st.session_state.ground_truth_file:
        st.error("❌ Please upload a ground truth JSON file")
    else:
        # Perform upload
        st.markdown("---")
        st.header("📤 Upload Progress")
        
        progress_container = st.container()
        
        with progress_container:
            # Create progress tracking
            total_files = len(st.session_state.database_files) + 1  # +1 for ground truth
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            upload_results = {
                "database_files": [],
                "ground_truth": None,
                "errors": []
            }
            
            # Upload database files
            status_text.text("Uploading database files...")
            
            for i, file in enumerate(st.session_state.database_files):
                try:
                    file.seek(0)
                    files = {
                        "file": (file.name, file.getvalue(), file.type)
                    }
                    
                    response = requests.post(
                        f"{BACKEND_URL}/upload-database-file/",
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        upload_results["database_files"].append({
                            "filename": file.name,
                            "status": "success",
                            "details": result
                        })
                        st.success(f"✅ {file.name} uploaded successfully")
                    else:
                        error_msg = f"Failed to upload {file.name}: {response.text}"
                        upload_results["errors"].append(error_msg)
                        st.error(f"❌ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Error uploading {file.name}: {str(e)}"
                    upload_results["errors"].append(error_msg)
                    st.error(f"❌ {error_msg}")
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
                time.sleep(0.1)  # Small delay for better UX
            
            # Upload ground truth file
            status_text.text("Uploading ground truth file...")
            
            try:
                file = st.session_state.ground_truth_file
                file.seek(0)
                files = {
                    "file": (file.name, file.getvalue(), file.type)
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/upload-ground-truth/",
                    files=files,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    upload_results["ground_truth"] = {
                        "filename": file.name,
                        "status": "success",
                        "details": result
                    }
                    st.success(f"✅ {file.name} uploaded successfully")
                else:
                    error_msg = f"Failed to upload ground truth: {response.text}"
                    upload_results["errors"].append(error_msg)
                    st.error(f"❌ {error_msg}")
                    
            except Exception as e:
                error_msg = f"Error uploading ground truth: {str(e)}"
                upload_results["errors"].append(error_msg)
                st.error(f"❌ {error_msg}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Upload completed!")
            
            # Summary
            st.markdown("### 📋 Upload Summary")
            
            success_count = len(upload_results["database_files"])
            if upload_results["ground_truth"] and upload_results["ground_truth"]["status"] == "success":
                success_count += 1
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                st.metric("Total Files", total_files)
            
            with col_summary2:
                st.metric("Successful", success_count)
            
            with col_summary3:
                st.metric("Errors", len(upload_results["errors"]))
            
            # Detailed results
            if upload_results["database_files"]:
                with st.expander("📊 Database Files Results", expanded=len(upload_results["errors"]) == 0):
                    for result in upload_results["database_files"]:
                        st.write(f"**{result['filename']}:** {result['status']}")
                        if result["status"] == "success" and "details" in result:
                            details = result["details"]
                            if "processed_data" in details:
                                data = details["processed_data"]
                                st.write(f"  - Rows: {data.get('rows', 'N/A')}")
                                st.write(f"  - Columns: {data.get('columns', 'N/A')}")
            
            if upload_results["ground_truth"]:
                with st.expander("🎯 Ground Truth Results", expanded=True):
                    gt = upload_results["ground_truth"]
                    st.write(f"**{gt['filename']}:** {gt['status']}")
                    if gt["status"] == "success" and "details" in gt:
                        details = gt["details"]
                        st.json(details.get("processed_data", {}))
            
            if upload_results["errors"]:
                with st.expander("❌ Errors", expanded=True):
                    for error in upload_results["errors"]:
                        st.error(error)
            
            # Mark upload as completed
            if len(upload_results["errors"]) == 0:
                st.session_state.upload_completed = True
                st.balloons()

# # Footer with backend connection test
# st.markdown("---")
# st.subheader("🔧 System Status")

# col_status1, col_status2 = st.columns(2)

# with col_status1:
#     if st.button("🔍 Test Backend Connection"):
#         try:
#             response = requests.get(f"{BACKEND_URL}/", timeout=5)
#             if response.status_code == 200:
#                 st.success("✅ Backend is reachable!")
#                 data = response.json()
#                 st.json(data)
#             else:
#                 st.error(f"❌ Backend returned status: {response.status_code}")
#         except requests.exceptions.ConnectionError:
#             st.error("❌ Could not connect to backend. Check if the service is running.")
#         except Exception as e:
#             st.error(f"❌ Connection error: {str(e)}")

# with col_status2:
#     st.info(f"**Backend URL:** {BACKEND_URL}")
#     if st.session_state.database_files or st.session_state.ground_truth_file:
#         st.info("💡 **Tip:** Files are stored in session. Clear cache if you encounter issues.")