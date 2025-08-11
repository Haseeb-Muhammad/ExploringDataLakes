import requests
import streamlit as st

API_BASE_URL = "http://backend:8000" 

def perform_clustering(cluster_method: str):
    """Perform clustering using the specified method"""
    try:
        response = requests.get(f"{API_BASE_URL}/cluster", params={"cluster_method": cluster_method})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error performing clustering: {str(e)}")
        return None


def upload_database_file(file):
    """Upload a database file to the backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "csv")}
        response = requests.post(f"{API_BASE_URL}/upload-database-file", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading database file: {str(e)}")
        return None

def upload_ground_truth_file(file):
    """Upload a ground truth file to the backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/json")}
        response = requests.post(f"{API_BASE_URL}/upload-ground-truth", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading ground truth file: {str(e)}")
        return None

def get_database_description():
    """Get the database description from the backend"""
    try:
        # This assumes you have an endpoint to get the database description
        # If not, you'll need to modify your backend to include this endpoint
        response = requests.get(f"{API_BASE_URL}/database-description")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching database description: {str(e)}")
        return None

def reset_database():
    """Reset the database"""
    try:
        response = requests.post(f"{API_BASE_URL}/reset-database")
        return response.json()
    except Exception as e:
        st.error(f"Error resetting database: {str(e)}")
        return None
