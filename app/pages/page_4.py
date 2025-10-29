import streamlit as st
import os
import re

st.set_page_config(
    page_title="Languages PCA Plotter",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Glossary")

file_path = 'shared/README_app.md'
    
if file_path and os.path.exists(file_path):
    # Read file from path
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Extract image paths from markdown
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images = re.findall(image_pattern, content)
        
        # Split content by images and display alternately
        parts = re.split(image_pattern, content)
        
        for i, part in enumerate(parts):
            if i % 3 == 0:  # Text content
                st.markdown(part)
            elif i % 3 == 2:  # Image path
                if os.path.exists(part):
                    st.image(part, caption=parts[i-1])
    except Exception as e:
        st.error(f"Error reading file: {e}")
