import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from common.constants import column_mapping,relevant_columns
from common.utilsstreamlit import display_grid, read_plot_info
from common.tracking import log_main
from common.visitor_tracker import VisitorTracker



pages_4 = {
    "About": [
        st.Page(page="pages/page_4.py", title = "📄 About")
    ],
    "Syntactic Embeddings": [
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
    ],
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ],
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ]
    
}

pages_2 = {
    
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ],
    "About": [
        st.Page(page="pages/page_4.py", title = "📄 About")
    ],
    "Syntactic Embeddings": [
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
    ],
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ]
    
}
pages_3 = {
    
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ],
    "About": [
        st.Page(page="pages/page_4.py", title = "📄 About")
    ],
    "Syntactic Embeddings": [
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
    ],
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ]
    
}
pages_5 = {
    
    "Syntactic Embeddings": [
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
    ],
    "About": [
        st.Page(page="pages/page_4.py", title = "📄 About")
    ],
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ],
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ]
    
}
pages_6 = {
    
    "Syntactic Embeddings": [
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
    ],
    "About": [
        st.Page(page="pages/page_4.py", title = "📄 About")
    ],
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ],
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ]
    
}


def run_app():
    # Get query parameters
    query_params = st.query_params
    page_key = query_params.get("page", "about")
    if page_key == 'page_2':
        pages  = pages_2  # Default to "about" if no query param
    elif page_key == 'page_3':
        pages  = pages_3  # Default to "about" if no query param
    elif page_key == 'page_5':
        pages  = pages_5  # Default to "about" if no query par
    elif page_key == 'page_6':
        pages  = pages_6  # Default to "about" if no query par
    else:
        pages  = pages_4  # Default to "about" if no query param
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    run_app()
