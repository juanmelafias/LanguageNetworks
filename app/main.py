import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from common.constants import column_mapping,relevant_columns
from common.utilsstreamlit import display_grid, read_plot_info
from common.tracking import log_main
from common.visitor_tracker import VisitorTracker

from loguru import logger


pages = {
    "Syntactic Embeddings": [
        st.Page(page="pages/page_5.py", title = "📊 Compare words in same language"),
        st.Page(page="pages/page_6.py", title = "🗺️ Compare words across languages"),
    ],
    "Mean Language Properties": [
        st.Page(page="pages/page_3.py", title = "📈 Compare vector representations of Language Syntax")
    ],
    "Syntax Network": [
        st.Page(page="pages/page_2.py", title = "🕸️ Syntax Network Plotter")
    ],
    "Glossary": [
        st.Page(page="pages/page_4.py", title = "📄 Glossary")
    ]
}

def run_app():
    pg = (st.navigation(pages))
    pg.run()

if __name__ == "__main__":
    run_app()
