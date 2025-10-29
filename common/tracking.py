import streamlit as st
import pandas as pd
import datetime
import os
import uuid
from dotenv import load_dotenv
from urllib.parse import quote_plus
from common.visitor_tracker import VisitorTracker


# Initialize visitor tracker with database connection

def get_client_ip():
    """Try to get the client's IP address from Streamlit context"""
    try:
        # For newer Streamlit versions, use session info
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_session_id'):
            return "streamlit_session"
        return "unknown"
    except:
        return "unknown"

def get_session_id():
    """Get or create a unique session ID for this user session"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def log_main(tracker: VisitorTracker, page_name="home"):
    # Track the visit at the beginning of the app
    user_data = {
        "page": page_name,
        "session_id": get_session_id(),
        "ip_address": get_client_ip(),
        "user_agent": st.session_state.get("user_agent", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "referrer": st.session_state.get("referrer", "")
    }
    
    # Always log the visit - this ensures a new record for each visit
    visit_count = tracker.log_visit(user_data)
    
    # Your main app content

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    db_url = os.environ.get('DATABASE_URL')
    password = os.environ.get('DATABASE_PW')
    password = quote_plus(password)
    conn_str = f"postgresql://postgres.iledvstjxakclwymxwhc:{password}@aws-0-eu-central-2.pooler.supabase.com:6543/postgres?sslmode=require&gssencmode=disable" 
    tracker = VisitorTracker(db_url=conn_str)
    user_data = {
        "page": "home",
        "session_id": get_session_id(),
        "ip_address": 'unknown',
        "user_agent": 'test',
        "timestamp": datetime.datetime.now().isoformat(),
        "referrer": "unknown"
    }
    tracker.log_visit(user_data)