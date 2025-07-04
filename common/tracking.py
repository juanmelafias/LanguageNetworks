import streamlit as st
import pandas as pd
from common.visitor_tracker import VisitorTracker
import datetime
import os
import uuid
from streamlit.web.server.websocket_headers import get_headers
from dotenv import load_dotenv

# Initialize visitor tracker with database connection
db_url = os.environ.get('DATABASE_URL')
password = os.environ.get('DATABASE_PASSWORD')
tracker = VisitorTracker(db_url=db_url)

def get_client_ip():
    """Try to get the client's IP address from headers if available"""
    try:
        headers = get_headers()
        return headers.get("X-Forwarded-For", headers.get("Remote-Addr", "unknown"))
    except:
        return "unknown"

def get_session_id():
    """Get or create a unique session ID for this user session"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def main():
    # Track the visit at the beginning of the app
    user_data = {
        "page": "home",
        "session_id": get_session_id(),
        "ip_address": get_client_ip(),
        "user_agent": st.session_state.get("user_agent", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "referrer": st.session_state.get("referrer", "")
    }
    
    # Always log the visit - this ensures a new record for each visit
    visit_count = tracker.log_visit(user_data)
    
    # Your main app content
    st.title("My Streamlit App")
    st.write("Welcome to my application!")
    
    # Display visitor stats
    st.sidebar.title("Visitor Stats")
    st.sidebar.metric("Total Visits", visit_count)
    
    # Show recent visitors (optional)
    if st.sidebar.checkbox("Show recent visitors"):
        recent_visits = tracker.get_visits(limit=10)
        if recent_visits:
            df = pd.DataFrame([
                {"Time": v["timestamp"].strftime("%Y-%m-%d %H:%M") if v["timestamp"] else "",
                 "Page": v["page"] if v["page"] else "unknown"}
                for v in recent_visits
            ])
            st.sidebar.dataframe(df)

# For multi-page apps, use this function
def log_page_view(page_name):
    """Log a view for a specific page in multi-page apps"""
    user_data = {
        "page": page_name,
        "session_id": get_session_id(),
        "ip_address": get_client_ip(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    tracker.log_visit(user_data)
    return True

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    db_url = os.environ.get('DATABASE_URL')
    password = os.environ.get('DATABASE_PASSWORD')
    conn_str = f"postgresql://postgres:{password}@{db_url}:5432/postgres"
    tracker = VisitorTracker(db_url=db_url)
    print(tracker._get_connection())