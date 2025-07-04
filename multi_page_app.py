import streamlit as st
import pandas as pd
from common.visitor_tracker import VisitorTracker
import datetime
import os
import uuid
from streamlit.web.server.websocket_headers import get_headers

# Initialize visitor tracker with database connection
db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/visitor_db')
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

def log_page_view(page_name):
    """Log a view for a specific page"""
    user_data = {
        "page": page_name,
        "session_id": get_session_id(),
        "ip_address": get_client_ip(),
        "user_agent": st.session_state.get("user_agent", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "referrer": st.session_state.get("referrer", "")
    }
    return tracker.log_visit(user_data)

def page_home():
    visit_count = log_page_view("home")
    
    st.title("Home Page")
    st.write("Welcome to the home page of our multi-page app!")
    st.metric("Total Visits", visit_count)

def page_about():
    visit_count = log_page_view("about")
    
    st.title("About Page")
    st.write("This is the about page of our application.")
    st.metric("Total Visits", visit_count)

def page_dashboard():
    visit_count = log_page_view("dashboard")
    
    st.title("Dashboard")
    st.write("Here's some interactive content for our dashboard.")
    
    # Display recent visitors for this page only
    recent_visits = tracker.get_visits(limit=5)
    if recent_visits:
        st.subheader("Recent Visitors")
        df = pd.DataFrame([
            {"Time": v["timestamp"].strftime("%Y-%m-%d %H:%M") if v["timestamp"] else "",
             "Page": v["page"] if v["page"] else "unknown"}
            for v in recent_visits
        ])
        st.dataframe(df)

def main():
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Dashboard"])
    
    # Display visitor stats in sidebar
    st.sidebar.title("Visitor Stats")
    st.sidebar.metric("Total Visits", tracker.get_visit_count())
    
    # Display the selected page
    if page == "Home":
        page_home()
    elif page == "About":
        page_about()
    elif page == "Dashboard":
        page_dashboard()

if __name__ == "__main__":
    main()