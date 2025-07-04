import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from common.visitor_tracker import VisitorTracker
import os

# Set page configuration
st.set_page_config(page_title="Visitor Analytics Dashboard", layout="wide")

# Initialize the visitor tracker with database connection
db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/visitor_db')
tracker = VisitorTracker(db_url=db_url)

def load_visit_data():
    """Load all visit data from the tracker"""
    visits = tracker.get_visits()
    if not visits:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(visits)
    
    # Extract datetime components
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    return df

def main():
    st.title("Visitor Analytics Dashboard")
    
    # Load data
    df = load_visit_data()
    
    if df.empty:
        st.warning("No visitor data available. Start using your app to collect data.")
        return
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Visits", tracker.get_visit_count())
    
    with col2:
        if not df.empty:
            today = datetime.datetime.now().date()
            today_visits = len(df[df['date'] == today])
            st.metric("Today's Visits", today_visits)
    
    with col3:
        if len(df) > 1:
            yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
            yesterday_visits = len(df[df['date'] == yesterday])
            st.metric("Yesterday's Visits", yesterday_visits)
    
    # Time series chart
    st.subheader("Visit Trends")
    
    # Use the tracker's get_daily_counts method to get data directly from the database
    daily_counts = tracker.get_daily_counts(days=30)
    
    if daily_counts:
        # Convert to DataFrame for plotting
        daily_df = pd.DataFrame(daily_counts)
        
        # Convert date strings to datetime
        if isinstance(daily_df['date'].iloc[0], str):
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            
        # Create time series chart
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=daily_df, x='date', y='count', ax=ax)
        ax.set_title('Daily Visits')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Visits')
        st.pyplot(fig)
        
    # Hour of day analysis
    st.subheader("Traffic by Hour of Day")
    
    if not df.empty and 'hour' in df.columns:
        # Group by hour and count visits
        hourly_visits = df.groupby('hour').size().reset_index(name='visits')
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=hourly_visits, x='hour', y='visits', ax=ax)
        ax.set_title('Visits by Hour of Day')
        ax.set_xlabel('Hour (24-hour format)')
        ax.set_ylabel('Number of Visits')
        st.pyplot(fig)
    
    # Day of week analysis
    st.subheader("Traffic by Day of Week")
    
    if not df.empty and 'day_of_week' in df.columns:
        # Define day order for sorting
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Group by day of week and count visits
        day_visits = df.groupby('day_of_week').size().reset_index(name='visits')
        
        # Create categorical type with custom order
        day_visits['day_of_week'] = pd.Categorical(day_visits['day_of_week'], categories=day_order, ordered=True)
        day_visits = day_visits.sort_values('day_of_week')
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=day_visits, x='day_of_week', y='visits', ax=ax)
        ax.set_title('Visits by Day of Week')
        ax.set_xlabel('Day')
        ax.set_ylabel('Number of Visits')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Raw data
    st.subheader("Recent Visits")
    
    if not df.empty:
        # Display most recent visits first
        recent_df = df.sort_values('timestamp', ascending=False).head(50)
        display_df = pd.DataFrame({
            'Time': recent_df['timestamp'],
            'Page': recent_df['page'],
            'Session ID': recent_df['session_id']
        })
        st.dataframe(display_df)

if __name__ == "__main__":
    main()