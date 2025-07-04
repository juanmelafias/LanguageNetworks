import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

class VisitorTracker:
    def __init__(self, db_url=None):
        """Initialize the visitor tracker with database connection info."""
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable or db_url parameter is required")
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.db_url)
        
    def log_visit(self, user_data=None):
        """
        Log a visit with timestamp and optional user data.
        
        Parameters:
        - user_data: dict, optional user information like IP address, user agent, etc.
        
        Returns:
        - int: total number of visits
        """
        user_data = user_data or {}
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Insert the visit record - always create a new record for each visit
                    cur.execute(
                        """
                        INSERT INTO visitors 
                        (page, session_id, ip_address, user_agent, referrer, additional_data)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            user_data.get('page'),
                            user_data.get('session_id'),
                            user_data.get('ip_address'),
                            user_data.get('user_agent'),
                            user_data.get('referrer'),
                            json.dumps(user_data) if user_data else None
                        )
                    )
                    visit_id = cur.fetchone()[0]
                    
                    # Get the total count
                    cur.execute("SELECT COUNT(*) FROM visitors")
                    total_visits = cur.fetchone()[0]
                    
                    return total_visits
        except Exception as e:
            print(f"Error logging visit: {e}")
            return None
    
    def get_visit_count(self):
        """Get the total number of visits."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM visitors")
                    return cur.fetchone()[0]
        except Exception as e:
            print(f"Error getting visit count: {e}")
            return 0

    def get_page_visit_count(self, page_name):
        """Get the visit count for a specific page."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM visitors WHERE page = %s", (page_name,))
                    return cur.fetchone()[0]
        except Exception as e:
            print(f"Error getting page visit count: {e}")
            return 0
    
    def get_visits(self, limit=None, page=None):
        """
        Get the visit records.
        
        Parameters:
        - limit: int, optional number of records to return (most recent first)
        - page: str, optional filter by page name
        
        Returns:
        - list: visit records
        """
        query = "SELECT * FROM visitors"
        params = []
        
        if page:
            query += " WHERE page = %s"
            params.append(page)
            
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
            
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    visits = cur.fetchall()
                    return visits
        except Exception as e:
            print(f"Error getting visits: {e}")
            return []
            
    def get_visits_by_date_range(self, start_date, end_date, page=None):
        """
        Get visits within a date range.
        
        Parameters:
        - start_date: datetime.date or string (YYYY-MM-DD)
        - end_date: datetime.date or string (YYYY-MM-DD)
        - page: str, optional filter by page name
        
        Returns:
        - list: visit records
        """
        query = "SELECT * FROM visitors WHERE timestamp::date BETWEEN %s AND %s"
        params = [start_date, end_date]
        
        if page:
            query += " AND page = %s"
            params.append(page)
            
        query += " ORDER BY timestamp DESC"
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        except Exception as e:
            print(f"Error getting visits by date range: {e}")
            return []
            
    def get_daily_counts(self, days=30, page=None):
        """
        Get daily visit counts for the last N days.
        
        Parameters:
        - days: int, number of days to include
        - page: str, optional filter by page name
        
        Returns:
        - list: daily counts
        """
        query = """
            SELECT 
                timestamp::date AS date, 
                COUNT(*) AS count
            FROM visitors
            WHERE timestamp >= NOW() - INTERVAL '%s days'
        """
        params = [days]
        
        if page:
            query += " AND page = %s"
            params.append(page)
            
        query += " GROUP BY date ORDER BY date"
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        except Exception as e:
            print(f"Error getting daily counts: {e}")
            return []
            
    def get_unique_visitors_count(self, days=None):
        """
        Get count of unique visitors (by session_id)
        
        Parameters:
        - days: int, optional number of days to limit results (None for all time)
        
        Returns:
        - int: count of unique visitors
        """
        query = "SELECT COUNT(DISTINCT session_id) FROM visitors"
        params = []
        
        if days:
            query += " WHERE timestamp >= NOW() - INTERVAL '%s days'"
            params.append(days)
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()[0]
        except Exception as e:
            print(f"Error getting unique visitors count: {e}")
            return 0