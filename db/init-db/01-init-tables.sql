-- Create visitor tracking table
CREATE TABLE IF NOT EXISTS visitors (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    page VARCHAR(255),
    session_id VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent TEXT,
    referrer TEXT,
    additional_data JSONB
);

-- Create index for faster querying by timestamp
CREATE INDEX IF NOT EXISTS visitors_timestamp_idx ON visitors (timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE visitors TO postgres;
GRANT USAGE, SELECT ON SEQUENCE visitors_id_seq TO postgres;