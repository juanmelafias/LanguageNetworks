#! /bin/bash

cd /src || exit

echo "[INFO]: Running Streamlit App"
export DATABASE_PW="vE0Mj8tx'ND-~fef4%&B"
python -m streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0