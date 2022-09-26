#! /bin/bash

cd /src || exit

echo "[INFO]: Running Streamlit App"
python -m streamlit run app/main.py --server.port $PORT