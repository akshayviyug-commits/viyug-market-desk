# Viyug.AI Market Desk

Streamlit app with two pages: **1 Learn** (what recent settled days and the exchange price history show) and
**2 Forecast & Split** (a block-by-block G-DAM / RTM split driven by a price forecast).

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Upload the settlement workbook in the sidebar (read in memory, never written to disk), or use "Load sample workbook"
for a synthetic file. Optional server-side setting for a folder of exchange price exports:
`streamlit run app.py -- --iex-dir <folder>` (users are never asked to upload these).

Tests (synthetic sample only; set MARKET_DESK_REAL_WORKBOOK to also run them on a real workbook):

    python test_price_model.py
    python test_split_price.py
    python test_iex_import.py
