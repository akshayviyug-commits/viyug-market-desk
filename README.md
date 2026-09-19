# Viyug.AI Market Desk

Streamlit app with two pages: **1 Learn** (what recent settled days and the price history in your workbook show) and
**2 Forecast & Split** (a block-by-block G-DAM / RTM split driven by a price forecast trained on that workbook).

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Upload the settlement workbook in the sidebar (read in memory, never written to disk), or use "Load sample workbook"
for a synthetic file. The forecast is built only from the days in the workbook you upload.

Tests (synthetic sample only; set MARKET_DESK_REAL_WORKBOOK to also run them on a real workbook):

    python test_price_model.py
    python test_split_price.py
