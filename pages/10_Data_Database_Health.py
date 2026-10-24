"""
Data and Database Health Page.
Displays PostgreSQL connection telemetry, table row counts, live ping tests,
data dictionary definitions, and data quality validation reports.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text

from src.ui_components import inject_custom_css, render_storage_status_banner, render_metric_card
from src.database.connection import check_connection_health, get_engine
from src.data.validate_data import get_data_dictionary, validate_raw_dataframe
from src.data.load_data import load_raw_data

st.set_page_config(page_title="Data & Database Health", page_icon="🏥", layout="wide")
inject_custom_css()

st.markdown("""
<div class="header-banner">
    <h1>Data & Database Telemetry</h1>
    <p>PostgreSQL Connection Health, Table Inventory, Schema Dictionaries & Data Integrity Audits</p>
</div>
""", unsafe_allow_html=True)

render_storage_status_banner()

health = check_connection_health()
engine = get_engine()

# Telemetry KPIs
h1, h2, h3 = st.columns(3)
with h1:
    render_metric_card("Connection Health", health["status"], health["storage_mode"], is_danger=(health["status"] != "HEALTHY"))
with h2:
    render_metric_card("Database Target", health["storage_mode"], health["database_url"])
with h3:
    render_metric_card("Registered Schema Tables", f"{len(health['tables_registered'])} Tables", "All ORM Models Active")

st.markdown("---")

tab_tables, tab_dict, tab_quality = st.tabs(["🗄️ Database Table Inventory", "📖 Data Dictionary", "🔍 Data Quality Report"])

with tab_tables:
    st.subheader("Persistent Tables & Row Counts")
    
    table_rows = []
    if engine is not None and health["status"] == "HEALTHY":
        with engine.connect() as conn:
            for tbl in sorted(health["tables_registered"]):
                try:
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                except Exception:
                    cnt = "Error"
                table_rows.append({"Table Name": tbl, "Row Count": cnt, "Storage Status": "Persistent"})
    else:
        for tbl in sorted(health["tables_registered"]):
            table_rows.append({"Table Name": tbl, "Row Count": "Offline", "Storage Status": "Disconnected"})

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

with tab_dict:
    st.subheader("Standardized Data Dictionary")
    data_dict = get_data_dictionary()
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True, hide_index=True)

with tab_quality:
    st.subheader("Raw Dataset Quality Validation Audit")
    try:
        raw_df = load_raw_data()
        val_res = validate_raw_dataframe(raw_df)

        st.markdown(f"""
        - **Total Records:** `{val_res['row_count']:,}`
        - **Total Columns:** `{val_res['column_count']}`
        - **Duplicate Customer Records:** `{val_res['duplicates']}`
        - **Blank String `TotalCharges` Handled:** `{val_res['blank_total_charges']}`
        - **Target Leakage Fields Identified & Dropped:** `{', '.join(val_res['leakage_columns_detected'])}`
        """)

        if val_res["warnings"]:
            st.info("**Data Pipeline Audit Notes:**\n" + "\n".join([f"- {w}" for w in val_res["warnings"]]))
    except Exception as e:
        st.error(f"Error profiling raw data: {e}")
