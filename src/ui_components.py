"""
Shared UI Styling and Component Library for Streamlit Application.
Provides a cohesive, modern visual theme, metric cards, risk badges, and navigation headers.
"""

import streamlit as st
from src.database.connection import get_active_storage_mode


def inject_custom_css():
    """Injects modern, responsive styling with clean glassmorphism and subtle gradients."""
    css = """
    <style>
    /* Global Container */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Banners */
    .header-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-banner h1 {
        color: white !important;
        margin: 0 0 6px 0;
        font-size: 26px;
        font-weight: 700;
    }
    .header-banner p {
        color: #E0E7FF;
        margin: 0;
        font-size: 14px;
    }

    /* Metric Card */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-subtext {
        font-size: 12px;
        color: #10B981;
        font-weight: 500;
    }
    .metric-subtext-danger {
        font-size: 12px;
        color: #EF4444;
        font-weight: 500;
    }

    /* Badges */
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    /* Storage Indicator */
    .storage-banner {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_storage_status_banner():
    """Renders persistent active database storage banner across all pages."""
    mode, msg = get_active_storage_mode()
    if mode == "POSTGRESQL":
        status_html = f"""
        <div class="storage-banner">
            <div>
                <strong>Active Persistence Layer:</strong> <span style="color:#166534; font-weight:700;">PostgreSQL 18 (Online)</span>
            </div>
            <div style="font-size:12px; color:#475569;">
                Source: <code>DATABASE_URL</code>
            </div>
        </div>
        """
    else:
        status_html = f"""
        <div class="storage-banner" style="background-color:#FEF2F2; border-color:#FECACA;">
            <div>
                <strong>Active Persistence Layer:</strong> <span style="color:#DC2626; font-weight:700;">PostgreSQL Offline</span>
                <span style="font-size:12px; color:#991B1B; margin-left:8px;">(Action required: verify DATABASE_URL and ensure PostgreSQL is running)</span>
            </div>
            <div style="font-size:12px; color:#DC2626;">
                Status: Disconnected
            </div>
        </div>
        """
    st.markdown(status_html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", is_danger: bool = False):
    """Renders polished metric card."""
    sub_class = "metric-subtext-danger" if is_danger else "metric-subtext"
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{sub_class}">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
