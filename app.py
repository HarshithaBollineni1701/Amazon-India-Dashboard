"""
AMAZON INDIA | AI BUSINESS INTELLIGENCE
======================================
A polished Streamlit dashboard using the user's AI skill set:
- LLM integration
- Retrieval-Augmented Generation (RAG)
- Role-based multi-agent orchestration
- Prompt engineering and grounded responses

No traditional ML model (Scikit-learn / TensorFlow / PyTorch) is used.
Pandas is used only for data processing and KPI calculations.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Dataset:
    Keep "Amazon Sales Data India.xlsx" beside app.py, or upload it
    using the sidebar.

AI setup:
    Add an OpenAI API key in the sidebar, or set OPENAI_API_KEY.
    The dashboard and charts work without an API key. The AI copilot
    requires a key and internet access.
"""

from pathlib import Path
import os
import re
import json
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# -------------------------------------------------------------------
# APP CONFIG + DESIGN
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon India | AI Business Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: #101a32; }
[data-testid="stSidebar"] * { color: #edf3ff !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #17233d !important; }
[data-testid="stSidebar"] input { color: #17233d !important; }
.hero {
  background: linear-gradient(115deg, #111d3a 0%, #172f59 55%, #285b91 100%);
  padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 18px;
  box-shadow: 0 12px 30px rgba(16,32,65,.15);
}
.hero-kicker { color: #a9d5ff; text-transform: uppercase; letter-spacing: 2px;
  font-size: 11px; font-weight: 700; }
.hero h1 { font-family: 'Manrope', sans-serif; color: white; font-size: 31px;
  line-height: 1.2; margin: 7px 0 8px 0; font-weight: 800; }
.hero p { color: #d9e8fb; margin: 0; font-size: 14px; }
.section-title { font-family: 'Manrope', sans-serif; color: #14213d;
  font-size: 21px; font-weight: 800; margin: 8px 0 3px 0; }
.section-sub { color: #6b7890; font-size: 13px; margin-bottom: 14px; }
.kpi {
  background: white; border: 1px solid #e7edf5; border-radius: 17px;
  padding: 16px 17px; box-shadow: 0 4px 16px rgba(23,43,77,.045);
  min-height: 105px;
}
.kpi-label { color: #77849a; font-size: 11px; font-weight: 700;
  letter-spacing: .7px; text-transform: uppercase; }
.kpi-value { color: #14213d; font-family: 'Manrope',sans-serif;
  font-size: 25px; font-weight: 800; margin-top: 7px; }
.kpi-note { color: #8290a5; font-size: 11px; margin-top: 2px; }
.panel {
  background: white; border: 1px solid #e7edf5; border-radius: 18px;
  padding: 18px 20px; box-shadow: 0 4px 16px rgba(23,43,77,.04);
}
div[data-testid="stTabs"] button { font-weight: 700; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: #2158a6; border-bottom-color: #3478d4;
}
div[data-testid="stMetric"] {
  background: white; border: 1px solid #e7edf5; padding: 13px 15px;
  border-radius: 15px; box-shadow: 0 4px 14px rgba(23,43,77,.04);
}
div[data-testid="stMetricLabel"] p { color: #718099; font-size: 12px; }
div[data-testid="stMetricValue"] { color: #14213d; }
div.stButton > button {
  border-radius: 11px; border: 0; font-weight: 700;
  background: #235ca8; color: white; padding: .55rem 1rem;
}
div.stButton > button:hover { background: #17457f; color: white; }
div[data-testid="stAlert"] { border-radius: 13px; }
.small-muted { color: #8190a6; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef6685",
          "#0ea5e9", "#84cc16", "#f97316"]
CHART_TEMPLATE = "plotly_white"

REQUIRED = [
    "Order_ID", "Order_Date", "Category", "Product", "Quantity",
    "Unit_Price_INR", "Discount_Pct", "Total_Sales_INR", "Profit_INR",
    "Payment_Method", "Fulfillment", "Order_Status", "Ship_State",
]


# -------------------------------------------------------------------
# DATA INGESTION + CLEANING
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes = None) -> pd.DataFrame:
    if file_bytes:
        from io import BytesIO
        raw = pd.read_excel(BytesIO(file_bytes))
    else:
        path = Path(__file__).with_name("Amazon Sales Data India.xlsx")
        if not path.exists():
            return pd.DataFrame()
        raw = pd.read_excel(path)

    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing required dataset columns: " + ", ".join(missing))

    data = raw[REQUIRED].copy()
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
    for col in ["Quantity", "Unit_Price_INR", "Discount_Pct",
                "Total_Sales_INR", "Profit_INR"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "Order_ID", "Order_Date", "Category", "Product",
        "Quantity", "Total_Sales_INR", "Profit_INR"
    ])
    data["Month"] = data["Order_Date"].dt.to_period("M").astype(str)
    data["Month_Start"] = data["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return data


def inr(value):
    if pd.isna(value):
        return "—"
    value = float(value)
    absolute = abs(value)
    if absolute >= 1e7:
        return f"₹{value/1e7:,.2f} Cr"
    if absolute >= 1e5:
        return f"₹{value/1e5:,.2f} L"
    return f"₹{value:,.0f}"


def pct(n, d):
    return (float(n) / float(d) * 100) if d else 0.0


def section(title: str, subtitle: str):
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def kpi_card(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def polish(fig, height=330):
    fig.update_layout(
        template=CHART_TEMPLATE, height=height,
        margin=dict(l=12, r=12, t=52, b=12),
        font=dict(family="DM Sans", color="#34425b", size=12),
        title=dict(font=dict(size=15, color="#17233d")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e8edf4")
    fig.update_yaxes(gridcolor="#edf1f6", zeroline=False)
    return fig


# -------------------------------------------------------------------
# RAG: BUILD RETRIEVABLE BUSINESS KNOWLEDGE FROM THE WORKBOOK
# -------------------------------------------------------------------
def build_knowledge_base(data: pd.DataFrame) -> List[Dict[str, str]]:
    """Create small, source-grounded text documents from the selected data."""
    docs = []

    total_sales = data["Total_Sales_INR"].sum()
    total_profit = data["Profit_INR"].sum()
    total_orders = data["Order_ID"].nunique()
    docs.append({
        "topic": "overall business performance",
        "text": (
            f"Overall performance for the selected data: {total_orders} unique orders; "
            f"recorded sales INR {total_sales:.2f}; recorded profit INR {total_profit:.2f}; "
            f"units {data['Quantity'].sum():.0f}; average order value "
            f"INR {(total_sales / total_orders if total_orders else 0):.2f}; "
            f"profit margin {pct(total_profit, total_sales):.2f}%."
        )
    })

    for category, group in data.groupby("Category"):
        docs.append({
            "topic": f"category {category}",
            "text": (
                f"Category {category}: recorded sales INR {group['Total_Sales_INR'].sum():.2f}; "
                f"profit INR {group['Profit_INR'].sum():.2f}; units {group['Quantity'].sum():.0f}; "
                f"unique orders {group['Order_ID'].nunique()}; "
                f"products: {', '.join(group['Product'].value_counts().head(8).index.astype(str))}."
            )
        })

    for product, group in data.groupby("Product"):
        docs.append({
            "topic": f"product {product}",
            "text": (
                f"Product {product}, category {group['Category'].mode().iloc[0]}: "
                f"recorded sales INR {group['Total_Sales_INR'].sum():.2f}; "
                f"profit INR {group['Profit_INR'].sum():.2f}; "
                f"units {group['Quantity'].sum():.0f}; "
                f"orders {group['Order_ID'].nunique()}."
            )
        })

    for state, group in data.groupby("Ship_State"):
        docs.append({
            "topic": f"state {state}",
            "text": (
                f"Shipping state {state}: recorded sales INR {group['Total_Sales_INR'].sum():.2f}; "
                f"profit INR {group['Profit_INR'].sum():.2f}; "
                f"orders {group['Order_ID'].nunique()}."
            )
        })

    for status, group in data.groupby("Order_Status"):
        docs.append({
            "topic": f"order status {status}",
            "text": (
                f"Order status {status}: {group['Order_ID'].nunique()} unique orders; "
                f"recorded sales INR {group['Total_Sales_INR'].sum():.2f}; "
                f"recorded profit INR {group['Profit_INR'].sum():.2f}."
            )
        })

    for payment, group in data.groupby("Payment_Method"):
        docs.append({
            "topic": f"payment method {payment}",
            "text": (
                f"Payment method {payment}: recorded sales INR {group['Total_Sales_INR'].sum():.2f}; "
                f"profit INR {group['Profit_INR'].sum():.2f}; "
                f"orders {group['Order_ID'].nunique()}."
            )
        })

    for fulfilment, group in data.groupby("Fulfillment"):
        docs.append({
            "topic": f"fulfillment {fulfilment}",
            "text": (
                f"Fulfillment method {fulfilment}: recorded sales INR "
                f"{group['Total_Sales_INR'].sum():.2f}; profit INR "
                f"{group['Profit_INR'].sum():.2f}; orders {group['Order_ID'].nunique()}."
            )
        })

    monthly = data.groupby("Month", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
        Orders=("Order_ID", "nunique"),
    )
    for _, row in monthly.iterrows():
        docs.append({
            "topic": f"month {row['Month']}",
            "text": (
                f"Month {row['Month']}: recorded sales INR {row['Sales']:.2f}; "
                f"profit INR {row['Profit']:.2f}; orders {row['Orders']}."
            )
        })
    return docs


def retrieve_documents(question: str, docs: List[Dict[str, str]], top_k=6):
    """
    Lightweight lexical retrieval for this structured dataset.
    This is the retrieval step in RAG; it does not use embeddings.
    """
    stop_words = {
        "what", "which", "where", "when", "how", "much", "many", "the",
        "and", "for", "are", "was", "were", "with", "from", "that", "this",
        "show", "tell", "about", "give", "can", "you", "our", "does", "did",
        "have", "has", "into", "over", "under", "compare", "please"
    }
    terms = [
        t for t in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if t not in stop_words and len(t) > 1
    ]
    if not terms:
        return docs[:top_k]

    scored = []
    for doc in docs:
        haystack = (doc["topic"] + " " + doc["text"]).lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]] or docs[:top_k]


# -------------------------------------------------------------------
# MULTI-AGENT ORCHESTRATION + LLM
# -------------------------------------------------------------------
def route_question(question: str) -> str:
    """Simple transparent router selects a specialist agent by user intent."""
    q = question.lower()
    if any(w in q for w in ["return", "cancel", "delivered", "shipped", "status", "lost"]):
        return "Order Intelligence Agent"
    if any(w in q for w in ["product", "category", "item", "best seller", "top selling"]):
        return "Product Intelligence Agent"
    if any(w in q for w in ["state", "region", "geography", "location", "payment", "fulfillment"]):
        return "Channel & Geography Agent"
    return "Business Performance Agent"


def ask_llm(question, retrieved_docs, api_key, model, base_url):
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing. Install requirements.txt again.")
    if not api_key:
        raise RuntimeError("Add your API key in the sidebar to use the AI copilot.")

    client_args = {"api_key": api_key}
    if base_url.strip():
        client_args["base_url"] = base_url.strip()
    client = OpenAI(**client_args)

    specialist = route_question(question)
    evidence = "\n\n".join(
        f"[{doc['topic']}] {doc['text']}" for doc in retrieved_docs
    )
    if not evidence:
        evidence = "No relevant evidence was retrieved from the selected dataset."

    # Agent 1: specialist analysis grounded in retrieved workbook evidence.
    analyst_prompt = f"""
You are the {specialist}. Analyze the user's question using ONLY the
retrieved evidence below. The evidence comes from the currently selected
Amazon India workbook rows and filters.

Rules:
- Do not invent figures, causes, customer details, or missing columns.
- Say when the evidence is insufficient.
- Clearly call sales "recorded sales" because the source includes multiple
  order statuses; do not automatically call all sales realized revenue.
- Use INR and concise language.
- Separate facts from recommendations.

RETRIEVED EVIDENCE:
{evidence}

USER QUESTION:
{question}

Write a concise analysis with: Answer, Evidence, and Limitations.
"""
    first = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful data analyst. Ground every claim in evidence."},
            {"role": "user", "content": analyst_prompt},
        ],
        temperature=0.2,
    )
    analysis = first.choices[0].message.content or ""

    # Agent 2: executive recommendation agent turns analysis into actions.
    recommendation_prompt = f"""
You are an executive recommendation agent supporting a non-technical manager.
Use the analyst's response below. Do not add new facts or unsupported numbers.
Give:
1. A one-sentence management takeaway.
2. Up to three practical next steps.
3. One data limitation to keep in mind.

ANALYST RESPONSE:
{analysis}
"""
    second = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You translate grounded analysis into clear business decisions."},
            {"role": "user", "content": recommendation_prompt},
        ],
        temperature=0.3,
    )
    recommendations = second.choices[0].message.content or ""
    return specialist, analysis, recommendations


# -------------------------------------------------------------------
# SIDEBAR: DATA, FILTERS, AI CONFIG
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📦 Amazon India")
    st.caption("AI Business Intelligence")
    st.markdown("---")

    uploaded = st.file_uploader("Upload sales workbook", type=["xlsx"])
    upload_bytes = uploaded.getvalue() if uploaded else None

try:
    data = load_data(upload_bytes)
except Exception as error:
    st.error(f"Dataset could not be loaded: {error}")
    st.stop()

if data.empty:
    st.warning(
        "Dataset not found. Place 'Amazon Sales Data India.xlsx' beside app.py "
        "or upload the workbook in the sidebar."
    )
    st.stop()

with st.sidebar:
    st.success(f"{len(data):,} records loaded")
    min_day = data["Order_Date"].min().date()
    max_day = data["Order_Date"].max().date()
    date_range = st.date_input(
        "Order date range",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
    )
    all_categories = sorted(data["Category"].dropna().unique())
    selected_categories = st.multiselect(
        "Categories", all_categories, default=all_categories
    )
    all_states = sorted(data["Ship_State"].dropna().unique())
    selected_states = st.multiselect("States", all_states, default=all_states)

    st.markdown("---")
    st.markdown("### 🤖 AI Copilot setup")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "API key", value=env_key, type="password",
        help="Optional. Dashboard charts work without a key; AI answers need an API key."
    )
    model_name = st.text_input("Model", value="gpt-4o-mini")
    base_url = st.text_input(
        "Base URL (optional)", value="",
        help="Leave blank for the default OpenAI endpoint. Use only if your provider supports the OpenAI SDK format."
    )
    st.caption("AI requests send retrieved aggregate workbook information to your chosen LLM provider.")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    date_from, date_to = date_range
else:
    date_from, date_to = min_day, max_day

view = data[
    (data["Order_Date"].dt.date >= date_from)
    & (data["Order_Date"].dt.date <= date_to)
    & (data["Category"].isin(selected_categories))
    & (data["Ship_State"].isin(selected_states))
].copy()

if view.empty:
    st.warning("No rows match these filters. Update the sidebar filters.")
    st.stop()


# -------------------------------------------------------------------
# HEADER + KPI STRIP
# -------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Sapphire IQ · Business intelligence project</div>
  <h1>Amazon India Sales Intelligence</h1>
  <p>Understand performance, explore business patterns, and ask questions
  using a retrieval-augmented AI copilot.</p>
</div>
""", unsafe_allow_html=True)

sales = view["Total_Sales_INR"].sum()
profit = view["Profit_INR"].sum()
orders = view["Order_ID"].nunique()
units = view["Quantity"].sum()
aov = sales / orders if orders else 0
margin = pct(profit, sales)
delivered_rows = view[view["Order_Status"].str.lower() == "delivered"]
delivered_sales = delivered_rows["Total_Sales_INR"].sum()

kpis = st.columns(6)
with kpis[0]: kpi_card("Recorded sales", inr(sales), "All order statuses")
with kpis[1]: kpi_card("Recorded profit", inr(profit), "From workbook")
with kpis[2]: kpi_card("Unique orders", f"{orders:,}", "Distinct Order_ID")
with kpis[3]: kpi_card("Units", f"{units:,.0f}", "Quantity total")
with kpis[4]: kpi_card("Average order value", inr(aov), "Sales ÷ unique orders")
with kpis[5]: kpi_card("Profit margin", f"{margin:.1f}%", "Profit ÷ recorded sales")

st.markdown(
    f'<div class="small-muted" style="margin:10px 2px 18px 2px;">'
    f'Filter view: {date_from} to {date_to} · '
    f'Delivered-order recorded sales: {inr(delivered_sales)}. '
    f'Recorded sales include every order status and are not automatically realized revenue.'
    f'</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------------
tabs = st.tabs([
    "📈 Overview",
    "🛍️ Products",
    "📦 Orders",
    "🌍 Channels & Regions",
    "✨ AI Copilot",
    "🧾 Data Explorer",
])

# OVERVIEW
with tabs[0]:
    section("Business performance", "A quick view of sales, profit and order activity.")
    monthly = view.groupby("Month_Start", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
        Orders=("Order_ID", "nunique"),
        Units=("Quantity", "sum"),
    ).sort_values("Month_Start")

    left, right = st.columns([1.65, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month_Start"], y=monthly["Sales"],
            mode="lines+markers", name="Recorded sales",
            line=dict(color="#2563eb", width=3),
            fill="tozeroy", fillcolor="rgba(37,99,235,.08)",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month_Start"], y=monthly["Profit"],
            mode="lines+markers", name="Recorded profit",
            line=dict(color="#14b8a6", width=3),
        ))
        fig.update_layout(title="Monthly sales & profit trend",
                          yaxis_title="INR", hovermode="x unified")
        st.plotly_chart(polish(fig, 370), use_container_width=True)
    with right:
        status = view.groupby("Order_Status", as_index=False).agg(
            Orders=("Order_ID", "nunique")
        )
        fig = px.pie(status, names="Order_Status", values="Orders",
                     hole=.62, color_discrete_sequence=COLORS)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(title="Order status mix", showlegend=False)
        st.plotly_chart(polish(fig, 370), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(monthly, x="Month_Start", y="Orders",
                     title="Monthly order volume", color_discrete_sequence=["#8b5cf6"])
        st.plotly_chart(polish(fig, 300), use_container_width=True)
    with c2:
        fig = px.bar(monthly, x="Month_Start", y="Units",
                     title="Monthly units sold", color_discrete_sequence=["#f59e0b"])
        st.plotly_chart(polish(fig, 300), use_container_width=True)

# PRODUCTS
with tabs[1]:
    section("Category & product performance",
            "Find the categories and products contributing most to recorded sales and profit.")
    cat = view.groupby("Category", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
        Units=("Quantity", "sum"),
    ).sort_values("Sales", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(cat.sort_values("Sales"), x="Sales", y="Category",
                     orientation="h", title="Recorded sales by category",
                     color="Sales", color_continuous_scale=["#bfdbfe", "#1d4ed8"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig, 340), use_container_width=True)
    with col2:
        fig = px.bar(cat.sort_values("Profit"), x="Profit", y="Category",
                     orientation="h", title="Recorded profit by category",
                     color="Profit", color_continuous_scale=["#a7f3d0", "#0f766e"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig, 340), use_container_width=True)

    product = view.groupby(["Category", "Product"], as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
        Units=("Quantity", "sum"),
        Orders=("Order_ID", "nunique"),
    ).sort_values("Sales", ascending=False)
    top10 = product.head(10).sort_values("Sales")
    fig = px.bar(top10, x="Sales", y="Product", color="Category",
                 orientation="h", title="Top 10 products by recorded sales",
                 color_discrete_sequence=COLORS)
    st.plotly_chart(polish(fig, 430), use_container_width=True)
    st.dataframe(cat, use_container_width=True, hide_index=True)

# ORDERS
with tabs[2]:
    section("Order status & revenue exposure",
            "Understand delivery progress, returns and cancellations.")
    status = view.groupby("Order_Status", as_index=False).agg(
        Orders=("Order_ID", "nunique"),
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
    )
    status_map = dict(zip(status["Order_Status"], status["Orders"]))
    return_count = status_map.get("Returned", 0)
    cancel_count = status_map.get("Cancelled", 0)
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi_card("Delivered", f"{status_map.get('Delivered', 0):,}", "Unique orders")
    with m2: kpi_card("Shipped", f"{status_map.get('Shipped', 0):,}", "Unique orders")
    with m3: kpi_card("Return rate", f"{pct(return_count, orders):.1f}%", f"{return_count:,} returned")
    with m4: kpi_card("Cancellation rate", f"{pct(cancel_count, orders):.1f}%", f"{cancel_count:,} cancelled")

    l, r = st.columns(2)
    with l:
        fig = px.bar(status, x="Order_Status", y="Orders",
                     color="Order_Status", title="Orders by status",
                     color_discrete_sequence=COLORS)
        st.plotly_chart(polish(fig, 330), use_container_width=True)
    with r:
        fig = px.bar(status, x="Order_Status", y="Sales",
                     color="Order_Status", title="Recorded sales by status",
                     color_discrete_sequence=COLORS)
        st.plotly_chart(polish(fig, 330), use_container_width=True)

    risk = view[view["Order_Status"].isin(["Returned", "Cancelled"])]
    risk_cat = risk.groupby(["Category", "Order_Status"], as_index=False).agg(
        Orders=("Order_ID", "nunique")
    )
    fig = px.bar(risk_cat, x="Category", y="Orders", color="Order_Status",
                 barmode="group", title="Returns & cancellations by category",
                 color_discrete_sequence=["#f59e0b", "#ef6685"])
    st.plotly_chart(polish(fig, 330), use_container_width=True)
    st.dataframe(status, use_container_width=True, hide_index=True)

# CHANNELS + GEOGRAPHY
with tabs[3]:
    section("Payment, fulfillment & geographic performance",
            "Compare the channels and shipping states represented in the workbook.")
    payment = view.groupby("Payment_Method", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum")
    )
    fulfil = view.groupby("Fulfillment", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"), Profit=("Profit_INR", "sum")
    )
    geo = view.groupby("Ship_State", as_index=False).agg(
        Sales=("Total_Sales_INR", "sum"),
        Profit=("Profit_INR", "sum"),
        Orders=("Order_ID", "nunique"),
    ).sort_values("Sales", ascending=False)

    a, b = st.columns(2)
    with a:
        fig = px.bar(payment, x="Payment_Method", y="Sales",
                     title="Recorded sales by payment method",
                     color_discrete_sequence=["#2563eb"])
        st.plotly_chart(polish(fig, 320), use_container_width=True)
        fig = px.bar(payment, x="Payment_Method", y="Profit",
                     title="Profit by payment method",
                     color_discrete_sequence=["#14b8a6"])
        st.plotly_chart(polish(fig, 320), use_container_width=True)
    with b:
        fig = px.bar(fulfil, x="Fulfillment", y="Sales",
                     title="Recorded sales by fulfillment",
                     color_discrete_sequence=["#8b5cf6"])
        st.plotly_chart(polish(fig, 320), use_container_width=True)
        fig = px.bar(fulfil, x="Fulfillment", y="Profit",
                     title="Profit by fulfillment",
                     color_discrete_sequence=["#f59e0b"])
        st.plotly_chart(polish(fig, 320), use_container_width=True)

    fig = px.bar(geo.sort_values("Sales"), x="Sales", y="Ship_State",
                 orientation="h", title="State-wise recorded sales",
                 color="Sales", color_continuous_scale=["#bfdbfe", "#1d4ed8"])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(polish(fig, 440), use_container_width=True)
    st.dataframe(geo, use_container_width=True, hide_index=True)

# AI COPILOT
with tabs[4]:
    section("Ask the Amazon India AI copilot",
            "A RAG workflow retrieves relevant evidence from the filtered workbook before the LLM responds.")
    st.markdown("""
    <div class="panel">
      <b>How the AI workflow works</b><br>
      <span style="color:#64748b;font-size:13px;">
      Your question → Intent router → Specialist analyst agent → RAG evidence retrieval
      → LLM grounded analysis → Executive recommendation agent
      </span>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history_amazon" not in st.session_state:
        st.session_state.chat_history_amazon = []

    for item in st.session_state.chat_history_amazon:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input(
        "Ask about sales, profit, products, returns, states or payment methods..."
    )
    if prompt:
        st.session_state.chat_history_amazon.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key:
                st.warning(
                    "Add an API key in the sidebar to activate the LLM copilot. "
                    "The dashboard and analytics remain available without it."
                )
            else:
                with st.spinner("Retrieving workbook evidence and coordinating AI agents..."):
                    try:
                        knowledge = build_knowledge_base(view)
                        retrieved = retrieve_documents(prompt, knowledge, top_k=7)
                        agent_name, analysis, recommendations = ask_llm(
                            prompt, retrieved, api_key, model_name, base_url
                        )
                        st.caption(f"Routed to: {agent_name}")
                        st.markdown("#### Analyst agent")
                        st.markdown(analysis)
                        st.markdown("#### Executive recommendation agent")
                        st.markdown(recommendations)
                        st.markdown("#### Retrieved evidence")
                        for doc in retrieved:
                            st.caption(f"**{doc['topic']}** — {doc['text']}")
                        answer_text = (
                            f"**Routed to:** {agent_name}\n\n"
                            f"**Analyst agent**\n\n{analysis}\n\n"
                            f"**Executive recommendation agent**\n\n{recommendations}"
                        )
                        st.session_state.chat_history_amazon.append(
                            {"role": "assistant", "content": answer_text}
                        )
                    except Exception as error:
                        st.error(f"AI request failed: {error}")

    if st.button("Clear AI conversation"):
        st.session_state.chat_history_amazon = []
        st.rerun()

    st.caption(
        "Privacy note: when you submit a question, retrieved aggregate information "
        "from the selected dataset is sent to the LLM provider configured by you. "
        "Do not enter personal or confidential information."
    )

# DATA EXPLORER
with tabs[5]:
    section("Explore the source data",
            "Inspect the filtered records, check data quality and export a CSV.")
    d1, d2, d3, d4 = st.columns(4)
    with d1: kpi_card("Rows in source", f"{len(data):,}")
    with d2: kpi_card("Rows after filters", f"{len(view):,}")
    with d3: kpi_card("Missing cells", f"{int(data.isna().sum().sum()):,}")
    with d4: kpi_card("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download filtered records (CSV)",
        data=csv_bytes,
        file_name="amazon_india_filtered_records.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="small-muted" style="text-align:center;padding:22px 0 5px 0;">'
    'Amazon India AI Business Intelligence · Learning portfolio project · '
    'Insights are limited to fields available in the supplied workbook.'
    '</div>',
    unsafe_allow_html=True,
)
