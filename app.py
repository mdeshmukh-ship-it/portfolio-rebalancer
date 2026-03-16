"""
Portfolio Rebalancer — Streamlit App
=====================================
Rebuilt from the Hex project YAML. Provides a self-serve, always-fresh
portfolio drift & rebalancing tool backed by BigQuery + Yahoo Finance.
"""

import streamlit as st
import pandas as pd
import requests
import urllib3
import os
from datetime import datetime, date, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

# Suppress SSL warnings for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────── page config ───────────────────────────
st.set_page_config(
    page_title="Portfolio Rebalancer",
    page_icon="⚖️",
    layout="wide",
)

# ─────────────────────────── constants ─────────────────────────────
BQ_PROJECT = "perennial-data-prod"
ACCOUNTS_TABLE = f"{BQ_PROJECT}.fidelity.accounts"
POSITIONS_TABLE = f"{BQ_PROJECT}.fidelity.daily_positions"
TARGETS_TABLE = f"{BQ_PROJECT}.rebalancer.portfolio_targets"

ASSET_CLASSES = ["Equity", "Fixed Income", "Cash", "Crypto"]

ASSET_CLASS_ICONS = {
    "Equity": "📈",
    "Fixed Income": "🏛️",
    "Cash": "💵",
    "Crypto": "🪙",
}


# ─────────────────────────── credentials ───────────────────────────
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcp-key.json")


# ─────────────────────────── helpers ───────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bq_client() -> bigquery.Client:
    """Return a BigQuery client. Uses Streamlit secrets (cloud) or local key file."""
    try:
        # Streamlit Community Cloud: credentials stored in st.secrets
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
    except (KeyError, FileNotFoundError):
        # Local development: use gcp-key.json file
        credentials = service_account.Credentials.from_service_account_file(
            KEY_FILE,
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
    return bigquery.Client(project=BQ_PROJECT, credentials=credentials)


def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Execute *sql* against BigQuery and return a DataFrame."""
    client = get_bq_client()
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return client.query(sql, job_config=job_config).to_dataframe()


@st.cache_data(ttl=300, show_spinner="Fetching stock prices …")
def get_ticker_prices(tickers: tuple) -> dict[str, float]:
    """Fetch current prices for *tickers* via Yahoo Finance chart API."""
    prices: dict[str, float] = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for t in tickers:
        if not t:
            continue
        try:
            symbol = t.strip().upper()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            resp = requests.get(url, headers=headers, verify=False, timeout=10)
            data = resp.json()
            price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            prices[symbol] = float(price) if price else 0.0
        except Exception:
            prices[t.strip().upper()] = 0.0
    return prices


def parse_currency(val) -> float:
    """Parse strings like '$1,234' or '+1,234' into a float."""
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ─────────────────────────── data loaders ──────────────────────────
@st.cache_data(ttl=600, show_spinner="Loading client list …")
def load_clients() -> list[str]:
    sql = f"""
        SELECT DISTINCT ClientName
        FROM `{ACCOUNTS_TABLE}`
        WHERE ClientName IS NOT NULL
        ORDER BY ClientName
    """
    df = run_query(sql)
    return df["ClientName"].tolist()


def load_existing_targets(family_name: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            family_name, category, label, target_weight, run_by, load_timestamp
        FROM `{TARGETS_TABLE}`
        WHERE family_name = @family_name
          AND load_timestamp = (
              SELECT MAX(load_timestamp)
              FROM `{TARGETS_TABLE}`
              WHERE family_name = @family_name
          )
        ORDER BY
            CASE category
                WHEN 'Entity'      THEN 1
                WHEN 'Account'     THEN 2
                WHEN 'Ticker'      THEN 3
                WHEN 'Asset Class' THEN 4
            END,
            target_weight DESC
    """
    params = [bigquery.ScalarQueryParameter("family_name", "STRING", family_name)]
    return run_query(sql, params)


def load_entity_options(family_name: str) -> list[str]:
    sql = f"""
        SELECT DISTINCT PrimaryAccountHolder AS Entity
        FROM `{ACCOUNTS_TABLE}`
        WHERE ClientName = @family_name
          AND PrimaryAccountHolder IS NOT NULL
        ORDER BY PrimaryAccountHolder
    """
    params = [bigquery.ScalarQueryParameter("family_name", "STRING", family_name)]
    df = run_query(sql, params)
    return df["Entity"].tolist() if len(df) > 0 else []


def load_account_options(family_name: str, entities: list[str]) -> pd.DataFrame:
    if not entities:
        return pd.DataFrame(columns=["AccountNumber", "AccountName"])
    entity_list = ", ".join(f"'{e}'" for e in entities)
    sql = f"""
        SELECT DISTINCT
            AccountNumber,
            COALESCE(FBSIShortName, AccountNumber) AS AccountName
        FROM `{ACCOUNTS_TABLE}`
        WHERE ClientName = @family_name
          AND PrimaryAccountHolder IN ({entity_list})
          AND AccountNumber IS NOT NULL
        ORDER BY AccountName
    """
    params = [bigquery.ScalarQueryParameter("family_name", "STRING", family_name)]
    return run_query(sql, params)


def load_actual_mv(
    family_name: str,
    account_names: list[str],
    tickers: list[str],
    portfolio_date: date,
) -> pd.DataFrame:
    """Run the main drift CTE query against BigQuery."""
    acct_list = ", ".join(f"'{a}'" for a in account_names) if account_names else "''"
    ticker_csv = ",".join(tickers) if tickers else ""

    sql = f"""
        WITH
        ClassifiedPositions AS (
          SELECT
            a.FBSIShortName AS Account,
            TRIM(dp.Symbol) AS Symbol,
            CASE
              WHEN TRIM(dp.Symbol) IN ('QJXAQ','FRGXX','QIWSQ') THEN 'Cash'
              WHEN TRIM(dp.Symbol) IN ('ISHUF','MUB','VTEB','NUVBX','NVHIX','PRIMX','VMLUX','AGG','CMF') THEN 'Fixed Income'
              WHEN dp.SecurityType IN ('0','1','2','9') THEN 'Equity'
              WHEN dp.SecurityType IN ('5','6','7') THEN 'Fixed Income'
              WHEN dp.SecurityType IN ('F','C') THEN 'Cash'
              ELSE 'Undefined'
            END AS SecurityType,
            dp.PositionMarketValue AS MarketValue
          FROM `{ACCOUNTS_TABLE}` a
          INNER JOIN `{POSITIONS_TABLE}` dp
            ON a.AccountNumber = dp.AccountNumber
          WHERE
            a.ClientName = @family_name
            AND a.FBSIShortName IN ({acct_list})
            AND dp.Date = @portfolio_date
        ),
        TickerList AS (
          SELECT TRIM(ticker) AS ticker
          FROM UNNEST(SPLIT(@ticker_csv, ',')) AS ticker
          WHERE TRIM(ticker) != ''
        ),
        PortfolioTotal AS (
          SELECT SUM(MarketValue) AS TotalMV
          FROM ClassifiedPositions
        ),
        TickerMV AS (
          SELECT Symbol AS Name, 'Ticker' AS Type, SUM(MarketValue) AS ActualMV
          FROM ClassifiedPositions
          WHERE Symbol IN (SELECT ticker FROM TickerList)
          GROUP BY Symbol
        ),
        AccountMV AS (
          SELECT Account AS Name, 'Account' AS Type,
            SUM(CASE WHEN Symbol NOT IN (SELECT ticker FROM TickerList) THEN MarketValue ELSE 0 END) AS ActualMV
          FROM ClassifiedPositions
          GROUP BY Account
        ),
        AssetClassMV AS (
          SELECT SecurityType AS Name, 'Asset Class' AS Type,
            SUM(CASE WHEN Symbol NOT IN (SELECT ticker FROM TickerList) THEN MarketValue ELSE 0 END) AS ActualMV
          FROM ClassifiedPositions
          WHERE SecurityType IN ('Cash', 'Equity', 'Fixed Income')
          GROUP BY SecurityType
        )
        SELECT Name, Type, ActualMV, pt.TotalMV
        FROM TickerMV CROSS JOIN PortfolioTotal pt
        UNION ALL
        SELECT Name, Type, ActualMV, pt.TotalMV
        FROM AccountMV CROSS JOIN PortfolioTotal pt
        UNION ALL
        SELECT Name, Type, ActualMV, pt.TotalMV
        FROM AssetClassMV CROSS JOIN PortfolioTotal pt
    """
    params = [
        bigquery.ScalarQueryParameter("family_name", "STRING", family_name),
        bigquery.ScalarQueryParameter("portfolio_date", "DATE", portfolio_date),
        bigquery.ScalarQueryParameter("ticker_csv", "STRING", ticker_csv),
    ]
    return run_query(sql, params)


# ═══════════════════════════════════════════════════════════════════
#                           MAIN APP
# ═══════════════════════════════════════════════════════════════════

st.title("⚖️ Portfolio Rebalancer")

# ────────────────── Step 1: Client & Date ──────────────────────────
st.markdown("### 📅 Step 1: Select Client & Date")
st.caption("Choose the **client family** and the **portfolio date** you want to analyze.")

col_family, col_date = st.columns(2)

with col_family:
    clients = load_clients()
    family_name = st.selectbox("Client Family", clients, index=None, placeholder="Choose a client…")

with col_date:
    portfolio_date = st.date_input("Portfolio Date", value=date.today() - timedelta(days=1))

if not family_name:
    st.info("👆 Select a client family above to get started.")
    st.stop()

# ────────────────── Step 2: Existing Targets ───────────────────────
st.divider()
st.markdown("### 📋 Step 2: Review Existing Targets")
st.caption("Below are the most recent target allocations saved for this family.")

existing_targets_df = load_existing_targets(family_name)

has_existing = len(existing_targets_df) > 0

if has_existing:
    last_updated = existing_targets_df["load_timestamp"].max()
    last_run_by = existing_targets_df["run_by"].iloc[0]
    st.success(f"**{family_name}** — Last updated: {last_updated} — Run by: {last_run_by}")

    summary = existing_targets_df[["category", "label", "target_weight"]].copy()
    summary.columns = ["Category", "Name", "Target Weight %"]
    summary["Target Weight %"] = summary["Target Weight %"].apply(lambda w: f"{w:.1f}%")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    existing_entities = existing_targets_df.loc[
        existing_targets_df["category"] == "Entity", "label"
    ].tolist()
    existing_accounts = existing_targets_df.loc[
        existing_targets_df["category"] == "Account", "label"
    ].tolist()
    existing_tickers = existing_targets_df.loc[
        existing_targets_df["category"] == "Ticker", "label"
    ].tolist()
    existing_weights: dict[str, float] = dict(
        zip(existing_targets_df["label"], existing_targets_df["target_weight"])
    )
else:
    st.warning("No existing targets found for this family. Set up new targets below.")
    existing_entities, existing_accounts, existing_tickers = [], [], []
    existing_weights = {}

# ────────────────── Step 3: Configure Filters ──────────────────────
st.divider()
st.markdown("### 🔧 Step 3: Configure Filters")

st.markdown("""
| Toggle Position | What Happens |
|----------------|-------------|
| **ON** (New Selection) | Pick fresh entities, accounts, and tickers from the dropdowns below |
| **OFF** | Reuse the last saved selections from the database |
""")

update_filters = st.toggle("New Selection", value=True)

if update_filters:
    st.info("🔄 **NEW SELECTION MODE** — Pick entities, accounts, and tickers below.")
else:
    if existing_accounts or existing_tickers:
        st.success(
            "✅ **USING SAVED SELECTIONS** from database  \n"
            + (f"Accounts: {', '.join(existing_accounts)}  \n" if existing_accounts else "")
            + (f"Tickers: {', '.join(existing_tickers)}" if existing_tickers else "")
        )
    else:
        st.warning("No saved selections found — switch to New Selection mode.")

# Load entity / account options for the "New Selection" path
entity_options = load_entity_options(family_name)

col_ent, col_acct, col_tick = st.columns(3)

with col_ent:
    selected_entities = st.multiselect(
        "Entities",
        entity_options,
        default=existing_entities if existing_entities else [],
        disabled=not update_filters,
    )

# Load accounts filtered by selected entities
account_options_df = load_account_options(family_name, selected_entities)
account_names_list = account_options_df["AccountName"].tolist() if len(account_options_df) > 0 else []

with col_acct:
    default_accounts = [a for a in existing_accounts if a in account_names_list] if not update_filters else []
    selected_accounts = st.multiselect(
        "Accounts",
        account_names_list,
        default=default_accounts,
        disabled=not update_filters,
    )

with col_tick:
    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(existing_tickers) if existing_tickers else "",
        disabled=not update_filters,
    )

# Resolve final filter values
if update_filters:
    final_accounts = selected_accounts
    final_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
else:
    final_accounts = existing_accounts
    final_tickers = existing_tickers

# ────────────────── Step 4: Target Weights ─────────────────────────
st.divider()
st.markdown("### ⚖️ Step 4: Set Target Weights")
st.caption("Enter the target allocation percentage for each row. **All weights must add up to exactly 100%.**")

# Build initial targets dataframe
rows: list[dict] = []
if update_filters:
    for a in final_accounts:
        rows.append({"target_name": a, "target_type": "Account", "weight_percent": 0.0})
    for t in final_tickers:
        rows.append({"target_name": t, "target_type": "Ticker", "weight_percent": 0.0})
    for ac in ASSET_CLASSES:
        rows.append({"target_name": ac, "target_type": "Asset Class", "weight_percent": 0.0})
else:
    for a in final_accounts:
        rows.append({"target_name": a, "target_type": "Account", "weight_percent": existing_weights.get(a, 0.0)})
    for t in final_tickers:
        rows.append({"target_name": t, "target_type": "Ticker", "weight_percent": existing_weights.get(t, 0.0)})
    for ac in ASSET_CLASSES:
        rows.append({"target_name": ac, "target_type": "Asset Class", "weight_percent": existing_weights.get(ac, 0.0)})

if not rows:
    st.warning("No accounts, tickers, or asset classes selected. Please configure filters above.")
    st.stop()

initial_targets_df = pd.DataFrame(rows)

targets_edited = st.data_editor(
    initial_targets_df,
    column_config={
        "target_name": st.column_config.TextColumn("Name", disabled=True),
        "target_type": st.column_config.TextColumn("Type", disabled=True),
        "weight_percent": st.column_config.NumberColumn(
            "Target Weight %",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.1f%%",
        ),
    },
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="targets_editor",
)

# Validate
total_weight = targets_edited["weight_percent"].sum()
validation_passed = abs(total_weight - 100.0) <= 0.01

if validation_passed:
    st.success(f"✅ **Validation passed** — Total weight: {total_weight:.2f}%")
else:
    diff = 100.0 - total_weight
    msg = f"Need to {'add' if diff > 0 else 'reduce by'} {abs(diff):.2f}%"
    st.error(f"❌ **Validation failed** — Total weight: {total_weight:.2f}% — {msg}")

# Weight breakdown
with st.expander("Weight breakdown by type"):
    for tt in ["Account", "Ticker", "Asset Class"]:
        subset = targets_edited.loc[targets_edited["target_type"] == tt, "weight_percent"]
        if len(subset) > 0:
            st.write(f"**{tt}:** {subset.sum():.2f}%")

# ────────────────── Step 5: Calculate Drift ────────────────────────
st.divider()
st.markdown("### 📊 Step 5: Calculate & Review Drift")
st.caption("Click **Calculate Drift** to fetch live prices and compare actual vs. target allocations.")

calc_drift = st.button("🧮 Calculate Drift", disabled=not validation_passed, type="primary")

if calc_drift and validation_passed:
    with st.spinner("Fetching live prices & querying BigQuery …"):
        # Yahoo Finance prices
        ticker_prices = get_ticker_prices(tuple(final_tickers))

        # BigQuery actual MVs
        actual_mv_df = load_actual_mv(family_name, final_accounts, final_tickers, portfolio_date)

    if len(actual_mv_df) > 0:
        total_mv = float(actual_mv_df["TotalMV"].iloc[0])
    else:
        total_mv = 0.0
        st.warning("⚠️ No actual market-value data found for the selected date / filters. Showing targets with $0 actuals.")

    actual_lookup: dict[str, float] = {}
    if len(actual_mv_df) > 0:
        for _, r in actual_mv_df.iterrows():
            actual_lookup[r["Name"]] = float(r["ActualMV"])

    # Build drift table
    drift_rows: list[dict] = []
    for _, r in targets_edited.iterrows():
        name = r["target_name"]
        ttype = r["target_type"]
        target_pct = float(r["weight_percent"])

        actual_mv_val = actual_lookup.get(name, 0.0)
        actual_pct = (actual_mv_val / total_mv * 100) if total_mv > 0 else 0.0
        target_mv_val = (target_pct / 100.0) * total_mv
        drift_mv_val = actual_mv_val - target_mv_val
        drift_pct_val = actual_pct - target_pct

        # Icon & price
        if ttype == "Ticker":
            icon = "🎯"
            price = ticker_prices.get(name, 0.0)
            if price == 0.0:
                st.warning(f"⚠️ Could not fetch price for **{name}** from Yahoo Finance.")
        elif ttype == "Account":
            icon = "🏦"
            price = 1.0  # $1 per unit so Qty = dollar amount
        else:
            icon = ASSET_CLASS_ICONS.get(name, "📊")
            price = 1.0  # $1 per unit so Qty = dollar amount

        drift_rows.append(
            {
                "Row Item": f"{icon} {name}",
                "Type": ttype,
                "Actual MV": actual_mv_val,
                "Actual %": actual_pct,
                "Target MV": target_mv_val,
                "Target %": target_pct,
                "Drift MV": drift_mv_val,
                "Drift %": drift_pct_val,
                "Qty": 0,
                "Price": price,
            }
        )

    # Total row
    drift_rows.append(
        {
            "Row Item": "📊 TOTAL",
            "Type": "",
            "Actual MV": total_mv,
            "Actual %": 100.0,
            "Target MV": total_mv,
            "Target %": 100.0,
            "Drift MV": 0.0,
            "Drift %": 0.0,
            "Qty": 0,
            "Price": 0.0,
        }
    )

    drift_df = pd.DataFrame(drift_rows)
    st.session_state["drift_df"] = drift_df
    st.session_state["total_mv"] = total_mv

# ────────────────── Step 6: Adjust Positions ───────────────────────
if "drift_df" in st.session_state:
    drift_df: pd.DataFrame = st.session_state["drift_df"]
    total_mv: float = st.session_state["total_mv"]

    st.divider()
    st.markdown("### ✏️ Step 6: Adjust Positions")

    st.markdown("""
| Drift Sign | Meaning | Action Needed |
|-----------|---------|--------------|
| **(−) Negative** | Underweight — below target | Buy / Add |
| **(+) Positive** | Overweight — above target | Sell / Reduce |

- **Stocks:** Enter share count in **Qty** (negative to sell). Price auto-fills.
- **Accounts / Asset Classes:** Enter **dollar amount** in Qty. Price defaults to $1.
    """)

    # Format display columns but keep Qty/Price editable as numbers
    display_df = drift_df.copy()
    display_df["Actual MV"] = display_df["Actual MV"].apply(lambda v: f"${v:,.0f}")
    display_df["Actual %"] = display_df["Actual %"].apply(lambda v: f"{v:.1f}%")
    display_df["Target MV"] = display_df["Target MV"].apply(lambda v: f"${v:,.0f}")
    display_df["Target %"] = display_df["Target %"].apply(lambda v: f"{v:.1f}%")
    display_df["Drift MV"] = display_df["Drift MV"].apply(
        lambda v: f"{'+' if v >= 0 else ''}{v:,.0f}"
    )
    display_df["Drift %"] = display_df["Drift %"].apply(
        lambda v: f"{'+' if v >= 0 else ''}{v:.1f}%"
    )

    edited_drift = st.data_editor(
        display_df,
        column_config={
            "Row Item": st.column_config.TextColumn("Row Item", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
            "Actual MV": st.column_config.TextColumn("Actual MV", disabled=True),
            "Actual %": st.column_config.TextColumn("Actual %", disabled=True),
            "Target MV": st.column_config.TextColumn("Target MV", disabled=True),
            "Target %": st.column_config.TextColumn("Target %", disabled=True),
            "Drift MV": st.column_config.TextColumn("Drift MV", disabled=True),
            "Drift %": st.column_config.TextColumn("Drift %", disabled=True),
            "Qty": st.column_config.NumberColumn("Qty", step=1),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="drift_editor",
    )

    # ── Compute Inflow / Outflow & Updated Portfolio ──────────────
    st.divider()
    st.markdown("### 📈 Updated Portfolio Analysis")

    results: list[dict] = []
    total_inflow = 0.0

    for _, row in edited_drift.iterrows():
        row_item = str(row["Row Item"])
        if "TOTAL" in row_item:
            continue
        qty = float(row["Qty"]) if pd.notna(row["Qty"]) else 0.0
        price = float(row["Price"]) if pd.notna(row["Price"]) and float(row["Price"]) != 0 else 1.0
        total_inflow += qty * price

    new_total = total_mv + total_inflow

    for _, row in edited_drift.iterrows():
        row_item = str(row["Row Item"])
        actual_mv_val = parse_currency(row["Actual MV"])
        actual_pct = parse_currency(row["Actual %"])
        target_mv_val = parse_currency(row["Target MV"])
        target_pct = parse_currency(row["Target %"])
        drift_mv_val = parse_currency(row["Drift MV"])
        drift_pct_val = parse_currency(row["Drift %"])
        qty = float(row["Qty"]) if pd.notna(row["Qty"]) else 0.0
        price = float(row["Price"]) if pd.notna(row["Price"]) and float(row["Price"]) != 0 else 1.0
        inflow = qty * price

        if "TOTAL" in row_item:
            updated_mv = new_total
            updated_pct = 100.0
            inflow = total_inflow
        else:
            updated_mv = actual_mv_val + inflow
            updated_pct = (updated_mv / new_total * 100) if new_total > 0 else 0.0

        results.append(
            {
                "Asset": row_item,
                "Actual MV": f"${actual_mv_val:,.0f}",
                "Actual %": f"{actual_pct:.1f}%",
                "Target %": f"{target_pct:.1f}%",
                "Drift %": f"{'+' if drift_pct_val >= 0 else ''}{drift_pct_val:.1f}%",
                "Inflow / Outflow": f"${inflow:,.2f}",
                "Updated MV": f"${updated_mv:,.0f}",
                "Updated %": f"{updated_pct:.1f}%",
            }
        )

    updated_df = pd.DataFrame(results)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Portfolio Total (Updated)", f"${new_total:,.0f}")
    col_m2.metric("Net Inflow / Outflow", f"${total_inflow:,.0f}")

    st.dataframe(updated_df, use_container_width=True, hide_index=True)

