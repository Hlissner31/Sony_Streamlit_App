#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import io

# -------------------- Page & Title --------------------
st.set_page_config(page_title="Sony YouTube Survival", layout="wide")
st.title("🎬 Sony Pictures YouTube – Survival Analysis")
st.write("Estimate how long videos stay relevant using survival analysis on views and revenue.")
st.caption("Use the filters on the left. Same filters apply to all tabs.")

# -------------------- Helpers --------------------
REQUIRED_COLS = {
    "date",
    "publish_date",
    "video_id",
    "channel_display_name",
    "Title Genre",
    "views",
    "estimated_partner_revenue",
}

def label_retention(days: float) -> str:
    if pd.isna(days):
        return ''
    if days < 8:
        return 'Weekly Rotation (0-7d)'
    elif days < 15:
        return 'Fast Rotation (8-14d)'
    elif days < 31:
        return 'Monthly Rotation (15-30d)'
    elif days < 91:
        return 'Medium-Term Rotation (31-90d)'
    else:
        return 'Long-Term Rotation (>90d)'

def km_median_for_group(durations, events):
    """Kaplan–Meier median survival; if not reached, use max observed time."""
    durations = np.asarray(durations)
    events = np.asarray(events).astype(bool)
    if durations.size == 0:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    m = kmf.median_survival_time_
    if np.isinf(m) or pd.isna(m):
        m = float(np.max(durations))
    return float(m)

def create_retention_excel(df: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    if df.empty:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({'No data for current filters': []}).to_excel(
                writer, index=False, sheet_name='Retention Groups'
            )
        output.seek(0)
        return output

    df = df.copy()
    df['Retention Group'] = df['median_survival_days'].apply(label_retention)

    order = [
        'Long-Term Rotation (>90d)',
        'Medium-Term Rotation (31-90d)',
        'Monthly Rotation (15-30d)',
        'Fast Rotation (8-14d)',
        'Weekly Rotation (0-7d)'
    ]
    df['Retention Group'] = pd.Categorical(df['Retention Group'], categories=order, ordered=True)
    df.sort_values(['Retention Group', 'median_survival_days'], ascending=[True, False], inplace=True)

    has_channel = 'channel_display_name' in df.columns
    has_genre = 'primary_genre' in df.columns

    def row_str(row):
        parts = []
        if has_channel:
            parts.append(str(row['channel_display_name']))
        if has_genre:
            parts.append(str(row.get('primary_genre', '')))
        parts.append(f"{int(round(row['median_survival_days']))}d")
        return " | ".join([p for p in parts if p])

    grouped = df.groupby('Retention Group').apply(
        lambda g: [row_str(r) for _, r in g.iterrows()]
    ).reset_index(name='Details')

    max_len = grouped['Details'].apply(len).max() if len(grouped) else 0
    wide_df = pd.DataFrame()
    for _, row in grouped.iterrows():
        col_name = row['Retention Group']
        items = row['Details']
        wide_df[col_name] = items + [''] * (max_len - len(items))

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wide_df.to_excel(writer, index=False, sheet_name='Retention Groups')
        wb = writer.book
        ws = writer.sheets['Retention Groups']

        colors = {
            'Long-Term Rotation (>90d)': '#DAE8FC',
            'Medium-Term Rotation (31-90d)': '#D5E8D4',
            'Monthly Rotation (15-30d)': '#F8CECC',
            'Fast Rotation (8-14d)': '#FCE4D6',
            'Weekly Rotation (0-7d)': '#FFF2CC'
        }

        for col_idx, col in enumerate(wide_df.columns):
            fmt = wb.add_format({'bg_color': colors.get(col, '#FFFFFF'), 'border': 1})
            for row_idx, val in enumerate(wide_df[col]):
                if val != '':
                    ws.write(row_idx + 1, col_idx, val, fmt)
            max_w = max([len(str(x)) for x in wide_df[col]] + [len(col)]) + 2
            ws.set_column(col_idx, col_idx, max_w)

    output.seek(0)
    return output

def create_comparison_excel(comp_df: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    df = comp_df.copy()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparison')
        wb = writer.book
        ws = writer.sheets['Comparison']

        cols = {c: i for i, c in enumerate(df.columns)}
        diff_col = cols.get('Difference (Rev - Views)')
        if diff_col is not None:
            ws.conditional_format(1, diff_col, len(df), diff_col, {
                'type': 'cell', 'criteria': '>=', 'value': 0, 'format': wb.add_format({'bg_color': '#C6EFCE'})
            })
            ws.conditional_format(1, diff_col, len(df), diff_col, {
                'type': 'cell', 'criteria': '<', 'value': 0, 'format': wb.add_format({'bg_color': '#FFC7CE'})
            })
        for i, col in enumerate(df.columns):
            max_w = max(df[col].astype(str).map(len).max(), len(col)) + 2
            ws.set_column(i, i, max_w)
    output.seek(0)
    return output

# -------------------- Upload (MAIN AREA) --------------------
st.markdown("### 📤 Upload data")
uploaded = st.file_uploader("CSV or Excel (.csv, .xlsx)", type=["csv", "xlsx"])

def read_uploaded_df(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, low_memory=False)
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()

def validate_columns(df: pd.DataFrame) -> bool:
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        st.error(f"Your file is missing required columns: {sorted(missing)}")
        st.info("Required columns: " + ", ".join(sorted(REQUIRED_COLS)))
        return False
    return True

def preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # Dates (date can be int YYYYMMDD or string)
    df['date'] = pd.to_datetime(df['date'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    df['publish_date'] = pd.to_datetime(df['publish_date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['date', 'publish_date'])
    # Non-negative day offsets
    df['days_since_publish'] = (df['date'] - df['publish_date']).dt.days
    df = df[df['days_since_publish'] >= 0]
    # one row per video per day
    df = df.drop_duplicates(subset=['video_id', 'days_since_publish'])
    # Primary genre parse (no dropping)
    df['Title Genre'] = df['Title Genre'].astype(str)
    df['primary_genre'] = (
        df['Title Genre'].str.split('/').str[0].str.strip()
           .replace(['', 'nan', 'None', 'NaN'], pd.NA)
    )
    return df

def build_survival_per_video(df: pd.DataFrame, metric_col: str, cap_at:int=100) -> pd.DataFrame:
    """Survival per video_id; event if a day's metric < 1% of that video's total."""
    daily = df.groupby(['video_id', 'days_since_publish'])[metric_col].sum().reset_index()
    totals = daily.groupby('video_id')[metric_col].sum().reset_index().rename(columns={metric_col: 'total_metric'})
    m = daily.merge(totals, on='video_id', how='left')
    m['event_occurred'] = m[metric_col] < (0.01 * m['total_metric'])

    first_evt = (
        m[m['event_occurred']]
        .groupby('video_id')['days_since_publish']
        .min().reset_index().rename(columns={'days_since_publish': 'event_day'})
    )
    last_day = (
        m.groupby('video_id')['days_since_publish']
        .max().reset_index().rename(columns={'days_since_publish': 'last_day'})
    )

    surv = last_day.merge(first_evt, on='video_id', how='left')
    surv['event_occurred'] = ~surv['event_day'].isna()
    surv['duration'] = surv.apply(lambda r: r['event_day'] if r['event_occurred'] else r['last_day'], axis=1)

    # Cap at 100 (your current analysis choice)
    surv['duration_capped'] = surv['duration']
    surv['event_capped'] = surv['event_occurred']
    mask = surv['duration'] > cap_at
    surv.loc[mask, 'duration_capped'] = cap_at
    surv.loc[mask, 'event_capped'] = False

    meta = df[['video_id', 'channel_display_name', 'primary_genre']].drop_duplicates('video_id')
    surv = surv.merge(meta, on='video_id', how='left')
    return surv

# ---------- Read + validate upload ----------
raw_df = read_uploaded_df(uploaded)
if uploaded is None:
    st.info("⬆️ Upload a dataset to begin. Required columns: " + ", ".join(sorted(REQUIRED_COLS)))
    data_ready = False
else:
    if raw_df.empty or not validate_columns(raw_df):
        data_ready = False
    else:
        data_ready = True

# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.markdown("### Filters")

    if not data_ready:
        st.info("Upload a file to enable filters.")
        # Disabled placeholders so the sidebar always shows controls
        selected_channels = st.multiselect("Select Channel(s)", ["All"], default=["All"], disabled=True)
        selected_genres   = st.multiselect("Select Genre(s)",   ["All"], default=["All"], disabled=True)
    else:
        # Build survival datasets first so we know the option lists
        df_clean = preprocess_raw(raw_df)
        survival_views   = build_survival_per_video(df_clean, 'views', cap_at=100)
        survival_revenue = build_survival_per_video(df_clean, 'estimated_partner_revenue', cap_at=100)

        channels = ["All"] + sorted(survival_views['channel_display_name'].dropna().unique())
        genres   = ["All"] + sorted(survival_views['primary_genre'].dropna().unique())

        selected_channels = st.multiselect("Select Channel(s)", channels, default=["All"])
        selected_genres   = st.multiselect("Select Genre(s)",   genres,   default=["All"])

# If no data yet, stop after showing disabled filters.
if not data_ready:
    st.stop()

# -------------------- Apply filters --------------------
def apply_filters(surv_df: pd.DataFrame) -> pd.DataFrame:
    df = surv_df.copy()
    if "All" not in selected_channels:
        df = df[df['channel_display_name'].isin(selected_channels)]
    if "All" not in selected_genres:
        df = df[df['primary_genre'].isin(selected_genres)]
    return df

filtered_views   = apply_filters(survival_views)
filtered_revenue = apply_filters(survival_revenue)

# -------------------- Tabs --------------------
tab_views, tab_rev, tab_combo = st.tabs(["📺 Views", "💰 Revenue", "🧮 Combined"])

# ======== VIEWS TAB ========
with tab_views:
    st.subheader("📈 Kaplan–Meier Survival (Views)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f5f5f5')

    if "All" in selected_channels:
        kmf_all = KaplanMeierFitter()
        kmf_all.fit(filtered_views['duration_capped'], event_observed=filtered_views['event_capped'], label="All Channels")
        kmf_all.plot(ax=ax, ci_show=False)

    for ch in selected_channels:
        if ch != "All":
            g = filtered_views[filtered_views['channel_display_name'] == ch]
            if not g.empty:
                kmf = KaplanMeierFitter()
                kmf.fit(g['duration_capped'], event_observed=g['event_capped'], label=ch)
                kmf.plot(ax=ax, ci_show=False)

    parts = []
    if "All" in selected_channels: parts.append("All Channels")
    else: parts += selected_channels
    if "All" not in selected_genres: parts.append(f"Genres: {', '.join(selected_genres)}")
    ax.set_title("Views | " + (" | ".join(parts) if parts else "All Data"), fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days Since Publish")
    ax.set_ylabel("Probability of Sustained Viewership")
    ax.grid(True)
    st.pyplot(fig)

    # KPI Row (Views)
    mv = km_median_for_group(filtered_views['duration_capped'], filtered_views['event_capped'])
    col1, col2 = st.columns(2)
    col1.metric("KM Median (Views)", f"{0 if pd.isna(mv) else round(mv,1)} days")
    col2.metric("Channels in view",  filtered_views['channel_display_name'].nunique())

    # KM median table (Views)
    st.markdown("#### KM Median Survival (Views)")
    median_df = pd.DataFrame()
    if "All" not in selected_channels:
        rows = []
        for (ch, gen), g in filtered_views.groupby(['channel_display_name', 'primary_genre']):
            rows.append({
                'channel_display_name': ch,
                'primary_genre': gen,
                'median_survival_days': km_median_for_group(g['duration_capped'], g['event_capped'])
            })
        median_df = pd.DataFrame(rows)
        median_df['Rotation Strategy'] = median_df['median_survival_days'].apply(label_retention)
    elif ("All" in selected_channels) and ("All" not in selected_genres):
        rows = []
        for gen, g in filtered_views.groupby(['primary_genre']):
            rows.append({
                'primary_genre': gen,
                'median_survival_days': km_median_for_group(g['duration_capped'], g['event_capped'])
            })
        median_df = pd.DataFrame(rows)
        median_df['Rotation Strategy'] = median_df['median_survival_days'].apply(label_retention)

    if not median_df.empty:
        st.dataframe(median_df.sort_values(
            ['channel_display_name','median_survival_days'],
            ascending=[True, False], na_position='last'
        ))
        st.download_button(
            "📥 Download Retention Groups (Excel)",
            data=create_retention_excel(median_df),
            file_name="views_retention_groups.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Select channels (not 'All') for channel×genre medians, or pick specific genre(s) with 'All' channels.")
    
    # -------------------- Overall Medians (Unfiltered, Views) --------------------
    st.subheader("🧭 Overall KM Medians by Channel (Views)")
    rows = []
    for ch, g in survival_views.groupby('channel_display_name'):
        rows.append({
            'channel_display_name': ch,
            'Median Survival Days (Views)': km_median_for_group(g['duration_capped'], g['event_capped'])
        })
    overall_views = pd.DataFrame(rows)

    # rename to the column name expected by create_retention_excel
    overall_views = overall_views.rename(columns={'Median Survival Days (Views)': 'median_survival_days'})
    overall_views['Rotation (Views)'] = overall_views['median_survival_days'].apply(label_retention)

    with st.expander("📊 Overall KM Medians (Views)"):
        st.dataframe(overall_views.sort_values('channel_display_name'))


    # ⬇️ grouped Excel, same format as retention groups
    st.download_button(
        "📥 Download Overall KM Medians (Grouped Excel)",
        data=create_retention_excel(overall_views[['channel_display_name', 'median_survival_days']]),
        file_name="overall_km_medians_views_grouped.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # Raw Data — Views
    with st.expander("📊 View Raw Data — Views"):
        st.write(f"**Total rows (filtered):** {len(filtered_views):,}")
        st.dataframe(
            filtered_views[['video_id', 'channel_display_name', 'primary_genre',
                            'duration', 'event_occurred', 'duration_capped', 'event_capped']].head(50)
        )

# ======== REVENUE TAB ========
with tab_rev:
    st.subheader("📈 Kaplan–Meier Survival (Revenue)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f5f5f5')

    if "All" in selected_channels:
        kmf_all = KaplanMeierFitter()
        kmf_all.fit(filtered_revenue['duration_capped'], event_observed=filtered_revenue['event_capped'], label="All Channels")
        kmf_all.plot(ax=ax, ci_show=False)

    for ch in selected_channels:
        if ch != "All":
            g = filtered_revenue[filtered_revenue['channel_display_name'] == ch]
            if not g.empty:
                kmf = KaplanMeierFitter()
                kmf.fit(g['duration_capped'], event_observed=g['event_capped'], label=ch)
                kmf.plot(ax=ax, ci_show=False)

    parts = []
    if "All" in selected_channels: parts.append("All Channels")
    else: parts += selected_channels
    if "All" not in selected_genres: parts.append(f"Genres: {', '.join(selected_genres)}")
    ax.set_title("Revenue | " + (" | ".join(parts) if parts else "All Data"), fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Days Since Publish")
    ax.set_ylabel("Probability of Sustained Revenue")
    ax.grid(True)
    st.pyplot(fig)

    # KPI Row (Revenue)
    mr = km_median_for_group(filtered_revenue['duration_capped'], filtered_revenue['event_capped'])
    col1, col2 = st.columns(2)
    col1.metric("KM Median (Revenue)", f"{0 if pd.isna(mr) else round(mr,1)} days")
    col2.metric("Channels in view",    filtered_revenue['channel_display_name'].nunique())

    # KM median table (Revenue)
    st.markdown("#### KM Median Survival (Revenue)")
    median_rev = pd.DataFrame()
    if "All" not in selected_channels:
        rows = []
        for (ch, gen), g in filtered_revenue.groupby(['channel_display_name', 'primary_genre']):
            rows.append({
                'channel_display_name': ch,
                'primary_genre': gen,
                'median_survival_days': km_median_for_group(g['duration_capped'], g['event_capped'])
            })
        median_rev = pd.DataFrame(rows)
        median_rev['Rotation Strategy'] = median_rev['median_survival_days'].apply(label_retention)
    elif ("All" in selected_channels) and ("All" not in selected_genres):
        rows = []
        for gen, g in filtered_revenue.groupby(['primary_genre']):
            rows.append({
                'primary_genre': gen,
                'median_survival_days': km_median_for_group(g['duration_capped'], g['event_capped'])
            })
        median_rev = pd.DataFrame(rows)
        median_rev['Rotation Strategy'] = median_rev['median_survival_days'].apply(label_retention)

    if not median_rev.empty:
        st.dataframe(median_rev.sort_values(
            ['channel_display_name','median_survival_days'],
            ascending=[True, False], na_position='last'
        ))
        st.download_button(
            "📥 Download Retention Groups (Excel)",
            data=create_retention_excel(median_rev),
            file_name="revenue_retention_groups.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Select channels (not 'All') for channel×genre medians, or pick specific genre(s) with 'All' channels.")

    
    # -------------------- Overall Medians (Unfiltered, Revenue) --------------------
    st.subheader("🧭 Overall KM Medians by Channel (Revenue)")
    rows = []
    for ch, g in survival_revenue.groupby('channel_display_name'):
        rows.append({
            'channel_display_name': ch,
            'Median Survival Days (Revenue)': km_median_for_group(g['duration_capped'], g['event_capped'])
        })
    overall_rev = pd.DataFrame(rows)
    # rename to the column name expected by create_retention_excel
    overall_rev = overall_rev.rename(columns={'Median Survival Days (Revenue)': 'median_survival_days'})
    overall_rev['Rotation (Revenue)'] = overall_rev['median_survival_days'].apply(label_retention)
        
    with st.expander("📊 Overall KM Medians (Revenue)"):
        st.dataframe(overall_rev.sort_values('channel_display_name'))
        
    # ⬇️ grouped Excel, same format as retention groups
    st.download_button(
        "📥 Download Overall KM Medians (Grouped Excel)",
        data=create_retention_excel(overall_rev[['channel_display_name', 'median_survival_days']]),
        file_name="overall_km_medians_revenue_grouped.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Raw Data — Revenue
    with st.expander("📊 View Raw Data — Revenue"):
        st.write(f"**Total rows (filtered):** {len(filtered_revenue):,}")
        st.dataframe(
            filtered_revenue[['video_id', 'channel_display_name', 'primary_genre',
                              'duration', 'event_occurred', 'duration_capped', 'event_capped']].head(50)
        )

# ======== COMBINED TAB ========
with tab_combo:
    st.subheader("📊 Views vs Revenue — KM Median Comparison (Filtered)")

    rows_v = []
    for ch, g in filtered_views.groupby('channel_display_name'):
        rows_v.append({'channel_display_name': ch,
                       'Median Survival (Views, KM)': km_median_for_group(g['duration_capped'], g['event_capped'])})
    median_views = pd.DataFrame(rows_v)

    rows_r = []
    for ch, g in filtered_revenue.groupby('channel_display_name'):
        rows_r.append({'channel_display_name': ch,
                       'Median Survival (Revenue, KM)': km_median_for_group(g['duration_capped'], g['event_capped'])})
    median_revenue = pd.DataFrame(rows_r)

    comp = pd.merge(median_views, median_revenue, on='channel_display_name', how='outer')
    comp['Difference (Rev - Views)'] = comp['Median Survival (Revenue, KM)'] - comp['Median Survival (Views, KM)']
    comp['Recommendation'] = np.where(
        comp['Difference (Rev - Views)'] < 0,
        'Revenue decays faster → rotate sooner',
        np.where(comp['Difference (Rev - Views)'] > 0,
                 'Views decay faster → rev can stay longer',
                 'Similar decay')
    )

    st.dataframe(comp.sort_values('channel_display_name'))

    # KPI Row (Combined)
    col1, col2, col3 = st.columns(3)
    mv = km_median_for_group(filtered_views['duration_capped'],     filtered_views['event_capped'])
    mr = km_median_for_group(filtered_revenue['duration_capped'],   filtered_revenue['event_capped'])
    delta = None if (pd.isna(mv) or pd.isna(mr)) else round(mr - mv, 1)

    col1.metric("KM Median (Views)",   f"{0 if pd.isna(mv) else round(mv,1)} days")
    col2.metric("KM Median (Revenue)", f"{0 if pd.isna(mr) else round(mr,1)} days",
                delta=None if delta is None else f"{delta} days",
                delta_color="inverse" if (delta is not None and delta < 0) else "normal")
    col3.metric("Channels in view", filtered_views['channel_display_name'].nunique())

