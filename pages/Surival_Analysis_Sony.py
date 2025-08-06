#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import io  
import seaborn as sns

st.set_page_config(page_title="Sony YouTube Survival", layout="wide")
st.title("🎬 Sony Pictures YouTube – Survival Analysis")
st.write("This app uses survival analysis to explore how long videos stay relevant, based on when daily views fall below 1% of total views.")
st.caption("Click the button above to download retention groups based on your filters.")

# === Load & Process Data ===
def load_data():
    df_clean = pd.read_csv("daily_df.csv", encoding='utf-8-sig')
    df_clean['date'] = pd.to_datetime(df_clean['date'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    df_clean['publish_date'] = pd.to_datetime(df_clean['publish_date'], format='%m/%d/%Y', errors='coerce')
    df_clean = df_clean.dropna(subset=['date', 'publish_date'])
    df_clean['days_since_publish'] = (df_clean['date'] - df_clean['publish_date']).dt.days
    df_clean = df_clean[df_clean['days_since_publish'] >= 0]
    df_clean = df_clean.drop_duplicates(subset=['video_id', 'days_since_publish'])

    # Group views by both video and channel
    daily_views = df_clean.groupby(['video_id', 'channel_display_name', 'days_since_publish'])['views'].sum().reset_index()
    # Compute total views per video-channel
    video_totals = daily_views.groupby(['video_id', 'channel_display_name'])['views'].sum().reset_index()
    video_totals.rename(columns={'views': 'total_views'}, inplace=True)
    # Merge total views back to daily views
    merged = pd.merge(daily_views, video_totals, on=['video_id', 'channel_display_name'], how='left')
    # Mark decay threshold
    merged['event_occurred'] = merged['views'] < (0.01 * merged['total_views'])
    # First event day per video-channel
    first_event_day = merged[merged['event_occurred']].groupby(['video_id', 'channel_display_name'])['days_since_publish'].min().reset_index()
    first_event_day.rename(columns={'days_since_publish': 'event_day'}, inplace=True)
    # Last day observed
    last_day = merged.groupby(['video_id', 'channel_display_name'])['days_since_publish'].max().reset_index()
    last_day.rename(columns={'days_since_publish': 'last_day'}, inplace=True)

    # Merge censoring + event info
    survival_df = pd.merge(last_day, first_event_day, on=['video_id', 'channel_display_name'], how='left')
    survival_df['event_occurred'] = ~survival_df['event_day'].isna()

    # Final duration
    survival_df['duration'] = survival_df.apply(
    lambda row: row['event_day'] if row['event_occurred'] else row['last_day'],
    axis=1)

    # Cap durations at 100, and update event status accordingly
    survival_df['duration_capped'] = survival_df['duration']
    survival_df['event_capped'] = survival_df['event_occurred']

    # If duration > 100, it's now censored at 100
    mask = survival_df['duration'] > 100
    survival_df.loc[mask, 'duration_capped'] = 100
    survival_df.loc[mask, 'event_capped'] = False

    # Make sure metadata includes both video_id and channel
    metadata = df_clean[['video_id', 'channel_display_name', 'Title Genre', 'IP_Type']].drop_duplicates()
    metadata['primary_genre'] = metadata['Title Genre'].str.split('/').str[0]

    # Merge with survival data
    survival_df = pd.merge(survival_df, metadata, on=['video_id', 'channel_display_name'], how='left')

    return survival_df

survival_df = load_data()

# === Sidebar ===

# 1. Get available channels and genres
channels =["All"] + sorted(survival_df['channel_display_name'].dropna().unique())
genres = ["All"] + sorted(survival_df['primary_genre'].dropna().unique())

# 2. Sidebar selectors
selected_channels = st.sidebar.multiselect("Select Channel(s)", channels, default=channels[:1])
selected_genre = st.sidebar.selectbox("Select Genre", genres)

# 3. Apply filters

if "All" not in selected_channels:
    filtered = survival_df[survival_df['channel_display_name'].isin(selected_channels)]
else:
    filtered = survival_df.copy()

if selected_genre != "All":
    filtered = filtered[filtered['primary_genre'] == selected_genre]

# === Kaplan-Meier Plot ===
st.subheader("📈 Kaplan-Meier Survival Curve")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#f5f5f5')

# Always plot All if selected
if "All" in selected_channels:
    kmf_all = KaplanMeierFitter()
    kmf_all.fit(survival_df['duration_capped'], event_observed=survival_df['event_capped'], label="All Channels")
    kmf_all.plot(ax=ax, ci_show=False)

# Always plot other selected channels
for channel in selected_channels:
    if channel != "All":
        data = filtered[filtered['channel_display_name'] == channel]
        if not data.empty and data['event_capped'].sum() > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(data['duration_capped'], event_observed=data['event_capped'], label=channel)
            kmf.plot(ax=ax, ci_show=False)
            
title_parts = []

if "All" in selected_channels:
    title_parts.append("All Channels")
else:
    title_parts += selected_channels

if selected_genre != "All":
    title_parts.append(f"Genre: {selected_genre}")

title_str = " | ".join(title_parts)

ax.set_title(f"Kaplan-Meier Survival Curve ({title_str})", fontsize=14)

ax.set_ylim(0, 1.05)
ax.set_xlabel("Days Since Publish")
ax.set_ylabel("Probability of Sustained Viewership")
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)


# === Compute median survival (only one of channel or genre) ===
# === Compute median survival (by channel & genre if specific channels selected) ===
median_df = pd.DataFrame()

def label_retention(days):
    if days < 7:
        return 'Weekly Rotation (<7d)'
    elif days < 14:
        return 'Fast Rotation (<14d)'
    elif days < 30:
        return 'Monthly Rotation (<30d)'
    elif days < 90:
        return 'Medium-Term Rotation (30-90d)'
    else:
        return 'Long-Term Rotation (>90d)'

# Case 1: Specific channels selected (not "All")
if "All" not in selected_channels:
    median_df = filtered.groupby(['channel_display_name', 'primary_genre'])['duration_capped'].median().reset_index()
    median_df.rename(columns={'duration_capped': 'median_survival_days'}, inplace=True)
    median_df['Rotation Strategy'] = median_df['median_survival_days'].apply(label_retention)

# Case 2: All channels selected, but a specific genre
elif selected_channels == ["All"] and selected_genre != "All":
    median_df = filtered.groupby('primary_genre')['duration_capped'].median().reset_index()
    median_df.rename(columns={'duration_capped': 'median_survival_days'}, inplace=True)
    median_df['Rotation Strategy'] = median_df['median_survival_days'].apply(label_retention)

# === Show table ===
with st.expander("📈 Median Survival by Channel & Genre"):
    if not median_df.empty:
        st.markdown(f"**Selection: {title_str}**")
        st.dataframe(median_df)

def create_retention_excel(df):
    df = df.copy()
    df['Retention Group'] = df['median_survival_days'].apply(label_retention)

    group_order = [
        'Long-Term Rotation (>90d)',
        'Medium-Term Rotation (30-90d)',
        'Monthly Rotation (<30d)',
        'Fast Rotation (<14d)',
        'Weekly Rotation (<7d)'
    ]

    df['Retention Group'] = pd.Categorical(df['Retention Group'], categories=group_order, ordered=True)
    df.sort_values('Retention Group', inplace=True)

    grouped = df.groupby('Retention Group').apply(
        lambda g: [f"{row['channel_display_name']} | {row.get('primary_genre', '')} | {int(row['median_survival_days'])}d"
                   for _, row in g.iterrows()]
    ).reset_index(name='Details')

    max_len = grouped['Details'].apply(len).max()
    wide_df = pd.DataFrame()

    for _, row in grouped.iterrows():
        group_name = row['Retention Group']
        items = row['Details']
        wide_df[group_name] = items + [''] * (max_len - len(items))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wide_df.to_excel(writer, index=False, sheet_name='Retention Groups')
        workbook = writer.book
        worksheet = writer.sheets['Retention Groups']

        colors = {
            'Long-Term Rotation (>90d)': '#DAE8FC',
            'Medium-Term Rotation (30-90d)': '#D5E8D4',
            'Monthly Rotation (<30d)': '#F8CECC',
            'Fast Rotation (<14d)': '#FCE4D6',
            'Weekly Rotation (<7d)': '#FFF2CC'
        }

        for col_idx, col in enumerate(wide_df.columns):
            color = colors.get(col, '#FFFFFF')
            cell_format = workbook.add_format({'bg_color': color, 'border': 1})
            for row_idx, value in enumerate(wide_df[col]):
                if value != '':
                    worksheet.write(row_idx + 1, col_idx, value, cell_format)

        # Auto-adjust column widths
        for idx, col in enumerate(wide_df.columns):
            max_len = max([len(str(cell)) for cell in wide_df[col]] + [len(col)])
            worksheet.set_column(idx, idx, max_len + 2)

    output.seek(0)
    return output


if not median_df.empty:
    excel_data = create_retention_excel(median_df)

    st.download_button(
        label="📥 Download Filtered Retention Groups as Excel",
        data=excel_data,
        file_name="filtered_retention_groups.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("To download the median survival, select either a specific genre OR specific channels.")

# === Overall Median Survival Time by Channel (not affected by filters) ===
st.subheader("📊 Overall Median Survival Time by Channel (Unfiltered)")

median_all_channels = survival_df.groupby('channel_display_name')['duration_capped'].median().reset_index()
median_all_channels.rename(columns={'duration_capped': 'Median Survival Days'}, inplace=True)

with st.expander("📊 Overall Median Survival Time by Channel (All Data)"):
    median_all_channels['Rotation Strategy'] = median_all_channels['Median Survival Days'].apply(label_retention)
    st.dataframe(median_all_channels)


# === Data Preview ===
with st.expander("📊 View Raw Data"):
    st.dataframe(filtered[['video_id', 'channel_display_name', 'duration', 'event_occurred']].head(50))



# In[3]:





# In[ ]:




