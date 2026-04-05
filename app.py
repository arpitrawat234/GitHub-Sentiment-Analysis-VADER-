"""
app.py  —  GitHub Sentiment Analysis Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

from github_fetcher   import GitHubFetcher
from sentiment_engine import analyse, summary_stats, top_items

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Sentiment Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(120deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(88,166,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(63,185,80,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero h1 { color: #f0f6fc; font-size: 1.8rem; margin: 0; font-weight: 800; letter-spacing: -0.5px; }
.hero p  { color: #8b949e; font-size: 0.85rem; margin: 6px 0 0;
           font-family: 'JetBrains Mono', monospace; }

/* ── metric cards ── */
.metric-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.2rem; }
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 1rem 1.4rem; flex: 1; min-width: 130px;
}
.metric-card .label { font-size: 0.68rem; color: #6e7681; text-transform: uppercase;
                       letter-spacing: 1.2px; font-family: 'JetBrains Mono', monospace; }
.metric-card .value { font-size: 1.7rem; font-weight: 800; color: #58a6ff;
                       font-family: 'JetBrains Mono', monospace; line-height: 1.2; }
.metric-card .sub   { font-size: 0.72rem; color: #8b949e; margin-top: 2px; }

/* ── section headers ── */
.section-hdr {
    font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
    color: #58a6ff; text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin: 1.4rem 0 0.8rem;
}

/* ── sentiment badges ── */
.badge-pos { background:#0d4429; color:#3fb950; border:1px solid #238636;
             border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.badge-neg { background:#4d1b1b; color:#f85149; border:1px solid #da3633;
             border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.badge-neu { background:#1c2128; color:#8b949e; border:1px solid #30363d;
             border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/* ── sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label { color: #8b949e !important; }

/* table */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* hide streamlit footer */
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

code { background:#161b22; border-radius:4px; padding:2px 6px;
       font-family:'JetBrains Mono',monospace; font-size:0.8em; color:#79c0ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Matplotlib theme
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#6e7681",
    "ytick.color":      "#6e7681",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "text.color":       "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
})

POS_COL = "#3fb950"
NEG_COL = "#f85149"
NEU_COL = "#8b949e"
ACC_COL = "#58a6ff"

# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 GitHub Sentiment Analyser</h1>
  <p>VADER-powered sentiment analysis · Issues · PR Comments · Commits</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    token = st.text_input("GitHub Token (optional but recommended)",
                          type="password",
                          help="Increases rate limit from 60 → 5000 req/hr")
    st.markdown("---")
    repo_input = st.text_input("Repository  `owner/repo`",
                               value="microsoft/vscode",
                               placeholder="e.g. facebook/react")
    st.markdown("**Data sources**")
    inc_issues   = st.checkbox("Issues",           value=True)
    inc_pr       = st.checkbox("PR Comments",      value=True)
    inc_ic       = st.checkbox("Issue Comments",   value=True)
    inc_commits  = st.checkbox("Commits",          value=False)
    max_pages    = st.slider("Pages per source (100 items/page)", 1, 10, 3)
    st.markdown("---")
    run = st.button("🚀  Run Analysis", use_container_width=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Main — wait for run
# ─────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Enter a GitHub repo and click **Run Analysis** to begin.")
    st.markdown("""
    **How it works**
    1. Fetches issues, PR comments, issue comments and/or commits via GitHub REST API  
    2. Cleans text — strips code blocks, URLs, mentions, HTML  
    3. Scores each item with **VADER** (Valence Aware Dictionary and sEntiment Reasoner)  
    4. Visualises distribution, trends, and top items  

    > No GitHub token needed for public repos, but adds rate-limit headroom.
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Fetch + Analyse
# ─────────────────────────────────────────────────────────────
if "/" not in repo_input:
    st.error("❌ Repo must be in `owner/repo` format.")
    st.stop()

owner, repo = repo_input.strip().split("/", 1)

with st.spinner(f"Fetching data from **{owner}/{repo}**…"):
    try:
        fetcher = GitHubFetcher(token=token or None)
        raw_df  = fetcher.fetch_all(
            owner, repo,
            include_issues         = inc_issues,
            include_pr_comments    = inc_pr,
            include_issue_comments = inc_ic,
            include_commits        = inc_commits,
            max_pages              = max_pages,
        )
    except Exception as e:
        st.error(f"❌ GitHub API error: {e}")
        st.stop()

if raw_df.empty:
    st.warning("No data returned. Try enabling more data sources or check the repo name.")
    st.stop()

with st.spinner("Running VADER sentiment analysis…"):
    df = analyse(raw_df)

if df.empty:
    st.warning("All fetched items had empty text after cleaning.")
    st.stop()

stats = summary_stats(df)

# ─────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="label">Total Items</div>
    <div class="value">{stats['total']:,}</div>
    <div class="sub">{owner}/{repo}</div>
  </div>
  <div class="metric-card">
    <div class="label">Positive</div>
    <div class="value" style="color:#3fb950">{stats['positive']:,}</div>
    <div class="sub">{stats['pct_positive']}% of total</div>
  </div>
  <div class="metric-card">
    <div class="label">Negative</div>
    <div class="value" style="color:#f85149">{stats['negative']:,}</div>
    <div class="sub">{stats['pct_negative']}% of total</div>
  </div>
  <div class="metric-card">
    <div class="label">Neutral</div>
    <div class="value" style="color:#8b949e">{stats['neutral']:,}</div>
    <div class="sub">{stats['pct_neutral']}% of total</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg Compound</div>
    <div class="value" style="color:#e3b341">{stats['avg_compound']}</div>
    <div class="sub">{stats['overall_mood']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Charts — Row 1
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">01 — Sentiment Distribution</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Donut chart
    fig, ax = plt.subplots(figsize=(4.5, 4), subplot_kw=dict(aspect="equal"))
    sizes  = [stats["positive"], stats["neutral"], stats["negative"]]
    clrs   = [POS_COL, NEU_COL, NEG_COL]
    lbls   = ["Positive", "Neutral", "Negative"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=lbls, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2),
        textprops=dict(color="#c9d1d9", fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color("#0d1117"); at.set_fontweight("bold")
    ax.set_title("Overall Sentiment Split", color="#f0f6fc", fontsize=10, pad=12)
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    # Bar chart by type
    type_sent = df.groupby(["type","label"]).size().unstack(fill_value=0)
    for col_name in ["positive","neutral","negative"]:
        if col_name not in type_sent.columns:
            type_sent[col_name] = 0
    type_sent = type_sent[["positive","neutral","negative"]]

    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    x = np.arange(len(type_sent))
    w = 0.25
    ax2.bar(x - w, type_sent["positive"], w, color=POS_COL, label="Positive", alpha=0.9)
    ax2.bar(x,     type_sent["neutral"],  w, color=NEU_COL, label="Neutral",  alpha=0.9)
    ax2.bar(x + w, type_sent["negative"], w, color=NEG_COL, label="Negative", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace("_"," ").title() for t in type_sent.index],
                         rotation=20, ha="right", fontsize=8)
    ax2.set_title("Sentiment by Source Type", color="#f0f6fc", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(axis="y")
    fig2.tight_layout()
    st.pyplot(fig2)

with col3:
    # Compound score histogram
    fig3, ax3 = plt.subplots(figsize=(4.5, 4))
    pos_d = df[df["label"]=="positive"]["compound"]
    neg_d = df[df["label"]=="negative"]["compound"]
    neu_d = df[df["label"]=="neutral"]["compound"]
    ax3.hist(pos_d, bins=30, color=POS_COL, alpha=0.7, label="Positive")
    ax3.hist(neu_d, bins=30, color=NEU_COL, alpha=0.5, label="Neutral")
    ax3.hist(neg_d, bins=30, color=NEG_COL, alpha=0.7, label="Negative")
    ax3.axvline(0.05,  color=POS_COL, linestyle="--", linewidth=1)
    ax3.axvline(-0.05, color=NEG_COL, linestyle="--", linewidth=1)
    ax3.set_title("Compound Score Distribution", color="#f0f6fc", fontsize=10)
    ax3.set_xlabel("Compound Score"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=7); ax3.grid(True)
    fig3.tight_layout()
    st.pyplot(fig3)

# ─────────────────────────────────────────────────────────────
# Charts — Row 2  (trend over time)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">02 — Sentiment Trend Over Time</div>', unsafe_allow_html=True)

df_time = df.dropna(subset=["month"])
if not df_time.empty and df_time["month"].nunique() > 1:
    monthly = df_time.groupby(["month","label"]).size().unstack(fill_value=0)
    for c in ["positive","neutral","negative"]:
        if c not in monthly.columns: monthly[c] = 0

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        fig4, ax4 = plt.subplots(figsize=(7, 3.5))
        ax4.plot(monthly.index, monthly["positive"], color=POS_COL, marker="o",
                 markersize=4, linewidth=1.5, label="Positive")
        ax4.plot(monthly.index, monthly["neutral"],  color=NEU_COL, marker="s",
                 markersize=4, linewidth=1.5, label="Neutral")
        ax4.plot(monthly.index, monthly["negative"], color=NEG_COL, marker="^",
                 markersize=4, linewidth=1.5, label="Negative")
        ax4.fill_between(monthly.index, monthly["positive"], alpha=0.1, color=POS_COL)
        ax4.fill_between(monthly.index, monthly["negative"], alpha=0.1, color=NEG_COL)
        step = max(1, len(monthly) // 8)
        ax4.set_xticks(monthly.index[::step])
        ax4.set_xticklabels(monthly.index[::step], rotation=35, ha="right", fontsize=7)
        ax4.set_title("Monthly Sentiment Counts", color="#f0f6fc", fontsize=10)
        ax4.legend(fontsize=7); ax4.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4)

    with col_t2:
        # Rolling avg compound score
        df_sorted = df_time.sort_values("created_at").dropna(subset=["compound"])
        rolling   = df_sorted["compound"].rolling(50, min_periods=5).mean()
        fig5, ax5 = plt.subplots(figsize=(7, 3.5))
        ax5.plot(range(len(rolling)), rolling, color=ACC_COL, linewidth=1.5)
        ax5.axhline(0.05,  color=POS_COL, linestyle="--", linewidth=0.8, alpha=0.7)
        ax5.axhline(-0.05, color=NEG_COL, linestyle="--", linewidth=0.8, alpha=0.7)
        ax5.axhline(0,     color="#6e7681", linestyle="-",  linewidth=0.5, alpha=0.5)
        ax5.fill_between(range(len(rolling)), rolling, 0,
                         where=(rolling >= 0), color=POS_COL, alpha=0.1)
        ax5.fill_between(range(len(rolling)), rolling, 0,
                         where=(rolling < 0),  color=NEG_COL, alpha=0.1)
        ax5.set_title("50-Item Rolling Avg Compound Score", color="#f0f6fc", fontsize=10)
        ax5.set_xlabel("Item index (chronological)"); ax5.set_ylabel("Compound")
        ax5.grid(True)
        fig5.tight_layout()
        st.pyplot(fig5)
else:
    st.info("Not enough time-range data to plot trends.")

# ─────────────────────────────────────────────────────────────
# Charts — Row 3  (author + heatmap)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">03 — Author & Heatmap Analysis</div>', unsafe_allow_html=True)
col_a1, col_a2 = st.columns(2)

with col_a1:
    # Top-10 most active authors avg compound
    author_stats = (df.groupby("author")
                      .agg(count=("compound","count"), avg=("compound","mean"))
                      .query("count >= 3")
                      .nlargest(10, "count"))
    if not author_stats.empty:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        bar_colors = [POS_COL if v >= 0.05 else NEG_COL if v <= -0.05 else NEU_COL
                      for v in author_stats["avg"]]
        bars = ax6.barh(author_stats.index, author_stats["avg"],
                        color=bar_colors, edgecolor="#0d1117", alpha=0.85)
        ax6.axvline(0, color="#6e7681", linewidth=1)
        ax6.set_title("Top Authors — Avg Compound Score", color="#f0f6fc", fontsize=10)
        ax6.set_xlabel("Avg Compound Score")
        ax6.grid(axis="x")
        fig6.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("Not enough per-author data (need ≥3 items per author).")

with col_a2:
    # Sentiment heatmap: month × type
    if df_time["month"].nunique() > 1 and df_time["type"].nunique() > 1:
        pivot = df_time.groupby(["month","type"])["compound"].mean().unstack(fill_value=0)
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            pivot.T, ax=ax7, cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor="#0d1117",
            cbar_kws={"label": "Avg Compound"},
            annot=(pivot.shape[0] <= 12), fmt=".2f", annot_kws={"size": 7},
        )
        step = max(1, pivot.shape[0] // 8)
        ax7.set_xticks(range(0, pivot.shape[0], step))
        ax7.set_xticklabels(pivot.index[::step], rotation=35, ha="right", fontsize=7)
        ax7.set_title("Sentiment Heatmap (Month × Source)", color="#f0f6fc", fontsize=10)
        ax7.set_ylabel(""); ax7.set_xlabel("")
        fig7.tight_layout()
        st.pyplot(fig7)
    else:
        st.info("Need multiple months and source types for heatmap.")

# ─────────────────────────────────────────────────────────────
# Top items
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">04 — Most Extreme Items</div>', unsafe_allow_html=True)

tab_pos, tab_neg, tab_neu = st.tabs(["🟢 Most Positive", "🔴 Most Negative", "⚪ Neutral Sample"])

with tab_pos:
    top_pos = top_items(df, "positive", 8)
    if not top_pos.empty:
        top_pos["compound"] = top_pos["compound"].round(4)
        top_pos["cleaned_text"] = top_pos["cleaned_text"].str[:120] + "…"
        st.dataframe(top_pos, use_container_width=True)

with tab_neg:
    top_neg = top_items(df, "negative", 8)
    if not top_neg.empty:
        top_neg["compound"] = top_neg["compound"].round(4)
        top_neg["cleaned_text"] = top_neg["cleaned_text"].str[:120] + "…"
        st.dataframe(top_neg, use_container_width=True)

with tab_neu:
    top_neu = top_items(df, "neutral", 8)
    if not top_neu.empty:
        top_neu["compound"] = top_neu["compound"].round(4)
        top_neu["cleaned_text"] = top_neu["cleaned_text"].str[:120] + "…"
        st.dataframe(top_neu, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Raw data download
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">05 — Export</div>', unsafe_allow_html=True)
csv = df[["type","author","cleaned_text","compound","label","date","month"]].to_csv(index=False)
st.download_button(
    label     = "⬇️  Download full results as CSV",
    data      = csv,
    file_name = f"{owner}_{repo}_sentiment.csv",
    mime      = "text/csv",
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6e7681;font-size:0.78rem;font-family:'JetBrains Mono',monospace">
  GitHub Sentiment Analyser · VADER · Python · Streamlit
</div>
""", unsafe_allow_html=True)
