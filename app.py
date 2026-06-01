import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Color palette ──────────────────────────────────────────────────────────────
BG        = "#0B1120"
CARD_BG   = "#111827"
GREEN     = "#10B981"
RED       = "#EF4444"
AMBER     = "#F59E0B"
BLUE      = "#3B82F6"
TEXT      = "#F1F5F9"
SUBTEXT   = "#64748B"

# ── Matplotlib global theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD_BG,
    "axes.edgecolor"   : "#1E293B",
    "axes.labelcolor"  : SUBTEXT,
    "xtick.color"      : SUBTEXT,
    "ytick.color"      : SUBTEXT,
    "text.color"       : TEXT,
    "grid.color"       : "#1E293B",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.6,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.titlecolor"  : TEXT,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.spines.left" : False,
    "axes.spines.bottom": False,
})

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️  UPI Payment Fraud Detection")
st.caption("Machine Learning powered fraud intelligence dashboard")
st.divider()

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel('upi_transactions.xlsx')

df = load_data()

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Total Transactions", f"{len(df):,}")
k2.metric("🚨 Fraud Cases",        f"{df['IS_FRAUD'].sum():,}",
          delta=f"{round(df['IS_FRAUD'].mean()*100,2)}% of total",
          delta_color="inverse")
k3.metric("⚠️ Fraud Rate",         f"{round(df['IS_FRAUD'].mean()*100,2)}%")
k4.metric("💰 Total Volume",       f"₹{round(df['AMOUNT'].sum()/1e7,1)} Cr")

st.divider()

# ── Amount distribution ────────────────────────────────────────────────────────
st.subheader("📈  Transaction Amount Distribution")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(wspace=0.35)

for ax, fraud_flag, color, title in [
    (axes[0], 0, GREEN, "Normal Transactions"),
    (axes[1], 1, RED,   "Fraudulent Transactions"),
]:
    data = df[df['IS_FRAUD'] == fraud_flag]['AMOUNT']
    if fraud_flag == 0:
        data = data.clip(upper=10000)

    n, bins, patches = ax.hist(data, bins=50, color=color,
                                alpha=0.9, edgecolor="none", zorder=3)
    # fade bars left→right
    for i, p in enumerate(patches):
        p.set_alpha(0.35 + 0.65 * (i / max(len(patches) - 1, 1)))

    # soft area fill
    mid = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    ax.fill_between(mid, n, alpha=0.07, color=color, zorder=2)

    # mean line
    ax.axvline(data.mean(), color=color, linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"Mean ₹{data.mean():,.0f}")

    ax.set_title(title, pad=12)
    ax.set_xlabel("Amount (₹)", labelpad=8)
    ax.set_ylabel("Count", labelpad=8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor="#1E293B", labelcolor=TEXT)

st.pyplot(fig)
plt.close(fig)

st.divider()

# ── Hourly pattern ─────────────────────────────────────────────────────────────
st.subheader("🕐  Fraud Activity by Hour of Day")

hourly = df.groupby(['HOUR_OF_DAY', 'IS_FRAUD']).size().unstack(fill_value=0)
hours  = np.arange(len(hourly))
w      = 0.38

fig2, ax2 = plt.subplots(figsize=(14, 4.5))
fig2.patch.set_facecolor(BG)

bars_n = ax2.bar(hours - w/2, hourly[0], w, color=GREEN, alpha=0.85,
                  label="Normal", zorder=3, linewidth=0)
bars_f = ax2.bar(hours + w/2, hourly[1], w, color=RED,   alpha=0.85,
                  label="Fraud",  zorder=3, linewidth=0)

# highlight peak fraud hour
if 1 in hourly.columns:
    peak_hour = hourly[1].idxmax()
    peak_idx  = list(hourly.index).index(peak_hour)
    ax2.bar(peak_idx + w/2, hourly[1].max(), w,
            color=RED, alpha=1.0, zorder=4, linewidth=0)
    ax2.annotate(f"Peak ▲ {hourly[1].max():,}",
                 xy=(peak_idx + w/2, hourly[1].max()),
                 xytext=(peak_idx + w/2 + 0.8, hourly[1].max() * 1.08),
                 color=RED, fontsize=8.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax2.set_xticks(hours)
ax2.set_xticklabels([f"{h:02d}:00" for h in hourly.index],
                     rotation=45, ha="right", fontsize=8)
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlabel("Hour of Day", labelpad=8)
ax2.set_ylabel("Transaction Count", labelpad=8)
ax2.legend(handles=[
    mpatches.Patch(color=GREEN, label="Normal"),
    mpatches.Patch(color=RED,   label="Fraud"),
], facecolor=CARD_BG, edgecolor="#1E293B", labelcolor=TEXT, fontsize=10)

plt.tight_layout(pad=1.5)
st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ── Model training ─────────────────────────────────────────────────────────────
st.subheader("🤖  Model Training & Evaluation")

c_btn, c_info = st.columns([1, 2])
with c_btn:
    train_clicked = st.button("⚡  Train Random Forest", use_container_width=True)
with c_info:
    st.info("**Config:** 100 estimators · SMOTE oversampling · 80/20 train-test split · 5 features")

if train_clicked:
    with st.spinner("Training in progress..."):
        features = ['AMOUNT', 'HOUR_OF_DAY', 'NEW_DEVICE',
                    'FAILED_ATTEMPTS_BEFORE', 'DIFFERENT_STATE']
        X   = StandardScaler().fit_transform(df[features])
        y   = df['is_fraud']
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(Xr, yr)
        yp  = clf.predict(Xte)

    st.success("✅  Model trained successfully!")
    st.divider()

    # metrics
    r = classification_report(yte, yp, output_dict=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Precision", f"{round(r['1']['precision']*100, 1)}%")
    m2.metric("🔍 Recall",    f"{round(r['1']['recall']*100, 1)}%")
    m3.metric("⚖️ F1 Score",  f"{round(r['1']['f1-score']*100, 1)}%")

    st.divider()

    # ── Confusion matrix + feature importance side by side ────────────────────
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(yte, yp)
        tn, fp, fn, tp = cm.ravel()

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor(BG)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap="YlGn",
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            ax=ax3,
            linewidths=2, linecolor=BG,
            annot_kws={"size": 18, "weight": "bold", "color": "#111"},
            cbar_kws={"shrink": 0.75},
        )
        ax3.set_xlabel("Predicted", labelpad=10)
        ax3.set_ylabel("Actual",    labelpad=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        total = cm.sum()
        st.caption(
            f"Accuracy: **{round((tn+tp)/total*100,1)}%** · "
            f"TN: {tn:,} · FP: {fp:,} · FN: {fn:,} · TP: {tp:,}"
        )

    with col_fi:
        st.markdown("**Feature Importance**")
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({
            "Feature"   : features,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig4, ax4 = plt.subplots(figsize=(5, 4))
        fig4.patch.set_facecolor(BG)

        colors = [GREEN if v == fi_df["Importance"].max() else BLUE
                  for v in fi_df["Importance"]]
        bars = ax4.barh(fi_df["Feature"], fi_df["Importance"],
                        color=colors, alpha=0.85, edgecolor="none", zorder=3)

        # value labels
        for bar, val in zip(bars, fi_df["Importance"]):
            ax4.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=9, color=TEXT)

        ax4.xaxis.grid(True)
        ax4.set_axisbelow(True)
        ax4.set_xlabel("Importance Score", labelpad=8)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

st.divider()
st.caption("UPI Fraud Shield · Random Forest · Built with Streamlit & scikit-learn")
