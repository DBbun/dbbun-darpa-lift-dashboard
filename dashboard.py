#!/usr/bin/env python
"""
DBbun DARPA Lift Challenge — Interactive Results Dashboard v1
=============================================================
Run:   streamlit run dashboard.py
Deps:  pip install streamlit plotly pandas pyarrow

Loads CSVs from output/ folder produced by the generator.
© 2026 DBbun LLC
"""

import os, warnings, re
import urllib.request
warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import fisher_exact
import pyarrow.csv as pv
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq

st.set_page_config(
    page_title="DBbun · DARPA Lift Challenge",
    page_icon="🚁", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE   = "#003087"
ORANGE = "#FF6600"
GREEN  = "#27AE60"
RED    = "#E74C3C"
AMBER  = "#F39C12"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_animals(raw: str) -> str:
    """'harpy_eagle,osprey' → 'Harpy Eagle + Osprey'"""
    if not isinstance(raw, str) or not raw.strip():
        return "—"
    return " + ".join(a.strip().replace("_", " ").title() for a in raw.split(",") if a.strip())

def fmt_traits(raw: str) -> list:
    if not isinstance(raw, str):
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]

def prize_badge(ratio: float) -> str:
    if ratio >= 4.0: return "🏆 FULL PRIZE (≥4:1)"
    if ratio >= 2.0: return "✅ Qualifies (<4:1, 50% prize)"
    return "❌ Below minimum"

def stars(n: int) -> str:
    n = max(0, min(5, int(n or 0)))
    return "★" * n + "☆" * (5 - n)

def compare_proportions(success_a: int, n_a: int, success_b: int, n_b: int):
    """
    Compares two success rates (e.g. WITH a trait vs WITHOUT it) using Fisher's
    exact test on the underlying 2x2 contingency table. Returns (p_value, sig_label).
    Fisher's exact is used instead of a chi-square/z-test because it's exact at any
    sample size — no large-sample approximation needed, which matters since some
    trait groups here can be small.
    """
    fail_a = n_a - success_a
    fail_b = n_b - success_b
    if n_a == 0 or n_b == 0:
        return None, "n/a"
    _, p = fisher_exact([[success_a, fail_a], [success_b, fail_b]])
    if p < 0.001:
        sig = "*** p<0.001"
    elif p < 0.01:
        sig = "** p<0.01"
    elif p < 0.05:
        sig = "* p<0.05"
    else:
        sig = "ns"
    return p, sig

def format_design_summary(text: str, animals_list: list) -> str:
    """Clean up auto-generated design summary text for display:
    - Replace underscore animal names ('harpy_eagle') with Title Case ('Harpy Eagle')
    - Capitalize material names ('titanium alloy' -> 'Titanium alloy')
    No bolding — plain text throughout, per request.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    out = text

    # 1. Replace each animal's underscore form with a clean Title Case name
    if isinstance(animals_list, list):
        for a in animals_list:
            if not a:
                continue
            pretty = a.replace("_", " ").title()
            out = re.sub(re.escape(a), pretty, out)
    # Catch any remaining generic snake_case animal-like tokens preceded by "by " or ", "
    out = re.sub(r'\b([a-z]+)_([a-z]+)\b',
                 lambda m: m.group(0) if m.group(0) in ("li_ion","li_s")
                           else f"{m.group(1)} {m.group(2)}", out)

    # 2. Capitalize material names wherever they appear. Single combined regex (longest
    #    phrase first) so e.g. "carbon composite" is matched whole, not re-matched a
    #    second time by the shorter standalone "composite" alternative afterward.
    material_names = [
        "carbon composite", "aluminum lithium", "titanium alloy", "magnesium alloy",
        "steel truss", "glass composite",
        "titanium", "steel", "aluminum", "composite",
    ]
    material_alt = "|".join(re.escape(m) for m in sorted(material_names, key=len, reverse=True))
    out = re.sub(
        rf'\b({material_alt})\b',
        lambda m: m.group(0)[0].upper() + m.group(0)[1:],
        out, flags=re.IGNORECASE
    )

    return out

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading designs…")
def load_designs() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "designs.csv"))
    df["animals_fmt"] = df["animals"].apply(fmt_animals)
    # Note: list columns computed outside cache to avoid unhashable-type errors
    return df

@st.cache_data(show_spinner="Loading missions…")
def load_missions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "missions.csv"))
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    if "prize_tier" not in df.columns:
        des = pd.read_csv(os.path.join(OUTPUT_DIR, "designs.csv"))[
            ["design_id", "payload_to_aircraft_ratio"]]
        df = df.merge(des, on="design_id", how="left", suffixes=("", "_d"))
        rc = "payload_to_aircraft_ratio_d" if "payload_to_aircraft_ratio_d" in df.columns \
             else "payload_to_aircraft_ratio"
        df["prize_tier"] = "none"
        df.loc[df["success"] == 1, "prize_tier"] = "partial"
        df.loc[(df["success"] == 1) & (df[rc] >= 4.0), "prize_tier"] = "full"
    return df

@st.cache_data(show_spinner=False)
def build_dm(designs: pd.DataFrame, missions: pd.DataFrame) -> pd.DataFrame:
    keep = ["design_id", "propulsion_architecture", "animals_fmt",
            "energy_system_type", "design_stars", "empty_mass_kg", "rotor_count",
            "trait_count", "traits", "battery_max_power_W",
            "motor_efficiency", "cruise_speed_mps", "structural_material",
            "payload_to_aircraft_ratio"]  # list cols excluded — added after cache
    keep = [c for c in keep if c in designs.columns]
    return missions.merge(designs[keep], on="design_id", how="left", suffixes=("", "_d"))

@st.cache_data(show_spinner="Loading telemetry for design…")
def telemetry_for_design(design_id: str) -> pd.DataFrame:
    parquet = os.path.join(OUTPUT_DIR, "missions_timeseries.parquet")
    if os.path.exists(parquet):
        return pq.read_table(parquet, filters=[("design_id", "=", str(design_id))]).to_pandas()
    csv = os.path.join(OUTPUT_DIR, "missions_timeseries.csv")
    chunks = []
    for chunk in pd.read_csv(csv, chunksize=50_000):
        sub = chunk[chunk["design_id"].astype(str) == str(design_id)]
        if not sub.empty:
            chunks.append(sub)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# ── Design Inspector card ─────────────────────────────────────────────────────
def show_design_inspector(did: str, designs: pd.DataFrame, missions: pd.DataFrame):
    row = designs[designs["design_id"] == did]
    if row.empty:
        st.warning("Design not found.")
        return
    d = row.iloc[0]
    m = missions[missions["design_id"] == did]

    ratio = float(d.get("payload_to_aircraft_ratio", 0))
    srate = float(d.get("design_success_rate", m["success"].mean() if len(m) else 0))
    n_m   = len(m)
    n_full = int((m.get("prize_tier", pd.Series()) == "full").sum()) if "prize_tier" in m.columns else 0

    # ── Header ──
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"### {did}")
        st.markdown(f"**Animals:** {d.get('animals_fmt', fmt_animals(str(d.get('animals',''))))}")
        prize_sc = d.get("prize_rank_score")
        eng_sc   = d.get("design_rank_score")
        score_bits = [f"**Rating:** {stars(d.get('design_stars', 0))}"]
        if pd.notna(prize_sc):
            score_bits.append(f"🏆 DARPA Prize Score: **{prize_sc:.3f}**")
        if pd.notna(eng_sc):
            score_bits.append(f"⚙️ Engineering Rank: {eng_sc:.3f}")
        st.markdown("  ·  ".join(score_bits))
    with col_badge:
        colour = GREEN if ratio >= 4 else (AMBER if ratio >= 2 else RED)
        st.markdown(
            f"<div style='background:{colour};color:white;padding:10px 14px;"
            f"border-radius:8px;font-size:14px;font-weight:bold;text-align:center'>"
            f"P/W  {ratio:.2f}:1<br>{prize_badge(ratio)}</div>",
            unsafe_allow_html=True,
        )

    # ── Natural-language summary (from generator) ──
    summary = d.get("design_summary", "")
    if summary:
        formatted = format_design_summary(summary, d.get("animals_list", []))
        with st.expander("📄 Design description", expanded=True):
            st.markdown(formatted)

    # ── Mission snapshot ──
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Missions",      f"{n_m}")
    mc2.metric("Success rate",  f"{srate*100:.1f}%")
    mc3.metric("Full prize runs", f"{n_full}")
    mc4.metric("Prize tier",
               "Full 🏆" if n_full > n_m * 0.5 else ("Partial ✅" if srate > 0 else "None ❌"))

    # ── All 66 fields in tabs ──
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📐 Mass & Structure", "⚡ Energy", "🔧 Propulsion",
        "🦅 Bio-Inspiration", "📊 Scores & Rules", "🖼 All Fields",
    ])

    def row2(col_a, label_a, val_a, col_b, label_b, val_b):
        col_a.metric(label_a, val_a)
        col_b.metric(label_b, val_b)

    with t1:
        r1a, r1b, r1c, r1d = st.columns(4)
        r1a.metric("Empty mass",    f"{d.get('empty_mass_kg',0):.2f} kg")
        r1b.metric("Payload mass",  f"{d.get('payload_mass_kg',0):.2f} kg")
        r1c.metric("MTOW",          f"{d.get('mtow_kg',0):.2f} kg")
        r1d.metric("P/W ratio",     f"{ratio:.2f}:1")
        r2a, r2b, r2c, r2d = st.columns(4)
        r2a.metric("Rotors",         str(d.get("rotor_count","—")))
        r2b.metric("Frame material", str(d.get("structural_material","—")).replace("_"," "))
        r2c.metric("Rotor blades",   str(d.get("rotor_blade_material","—")).replace("_"," "))
        r2d.metric("Landing gear",   str(d.get("landing_gear_material","—")).replace("_"," "))
        r3a, r3b, r3c, r3d = st.columns(4)
        r3a.metric("Frame stiffness",  f"{d.get('frame_stiffness_longitudinal',0):.3f}")
        r3b.metric("Tendon fraction",   f"{d.get('tendon_cable_fraction',0):.2f}")
        r3c.metric("Gear mass",         f"{d.get('landing_gear_mass_kg',0):.2f} kg")
        r3d.metric("Max touchdown v",   f"{d.get('max_touchdown_velocity_mps',0):.2f} m/s")

    with t2:
        r1a, r1b, r1c, r1d = st.columns(4)
        r1a.metric("Energy system",     str(d.get("energy_system_type","—")))
        r1b.metric("Battery mass",      f"{d.get('battery_mass_kg',0):.2f} kg")
        r1c.metric("Spec energy",       f"{d.get('battery_spec_energy_Wh_per_kg',0):.0f} Wh/kg")
        r1d.metric("Total energy",      f"{d.get('battery_energy_Wh',0):.0f} Wh")
        r2a, r2b, r2c, r2d = st.columns(4)
        r2a.metric("Peak power",        f"{d.get('battery_max_power_W',0)/1000:.1f} kW")
        r2b.metric("Voltage",           f"{d.get('battery_nominal_voltage_V',0):.1f} V")
        r2c.metric("Supercap mass",     f"{d.get('supercap_mass_kg',0):.3f} kg")
        r2d.metric("Supercap energy",   f"{d.get('supercap_energy_Wh',0):.1f} Wh")
        r3a, r3b = st.columns(2)
        r3a.metric("Supercap peak power", f"{d.get('supercap_max_power_W',0):.0f} W")
        r3b.metric("Energy density class", str(d.get("energy_density_class","—")))
        desc = d.get("energy_system_description","")
        if desc:
            st.info(f"💡 {desc}")

    with t3:
        r1a, r1b, r1c, r1d = st.columns(4)
        r1a.metric("Architecture",      str(d.get("propulsion_architecture","—")).replace("_"," "))
        r1b.metric("Motor type",        str(d.get("motor_type","—")).replace("_"," "))
        r1c.metric("Motor efficiency",  f"{d.get('motor_efficiency',0):.3f}")
        r1d.metric("ESC efficiency",    f"{d.get('esc_efficiency',0):.3f}")
        r2a, r2b, r2c, r2d = st.columns(4)
        r2a.metric("ESC current rating", f"{d.get('esc_current_rating_A',0):.0f} A")
        r2b.metric("Max TWR",            f"{d.get('max_twr',0):.2f}")
        r2c.metric("Cruise speed",       f"{d.get('cruise_speed_mps',0):.1f} m/s")
        r2d.metric("Climb rate",         f"{d.get('climb_rate_mps',0):.1f} m/s")
        r3a, r3b, r3c, r3d = st.columns(4)
        r3a.metric("Burst power factor", f"{d.get('burst_power_factor',0):.2f}")
        r3b.metric("Burst duration",     f"{d.get('burst_duration_s',0):.1f} s")
        r3c.metric("Flight modes",       str(d.get("mode_count","—")))
        r3d.metric("Wing foldable",      "Yes" if d.get("wing_foldable") else "No")
        if d.get("wing_foldable"):
            st.caption(f"Deploy time: {d.get('wing_deploy_time_s',0):.1f}s  |  "
                       f"Deploy failure risk: {d.get('wing_deploy_failure_risk',0):.3f}")

    with t4:
        st.markdown(f"**All animals:** {d.get('animals_fmt', '—')}")
        st.markdown(f"**Animal count:** {d.get('animal_count', '—')}")
        st.divider()
        traits = fmt_traits(str(d.get("traits", "")))
        if traits:
            st.markdown("**Active traits:**")
            cols = st.columns(3)
            for i, tr in enumerate(traits):
                cols[i % 3].success(f"✓ {tr.replace('_',' ')}")
        st.markdown(f"\n**Trait count:** {d.get('trait_count','—')}")
        gust = d.get("gust_rejection_gain", 0)
        unsteady = d.get("unsteady_lift_gain", 0)
        c1, c2 = st.columns(2)
        c1.metric("Gust rejection gain",    f"{gust:.3f}")
        c2.metric("Unsteady lift gain",     f"{unsteady:.3f}")

    with t5:
        r0a, r0b = st.columns(2)
        r0a.metric("🏆 DARPA Prize Score", f"{d.get('prize_rank_score',0):.3f}",
                   help="50% full-prize rate + 25% success rate + 15% payload ratio (capped at 4:1) + 10% robustness")
        r0b.metric("⚙️ Engineering Rank Score", f"{d.get('design_rank_score',0):.3f}",
                   help="From the report generator: 55% success rate + 25% qualifying rate + 20% low rule-violations. Does not weight the 4:1 prize threshold.")
        with st.expander("ℹ️ What do these two scores mean, and why are there two?", expanded=False):
            st.markdown(
                "**⚙️ Engineering Rank Score** comes from the report generator itself. It's a general "
                "reliability/compliance score: **55% mission success rate** + **25% qualifying rate** "
                "(success rate, but counted as zero if the design never meets the basic mass/payload "
                "rules in the first place) + **20% rule compliance** (fewer rule violations is better). "
                "It treats any successful, rule-compliant mission the same — it doesn't care whether "
                "the payload ratio cleared 2:1 or 6:1, only that the mission succeeded."
            )
            st.markdown(
                "**🏆 DARPA Prize Score** is computed here in the dashboard specifically to answer "
                "'which design is most likely to win prize money.' The actual competition pays out "
                "based on hitting payload-ratio tiers — a 2:1 design only qualifies, a 4:1 design wins "
                "the full prize — so this score weights that directly: **50% full-prize rate** "
                "(fraction of missions hitting the 4:1 tier) + **25% success rate** + **15% payload "
                "ratio** (capped at 4:1, so going beyond 4:1 doesn't add further credit) + "
                "**10% robustness** (designs that fail in only one or two ways score higher than ones "
                "that fail in many different ways, even at a similar overall success rate)."
            )
            st.markdown(
                "**Why they can disagree:** a design with a 100%-reliable 2.9:1 payload ratio can "
                "outrank a less-reliable 4:1 design on Engineering Rank (since that score doesn't "
                "see the prize threshold at all), while DARPA Prize Score would rank them the other "
                "way around, since only the 4:1 design can win full prize money. The dashboard's "
                "'best design' defaults (highlighted bubble on Leaderboard, auto-selected design "
                "everywhere) use **DARPA Prize Score**, since that's the one that reflects the actual "
                "competition payout."
            )
        r1a, r1b, r1c, r1d = st.columns(4)
        r1a.metric("Stars",              stars(d.get("design_stars",0)))
        r1b.metric("Design success rate",f"{d.get('design_success_rate',0)*100:.1f}%")
        r1c.metric("Qualifying rate",    f"{d.get('design_qualifying_rate',0)*100:.1f}%")
        r1d.metric("Full prize rate",    f"{d.get('full_prize_rate',0)*100:.1f}%")
        r2a, r2b, r2c, r2d = st.columns(4)
        r2a.metric("Rule: empty mass OK",  "✅" if d.get("rule_empty_mass_ok") else "❌")
        r2b.metric("Rule: payload OK",     "✅" if d.get("rule_payload_ok") else "❌")
        r2c.metric("Qualifying design",    "✅" if d.get("design_qualifying") else "❌")
        r2d.metric("Qualifying score",     f"{d.get('design_qualifying_score',0):.3f}")

    with t6:
        # All 66 fields as a scrollable table
        all_fields = d.to_dict()
        field_df = pd.DataFrame([
            {"Field": k, "Value": (f"{v:.3f}" if isinstance(v, float) else str(v))[:120]}
            for k, v in all_fields.items()
        ])
        st.dataframe(field_df, use_container_width=True, hide_index=True, height=400)


# ── Click-to-inspect helper ─────────────────────────────────────────────────
def clicked_id_from_event(event) -> str | None:
    """Extract design_id from a plotly on_select event. Returns None if nothing clicked."""
    try:
        pts = event.selection.get("points", []) if event and hasattr(event, "selection") else []
        if not pts:
            return None
        pt = pts[0]
        # hover_name maps to 'hovertext'; fallback to text/label
        did = (pt.get("hovertext") or pt.get("text") or pt.get("label") or "").strip()
        return did if did else None
    except Exception:
        return None

def sync_click_selection(widget_key: str, designs: pd.DataFrame):
    """Call BEFORE building a chart that uses this widget_key, so the highlight ring
    reflects THIS click immediately rather than lagging one click behind."""
    prior = st.session_state.get(widget_key)
    did = clicked_id_from_event(prior)
    if did and did in designs["design_id"].values:
        st.session_state["selected_design"] = did

def show_selected_inspector(designs: pd.DataFrame, missions: pd.DataFrame):
    """Show the inspector for the currently selected design (session_state), defaulting
    to the top-ranked design. Call this AFTER the chart — selection was already synced."""
    sel = st.session_state.get("selected_design")
    if not sel or sel not in designs["design_id"].values:
        sel = designs.sort_values("prize_rank_score", ascending=False)["design_id"].iloc[0]
        st.session_state["selected_design"] = sel
        st.caption("💡 Showing top-ranked design. Click any bubble to select a different one.")
    st.markdown("---")
    st.markdown(f"### 🔍 Design Inspector — *{sel}*")
    show_design_inspector(sel, designs, missions)

def inspect_on_click(event, designs, missions, key=""):
    """Legacy wrapper kept for pages that haven't been migrated to sync_click_selection."""
    did = clicked_id_from_event(event)
    if did and did in designs["design_id"].values:
        st.session_state["selected_design"] = did
    show_selected_inspector(designs, missions)

# ── Chart explanation helper ──────────────────────────────────────────────────
def chart_help(title: str, what_it_shows: str, key_takeaway: str, action: str):
    with st.expander(f"📖 {title}", expanded=False):
        st.markdown(f"**What this shows:** {what_it_shows}")
        st.markdown(f"**Key takeaway:** {key_takeaway}")
        st.markdown(f"**What to do with this:** {action}")


# ── Load everything ───────────────────────────────────────────────────────────
try:
    designs  = load_designs()
    # Compute list columns here (outside cache) to avoid unhashable-type hash errors
    designs["animals_list"] = designs["animals"].fillna("").apply(
        lambda x: [a.strip() for a in x.split(",") if a.strip()])
    designs["traits_list"] = designs["traits"].apply(fmt_traits)
    missions = load_missions()
    dm       = build_dm(designs, missions)
    dm["animals_list"] = dm["design_id"].map(
        designs.set_index("design_id")["animals_list"])
except FileNotFoundError as e:
    st.error(f"**CSV not found:** {e}\n\nRun `darpa_lift_challenge_generator_v1_2.py` first.")
    st.stop()

# ── Prize-aware ranking score ──────────────────────────────────────────────
# The generator's design_rank_score (success/qualifying/rule-penalty weighted) does
# NOT account for the 4:1 full-prize threshold, so a 100%-reliable 2.9:1 design can
# outrank a less-reliable 4:1 design. For dashboard "best design" purposes, compute
# a separate prize-focused score:
#   50% full-prize rate + 25% mission success rate + 15% payload-ratio (capped at 4:1)
#   + 10% robustness (1 - breadth of distinct failure modes observed)
_m_per_design = missions.groupby("design_id").agg(
    success_rate=("success", "mean"),
    full_prize_rate=("prize_tier", lambda x: (x == "full").mean()) if "prize_tier" in missions.columns else ("success", "mean"),
).reset_index()
_n_failure_types_global = missions.loc[missions["success"] == 0, "failure_reason"].nunique()
_fail_breadth = (
    missions[missions["success"] == 0]
    .groupby("design_id")["failure_reason"].nunique()
    .reindex(designs["design_id"]).fillna(0)
)
_robustness = 1.0 - (_fail_breadth / max(_n_failure_types_global, 1)).clip(0, 1)
_m_per_design = _m_per_design.merge(
    _robustness.rename("robustness").reset_index(), on="design_id", how="left"
)
designs = designs.merge(_m_per_design, on="design_id", how="left")
designs["robustness"] = designs["robustness"].fillna(1.0)
designs["full_prize_rate"] = designs["full_prize_rate"].fillna(0.0)
designs["success_rate"] = designs["success_rate"].fillna(0.0)
designs["payload_ratio_norm"] = (designs["payload_to_aircraft_ratio"] / 4.0).clip(0, 1)
designs["prize_rank_score"] = (
    0.50 * designs["full_prize_rate"] +
    0.25 * designs["success_rate"] +
    0.15 * designs["payload_ratio_norm"] +
    0.10 * designs["robustness"]
)

# One-time parquet conversion — uses pyarrow's native CSV reader (not pandas) to build
# the Parquet index, since skipping the pandas round-trip is ~3.7x faster on this file.
# If you're deploying online, see prebuild_parquet.py to do this step ahead of time
# instead of on the deployed app's first run (see notes there on why that matters).
#
# DEPLOYMENT NOTE: missions_timeseries.parquet is too large for GitHub's normal 25MB
# browser-upload limit. If you've hosted it as a GitHub Release asset instead (see
# deployment instructions), set PARQUET_DOWNLOAD_URL below to that asset's download
# link and the app will fetch it automatically on first run. Leave it as None if the
# parquet is already committed directly in the repo's output/ folder.
PARQUET_DOWNLOAD_URL = None  # e.g. "https://github.com/DBbun/dbbun-darpa-lift-dashboard/releases/download/data-v1/missions_timeseries.parquet"

_pq = os.path.join(OUTPUT_DIR, "missions_timeseries.parquet")
_csv = os.path.join(OUTPUT_DIR, "missions_timeseries.csv")
if not os.path.exists(_pq) and os.path.exists(_csv):
    with st.spinner("One-time setup: building fast telemetry index (~10–25 s)…"):
        pq.write_table(pv.read_csv(_csv), _pq)
    st.rerun()
elif not os.path.exists(_pq) and not os.path.exists(_csv) and PARQUET_DOWNLOAD_URL:
    with st.spinner("One-time setup: downloading telemetry data…"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        urllib.request.urlretrieve(PARQUET_DOWNLOAD_URL, _pq)
    st.rerun()

n_d, n_m = len(designs), len(missions)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚁 DBbun · DARPA Lift\n*Dashboard v1*")
st.sidebar.markdown(f"**{n_d:,} designs · {n_m:,} missions**")
sr = missions["success"].mean()
pct_full = (missions.get("prize_tier", pd.Series()) == "full").mean() \
           if "prize_tier" in missions.columns else 0
st.sidebar.metric("Overall success rate", f"{sr*100:.1f}%")
st.sidebar.metric("Full prize rate (≥4:1)", f"{pct_full*100:.1f}%")
st.sidebar.divider()

PAGE = st.sidebar.radio("Navigation", [
    "📋  Executive Summary",
    "🏆  Competition Leaderboard",
    "💥  Failure Autopsy",
    "🦅  Bio-Inspiration Benchmark",
    "📡  Flight Recorder",
    "🔬  Design DNA",
], label_visibility="collapsed")

st.sidebar.divider()
# ── Trait set computed once at module level (used by multiple pages) ────────────
all_traits_set = set()
for _ts in dm["traits"].fillna(""):
    for _t in str(_ts).split(","):
        _t = _t.strip()
        if _t: all_traits_set.add(_t)

# ── Universal design inspector in sidebar ────────────────────────────────────
st.sidebar.markdown("### 🔍 Inspect Any Design")
all_ids = designs.sort_values("prize_rank_score", ascending=False)["design_id"].tolist()

def design_label(did):
    r = designs[designs["design_id"] == did]
    if r.empty: return did
    d = r.iloc[0]
    # Build the animal list without ever cutting a name mid-word: add whole animal
    # names until the character budget is used up, then show "+N more" if needed.
    animals_full = str(d.get('animals_fmt','') or '')
    animal_parts = [a.strip() for a in animals_full.split('+') if a.strip()]
    budget = 28
    shown = []
    for a in animal_parts:
        candidate = " + ".join(shown + [a])
        if len(candidate) > budget and shown:
            break
        shown.append(a)
    animals_disp = " + ".join(shown)
    n_more = len(animal_parts) - len(shown)
    if n_more > 0:
        animals_disp += f" +{n_more} more"
    return (f"{did}  {stars(d.get('design_stars',0))}  "
            f"P/W {d.get('payload_to_aircraft_ratio',0):.2f}  "
            f"| {animals_disp}")

inspect_id = st.sidebar.selectbox("Select design:", all_ids,
                                   format_func=design_label,
                                   key="sidebar_inspector")
if st.sidebar.button("🔎 Jump to this design"):
    st.session_state["selected_design"] = inspect_id
st.sidebar.caption("Opens the full design inspector on the Leaderboard or Design DNA page.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════
if PAGE == "📋  Executive Summary":
    st.title("📋 Executive Summary")
    st.caption("Auto-generated analysis of the full simulation run. Synthesizes all designs and missions.")

    # ── KPI banner ──
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Designs",        f"{n_d:,}")
    k2.metric("Missions",       f"{n_m:,}")
    k3.metric("Success rate",   f"{sr*100:.1f}%")
    k4.metric("Full prize (4:1)", f"{pct_full*100:.1f}%")
    best_ratio = designs["payload_to_aircraft_ratio"].max()
    k5.metric("Best P/W ratio", f"{best_ratio:.2f}:1")
    top_star = int(designs["design_stars"].max())
    k6.metric("Top star rating", stars(top_star))
    st.divider()

    # ── Score explanation — visible directly here, not buried in an inspector tab ──
    _best = designs.sort_values("prize_rank_score", ascending=False).iloc[0]
    sc1, sc2 = st.columns(2)
    sc1.metric("🏆 DARPA Prize Score (best design)", f"{_best.get('prize_rank_score',0):.3f}")
    sc2.metric("⚙️ Engineering Rank Score (best design)", f"{_best.get('design_rank_score',0):.3f}")
    st.markdown(
        "**⚙️ Engineering Rank Score** comes from the report generator itself: "
        "**55% mission success rate** + **25% qualifying rate** (success rate, but counted as "
        "zero if the design never meets the basic mass/payload rules) + **20% rule compliance**. "
        "It treats any successful, rule-compliant mission the same — it doesn't care whether the "
        "payload ratio cleared 2:1 or 6:1, only that the mission succeeded."
    )
    st.markdown(
        "**🏆 DARPA Prize Score** is computed here in the dashboard specifically to answer "
        "'which design is most likely to win prize money.' The competition pays out based on "
        "hitting payload-ratio tiers — a 2:1 design only qualifies, a 4:1 design wins the full "
        "prize — so this score weights that directly: **50% full-prize rate** (hitting the 4:1 "
        "tier) + **25% success rate** + **15% payload ratio** (capped at 4:1) + **10% robustness** "
        "(failing in fewer distinct ways)."
    )
    st.markdown(
        "**Why they can disagree:** a 100%-reliable 2.9:1 design can outrank a less-reliable 4:1 "
        "design on Engineering Rank (it never sees the prize threshold), while DARPA Prize Score "
        "ranks them the other way — only the 4:1 design can win full prize money. The dashboard's "
        "'best design' defaults everywhere use **DARPA Prize Score**, since that's the one tied "
        "to actual competition payout."
    )
    st.divider()

    # ── Compute insights ──
    # NOTE: designs already carries a canonical "success_rate" column from the global
    # prize_rank_score setup above — do NOT re-derive it here, or pandas will silently
    # rename both copies to success_rate_x/success_rate_y on merge (collision bug).
    m_agg = missions.groupby("design_id").agg(
        n=("success","count"),
        full_rate=("prize_tier", lambda x: (x=="full").mean()) if "prize_tier" in missions.columns else ("success","mean")
    ).reset_index()
    dna = designs.merge(m_agg, on="design_id", how="left")

    numeric_for_corr = ["battery_max_power_W","battery_spec_energy_Wh_per_kg",
                        "cruise_speed_mps","motor_efficiency","payload_to_aircraft_ratio",
                        "rotor_count","empty_mass_kg","max_twr","battery_energy_Wh"]
    numeric_for_corr = [c for c in numeric_for_corr if c in dna.columns]
    corr = dna[numeric_for_corr + ["success_rate"]].corr()["success_rate"].drop("success_rate").sort_values(ascending=False)

    top_corr_pos  = corr[corr > 0].index[0] if (corr > 0).any() else "—"
    top_corr_neg  = corr[corr < 0].index[-1] if (corr < 0).any() else "—"
    top_corr_r    = corr.iloc[0]

    # Rank correlations by absolute magnitude (a strong negative matters as much as positive)
    corr_abs_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    top_predictor      = corr_abs_sorted.index[0]
    top_predictor_r    = corr_abs_sorted.iloc[0]
    top_predictor_label = top_predictor.replace("_", " ").title()

    def _strength_word(r):
        ar = abs(r)
        if ar >= 0.40: return "strong"
        if ar >= 0.20: return "moderate"
        if ar >= 0.10: return "weak"
        return "negligible"

    _strength = _strength_word(top_predictor_r)
    _direction = "higher" if top_predictor_r > 0 else "lower"

    # Trait impact
    trait_rows = []
    for trait in all_traits_set:
        has   = dm[dm["traits"].fillna("").str.contains(trait, regex=False)]
        hasnt = dm[~dm["traits"].fillna("").str.contains(trait, regex=False)]
        if len(has) > 4 and len(hasnt) > 4:
            trait_rows.append({"trait":trait,
                                "delta":has["success"].mean()-hasnt["success"].mean(),
                                "with_rate":has["success"].mean()})
    trait_df = pd.DataFrame(trait_rows).sort_values("delta", ascending=False)

    # Top animals
    animal_rows = []
    for _, row in dna.iterrows():
        for a in str(row.get("animals","")).split(","):
            a=a.strip()
            if a: animal_rows.append({"animal":a,"success":row.get("success_rate",0)})
    animal_df = pd.DataFrame(animal_rows).groupby("animal")["success"].mean().sort_values(ascending=False)

    # Architecture
    arch_stats = dm.groupby("propulsion_architecture").agg(
        success_rate=("success","mean"),
        full_rate=("prize_tier",lambda x:(x=="full").mean()) if "prize_tier" in dm.columns else ("success","mean")
    ).sort_values("full_rate", ascending=False)

    best_arch     = arch_stats.index[0]
    best_arch_sr  = arch_stats.iloc[0]["full_rate"]
    worst_failure = missions[missions["success"]==0]["failure_reason"].mode()[0] \
                    if len(missions[missions["success"]==0]) > 0 else "—"

    top5_designs = dna.nlargest(5,"success_rate") if len(dna)>0 else dna
    avg_batt_top  = top5_designs["battery_max_power_W"].mean()/1000
    avg_ratio_top = top5_designs["payload_to_aircraft_ratio"].mean()
    avg_eff_top   = top5_designs["motor_efficiency"].mean()

    # ── What the data says ──
    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader("🔍 What the data says")

        findings = [
            (f"**{top_predictor_label}** shows the {_strength} linear association with mission "
             f"success in this run (Pearson r = {top_predictor_r:.3f}; {_direction} values associate "
             f"with more successes). "
             + (f"With |r| in the {_strength} range, treat this as a contributing factor rather than "
                f"a dominant driver — no single parameter explains most of the outcome variance here. "
                if _strength in ("weak", "negligible") else
                f"This is the clearest single-parameter signal in the dataset for this run. ")
             + f"Battery peak power remains an important constraint specifically for hover turns and "
               f"power saturation, independent of its correlation ranking.",
             GREEN),

            (f"**{best_arch.replace('_',' ').title()}** achieves the highest full-prize rate "
             f"({best_arch_sr*100:.1f}% of missions reach 4:1). "
             f"This architecture's combination of energy density and power delivery "
             f"best matches the 5-nautical-mile competition course.",
             BLUE),

            (f"**The 4:1 threshold** is the critical design inflection. "
             f"Only {pct_full*100:.1f}% of all simulated missions achieve it. "
             f"The top designs reach it by targeting payload mass ≥ {4*designs['empty_mass_kg'].median():.0f} kg "
             f"— build the lightest possible airframe first, then maximise payload.",
             AMBER),

            (f"**{worst_failure.replace('_',' ').title()}** is the most common failure reason "
             f"({(missions['failure_reason']==worst_failure).mean()*100:.1f}% of all failures). "
             + ("Increase battery peak power or add supercapacitor burst assist." 
                if "power" in worst_failure else 
                "Reduce total mission time by increasing cruise speed or reducing climb altitude."
                if "time" in worst_failure else
                "Improve landing gear design or reduce descent rate."),
             RED),

            (f"**Hover turns** (16–20 per mission) are a significant energy drain. "
             f"Designs with high battery peak power AND high motor efficiency fare best "
             f"because each turn briefly demands near-hover power. "
             f"Top 5 designs average motor efficiency = {avg_eff_top:.3f}.",
             BLUE),
        ]

        for text, colour in findings:
            _html = __import__('re').sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            st.markdown(
                f"<div style='border-left:4px solid {colour};padding:10px 14px;"
                f"margin-bottom:10px;background:#f8f9fa;border-radius:0 6px 6px 0;"
                f"text-align:left'>"
                f"{_html}</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.subheader("📊 Prize tier overview")
        if "prize_tier" in missions.columns:
            tc = missions["prize_tier"].value_counts().reset_index()
            tc.columns=["tier","count"]
            fig = px.pie(tc, names="tier", values="count", hole=0.5,
                         color="tier",
                         color_discrete_map={"full":GREEN,"partial":AMBER,"none":RED})
            fig.update_layout(height=220, showlegend=True, margin=dict(t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏗 Architecture full-prize rates")
        fig2 = px.bar(arch_stats.reset_index(),
                      x="propulsion_architecture", y="full_rate",
                      color="full_rate",
                      color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                      text=arch_stats["full_rate"].apply(lambda v:f"{v*100:.1f}%").values)
        fig2.update_traces(textposition="outside")
        fig2.update_yaxes(tickformat=".0%", title="Full prize rate")
        fig2.update_xaxes(tickangle=-30, title="")
        fig2.update_layout(height=240,coloraxis_showscale=False,margin=dict(t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Recommendations ──
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("✅ What to do to build a winning drone")
        recs = [
            f"**Target battery peak power ≥ {avg_batt_top*1.1:.0f} kW.** "
            f"The top 5 designs average {avg_batt_top:.1f} kW. Undersized batteries fail at hover turns.",

            f"**Aim for P/W ratio ≥ 4.0 ({4*designs['empty_mass_kg'].median():.0f} kg payload at median empty mass).** "
            f"This unlocks 100% of the prize tier vs 50% below 4:1.",

            f"**Maximise motor efficiency (target ≥ {avg_eff_top:.2f}).** "
            f"High-efficiency motors reduce heat buildup and extend effective mission time.",

            f"**Design for cruise speed ≥ {top5_designs['cruise_speed_mps'].mean():.0f} m/s.** "
            f"Time-limit failures account for a large share of losses. "
            f"Faster cruise is often more impactful than lighter weight.",

            f"**Use {best_arch.replace('_',' ')} propulsion** or hybridise with gas/ICE "
            f"for higher energy density. FAA confirmed all propulsion types are legal.",

            "**Test hover turns explicitly.** 16–20 turns per mission each demand near-hover "
            "power for ~12 seconds. Your battery must handle sustained bursts, not just peak.",
        ]
        for i, r in enumerate(recs, 1):
            st.markdown(f"**{i}.** {r}")
            st.markdown("")

    with c4:
        st.subheader("❌ What to avoid")
        avoids = [
            f"**Undersized battery peak power.** "
            f"Battery_max_power_W has the highest correlation with success (r = {corr.get('battery_max_power_W',0):.3f}). "
            f"A large energy capacity (Wh) does not help if C-rate is too low to deliver hover power.",

            f"**High empty mass without payload compensation.** "
            f"Empty mass has correlation r = {corr.get('empty_mass_kg',0):.3f} with success. "
            f"Every extra kilogram of airframe reduces your payload margin and P/W ratio.",

            "**Ignoring the return leg.** The 1 nmi unloaded return requires stable flight at a "
            "very different center of mass. Designs tuned only for loaded flight fail here.",

            "**Single-point energy failure.** "
            f"{(missions['failure_reason']=='energy_system_fault').mean()*100:.1f}% of failures "
            "are energy system faults. Add supercapacitor burst assist for redundancy.",

            "**Complex wing deployment with poor reliability.** "
            f"Wing deploy failures account for "
            f"{(missions['failure_reason']=='wing_deploy_failure').mean()*100:.1f}% of losses. "
            "If using foldable wings, test deployment extensively under turbulence.",

            "**Over-counting burst capability.** Burst power boosts available energy on the supply side "
            "— it does not reduce physical hover power demand. "
            "Size your baseline battery to sustain hover without burst.",
        ]
        for i, a in enumerate(avoids, 1):
            st.markdown(f"**{i}.** {a}")
            st.markdown("")

    st.divider()
    show_selected_inspector(designs, missions)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Leaderboard
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "🏆  Competition Leaderboard":
    st.title("🏆 Competition Leaderboard")

    with st.expander("Filters"):
        fc1, fc2, fc3 = st.columns(3)
        prop_opts = ["All"] + sorted(designs["propulsion_architecture"].dropna().unique())
        sel_prop  = fc1.selectbox("Propulsion", prop_opts)
        animal_flat = sorted({a for lst in designs["animals_list"] for a in lst})
        sel_animal  = fc2.selectbox("Animal archetype (any)", ["All"] + animal_flat)
        star_opts   = {"All ratings": 0, "⭐⭐⭐⭐⭐  5 stars only": 5,
                       "⭐⭐⭐⭐+  4 stars or above": 4,
                       "⭐⭐⭐+  3 stars or above": 3,
                       "⭐⭐+  2 stars or above": 2}
        sel_stars_label = fc3.selectbox("Minimum star rating", list(star_opts.keys()))
        min_stars = star_opts[sel_stars_label]

    df_d = designs.copy()
    if sel_prop   != "All": df_d = df_d[df_d["propulsion_architecture"] == sel_prop]
    if sel_animal != "All": df_d = df_d[df_d["animals_list"].apply(lambda lst: sel_animal in lst)]
    if min_stars == 5:      df_d = df_d[df_d["design_stars"] == 5]
    elif min_stars > 0:     df_d = df_d[df_d["design_stars"] >= min_stars]

    # NOTE: designs already carries a canonical "success_rate" column globally —
    # do not re-derive it here (causes a merge-collision rename to success_rate_x/_y).
    m_agg = missions.groupby("design_id").agg(
        n_missions=("mission_id","count"),
        full_rate=("prize_tier",lambda x:(x=="full").mean()) if "prize_tier" in missions.columns else ("success","mean"),
    ).reset_index()
    df_d = df_d.merge(m_agg, on="design_id", how="left")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Designs shown",       f"{len(df_d)}")
    k2.metric("Avg success rate",    f"{df_d['success_rate'].mean()*100:.1f}%")
    k3.metric("Full prize (≥4:1)",   f"{df_d['full_rate'].mean()*100:.1f}%")
    k4.metric("Best P/W ratio",      f"{df_d['payload_to_aircraft_ratio'].max():.2f}:1")
    k5.metric("5-star designs",      str(int((df_d["design_stars"]==5).sum())))
    st.divider()

    # ── Full-width scatter ────────────────────────────────────────────────────
    sync_click_selection("lb_scatter", designs)   # read THIS click before drawing, no lag
    # If nothing has ever been selected (e.g. first page load), default to the top-ranked
    # design NOW, before the chart is built — otherwise the highlight ring would lag one
    # interaction behind, only appearing after the user's first click.
    if (not st.session_state.get("selected_design")
            or st.session_state.get("selected_design") not in designs["design_id"].values):
        st.session_state["selected_design"] = (
            designs.sort_values("prize_rank_score", ascending=False)["design_id"].iloc[0]
        )

    sdf = df_d.sample(min(500,len(df_d)),random_state=42) if len(df_d)>500 else df_d
    if len(df_d)>500:
        st.caption(f"⚡ Sampling 500 of {len(df_d):,} for chart speed.")
    sdf = sdf.copy()
    sdf["hover_animals"] = sdf["animals_fmt"]

    # Trait list truncation: never cut mid-word — add whole traits until the budget is
    # used up, then show "+N more" instead of trailing off mid-trait-name.
    def _truncate_traits(s, budget=140):
        parts = [p.strip() for p in str(s or "").split(",") if p.strip()]
        shown = []
        for p in parts:
            candidate = ", ".join(shown + [p])
            if len(candidate) > budget and shown:
                break
            shown.append(p)
        disp = ", ".join(shown)
        n_more = len(parts) - len(shown)
        if n_more > 0:
            disp += f" (+{n_more} more)"
        return disp
    sdf["hover_traits"] = sdf["traits"].fillna("").apply(_truncate_traits)

    # Humanized propulsion label for the legend: "pure_rotor_electric" -> "Pure rotor electric"
    sdf["propulsion_label"] = sdf["propulsion_architecture"].apply(
        lambda s: str(s).replace("_"," ").capitalize()
    )

    # Manually build the full hover text with " = " spacing throughout — Plotly Express's
    # automatic hover_data formatting always renders "Label=Value" with no spaces, which
    # isn't adjustable through hover_data/labels alone.
    sdf["hover_text"] = (
        "<b>" + sdf["design_id"].astype(str) + "</b><br>" +
        "Animals = " + sdf["hover_animals"].astype(str) + "<br>" +
        "Traits = " + sdf["hover_traits"].astype(str) + "<br>" +
        "P/W Ratio = " + sdf["payload_to_aircraft_ratio"].round(2).astype(str) + "<br>" +
        "Success Rate = " + (sdf["success_rate"]*100).round(1).astype(str) + "%<br>" +
        "Full prize rate = " + (sdf["full_rate"]*100).round(2).astype(str) + "%<br>" +
        "Empty mass (kg) = " + sdf["empty_mass_kg"].round(1).astype(str) + "<br>" +
        "Battery peak (W) = " + sdf["battery_max_power_W"].round(0).astype(int).astype(str) + "<br>" +
        "Motor eff. = " + sdf["motor_efficiency"].round(3).astype(str) + "<br>" +
        "Cruise (m/s) = " + sdf["cruise_speed_mps"].round(1).astype(str) + "<br>" +
        "Rotor count = " + sdf["rotor_count"].astype(str) + "<br>" +
        "Stars = " + sdf["design_stars"].astype(str)
    )

    fig = px.scatter(sdf,
        x="payload_to_aircraft_ratio", y="success_rate",
        size="design_stars", color="propulsion_label",
        hover_name="design_id",   # required for click-to-select (clicked_id_from_event reads
                                   # this via pt["hovertext"]) — NOT for the visual tooltip,
                                   # which is fully overridden by the custom hovertemplate below
        hover_data={"hover_text": False},   # populates customdata without an auto-generated row
        labels={"payload_to_aircraft_ratio":"P/W Ratio","success_rate":"Success Rate",
                "propulsion_label":"Propulsion"},
        size_max=26, opacity=0.82,
        title="Click any bubble to inspect that design ▼",
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>",
                      hoverlabel=dict(align="left"))
    fig.add_vline(x=4.0, line_dash="dash", line_color=GREEN,
                  annotation_text="4:1 full prize", annotation_font_size=12,
                  annotation_bgcolor="white", annotation_bordercolor=GREEN,
                  annotation_borderwidth=1.5, annotation_borderpad=3)
    fig.add_vline(x=2.0, line_dash="dot", line_color=AMBER,
                  annotation_text="2:1 qualifying", annotation_font_size=12,
                  annotation_bgcolor="white", annotation_bordercolor=AMBER,
                  annotation_borderwidth=1.5, annotation_borderpad=3)

    # ── Selected design marker ────────────────────────────────────────────────
    _sel_id = st.session_state.get("selected_design")
    if _sel_id and _sel_id in df_d["design_id"].values:
        _sel_row = df_d[df_d["design_id"] == _sel_id].iloc[0]
        _x = float(_sel_row["payload_to_aircraft_ratio"])
        _y = float(_sel_row["success_rate"])
        # Outer orange ring
        fig.add_trace(go.Scatter(x=[_x], y=[_y], mode="markers", hoverinfo="skip",
            marker=dict(size=52, symbol="circle-open", color="rgba(0,0,0,0)",
                        line=dict(color="#FF4500", width=5)),
            showlegend=False, name=""))
        # Inner black square
        fig.add_trace(go.Scatter(x=[_x], y=[_y], mode="markers", hoverinfo="skip",
            marker=dict(size=36, symbol="square-open", color="rgba(0,0,0,0)",
                        line=dict(color="black", width=3)),
            showlegend=False, name=""))
        # Arrow annotation
        fig.add_annotation(x=_x, y=_y,
            text=f"<b>▶ {_sel_id[:16]}</b>",
            showarrow=True, arrowhead=3, arrowcolor="#FF4500", arrowwidth=2,
            ax=60, ay=-50, bgcolor="white", bordercolor="#FF4500",
            borderwidth=2, font=dict(size=12, color="#FF4500"))

    # Small padding beyond 0%/100% so bubbles centered at the extremes aren't visually
    # clipped by the plot boundary — ticks still land exactly on 0%, 20%, ... 100%.
    fig.update_yaxes(tickformat=".0%", range=[-0.09, 1.09], dtick=0.2, tick0=0)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
        margin=dict(t=50, b=10),
    )
    ev_lb = st.plotly_chart(fig, on_select="rerun", key="lb_scatter",
                            use_container_width=True)

    chart_help(
        "Reading this scatter plot",
        "Each bubble is one design. X-axis = payload-to-weight ratio (higher → heavier loads). "
        "Y-axis = fraction of missions that succeeded. Bubble size = star rating (1–5). Colour = propulsion type.",
        "Designs in the upper-right quadrant beyond the green dashed line are prize contenders: "
        "they carry ≥4× their own weight AND complete missions reliably.",
        "Use the Filters above to narrow by propulsion or animal archetype. "
        "Click any bubble — an orange ring + black square + label marks the selection, "
        "and the full design card appears below."
    )

    # ── Prize tier summary — compact row below the scatter ────────────────────
    if "prize_tier" in missions.columns:
        st.divider()
        tc   = missions["prize_tier"].value_counts()
        total_m = len(missions)
        p0, pa, pb, pc_ = st.columns(4)
        p0.metric("📊 Total missions", f"{total_m:,}")
        pa.metric("🏆 Full prize  (≥4:1)",
                  f"{tc.get('full',0):,} missions",
                  f"{tc.get('full',0)/total_m*100:.1f}% of all")
        pb.metric("🟡 Partial  (2–4:1)",
                  f"{tc.get('partial',0):,} missions",
                  f"{tc.get('partial',0)/total_m*100:.1f}% of all")
        pc_.metric("🔴 None  (failed/unqualified)",
                   f"{tc.get('none',0):,} missions",
                   f"{tc.get('none',0)/total_m*100:.1f}% of all")

    show_selected_inspector(designs, missions)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Failure Autopsy
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "💥  Failure Autopsy":
    st.title("💥 Failure Autopsy")
    st.caption("Why do missions fail — and what does that mean for your design?")

    failed = missions[missions["success"]==0]
    total, n_fail = len(missions), len(failed)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total missions", f"{total:,}")
    k2.metric("Failed",         f"{n_fail:,}")
    k3.metric("Failure rate",   f"{n_fail/total*100:.1f}%")
    top_r = failed["failure_reason"].value_counts().idxmax() if len(failed) else "—"
    k4.metric("Top failure",    top_r.replace("_"," ").title() if isinstance(top_r,str) else "—")
    st.divider()

    c1,c2 = st.columns(2)
    with c1:
        rc = failed["failure_reason"].value_counts().reset_index()
        rc.columns=["reason","count"]
        rc["pct"] = rc["count"]/n_fail*100
        rc["label"] = rc["reason"].str.replace("_"," ").str.title()
        rc["fix"] = rc["reason"].map({
            "power_saturation":     "↑ Battery C-rate or add supercaps",
            "time_limit_exceeded":  "↑ Cruise speed or ↓ hover altitude",
            "hard_touchdown":       "↑ Gear rating or ↓ descent rate",
            "gust_induced_instability": "↑ Gust rejection gain / add STAB traits",
            "energy_system_fault":  "Add energy system redundancy",
            "structural_overload":  "↑ Frame stiffness or ↓ payload",
            "wing_deploy_failure":  "Test wing deployment in turbulence",
            "propulsion_architecture_fault": "↑ Motor redundancy (more rotors)",
            "control_saturation":   "↑ Control authority / ↓ payload CG offset",
        }).fillna("Review design parameters")
        fig = px.bar(rc, x="count", y="label", orientation="h",
                     color="pct", color_continuous_scale=[[0,AMBER],[1,RED]],
                     hover_data={"fix":True,"pct":":.1f"},
                     title="Failure Reasons — hover for fix",
                     labels={"count":"Missions","label":"","pct":"% of failures","fix":"Suggested fix"})
        fig.update_layout(height=380,coloraxis_showscale=False,
                          yaxis={"categoryorder":"total ascending"})
        ev_fa = st.plotly_chart(fig, on_select="rerun", key="fa_reasons", use_container_width=True)

    chart_help(
        "Reading the failure reason chart",
        "Each bar = number of missions that failed for that reason. Hover over any bar to see a suggested design fix.",
        "Your dominant failure reason is the single most important thing to fix first. "
        "Addressing it unlocks the next failure mode layer beneath.",
        "Map each failure reason to a specific design parameter change using the fix column. "
        "Then re-run the generator with that parameter adjusted and compare success rates."
    )

    with c2:
        pc = failed["failure_phase"].fillna("unknown").value_counts().reset_index()
        pc.columns=["phase","count"]
        pc["label"] = pc["phase"].str.replace("_"," ").str.title()
        fig2 = px.pie(pc,names="label",values="count",hole=0.45,
                      title="Failure Phase Distribution")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    chart_help(
        "Reading the failure phase chart",
        "Shows in which flight phase failures occur most often. "
        "Takeoff = power problem. Loaded cruise = energy or gust problem. Landing = gear problem.",
        "If most failures happen early (takeoff/climb), it's a power problem. "
        "If late (cruise/landing), it's energy budget or structural.",
        "Target your design changes at the phase where failures concentrate."
    )

    st.subheader("Failure Rate by Environmental Condition")
    st.caption("Simpler view: how does failure rate change as wind or turbulence increases?")

    dm2 = dm.copy()
    dm2["wind_bin"] = pd.cut(dm2["wind_speed_kts"],  bins=5, precision=0)
    dm2["turb_bin"] = pd.cut(dm2["turbulence_index"], bins=5, precision=2)

    wind_fail = dm2.groupby("wind_bin", observed=True)["success"].apply(lambda x: 1-x.mean()).reset_index()
    wind_fail["wind_str"] = wind_fail["wind_bin"].astype(str)
    turb_fail = dm2.groupby("turb_bin", observed=True)["success"].apply(lambda x: 1-x.mean()).reset_index()
    turb_fail["turb_str"] = turb_fail["turb_bin"].astype(str)

    cw1, cw2 = st.columns(2)
    with cw1:
        figw = px.bar(wind_fail, x="wind_str", y="success",
                      title="Failure Rate vs Wind Speed",
                      labels={"wind_str":"Wind speed band (kts)","success":"Failure rate"},
                      color="success", color_continuous_scale=[[0,GREEN],[0.5,AMBER],[1,RED]])
        figw.update_yaxes(tickformat=".0%")
        figw.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(figw, use_container_width=True)
    with cw2:
        figt = px.bar(turb_fail, x="turb_str", y="success",
                      title="Failure Rate vs Turbulence",
                      labels={"turb_str":"Turbulence index band","success":"Failure rate"},
                      color="success", color_continuous_scale=[[0,GREEN],[0.5,AMBER],[1,RED]])
        figt.update_yaxes(tickformat=".0%")
        figt.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(figt, use_container_width=True)

    chart_help(
        "Reading these two charts",
        "Each bar shows the failure rate for missions flown in that wind speed or turbulence band. "
        "Taller red bars on the right = this design population struggles in worse conditions.",
        "A roughly flat set of bars means this population of designs is robust to environmental "
        "variation. A steep upward slope means wind or turbulence is a major failure driver.",
        "If turbulence drives failures more than wind, prioritise STAB_GUST_REJECTION and "
        "PAYLOAD_DAMPED traits. If wind drives failures, prioritise WING_MORPHING_MEMBRANE "
        "and higher battery peak power to overcome headwind power penalties."
    )

    st.info(
        "ℹ️ **Why this recommendation can differ from the Bio-Inspiration Benchmark page:** "
        "the trait impact chart on that page shows each trait's AVERAGE effect across all missions "
        "(calm and stormy combined). STAB_GUST_REJECTION and PAYLOAD_DAMPED can show a negative "
        "average there while still being the correct fix here — they're condition-specific traits "
        "whose benefit concentrates in high-wind/high-turbulence missions and gets diluted by all "
        "the calm-weather missions where they add weight without payoff. If most of your competition "
        "runs are expected in calm conditions, weight the average-effect chart more heavily; if you "
        "expect wind or turbulence on competition day, weight this page's condition-specific guidance."
    )

    # Sensitivity summary
    wind_sensitivity = wind_fail["success"].iloc[-1] - wind_fail["success"].iloc[0] if len(wind_fail) > 1 else 0
    turb_sensitivity = turb_fail["success"].iloc[-1] - turb_fail["success"].iloc[0] if len(turb_fail) > 1 else 0
    st.info(
        f"**Wind sensitivity:** failure rate rises by **{wind_sensitivity*100:.1f}%** from calm to max wind.  \n"
        f"**Turbulence sensitivity:** failure rate rises by **{turb_sensitivity*100:.1f}%** from calm to max turbulence.  \n"
        f"To reduce both: add **STAB_GUST_REJECTION** (Osprey/Dragonfly) and "
        f"**WING_MORPHING_MEMBRANE** (Bat) — both directly reduce environmental power overhead."
    )

    # Note: failure reason bars don't map to a single design.
    # Use the Leaderboard or Bio-Inspiration pages to inspect individual designs.


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Bio-Inspiration Benchmark
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "🦅  Bio-Inspiration Benchmark":
    st.title("🦅 Bio-Inspiration Benchmark")
    st.caption("Which animals and traits actually improve performance — and by how much?")

    # Explode all animals (not just primary)
    animal_rows = []
    for _, row in dm.iterrows():
        for a in str(row.get("animals_fmt","")).split(" + "):
            a=a.strip()
            if a and a!="—":
                animal_rows.append({"animal":a,"success":row["success"],
                                    "payload_to_aircraft_ratio":row.get("payload_to_aircraft_ratio",0),
                                    "design_stars":row.get("design_stars",0)})
    adm = pd.DataFrame(animal_rows)

    if not adm.empty:
        ast = adm.groupby("animal").agg(
            success_rate=("success","mean"),
            n=("success","count"),
            n_success=("success","sum"),
            avg_ratio=("payload_to_aircraft_ratio","mean"),
            avg_stars=("design_stars","mean"),
        ).reset_index().sort_values("success_rate",ascending=False)

        # p-value for each animal vs. all other animal-appearances combined (Fisher's exact)
        total_n, total_success = len(adm), int(adm["success"].sum())
        def _animal_p(row):
            rest_n = total_n - row["n"]
            rest_success = total_success - row["n_success"]
            p, sig = compare_proportions(int(row["n_success"]), int(row["n"]), int(rest_success), int(rest_n))
            return pd.Series({"p_value": p, "significance": sig})
        ast = ast.join(ast.apply(_animal_p, axis=1))
        ast["p_display"] = ast["p_value"].apply(lambda p: f"{p:.4f}" if p is not None and p>=0.0001 else ("<0.0001" if p is not None else "n/a"))

        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(ast, x="animal", y="success_rate",
                         color="success_rate",
                         color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                         text=ast["success_rate"].apply(lambda v:f"{v*100:.1f}%"),
                         title="Success Rate by Animal Archetype (all appearances, not just primary)",
                         hover_data={"avg_ratio":":.2f","avg_stars":":.1f","n":True,
                                     "p_display":True,"significance":True},
                         labels={"success_rate":"Success Rate","animal":"Animal",
                                 "avg_ratio":"Avg P/W","avg_stars":"Avg Stars",
                                 "p_display":"p-value (vs other animals)","significance":"Significance"})
            fig.update_traces(textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=400,coloraxis_showscale=False,
                              xaxis={"categoryorder":"total descending"})
            ev_bio = st.plotly_chart(fig, on_select="rerun", key="bio_animals", use_container_width=True)
            st.caption(
                "p-values compare each animal's success rate against all other animal-appearances "
                "combined (Fisher's exact test). Hover any bar for the exact value."
            )

        chart_help(
            "Reading the animal benchmark",
            "Each bar shows the average mission success rate across all designs that include "
            "that animal in their bio-inspiration mix. Multiple animals per design are counted separately.",
            "Animals at the top of the chart consistently improve mission outcomes. "
            "Animals at the bottom either add weight/complexity without proportionate benefit, "
            "or are better suited to specific conditions.",
            "Prioritise high-performing animal archetypes when composing your design's trait mix. "
            "Note that combining multiple animals has multiplicative effects — "
            "check the trait chart below for individual trait contributions."
        )

        with c2:
            # Radar
            top6 = ast.head(6)["animal"].tolist()
            r_df = ast[ast["animal"].isin(top6)].copy()
            for col in ["success_rate","avg_ratio","avg_stars"]:
                mn,mx = r_df[col].min(), r_df[col].max()
                r_df[f"{col}_n"] = (r_df[col]-mn)/max(mx-mn,1e-9)
            cats = ["Success Rate","Avg P/W Ratio","Avg Stars"]
            fig2 = go.Figure()
            for _,row in r_df.iterrows():
                vals = [row["success_rate_n"],row["avg_ratio_n"],row["avg_stars_n"]]
                fig2.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]],
                    fill="toself", name=row["animal"], opacity=0.6))
            fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                               title="Top Animal Profiles (normalised 0–1)",
                               height=400,showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Individual Trait Impact on Success Rate")
        st.warning(
            "⚠️ **Trait impact is observational, not causal.** Traits are not randomly assigned — "
            "a trait can show a negative average delta simply because it tends to co-occur with "
            "heavier airframes, more complex architectures, or other correlated design choices, "
            "not because the trait itself is harmful. A trait may still be the right choice for a "
            "specific design or condition even if its average effect across all designs is negative. "
            "Use the Trait Dictionary below for the physical trade-offs behind each number."
        )
        if not trait_df.empty if 'trait_df' in dir() else True:
            # Recompute here
            _t_rows = []
            for trait in all_traits_set:
                has   = dm[dm["traits"].fillna("").str.contains(trait,regex=False)]
                hasnt = dm[~dm["traits"].fillna("").str.contains(trait,regex=False)]
                if len(has)>4 and len(hasnt)>4:
                    p_val, sig_label = compare_proportions(
                        int(has["success"].sum()), len(has),
                        int(hasnt["success"].sum()), len(hasnt)
                    )
                    _t_rows.append({"trait":trait,
                                    "delta":has["success"].mean()-hasnt["success"].mean(),
                                    "with_rate":has["success"].mean(),
                                    "without_rate":hasnt["success"].mean(),
                                    "p_value":p_val, "significance":sig_label})
            _tdf = pd.DataFrame(_t_rows).sort_values("delta",ascending=False)

            _tdf["trait_label"] = _tdf["trait"].str.replace("_"," ")
            _tdf["p_display"] = _tdf["p_value"].apply(lambda p: f"{p:.4f}" if p>=0.0001 else "<0.0001")
            fig3 = px.bar(_tdf, x="delta", y="trait_label", orientation="h",
                          color="delta",
                          color_continuous_scale=[[0,RED],[0.5,"#cccccc"],[1,GREEN]],
                          hover_data={"with_rate":":.1%","without_rate":":.1%",
                                      "p_display":True,"significance":True},
                          title="Trait Impact: Success Rate Δ  (green = helps on average, red = hurts on average)",
                          labels={"delta":"Success Rate Δ","trait_label":"",
                                  "with_rate":"With trait","without_rate":"Without trait",
                                  "p_display":"p-value","significance":"Significance"})
            fig3.add_vline(x=0,line_color="black",line_width=1)
            fig3.update_xaxes(tickformat="+.0%")
            fig3.update_layout(height=480,coloraxis_showscale=False,
                               yaxis={"categoryorder":"total ascending"})
            # Mark statistically significant bars (p<0.05) with an asterisk label
            for _, r in _tdf.iterrows():
                if r["p_value"] is not None and r["p_value"] < 0.05:
                    fig3.add_annotation(x=r["delta"], y=r["trait_label"],
                                        text="✱", showarrow=False,
                                        xshift=18 if r["delta"]>=0 else -18,
                                        font=dict(size=14, color="black"))
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "✱ = statistically significant difference (Fisher's exact test, p < 0.05) between "
                "the WITH-trait and WITHOUT-trait success rates. Hover any bar for the exact p-value. "
                "Bars without ✱ may still reflect a real effect too small to detect at this sample "
                "size — they aren't proof of *no* effect, just insufficient evidence either way."
            )

        chart_help(
            "Reading the trait impact chart",
            "Each bar shows the univariate (single-variable) association between having a trait "
            "and mission success rate, compared to designs without it. This is a simple group-mean "
            "comparison, not a controlled experiment — it does not isolate the trait's effect from "
            "other design factors that happen to correlate with it.",
            "A negative bar does not mean 'avoid this trait.' It means designs carrying this trait "
            "average lower success in THIS dataset — often because the trait tends to appear "
            "alongside heavier, more complex, or less-tuned designs, not because the trait itself "
            "is the cause. Cross-check against the Failure Autopsy page: a trait can still be the "
            "right fix for a specific failure mode (e.g. gust instability) even with a negative "
            "average bar here.",
            "Don't treat this chart as a ranked shopping list. Use it alongside the Trait Dictionary's "
            "physical trade-offs and the Failure Autopsy's condition-specific recommendations to "
            "decide which traits suit YOUR design's specific failure modes and operating conditions."
        )

    # Note: this page presents aggregate analysis across ALL designs, not a single design.
    # Use the Leaderboard or Design DNA pages to inspect an individual design's full card.

    # ── Trait Dictionary ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📖 Trait Dictionary — What Each Trait Does for Your Design")
    st.caption("Each trait is derived from a bio-inspired archetype. The table below shows what the trait does physically, "
               "its engineering trade-off, and which animals contribute it.")

    TRAIT_DICT = {
        "LIFT_BURST_POWER":          ("Harpy Eagle, Golden Eagle, Cheetah, Tiger",
                                       "Boosts available battery/supercap power by 1–3× during takeoff climb.",
                                       "Critical for passing the 16–20 hover turns. Without burst, undersized batteries hit power saturation.",
                                       "Add supercapacitors alongside the main battery to supply burst power independently."),
        "LIFT_HIGH_CONTINUOUS":      ("Harpy Eagle, Golden Eagle",
                                       "Sustained high-power hover capability — reduces power sag during long loaded cruise.",
                                       "Heavier motor and battery sizing, but fewer mid-cruise power failures.",
                                       "Pair with high C-rate battery (≥8C) for continuous hover demand across the full 4 nmi loaded leg."),
        "LIFT_ENERGY_DENSE":         ("Albatross",
                                       "Biases energy system sampling toward high spec-energy chemistries (Li-S, solid-state).",
                                       "More mission range per kg of battery, but lower peak C-rate — may need supercap assist.",
                                       "Best for designs where range or time-limit failures dominate over power saturation."),
        "LIFT_UNSTEADY_AERO":        ("Bee",
                                       "Clap-and-fling insect aerodynamics reduce hover power demand by up to 12.5%.",
                                       "Requires very high rotor RPM and small-diameter rotors — difficult to scale.",
                                       "Most beneficial for very small designs (<10 kg empty). At DARPA Lift scale, effect is moderate."),
        "PAYLOAD_CENTRAL_TALON":     ("Harpy Eagle, Golden Eagle",
                                       "Payload mounted directly under the center of mass — eliminates pendulum dynamics.",
                                       "Simplest control, lowest gust coupling. Trade-off: payload must be rigid and fits the cradle.",
                                       "Best choice for the DARPA Lift barbell-plate payload. Minimises gust-induced instability failures."),
        "PAYLOAD_SLING_LOAD":        ("Osprey",
                                       "Hanging cable payload introduces pendulum sway (f = 1/2π√(g/L)).",
                                       "Increases gust-induced instability failures by 10% and structural failures by 5%.",
                                       "Avoid unless the mission requires remote drop delivery. If used, keep cable length < 0.4 m."),
        "PAYLOAD_DAMPED":            ("Osprey",
                                       "Rubber-isolated payload mount absorbs vibration and sway before it reaches the airframe.",
                                       "Reduces gust and control saturation failures by 8% each.",
                                       "Easy to add as a passive upgrade to any mounting system. Use Shore A 40–60 elastomer pads."),
        "STAB_GUST_REJECTION":       ("Osprey, Dragonfly",
                                       "Active disturbance suppression reduces environmental power overhead and cuts gust failures 25%.",
                                       "Requires a flight controller with at least 10 Hz attitude bandwidth. Negligible mass penalty.",
                                       "Tune gust_rejection_gain ≥ 1.2 in the simulator to capture the full benefit."),
        "STAB_DISTRIBUTED_THRUST":   ("Dragonfly, Bee",
                                       "Multiple independent rotors add 3% hover power overhead but cut control saturation failures by 20%.",
                                       "Requires ≥8 rotors. More rotors = more motor controllers = more potential single-point failures.",
                                       "Sweet spot is 12–16 rotors for this payload class. Beyond 20 rotors, management complexity outweighs gains."),
        "STAB_COMPLIANT_SPINE":      ("Cheetah",
                                       "Flexible airframe absorbs dynamic loads — frame stiffness capped at 0.70.",
                                       "Reduces peak structural stress during gusts but increases sensor noise and requires vibration isolation.",
                                       "Use only if structural overload failures dominate. Most designs perform better with stiffer frames."),
        "WING_HIGH_ASPECT":          ("Albatross",
                                       "High-aspect-ratio auxiliary surfaces reduce cruise power by 8% through improved L/D.",
                                       "Adds wing mass and folding complexity. Benefit scales with cruise speed — most useful above 15 m/s.",
                                       "Stack with MISSION_GLIDE_SEGMENTS for 17–22% total cruise power reduction."),
        "WING_MORPHING_MEMBRANE":    ("Bat",
                                       "Adaptive camber bat-wing surfaces cut cruise power by up to 10% in turbulent conditions.",
                                       "Complex fabrication. Benefit only appears in wind — zero gain in calm air.",
                                       "Valuable if your test site has sustained wind. Combine with STAB_GUST_REJECTION for resilience."),
        "WING_FOLDABLE":             ("Bat",
                                       "Wings fold for compact hover footprint; deploy for cruise. 1.5% hover saving, 6% cruise saving.",
                                       "Introduces wing deployment failure risk — scales with turbulence intensity.",
                                       "Test deployment under maximum turbulence conditions before the competition. "
                                       "Consider fixed wings if deployment reliability < 99%."),
        "STRUCT_TENDON_CABLES":      ("Cheetah, Tiger",
                                       "Tension cable primary structure: mass-efficient for arms and booms under bending loads.",
                                       "Cables carry tension only — compression members still needed. Complex assembly.",
                                       "Good for very large, spider-like rotor arm geometries where bending moment is the sizing load."),
        "STRUCT_ROBUST_GEAR":        ("Tiger",
                                       "Reinforced landing gear: 20% heavier but 50% higher touchdown velocity tolerance.",
                                       "Absorbs hard landings that would destroy standard gear. Trade-off: reduces payload margin.",
                                       "Essential if your design has a high descent rate or operates from uneven terrain."),
        "MISSION_GLIDE_SEGMENTS":    ("Albatross",
                                       "Mission planning uses partial-power cruise segments. 10–15% cruise power reduction.",
                                       "Zero benefit in calm air — all savings come from wind-assisted glide. No hardware change needed.",
                                       "Free gain for any design with a capable flight controller. Implement as a variable-throttle cruise mode."),
        "MISSION_MULTI_GAIT":        ("Cheetah",
                                       "Multiple flight modes (e.g., hover, transition, cruise, burst) optimised for each phase.",
                                       "Requires a sophisticated flight controller with mode-switching logic.",
                                       "High payoff for designs that span very different aerodynamic regimes (e.g. multirotor + fixed wing)."),
    }

    for trait, (animals, what_it_does, trade_off, tip) in TRAIT_DICT.items():
        # Show only traits that exist in this dataset
        if trait not in all_traits_set:
            continue
        # Compute success delta and statistical significance for this trait
        has   = dm[dm["traits"].fillna("").str.contains(trait, regex=False)]
        hasnt = dm[~dm["traits"].fillna("").str.contains(trait, regex=False)]
        delta = has["success"].mean() - hasnt["success"].mean() if len(has) > 4 and len(hasnt) > 4 else 0
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        colour = GREEN if delta >= 0.02 else (RED if delta < -0.02 else AMBER)
        p_val, sig_label = (compare_proportions(int(has["success"].sum()), len(has),
                                                 int(hasnt["success"].sum()), len(hasnt))
                            if len(has) > 4 and len(hasnt) > 4 else (None, "n/a"))

        with st.expander(f"{'✅' if delta>=0.02 else ('⚠️' if delta>=-0.02 else '❌')}  **{trait.replace('_',' ')}**  "
                         f"— impact on success: {delta_str}  |  Animals: {animals}", expanded=False):
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"**What it does:** {what_it_does}")
                st.markdown(f"**Trade-off:** {trade_off}")
            with ic2:
                st.markdown(f"**💡 Design tip:** {tip}")
                # Show with/without rates
                if len(has) > 4 and len(hasnt) > 4:
                    st.metric("Success WITH trait",    f"{has['success'].mean()*100:.1f}%",
                              delta=f"{delta*100:+.1f}% vs without")
                    p_str = f"{p_val:.4f}" if p_val is not None and p_val >= 0.0001 else "<0.0001"
                    st.caption(f"Fisher's exact test: p = {p_str} ({sig_label}, n={len(has)} with / {len(hasnt)} without)")




# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Flight Recorder
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "📡  Flight Recorder":
    st.title("📡 Flight Recorder")
    st.caption("All missions for the selected design overlaid — green = pass, red = fail. No dropdowns needed.")

    sel_design = st.selectbox("Select design:", all_ids, format_func=design_label, key="fr_design")

    drow = designs[designs["design_id"]==sel_design].iloc[0]
    ic1,ic2,ic3,ic4,ic5 = st.columns(5)
    ic1.metric("Animals",    drow.get("animals_fmt","—")[:35])
    ic2.metric("Propulsion", str(drow.get("propulsion_architecture","—")).replace("_"," "))
    ic3.metric("P/W ratio",  f"{drow.get('payload_to_aircraft_ratio',0):.2f}:1")
    ic4.metric("Battery",    f"{drow.get('battery_max_power_W',0)/1000:.1f} kW peak")
    ic5.metric("Stars",      stars(drow.get("design_stars",0)))
    st.caption(str(drow.get("design_summary",""))[:250]+"…")

    m4d = missions[missions["design_id"]==sel_design]
    n_pass = int(m4d["success"].sum());  n_fail = len(m4d) - n_pass
    st.markdown(f"**{len(m4d)} missions — {n_pass} ✅ passed · {n_fail} ❌ failed**  "
                f"| Wind {m4d['wind_speed_kts'].min():.0f}–{m4d['wind_speed_kts'].max():.0f} kts  "
                f"| Turbulence {m4d['turbulence_index'].min():.2f}–{m4d['turbulence_index'].max():.2f}")

    ts_full = telemetry_for_design(str(sel_design))
    if ts_full.empty:
        st.warning("No timeseries data found. Make sure missions_timeseries.csv exists.")
        st.stop()

    # ── Mission Phase Glossary — always visible, explains every phase and shape ──
    with st.expander("📖 Phase glossary — what each part of the flight means", expanded=False):
        st.markdown(
            "The DARPA Lift course is a **4 nautical mile payload (outbound) leg**, a payload "
            "drop, and a **1 nautical mile return leg** — about 5 nmi round trip. Every mission "
            "passes through some or all of these phases, shown as colored bands on the "
            "single-mission charts below:"
        )
        phase_glossary = [
            ("🔵", "Takeoff / Climb", "takeoff_climb",
             "Vertical ascent from the launch pad to cruise altitude, carrying the full payload. "
             "Power demand is high — this is one of two places (along with hover turns) where the "
             "aircraft needs its peak power."),
            ("🟢", "Loaded Cruise", "loaded_cruise",
             "The 4 nautical mile outbound leg, flown with the payload attached. This phase includes "
             "16–20 scheduled hover turns — brief stops where the aircraft pauses mid-flight, hovers "
             "for about 12 seconds, then resumes cruise speed. That's what produces the repeating "
             "square-wave pattern you'll see on the Speed chart (see 'Reading the shapes' below)."),
            ("🟡", "Drop Descent", "drop_descent",
             "Controlled descent toward the payload release point, reached at the 4 nmi mark."),
            ("🟠", "Drop Hover", "drop_hover",
             "The aircraft hovers in place while releasing the payload. Watch the Mass chart — this is "
             "exactly where mass drops sharply, since the payload (roughly 70–80% of total mass) "
             "is released here."),
            ("🟣", "Post-Drop Climb", "post_drop_climb",
             "Climb back to cruise altitude now that the payload is gone. Noticeably quicker and lower "
             "power than the loaded climb, since the aircraft is much lighter."),
            ("🩵", "Empty Cruise", "empty_cruise",
             "The 1 nautical mile return leg, flown without payload. Lower power demand than the "
             "loaded leg — no hover turns are scheduled here."),
            ("🔴", "Descent / Landing", "descent_landing",
             "Final descent and touchdown at the home base. A landing that's too fast for the design's "
             "gear rating shows up as a 'hard touchdown' failure."),
            ("⛔", "Emergency Descent", "emergency_descent",
             "Only appears in failed missions — triggered when a failure condition (power saturation, "
             "energy depletion, structural overload, etc.) forces an immediate, uncontrolled descent "
             "before the mission can complete normally."),
        ]
        for emoji, name, key, desc in phase_glossary:
            st.markdown(f"**{emoji} {name}** — {desc}")

        st.markdown("---")
        st.markdown("**Reading the shapes:**")
        st.markdown(
            "- **Square-wave pattern on the Speed chart** (alternating between ~20 m/s and ~0 m/s, "
            "repeated 16–20 times): this is the hover-turn sequence inside Loaded Cruise. The aircraft "
            "isn't falling — it's a VTOL design, so lift comes from the rotors, not forward airspeed. "
            "Speed genuinely drops to zero while the rotors hold the aircraft in place, then it "
            "re-accelerates. The transitions look 'square' rather than smooth because the ramp up/down "
            "is fast relative to how long the aircraft spends at each speed.\n"
            "- **Sharp single step-down on the Mass chart**: the payload release, always at the boundary "
            "between Drop Hover and Post-Drop Climb.\n"
            "- **Spikes on the Power Used chart**: each hover turn briefly demands near-hover power even "
            "mid-cruise, since the aircraft has to stop and hold position. A design with insufficient "
            "power margin shows these spikes pushing past the Power Available line — that's a "
            "power-saturation failure in progress.\n"
            "- **Flat segment with a slight downward slope on the Distance chart, ending in a sharp "
            "plateau, then a slower second slope**: outbound leg (4 nmi) → drop point (plateau during "
            "the hover) → return leg (1 nmi, usually steeper/faster since the aircraft is lighter)."
        )

    PASS_COL = "rgba(39,174,96,0.5)"
    FAIL_COL = "rgba(231,76,60,0.5)"
    success_map = dict(zip(m4d["mission_id"].astype(str), m4d["success"]))

    # ── View mode toggle ──────────────────────────────────────────────────
    view_mode = st.radio("View mode:", ["🔀 All missions overlaid", "🔍 Single mission detail"],
                         horizontal=True, key="fr_view_mode")

    def overlay_chart(y_col, ylabel, title, show_envelope=False):
        if y_col not in ts_full.columns:
            return
        fig = go.Figure()

        if show_envelope:
            # Reduce raw-line clutter; add a 5-95% percentile band + median line on top.
            line_alpha = 0.18
        else:
            line_alpha = 0.5
        pass_col = f"rgba(39,174,96,{line_alpha})"
        fail_col = f"rgba(231,76,60,{line_alpha})"

        for mid, grp in ts_full.groupby("mission_id"):
            grp = grp.sort_values("t_s")
            ok  = success_map.get(str(mid), 0)
            fig.add_trace(go.Scatter(
                x=grp["t_s"], y=grp[y_col], mode="lines",
                line=dict(color=pass_col if ok else fail_col, width=1.3),
                name="PASS" if ok else "FAIL", showlegend=False,
                hovertemplate=f"Mission {mid}<br>{ylabel}=%{{y:.1f}}<extra></extra>"
            ))

        if show_envelope:
            # Bin by integer t_s (1 s sampling) and compute median + 5th/95th percentile.
            env = ts_full.groupby("t_s")[y_col].agg(
                p05=lambda x: x.quantile(0.05),
                median="median",
                p95=lambda x: x.quantile(0.95),
            ).reset_index().sort_values("t_s")
            fig.add_trace(go.Scatter(
                x=pd.concat([env["t_s"], env["t_s"][::-1]]),
                y=pd.concat([env["p95"], env["p05"][::-1]]),
                fill="toself", fillcolor="rgba(0,48,135,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="5–95% range",
                showlegend=True, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=env["t_s"], y=env["median"], mode="lines",
                line=dict(color=BLUE, width=2.5), name="Median (all missions)",
                showlegend=True,
            ))

        fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
            line=dict(color="rgba(39,174,96,0.9)",width=3),name="✅ Pass"))
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
            line=dict(color="rgba(231,76,60,0.9)",width=3),name="❌ Fail"))
        fig.update_layout(title=title, height=260 if show_envelope else 230,
                          margin=dict(t=40,b=10),
                          xaxis_title="Time (s)", yaxis_title=ylabel,
                          legend=dict(orientation="h",y=1.25,x=1,xanchor="right"))
        st.plotly_chart(fig, use_container_width=True)

    if view_mode.startswith("🔀"):
        overlay_chart("altitude_m",           "Altitude (m)",   "Altitude — all missions  (green=pass, red=fail)")
        overlay_chart("speed_mps",            "Speed (m/s)",    "Horizontal Speed — spread shows environmental sensitivity")
        overlay_chart("power_used_W",         "Power Used (W)",
                      "Power Used — median + 5–95% range (raw lines faded; switch to single-mission view for one mission's detail)",
                      show_envelope=True)
        overlay_chart("battery_remaining_Wh", "Battery (Wh)",   "Battery Remaining — steeper slope = more energy hungry")

        # ── New chart types ──
        ts_full["distance_nmi"] = ts_full["distance_m"] / 1852.0
        def overlay_chart_nmi():
            """Distance chart with mile-marker reference lines (built separately — needs nmi conversion)."""
            fig = go.Figure()
            for mid, grp in ts_full.groupby("mission_id"):
                grp = grp.sort_values("t_s")
                ok = success_map.get(str(mid), 0)
                fig.add_trace(go.Scatter(
                    x=grp["t_s"], y=grp["distance_nmi"], mode="lines",
                    line=dict(color=PASS_COL if ok else FAIL_COL, width=1.3),
                    name="PASS" if ok else "FAIL", showlegend=False,
                    hovertemplate="Mission %{text}<br>Distance=%{y:.2f} nmi<extra></extra>", text=[mid]*len(grp)
                ))
            for nmi_mark, label in [(1,"Mile 1"),(2,"Mile 2"),(3,"Mile 3"),(4,"Mile 4 — Drop Point")]:
                fig.add_hline(y=nmi_mark, line_dash="dot", line_color="gray", line_width=1,
                              annotation_text=label, annotation_position="right")
            fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
                line=dict(color="rgba(39,174,96,0.9)",width=3),name="✅ Pass"))
            fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
                line=dict(color="rgba(231,76,60,0.9)",width=3),name="❌ Fail"))
            fig.update_layout(title="Distance Traveled — 4 nmi outbound + 1 nmi return, mile markers shown",
                              height=260, margin=dict(t=40,b=10),
                              xaxis_title="Time (s)", yaxis_title="Distance (nmi)",
                              legend=dict(orientation="h",y=1.25,x=1,xanchor="right"))
            st.plotly_chart(fig, use_container_width=True)
        overlay_chart_nmi()

        overlay_chart("mass_kg", "Mass (kg)",
                      "Aircraft Mass — sharp step down = payload released at the drop point")
        overlay_chart("energy_used_Wh_cum", "Cumulative Energy (Wh)",
                      "Cumulative Energy Used — pure consumption curve (vs. Battery Remaining, which also reflects capacity)")

        st.caption("💡 The power chart above is faded with a median + 5–95% band to stay readable across "
                   "many missions. For per-mission detail — including exactly where hover-turn power spikes "
                   "occur — switch to **🔍 Single mission detail** above.")

        chart_help(
            "Reading the all-missions overlay",
            "Each line is one complete mission. Green lines completed successfully; red lines failed. "
            "The spread between lines shows how sensitive this design is to wind and turbulence variation. "
            "The power chart shows a median line and a shaded 5th–95th percentile band instead of dense "
            "raw lines, since hover-turn spikes from 16-20 turns per mission made the raw overlay hard to read.",
            "Tight green cluster + isolated red outliers = robust design failing only in extreme conditions. "
            "Wide spread or mostly red = fundamental design issue. "
            "A median power line that tracks close to the available-power ceiling (visible in single-mission "
            "view) signals the design is running with little margin during hover turns. "
            "Missions that end (their lines stop) before reaching the Mile 4 marker on the Distance chart "
            "failed before ever releasing the payload.",
            "Look at the power chart first: if the 5-95% band's upper edge spikes sharply → some missions "
            "are close to power saturation, increase battery C-rate or add supercaps. "
            "If red missions run out of battery mid-flight → increase total energy (battery mass × spec energy). "
            "Switch to single-mission view to see the exact power-vs-available trace for any one failed mission."
        )

    else:
        # ── Single mission detail ─────────────────────────────────────────
        def mlabel(row):
            ok  = "✅ PASS" if row["success"]==1 else "❌ FAIL"
            rsn = f" ({row['failure_reason'].replace('_',' ')})" if row["success"]==0 else ""
            tier = f" | {row.get('prize_tier','').upper()}" \
                   if "prize_tier" in row and row.get("prize_tier","")!="none" else ""
            return (f"Mission {str(row['mission_id'])[-8:]}  {ok}{rsn}{tier}  "
                    f"🌬{row['wind_speed_kts']:.0f}kts  T={row['turbulence_index']:.2f}")

        mopts    = m4d.apply(mlabel, axis=1).tolist()
        sel_label = st.selectbox("Select mission:", mopts, key="fr_single_mission")
        sel_mrow  = m4d.iloc[mopts.index(sel_label)]
        sel_mid   = str(sel_mrow["mission_id"])

        if sel_mrow["success"] == 1:
            st.success(f"✅ MISSION PASSED — Time: {sel_mrow['total_time_s']:.0f}s  |  "
                       f"Energy used: {sel_mrow['energy_used_Wh']:.0f} Wh  |  "
                       f"Prize: {sel_mrow.get('prize_tier','—').upper()}")
        else:
            st.error(f"❌ MISSION FAILED — Phase: {sel_mrow['failure_phase']}  |  "
                     f"Reason: {sel_mrow['failure_reason'].replace('_',' ').title()}")

        ts_single = ts_full[ts_full["mission_id"].astype(str)==sel_mid].sort_values("t_s")
        if ts_single.empty:
            st.warning(f"No telemetry found for this mission.")
        else:
            PHASE_COLORS = {
                "takeoff_climb":"rgba(0,100,200,0.09)","loaded_cruise":"rgba(0,180,100,0.09)",
                "drop_descent":"rgba(200,150,0,0.09)","drop_hover":"rgba(200,100,0,0.09)",
                "post_drop_climb":"rgba(100,50,200,0.09)","empty_cruise":"rgba(50,180,180,0.09)",
                "descent_landing":"rgba(200,50,50,0.09)","emergency_descent":"rgba(220,0,0,0.18)",
            }
            PHASE_SHORT_LABELS = {
                "takeoff_climb":"Takeoff","loaded_cruise":"Loaded Cruise",
                "drop_descent":"Drop Descent","drop_hover":"Drop Hover",
                "post_drop_climb":"Post-Drop Climb","empty_cruise":"Empty Cruise",
                "descent_landing":"Landing","emergency_descent":"Emergency",
            }
            def add_bands(fig, ts, label_phases=False):
                # Pass 1: detect every contiguous phase segment (start, end, phase name)
                segments = []
                prev, start = None, None
                for _,row in ts.iterrows():
                    ph=row["phase"]
                    if ph!=prev:
                        if prev and start is not None:
                            segments.append((prev, start, row["t_s"]))
                        start,prev=row["t_s"],ph
                if prev and start:
                    segments.append((prev, start, ts["t_s"].max()))

                # Always draw every band's color, regardless of labeling decisions below.
                for p, x0, x1 in segments:
                    fig.add_vrect(x0=x0, x1=x1,
                                  fillcolor=PHASE_COLORS.get(p,"rgba(128,128,128,0.05)"),
                                  line_width=0, layer="below")

                if not label_phases or not segments:
                    return

                # Pass 2: decide which segments get a text label. A segment is only labeled
                # if its center sits far enough from the PREVIOUSLY LABELED segment's center —
                # this is what actually prevents crowding (e.g. Drop Descent and Post-Drop
                # Climb can each individually be "wide enough" on their own, but still sit too
                # close together for two labels to fit without colliding).
                t_span = ts["t_s"].max() - ts["t_s"].min() if len(ts) else 1
                min_gap = max(t_span * 0.08, 20)   # min separation between label centers
                last_label_center = None
                for p, x0, x1 in segments:
                    center = (x0 + x1) / 2
                    if last_label_center is not None and (center - last_label_center) < min_gap:
                        continue   # too close to the previous label — skip this one
                    fig.add_annotation(
                        x=center, y=0.99, xref="x", yref="paper",
                        text=PHASE_SHORT_LABELS.get(p, p), showarrow=False,
                        font=dict(size=11, color="#333"), yanchor="top",
                        bgcolor="rgba(255,255,255,0.75)", borderpad=2,
                    )
                    last_label_center = center

            t = ts_single["t_s"]

            def single_chart(y_cols_names, title, height=230, label_phases=False):
                fig = go.Figure()
                colors = [BLUE, AMBER, GREEN, RED, "steelblue", "royalblue"]
                for i,(col,lbl) in enumerate(y_cols_names):
                    if col in ts_single.columns:
                        fig.add_trace(go.Scatter(x=t, y=ts_single[col], name=lbl,
                                                  line=dict(color=colors[i%len(colors)], width=2)))
                add_bands(fig, ts_single, label_phases=label_phases)
                fig.update_layout(title=title, height=height, margin=dict(t=40,b=10),
                                  xaxis_title="Time (s)",
                                  legend=dict(orientation="h",y=1.15))
                if label_phases:
                    # Extra headroom above the highest data point so the line never crosses
                    # into the label zone at the top of the plot.
                    y_vals = pd.concat([ts_single[c] for c,_ in y_cols_names if c in ts_single.columns])
                    y_min, y_max = float(y_vals.min()), float(y_vals.max())
                    pad = (y_max - y_min) * 0.18 if y_max > y_min else 1.0
                    fig.update_yaxes(range=[y_min - pad*0.15, y_max + pad])
                st.plotly_chart(fig, use_container_width=True)

            # Find the payload-drop event for this specific mission, if it occurs
            drop_t = None
            if "payload_attached" in ts_single.columns:
                attached = ts_single["payload_attached"].astype(bool)
                if attached.any() and (~attached).any():
                    drop_idx = (attached.astype(int).diff() == -1).idxmax()
                    if drop_idx in ts_single.index:
                        drop_t = ts_single.loc[drop_idx, "t_s"]

            single_chart([("altitude_m","Altitude (m)")],
                         "Altitude (m) — phase names labeled, bands aligned across all charts below",
                         label_phases=True)

            single_chart([("speed_mps","Horiz speed"),("speed_total_mps","Total speed"),
                          ("vertical_rate_mps","Vertical rate")],
                         "Speed (m/s) — square-wave segment = hover turns (see Phase Glossary above)")
            st.caption(
                "**Why the square shape?** This is a VTOL design — lift comes from the rotors, "
                "not forward airspeed. During Loaded Cruise the aircraft makes 16–20 scheduled "
                "hover turns: it briefly stops, hovers in place (speed → 0), then re-accelerates "
                "to cruise speed. It isn't falling — altitude stays level through each one (check "
                "the Altitude chart above). The shape looks 'square' because the speed-up/slow-down "
                "transitions are fast relative to how long it spends at each speed."
            )

            single_chart([("power_used_W","Power used"),("power_requested_W","Requested"),
                          ("power_available_W","Available")],
                         "Power (W) — if used > available for >6 s → failure", height=250)
            st.caption(
                "**Why the spikes?** Each hover turn above briefly demands near-hover power even "
                "mid-cruise, since the aircraft has to stop and hold position — that's what produces "
                "the repeated spikes here. If a spike pushes the blue 'Power used' line above the "
                "green 'Available' line for more than 6 seconds, that's a power-saturation failure."
            )

            single_chart([("battery_remaining_Wh","Battery remaining")],
                         "Battery Remaining (Wh)")

            # ── New chart types ──
            ts_single = ts_single.copy()
            ts_single["distance_nmi"] = ts_single["distance_m"] / 1852.0

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=t, y=ts_single["distance_nmi"], name="Distance (nmi)",
                                          line=dict(color=BLUE, width=2)))
            add_bands(fig_dist, ts_single)
            for nmi_mark, label in [(1,"Mile 1"),(2,"Mile 2"),(3,"Mile 3"),(4,"Mile 4 — Drop Point")]:
                fig_dist.add_hline(y=nmi_mark, line_dash="dot", line_color="gray", line_width=1,
                                   annotation_text=label, annotation_position="right")
            fig_dist.update_layout(title="Distance Traveled (nmi) — 4 nmi outbound + 1 nmi return",
                                   height=230, margin=dict(t=40,b=10), xaxis_title="Time (s)",
                                   legend=dict(orientation="h",y=1.15))
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_mass = go.Figure()
            fig_mass.add_trace(go.Scatter(x=t, y=ts_single["mass_kg"], name="Mass (kg)",
                                          line=dict(color=AMBER, width=2)))
            add_bands(fig_mass, ts_single)
            if drop_t is not None:
                fig_mass.add_vline(x=drop_t, line_dash="dash", line_color=RED, line_width=2,
                                   annotation_text="Payload Released", annotation_position="top")
            fig_mass.update_layout(title="Aircraft Mass (kg) — step down = payload release",
                                   height=230, margin=dict(t=40,b=10), xaxis_title="Time (s)",
                                   legend=dict(orientation="h",y=1.15))
            st.plotly_chart(fig_mass, use_container_width=True)

            single_chart([("energy_used_Wh_cum","Cumulative energy used")],
                         "Cumulative Energy Used (Wh) — pure consumption, always increasing")

            st.caption("Phase bands: 🔵 Takeoff  🟢 Loaded cruise  🟡 Drop descent  "
                       "🟠 Drop hover  🟣 Post-drop climb  🩵 Empty cruise  🔴 Landing  ⛔ Emergency  "
                       "— see the **📖 Phase glossary** above for what each one means.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Design DNA
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔬  Design DNA":
    st.title("🔬 Design DNA")
    st.caption("What separates high-performing designs from low-performing ones?")

    # NOTE: designs already carries a canonical "success_rate" column globally —
    # do not re-derive it here (causes a merge-collision rename to success_rate_x/_y).
    m_agg = missions.groupby("design_id").agg(
        full_rate=("prize_tier",lambda x:(x=="full").mean()) if "prize_tier" in missions.columns else ("success","mean")
    ).reset_index()
    dna = designs.merge(m_agg, on="design_id", how="left")

    num_cols = [c for c in ["battery_max_power_W","battery_spec_energy_Wh_per_kg",
        "cruise_speed_mps","motor_efficiency","payload_to_aircraft_ratio",
        "rotor_count","empty_mass_kg","max_twr","battery_energy_Wh",
        "esc_efficiency","climb_rate_mps","gust_rejection_gain"] if c in dna.columns]

    corr = dna[num_cols+["success_rate"]].corr()["success_rate"].drop("success_rate").sort_values(ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns=["parameter","r"]
    corr_df["label"] = corr_df["parameter"].str.replace("_"," ").str.title()
    corr_df["interpretation"] = corr_df["r"].apply(
        lambda v: "Strong positive — more = better" if v > 0.4
        else "Moderate positive" if v > 0.15
        else "Negligible" if abs(v) < 0.05
        else "Moderate negative" if v > -0.15
        else "Strong negative — more = worse")

    fig = px.bar(corr_df,x="r",y="label",orientation="h",
                 color="r",color_continuous_scale=[[0,RED],[0.5,"#cccccc"],[1,GREEN]],
                 hover_data={"interpretation":True,"r":":.3f"},
                 title="Correlation with Mission Success Rate — which parameters matter most?",
                 labels={"r":"Pearson r","label":"","interpretation":"Interpretation"})
    fig.add_vline(x=0,line_color="black",line_width=1)
    fig.update_layout(height=420,coloraxis_showscale=False,
                      yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    _corr_abs = corr.reindex(corr.abs().sort_values(ascending=False).index)
    _top2 = _corr_abs.head(2)
    _top2_desc = " and ".join(
        f"{name.replace('_',' ')} (r = {val:.3f})" for name, val in _top2.items()
    )
    chart_help(
        "Reading the correlation chart",
        "Pearson r measures how much each design parameter is linearly associated with mission success rate. "
        "r = +1 means 'always better when higher'. r = −1 means 'always worse when higher'. "
        "r near 0 means the parameter barely matters. Treat these as observed associations in this "
        "specific run, not proof of causation — correlated design choices (e.g. heavier airframes "
        "tending to use certain propulsion types) can shift these numbers.",
        f"In this run, the strongest associations are {_top2_desc}. "
        "If both values are below |r|=0.2, no single parameter dominates — success likely "
        "depends on the combination of several factors rather than any one variable.",
        "Invest in the top 2–3 |r|-ranked parameters first, but validate with the Top 25% vs "
        "Bottom 25% comparison below before committing — a single correlation can be misleading "
        "in isolation."
    )

    st.subheader("Top 25% vs Bottom 25% — Side-by-Side Comparison")
    thr_h = dna["success_rate"].quantile(0.75)
    thr_l = dna["success_rate"].quantile(0.25)
    top_q = dna[dna["success_rate"]>=thr_h]
    bot_q = dna[dna["success_rate"]<=thr_l]

    cmp_rows = []
    for col in num_cols:
        t_avg = top_q[col].mean()
        b_avg = bot_q[col].mean()
        pct_better = (t_avg-b_avg)/max(abs(b_avg),1e-9)*100
        cmp_rows.append({
            "Parameter": col.replace("_"," ").title(),
            "Top 25% avg": f"{t_avg:.3g}",
            "Bottom 25% avg": f"{b_avg:.3g}",
            "Difference": f"{pct_better:+.1f}%",
            "Insight": "↑ Higher in winners" if pct_better>5
                       else "↓ Lower in winners" if pct_better<-5
                       else "Similar in both groups",
        })
    st.dataframe(pd.DataFrame(cmp_rows).sort_values("Difference",ascending=False,
                 key=lambda c: c.str.replace("+","").str.replace("%","").astype(float,errors="ignore")),
                 use_container_width=True, hide_index=True)

    st.subheader("Propulsion Architecture — Success vs Full Prize Rate")
    arch = dm.groupby("propulsion_architecture").agg(
        success_rate=("success","mean"),
        full_rate=("prize_tier",lambda x:(x=="full").mean()) if "prize_tier" in dm.columns else ("success","mean"),
        n=("success","count")
    ).reset_index().sort_values("success_rate",ascending=False)

    fig2=go.Figure()
    fig2.add_trace(go.Bar(name="Success rate",x=arch["propulsion_architecture"],
                          y=arch["success_rate"],marker_color=BLUE))
    fig2.add_trace(go.Bar(name="Full prize rate (4:1)",x=arch["propulsion_architecture"],
                          y=arch["full_rate"],marker_color=GREEN))
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_xaxes(tickangle=-25)
    fig2.update_layout(barmode="group",height=360,
                       title="Which propulsion wins?",
                       legend=dict(orientation="h"))
    ev_dna = st.plotly_chart(fig2, on_select="rerun", key="dna_arch", use_container_width=True)

    chart_help(
        "Reading the propulsion comparison",
        "Blue = overall mission success rate. Green = rate of achieving the 4:1 full-prize threshold. "
        "An architecture can have high success rate but low full-prize rate if it tends to carry lighter payloads.",
        "The ideal propulsion for the DARPA Lift Challenge maximises both bars simultaneously — "
        "not just completing missions, but completing them with enough payload to win the full prize.",
        "Consider which architecture best matches your fabrication capability and available components. "
        "Remember the FAA confirmed all propulsion types (electric, gas, hybrid) are legal under Part 107."
    )

    show_selected_inspector(designs, missions)

    st.divider()
    st.subheader("🛠 Development Roadmap — Failure-Specific Action Guide")
    st.caption("Based on the failure distribution in this simulation run. Follow these steps in order. "
               "Trait recommendations below target SPECIFIC failure modes — they can differ from "
               "the Bio-Inspiration Benchmark page's AVERAGE-effect trait chart; see that page's "
               "info box for why both can be correct simultaneously.")

    # Compute failure breakdown
    failed = missions[missions["success"]==0]
    total  = len(missions)
    def fail_pct(reason):
        return (missions["failure_reason"]==reason).sum()/max(total,1)*100

    action_guide = [
        ("power_saturation",
         "💥 Power Saturation",
         f"{fail_pct('power_saturation'):.1f}% of missions",
         "Battery cannot deliver enough peak power for hover. "
         "Happens at takeoff, hover turns, and gust recovery.",
         ["Raise BATTERY_MAX_C_RATE to at least 8C (currently the simulation floor is 5C).",
          "Add supercapacitors (supercap_mass_kg 0.5–2.5 kg) — they deliver burst power independently of battery chemistry.",
          "Add LIFT_BURST_POWER trait (Harpy Eagle / Tiger archetype) to activate supercap burst assist.",
          "Check battery_max_power_W > hover demand × 1.3 safety margin."]),
        ("time_limit_exceeded",
         "⏱ Time Limit Exceeded",
         f"{fail_pct('time_limit_exceeded'):.1f}% of missions",
         "Mission exceeds 30 minutes. Usually a cruise speed problem, not a power problem.",
         ["Increase cruise_speed_mps — target ≥ 15 m/s for the 5 nmi course.",
          "Add WING_HIGH_ASPECT (Albatross) + MISSION_GLIDE_SEGMENTS for 17–22% cruise power reduction, enabling faster flight.",
          "Reduce climb altitude to shorten time in climb phases.",
          "Verify motor_efficiency ≥ 0.88 — inefficient motors waste energy that could fund faster cruise."]),
        ("hard_touchdown",
         "💥 Hard Touchdown",
         f"{fail_pct('hard_touchdown'):.1f}% of missions",
         "Descent rate at landing exceeds gear tolerance.",
         ["Add STRUCT_ROBUST_GEAR trait (Tiger archetype): +50% touchdown velocity tolerance for +20% gear mass.",
          "Reduce climb_rate_mps — the simulator uses this for descent rate too.",
          "Increase max_touchdown_velocity_mps if your physical gear can absorb higher impact loads.",
          "Fit crushable foam foot pads — simple, effective, low mass."]),
        ("gust_induced_instability",
         "🌪 Gust Instability",
         f"{fail_pct('gust_induced_instability'):.1f}% of missions",
         "Aircraft cannot maintain attitude under turbulence.",
         ["Add STAB_GUST_REJECTION (Osprey/Dragonfly) — cuts gust failures by 25%.",
          "Switch payload mount to PAYLOAD_CENTRAL_TALON (not sling load) to eliminate pendulum coupling.",
          "Add PAYLOAD_DAMPED for passive vibration isolation — 8% fewer gust failures.",
          "Increase gust_rejection_gain ≥ 1.2 in simulation to model a better-tuned flight controller."]),
        ("energy_system_fault",
         "⚡ Energy System Fault",
         f"{fail_pct('energy_system_fault'):.1f}% of missions",
         "Battery or energy system failure mid-mission.",
         ["Add supercapacitor redundancy — if main battery faults, supercap bridges the gap.",
          "Consider fuel_cell_li_ion_hybrid energy system for built-in redundancy.",
          "Derate the battery to 80% of rated C-rate in practice — thermal runaway accelerates near the ceiling.",
          "Add LIFT_ENERGY_DENSE (Albatross) to bias toward more stable high-spec-energy chemistries."]),
    ]

    # Sort by descending failure percentage
    action_guide_sorted = sorted(action_guide, key=lambda x: float(x[2].split("%")[0]), reverse=True)

    for reason, title, pct, explanation, steps in action_guide_sorted:
        n_this = (missions["failure_reason"]==reason).sum()
        if n_this == 0:
            continue
        with st.expander(f"{title} — **{pct}** of all missions", expanded=(n_this > total*0.05)):
            st.markdown(f"**Root cause:** {explanation}")
            st.markdown("**How to fix it:**")
            for i, step in enumerate(steps, 1):
                st.markdown(f"&nbsp;&nbsp;&nbsp;**{i}.** {step}")

    st.divider()
    st.subheader("🎯 Parameter Targets from Top-Performing Designs")
    st.caption("Based on the top 10% of designs by success rate in this run.")

    top10pct = dna.nlargest(max(1, len(dna)//10), "success_rate")
    target_params = [
        ("Battery peak power",      "battery_max_power_W",           "W",    1, "/1000", "kW"),
        ("Motor efficiency",        "motor_efficiency",               "",     3, "*100",  "%"),
        ("ESC efficiency",          "esc_efficiency",                 "",     3, "*100",  "%"),
        ("Cruise speed",            "cruise_speed_mps",               "m/s",  1, "",      "m/s"),
        ("Payload-to-weight ratio", "payload_to_aircraft_ratio",      ":1",   2, "",      ":1"),
        ("Battery spec energy",     "battery_spec_energy_Wh_per_kg", "Wh/kg",1, "",      "Wh/kg"),
        ("Empty mass",              "empty_mass_kg",                  "kg",   2, "",      "kg"),
    ]

    tgt_rows = []
    for label, col, unit, dec, transform, unit2 in target_params:
        if col not in top10pct.columns:
            continue
        top_avg  = top10pct[col].mean()
        all_avg  = dna[col].mean()
        if transform == "/1000":
            top_avg /= 1000; all_avg /= 1000
        elif transform == "*100":
            top_avg *= 100; all_avg *= 100
        tgt_rows.append({
            "Parameter":       label,
            "Top 10% target":  f"{top_avg:.{dec}f} {unit2}",
            "All-design avg":  f"{all_avg:.{dec}f} {unit2}",
            "Direction":       "↑ Higher is better" if top_avg > all_avg else "↓ Lower is better",
        })
    st.dataframe(pd.DataFrame(tgt_rows), use_container_width=True, hide_index=True)
