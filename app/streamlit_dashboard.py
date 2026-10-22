"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 4 - Module 1: Institutional Multi-Tab Streamlit Benchmarking Terminal
File: app/streamlit_dashboard.py
Lead Architect: Kaustav Goswami

Launch Command:
    streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.econometrics.cbam_article6 import CrossBorderCBAMArticle6Engine
from src.econometrics.price_discovery import CorrelatedPriceDiscoveryEngine
from src.ingestion.pipeline import CarbonDataIngestionPipeline
from src.mechanics.cap_vs_intensity import StructuralDecouplingEngine
from src.mechanics.stability_reserves import MarketStabilityContainmentEngine

st.set_page_config(
    page_title="Global Carbon Compliance Architecture (2026-2035)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMINAL_CSS = """
<style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 6px;
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #94A3B8 !important;
        font-size: 0.80rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] > div {
        color: #F8FAFC !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .terminal-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-left: 5px solid #38BDF8;
        padding: 18px 24px;
        border-radius: 6px;
        margin-bottom: 18px;
    }
    .terminal-title {
        color: #F8FAFC;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .terminal-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .formula-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#0F172A",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#CBD5E1",
    "text.color": "#F8FAFC",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#1E293B",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.size": 9.5,
})


@st.cache_data(show_spinner="Executing 4-Layer Quantitative Carbon Engine...")
def load_all_engine_artifacts() -> Dict:
  pipeline = CarbonDataIngestionPipeline(project_root=PROJECT_ROOT)
  ingested = pipeline.run_pipeline()

  decoupling_eng = StructuralDecouplingEngine(project_root=PROJECT_ROOT)
  df_ind_panel, df_ind_summary = (
      decoupling_eng.simulate_india_9sectors_decoupling()
  )
  df_china_pivot = decoupling_eng.simulate_china_dual_control_pivot()

  stability_eng = MarketStabilityContainmentEngine(project_root=PROJECT_ROOT)
  df_eu_msr = stability_eng.simulate_eu_msr_trajectory_2026_2035()
  df_stress = stability_eng.run_cross_hub_stress_test()

  mc_eng = CorrelatedPriceDiscoveryEngine(
      project_root=PROJECT_ROOT, n_paths=10000, steps_per_year=12, random_seed=42
  )
  paths_dict, df_mc_annual, df_spreads = mc_eng.run_monte_carlo_simulation()

  cbam_eng = CrossBorderCBAMArticle6Engine(project_root=PROJECT_ROOT)
  df_cbam_panel, df_cbam_summary = cbam_eng.simulate_cbam_4sectors_exposure(
      df_mc_annual=df_mc_annual
  )
  df_art6 = cbam_eng.simulate_article6_arbitrage(df_mc_annual=df_mc_annual)

  with open(
      PROJECT_ROOT / "config" / "jurisdictions.json", "r", encoding="utf-8-sig"
  ) as f:
    cfg = json.load(f)

  return {
      "config": cfg,
      "ingested": ingested,
      "ind_decoupling_panel": df_ind_panel,
      "ind_decoupling_summary": df_ind_summary,
      "china_pivot": df_china_pivot,
      "eu_msr": df_eu_msr,
      "containment_stress": df_stress,
      "mc_annual": df_mc_annual,
      "basis_spreads": df_spreads,
      "cbam_panel": df_cbam_panel,
      "cbam_summary": df_cbam_summary,
      "art6_matrix": df_art6,
  }


data_bundle = load_all_engine_artifacts()
cfg = data_bundle["config"]

with st.sidebar:
  st.markdown("### ⚙️ TERMINAL CONTROLS")
  selected_year = st.slider(
      "Evaluation Horizon Year",
      min_value=2026,
      max_value=2035,
      value=2030,
      step=1,
  )
  currency_view = st.radio(
      "Standardized Currency Basis",
      options=["EUR (€/tCO2e)", "USD ($/tCO2e)", "Local Currency"],
      index=0,
  )
  risk_quantile = st.selectbox(
      "Carbon-VaR Risk Metric",
      options=["Median (P50)", "95% Carbon-VaR (P95)", "99% Carbon-VaR (P99)"],
      index=1,
  )
  st.markdown("---")
  st.markdown("**Lead Quantitative Architect:**\n\nKaustav Goswami")
  st.caption(
      "Data Lineage: EEA EUTL, World Bank Carbon Pricing Dashboard (0042051), "
      "ICAP APE, CARB WCI, UK-ETS Registry, NZ-ETS MfE, Korea GIR, and India"
      " BEE/ICM."
  )

st.markdown(
    """<div class="terminal-header"><p class="terminal-title">GLOBAL CARBON COMPLIANCE ARCHITECTURE: INDIA'S CCTS VS. THE WORLD (2026-2035)</p><p class="terminal-subtitle">STRUCTURAL MECHANICS &bull; PIECEWISE CONTAINMENT COLLARS &bull; 10,000-PATH OU JUMP-DIFFUSION &bull; EU/UK-CBAM &amp; ARTICLE 6.4 ARBITRAGE</p></div>""",
    unsafe_allow_html=True,
)

df_sp = data_bundle["basis_spreads"]
sp_row = df_sp[df_sp["year"] == selected_year].iloc[0]
df_ind_sum = data_bundle["ind_decoupling_summary"]
df_cbam_sum = data_bundle["cbam_summary"]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
  st.metric(
      label=f"{selected_year} EU-ETS vs. India CCTS Spread",
      value=f"€{sp_row['eu_vs_india_spread_median_eur']:.2f}/t",
      delta=f"99% VaR: €{sp_row['eu_vs_india_spread_var99_eur']:.2f}/t",
  )
with kpi2:
  st.metric(
      label="India 9-Sector 10-Yr Decoupling Wedge",
      value=f"+{df_ind_sum['cumulative_decoupling_wedge_mt'].sum():,.1f} Mt",
      delta="2035 Intensity vs. 4.3% Abs Cap",
  )
with kpi3:
  st.metric(
      label="CBAM Article 9 Intensity Haircut",
      value=f"{df_cbam_sum['mean_intensity_haircut_pct'].mean():.2f}%",
      delta=(
          f"{df_cbam_sum[df_cbam_sum['sector_code']!='cement']['mean_unhedged_cbam_share_pct'].mean():.2f}%"
          " Unhedged Tariff"
      ),
  )
with kpi4:
  st.metric(
      label="10-Yr Cumulative EU + UK CBAM Liability",
      value=(
          f"€{df_cbam_sum['cumulative_eu_cbam_liability_2026_2035_meur'].sum():,.1f}M"
      ),
      delta=(
          f"UK-CBAM:"
          f" £{df_cbam_sum['cumulative_uk_cbam_liability_2026_2035_mgbp'].sum():,.1f}M"
      ),
  )

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Global Architecture & Containment Collars",
    "2. Macro Decoupling Engine (India & China)",
    "3. Econometric Price Discovery & Carbon-VaR",
    "4. Cross-Border CBAM & Article 6 Arbitrage",
])

with tab1:
  st.subheader(
      "Structural & Containment Benchmarking Across All 7 Compliance Hubs"
  )
  df_mc = data_bundle["mc_annual"]
  df_yr = df_mc[df_mc["year"] == selected_year].copy()

  arch_rows = []
  for _, r in df_yr.iterrows():
    hub = r["jurisdiction"]
    j_cfg = cfg["jurisdictions"][hub]
    arch_rows.append({
        "Jurisdiction": hub,
        "Regime Type": j_cfg["regime_type"].upper(),
        "Unit": j_cfg["allowance_unit"],
        "Currency": r["currency"],
        "2026 Cap / Covered (Mt)": j_cfg.get(
            "base_cap_2026_mt", j_cfg.get("covered_emissions_2026_mt")
        ),
        "Stability Mechanism": j_cfg["stability_mechanism"]["type"],
        f"{selected_year} Floor (Local)": r["active_floor_local"],
        f"{selected_year} Median (Local)": r["median_price_local"],
        f"{selected_year} 95% VaR (Local)": r["var95_price_local"],
        f"{selected_year} Median (€/t)": r["median_price_eur"],
        f"{selected_year} 95% VaR (€/t)": r["var95_price_eur"],
        "Collar / Cliff Trigger Prob (%)": (
            r["containment_or_penalty_trigger_prob_pct"]
        ),
    })
  st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

  col_left, col_right = st.columns(2)
  with col_left:
    st.markdown(
        "#### EU-ETS Market Stability Reserve (MSR) Intake & Invalidation"
        " (2026-2035)"
    )
    df_msr = data_bundle["eu_msr"]
    fig1, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax1.plot(
        df_msr["year"],
        df_msr["tnac_prior_mt"],
        marker="o",
        color="#38BDF8",
        linewidth=2.2,
        label="TNAC in Circulation (Mt)",
    )
    ax1.bar(
        df_msr["year"] - 0.15,
        -df_msr["msr_intake_or_release_mt"],
        width=0.3,
        color="#F59E0B",
        alpha=0.85,
        label="MSR Annual Intake Withdrawn (Mt)",
    )
    ax1.bar(
        df_msr["year"] + 0.15,
        df_msr["msr_invalidated_mt"],
        width=0.3,
        color="#EF4444",
        alpha=0.85,
        label="MSR Holdings Invalidated (Mt)",
    )
    ax1.axhline(
        833.0,
        color="#94A3B8",
        linestyle="--",
        linewidth=1.0,
        label="833 Mt Upper Threshold",
    )
    ax1.set_xlabel("Compliance Year")
    ax1.set_ylabel("Million Tonnes CO2e (Mt)")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    st.pyplot(fig1)

  with col_right:
    st.markdown(
        "#### Statutory Containment Manifold Stress Test: WCI APCR vs. India"
        " Section 26 Cliff"
    )
    df_stress = data_bundle["containment_stress"]
    scenario_pick = st.selectbox(
        "Select Stress-Test Regime",
        options=[
            "SCARCITY_SPIKE_TEST",
            "HIGH_DEMAND_2030",
            "RECESSION_FLOOR_TEST",
            "BASELINE_2026",
        ],
        index=0,
    )
    sub_stress = df_stress[df_stress["scenario"] == scenario_pick][[
        "jurisdiction",
        "currency",
        "unconstrained_shadow_price",
        "constrained_clearing_price",
        "active_floor_price",
        "hard_ceiling_or_penalty_price",
        "reserve_supply_injected_mt",
        "regime_state",
        "physical_abatement_backed",
    ]]
    st.dataframe(sub_stress, use_container_width=True, hide_index=True)
    st.info(
        "**Microstructure Insight:** Under `SCARCITY_SPIKE_TEST` and"
        " `HIGH_DEMAND_2030`, California WCI, UK-ETS, NZ-ETS, and K-ETS inject"
        " backed reserve allowances (`physical_abatement_backed = True`). By"
        " contrast, India's Energy Conservation Act Section 26(1A) caps"
        " financial liability at **2.0x average clearing price** without"
        " retiring physical allowances (`physical_abatement_backed = False`)."
    )

with tab2:
  st.subheader(
      "ODE Decoupling Proof: Output-Adjusted Intensity Baseline vs. Fixed"
      " Absolute Cap"
  )
  st.markdown(
      r"""
        <div class="formula-box">
        <b>Continuous-Time Decoupling Differential:</b>
        $$\frac{\dot{E}_t^{\text{int}}}{E_t^{\text{int}}} = \frac{\dot{Y}_t}{Y_t} + \frac{\dot{\iota}_t^*}{\iota_t^*} = g_Y(t) - \gamma_\iota(t) \quad \Longrightarrow \quad \dot{E}_t^{\text{int}} > 0 \;\;\forall\; g_Y(t) > \gamma_\iota(t)$$
        </div>
        """,
      unsafe_allow_html=True,
  )

  ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
  with ctrl1:
    sector_options = {"ALL_9_SECTORS": "All 9 Notified BEE Sectors (Aggregate)"}
    for k, v in cfg["india_ccts_sectors"].items():
      sector_options[k] = v["sector_name"]
    chosen_sec = st.selectbox(
        "Select Indian CCTS Sector",
        options=list(sector_options.keys()),
        format_func=lambda x: sector_options[x],
    )
  with ctrl2:
    val_gy = st.slider(
        "Real Output Growth g_Y (%/yr)",
        min_value=2.0,
        max_value=10.0,
        value=6.8,
        step=0.1,
    )
    g_y_override = val_gy / 100.0
  with ctrl3:
    val_gp2 = st.slider(
        "Phase 2 Intensity Tightening gamma_iota (%/yr)",
        min_value=2.0,
        max_value=12.0,
        value=5.0,
        step=0.1,
    )
    gamma_p2_override = val_gp2 / 100.0
  with ctrl4:
    val_lrf = st.slider(
        "Counterfactual Absolute Cap LRF (%/yr)",
        min_value=2.0,
        max_value=7.0,
        value=4.3,
        step=0.1,
    )
    lrf_override = val_lrf / 100.0

  t_grid = np.arange(0, 10, dtype=float)
  years_grid = np.arange(2026, 2036, dtype=int)
  if chosen_sec == "ALL_9_SECTORS":
    e0_val = float(df_ind_sum["base_emissions_2026_mt"].sum())
    gam_p1_val = 0.028
  else:
    s_cfg = cfg["india_ccts_sectors"][chosen_sec]
    e0_val = float(s_cfg["base_output_2026_mt_product"]) * float(
        s_cfg["baseline_intensity_2026_tco2_per_t"]
    )
    gam_p1_val = float(s_cfg["annual_target_tightening_gamma_phase1"])

  _, _, e_int_custom = StructuralDecouplingEngine.solve_coupled_sector_ode(
      y0=e0_val,
      iota0=1.0,
      g_y=g_y_override,
      gamma_phase1=gam_p1_val,
      gamma_phase2=gamma_p2_override,
      t_eval=t_grid,
      phase_split_t=2.0,
  )
  e_abs_custom = np.array([
      StructuralDecouplingEngine.absolute_cap_supply(e0_val, lrf_override, t)[0]
      for t in t_grid
  ])
  gam_inst_star, gam_disc_star, gam_cum_star = (
      StructuralDecouplingEngine.compute_threshold_gamma_star(
          e0=e0_val,
          g_y=g_y_override,
          alpha_lrf_benchmark=lrf_override,
          horizon_years=9,
      )
  )

  m1, m2, m3, m4 = st.columns(4)
  m1.metric(
      "2035 Intensity Emissions",
      f"{e_int_custom[-1]:,.2f} Mt",
      f"2026 Base: {e0_val:,.2f} Mt",
  )
  m2.metric(
      "2035 Counterfactual Abs Cap",
      f"{e_abs_custom[-1]:,.2f} Mt",
      f"LRF: {lrf_override*100:.1f}%/yr",
  )
  m3.metric(
      "10-Yr Cumulative Decoupling Wedge",
      f"+{np.sum(e_int_custom - e_abs_custom):,.2f} Mt",
      "Intensity minus Abs Cap",
  )
  m4.metric(
      "Peaking Threshold gamma_iota*",
      f"{gam_inst_star*100:.2f}%/yr",
      f"Cap-Equiv: {gam_cum_star*100:.2f}%/yr",
  )

  c_plot1, c_plot2 = st.columns(2)
  with c_plot1:
    st.markdown(
        f"#### {sector_options[chosen_sec]}: Intensity Baseline vs. Absolute"
        " Cap (2026-2035)"
    )
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.8))
    ax2.plot(
        years_grid,
        e_int_custom,
        marker="o",
        color="#F59E0B",
        linewidth=2.4,
        label=(
            f"Intensity Regime (g_Y={g_y_override*100:.1f}%,"
            f" gamma_P2={gamma_p2_override*100:.1f}%)"
        ),
    )
    ax2.plot(
        years_grid,
        e_abs_custom,
        marker="s",
        color="#38BDF8",
        linewidth=2.4,
        label=f"Absolute Cap Counterfactual (LRF={lrf_override*100:.1f}%)",
    )
    ax2.fill_between(
        years_grid,
        e_abs_custom,
        e_int_custom,
        color="#F59E0B",
        alpha=0.18,
        label="Unabated Decoupling Wedge",
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Emissions (MtCO2e)")
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    st.pyplot(fig2)

  with c_plot2:
    st.markdown(
        "#### China National ETS: 2028 Dual-Control Pivot to Absolute Cap"
    )
    df_chn = data_bundle["china_pivot"]
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
    ax3.plot(
        df_chn["year"],
        df_chn["emissions_unreformed_intensity_mt"],
        marker="^",
        linestyle="--",
        color="#EF4444",
        linewidth=2.0,
        label="Unreformed Intensity Baseline (g_Y=4.5%, gamma=3.2%)",
    )
    ax3.plot(
        df_chn["year"],
        df_chn["emissions_dual_control_pivot_mt"],
        marker="o",
        color="#10B981",
        linewidth=2.4,
        label="2028 Dual-Control Absolute Cap Pivot (LRF=5.0%)",
    )
    ax3.axvline(
        2028,
        color="#38BDF8",
        linestyle=":",
        linewidth=1.4,
        label="2028 Statutory Cap Pivot",
    )
    ax3.fill_between(
        df_chn["year"],
        df_chn["emissions_dual_control_pivot_mt"],
        df_chn["emissions_unreformed_intensity_mt"],
        color="#10B981",
        alpha=0.18,
        label="Avoided Emissions (18,525 Mt Cum.)",
    )
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Covered Emissions (MtCO2e)")
    ax3.grid(True)
    ax3.legend(loc="lower left", frameon=False, fontsize=8)
    st.pyplot(fig3)

  st.markdown(
      "#### All 9 Notified BEE CCTS Sectors Decoupling & Peaking Summary Table"
  )
  st.dataframe(df_ind_sum, use_container_width=True, hide_index=True)

with tab3:
  st.subheader(
      "10,000-Path Correlated OU Jump-Diffusion Price Discovery & Basis Spreads"
  )
  p_col1, p_col2 = st.columns([1, 3])
  with p_col1:
    hub_pick = st.selectbox(
        "Select Compliance Hub for Fan Chart",
        options=cfg["correlation_matrix"]["hub_order"],
        index=6,
    )
    show_collars = st.checkbox(
        "Overlay Statutory Floor & Ceiling/Penalty Collars", value=True
    )

  df_hub_mc = df_mc[df_mc["jurisdiction"] == hub_pick].copy()
  if "EUR" in currency_view:
    scale_col_med, scale_col_v95, scale_col_v99 = (
        "median_price_eur",
        "var95_price_eur",
        "var99_price_eur",
    )
    p25_vals = df_hub_mc["p25_price_local"] * df_hub_mc["fx_to_eur"]
    p75_vals = df_hub_mc["p75_price_local"] * df_hub_mc["fx_to_eur"]
    fl_vals = df_hub_mc["active_floor_local"] * df_hub_mc["fx_to_eur"]
    cl_vals = (
        df_hub_mc["active_ceiling_or_penalty_local"] * df_hub_mc["fx_to_eur"]
    )
    unit_label = "EUR (€/tCO2e)"
  elif "USD" in currency_view:
    scale_col_med, scale_col_v95, scale_col_v99 = (
        "median_price_usd",
        "var95_price_usd",
        "var99_price_usd",
    )
    p25_vals = df_hub_mc["p25_price_local"] * df_hub_mc["fx_to_usd"]
    p75_vals = df_hub_mc["p75_price_local"] * df_hub_mc["fx_to_usd"]
    fl_vals = df_hub_mc["active_floor_local"] * df_hub_mc["fx_to_usd"]
    cl_vals = (
        df_hub_mc["active_ceiling_or_penalty_local"] * df_hub_mc["fx_to_usd"]
    )
    unit_label = "USD ($/tCO2e)"
  else:
    scale_col_med, scale_col_v95, scale_col_v99 = (
        "median_price_local",
        "var95_price_local",
        "var99_price_local",
    )
    p25_vals = df_hub_mc["p25_price_local"]
    p75_vals = df_hub_mc["p75_price_local"]
    fl_vals = df_hub_mc["active_floor_local"]
    cl_vals = df_hub_mc["active_ceiling_or_penalty_local"]
    unit_label = f"{df_hub_mc['currency'].iloc[0]}/tCO2e"

  fan_left, fan_right = st.columns(2)
  with fan_left:
    st.markdown(
        f"#### {hub_pick} 10,000-Path Monte Carlo Fan Chart ({unit_label})"
    )
    fig4, ax4 = plt.subplots(figsize=(6.5, 3.9))
    yrs = df_hub_mc["year"]
    ax4.plot(
        yrs,
        df_hub_mc[scale_col_med],
        marker="o",
        color="#38BDF8",
        linewidth=2.4,
        label="Median (P50)",
    )
    ax4.fill_between(
        yrs,
        p25_vals,
        p75_vals,
        color="#38BDF8",
        alpha=0.28,
        label="IQR (P25-P75)",
    )
    ax4.plot(
        yrs,
        df_hub_mc[scale_col_v95],
        linestyle="--",
        color="#F59E0B",
        linewidth=1.8,
        label="95% Carbon-VaR (P95)",
    )
    ax4.plot(
        yrs,
        df_hub_mc[scale_col_v99],
        linestyle=":",
        color="#EF4444",
        linewidth=1.8,
        label="99% Carbon-VaR (P99)",
    )
    if show_collars:
      ax4.plot(
          yrs,
          fl_vals,
          linestyle="-.",
          color="#10B981",
          linewidth=1.3,
          label="Statutory Floor",
      )
      ax4.plot(
          yrs,
          cl_vals,
          linestyle="-.",
          color="#EC4899",
          linewidth=1.3,
          label="Statutory Ceiling / Sec. 26 Cliff",
      )
    ax4.set_xlabel("Compliance Year")
    ax4.set_ylabel(unit_label)
    ax4.grid(True)
    ax4.legend(loc="upper left", frameon=False, fontsize=8)
    st.pyplot(fig4)

  with fan_right:
    st.markdown(
        "#### Cross-Border Basis Spread Trajectory vs. India CCTS (€/tCO2e)"
    )
    fig5, ax5 = plt.subplots(figsize=(6.5, 3.9))
    ax5.plot(
        df_sp["year"],
        df_sp["eu_vs_india_spread_median_eur"],
        marker="o",
        color="#38BDF8",
        linewidth=2.4,
        label="EU-ETS minus India CCC (Median €)",
    )
    ax5.fill_between(
        df_sp["year"],
        df_sp["eu_vs_india_spread_median_eur"],
        df_sp["eu_vs_india_spread_var99_eur"],
        color="#EF4444",
        alpha=0.20,
        label="EU-India Spread 95%-99% VaR Tail",
    )
    ax5.plot(
        df_sp["year"],
        df_sp["uk_vs_india_spread_median_eur"],
        marker="s",
        color="#A855F7",
        linewidth=2.0,
        label="UK-ETS minus India CCC (Median €)",
    )
    ax5.set_xlabel("Compliance Year")
    ax5.set_ylabel("Cross-Border Basis Spread (€/tCO2e)")
    ax5.grid(True)
    ax5.legend(loc="upper left", frameon=False, fontsize=8)
    st.pyplot(fig5)

  st.markdown("#### Cross-Border Basis Spread Matrix (2026-2035)")
  st.dataframe(df_sp, use_container_width=True, hide_index=True)

with tab4:
  st.subheader(
      "EU/UK-CBAM Intensity Recognition Haircut & UNFCCC Article 6.4 PACM"
      " Arbitrage"
  )
  cbam_panel = data_bundle["cbam_panel"]

  col_cbam1, col_cbam2 = st.columns(2)
  with col_cbam1:
    st.markdown(
        "#### 4-Sector Unhedged EU-CBAM Tariff Trajectory (€/t Product,"
        " 2026-2035)"
    )
    fig6, ax6 = plt.subplots(figsize=(6.5, 3.9))
    palette = {
        "iron_and_steel": "#38BDF8",
        "aluminum": "#F59E0B",
        "fertilizers": "#10B981",
        "cement": "#EC4899",
    }
    for scode, grp in cbam_panel.groupby("sector_code"):
      ax6.plot(
          grp["year"],
          grp["net_unhedged_eu_cbam_tariff_eur_per_t_product"],
          marker="o",
          linewidth=2.2,
          color=palette.get(scode, "#38BDF8"),
          label=grp["sector_name"].iloc[0],
      )
    ax6.set_xlabel(
        "Compliance Year (EU Free Allocation Phase-Out Phi_CBAM: 97.5% -> 0%)"
    )
    ax6.set_ylabel("Net Unhedged EU-CBAM Tariff (€/t Product)")
    ax6.grid(True)
    ax6.legend(loc="upper left", frameon=False, fontsize=8)
    st.pyplot(fig6)

  with col_cbam2:
    st.markdown(
        "#### Interactive UNFCCC Article 6.2 / 6.4 PACM ITMO Arbitrage"
        " Calculator"
    )
    a_col1, a_col2 = st.columns(2)
    with a_col1:
      mac_input = st.slider(
          "Host Project MAC ($/tCO2e)",
          min_value=8.0,
          max_value=45.0,
          value=16.5,
          step=0.5,
      )
      omega_input = st.slider(
          "Sovereign NDC Opportunity Cost Omega_NDC ($/t)",
          min_value=0.0,
          max_value=40.0,
          value=11.5,
          step=0.5,
      )
    with a_col2:
      sop_val = st.slider(
          "Article 6.4 Adaptation SOP Levy (%)",
          min_value=0.0,
          max_value=10.0,
          value=5.0,
          step=0.5,
      )
      sop_input = sop_val / 100.0
      omge_val = st.slider(
          "Mandatory OMGE Cancellation (%)",
          min_value=0.0,
          max_value=10.0,
          value=2.0,
          step=0.5,
      )
      omge_input = omge_val / 100.0

    gross_up = 1.0 / max(1e-4, 1.0 - sop_input - omge_input)
    min_itmo_offer = (mac_input + omega_input) * gross_up + 0.30

    st.markdown(
        f"**PACM Gross-Up Multiplier:** `{gross_up:.5f}x` | "
        f"**Minimum Viable ITMO Export Price:** `${min_itmo_offer:.2f}/tCO2e`"
    )

    df_art6_yr = data_bundle["art6_matrix"]
    df_art6_sub = df_art6_yr[df_art6_yr["year"] == selected_year].copy()
    df_art6_sub["custom_min_itmo_offer_usd"] = round(min_itmo_offer, 2)
    df_art6_sub["custom_net_arbitrage_spread_usd"] = (
        df_art6_sub["buyer_median_price_usd"] - min_itmo_offer
    ).round(2)
    df_art6_sub["custom_viable"] = (
        df_art6_sub["custom_net_arbitrage_spread_usd"] >= 0.0
    )
    st.dataframe(
        df_art6_sub[[
            "buyer_jurisdiction",
            "buyer_median_price_usd",
            "india_domestic_ccc_price_usd",
            "custom_min_itmo_offer_usd",
            "custom_net_arbitrage_spread_usd",
            "max_tolerable_sovereign_omega_ca_usd",
            "custom_viable",
        ]],
        use_container_width=True,
        hide_index=True,
    )

  st.markdown(
      "#### 4-Sector EU-CBAM & UK-CBAM 10-Year Liability & Intensity Recognition"
      " Haircut Summary"
  )
  st.dataframe(df_cbam_sum, use_container_width=True, hide_index=True)