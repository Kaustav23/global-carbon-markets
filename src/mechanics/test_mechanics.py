"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 2 Verification Suite: Unit & Integration Assertions for Mechanics Modules
File: src/mechanics/test_mechanics.py
Lead Architect: Kaustav Goswami
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from src.mechanics.cap_vs_intensity import StructuralDecouplingEngine
    from src.mechanics.stability_reserves import MarketStabilityContainmentEngine
except ModuleNotFoundError:
    from cap_vs_intensity import StructuralDecouplingEngine
    from stability_reserves import MarketStabilityContainmentEngine


def run_mechanics_verification() -> None:
    print("=" * 88)
    print("RUNNING LAYER 2 STRUCTURAL & CONTAINMENT MECHANICS VERIFICATION SUITE")
    print("=" * 88)

    # 1. Verify Structural Decoupling Engine (Dimension 1)
    decoupling_engine = StructuralDecouplingEngine(project_root=PROJECT_ROOT)
    df_ind_panel, df_ind_summary = decoupling_engine.simulate_india_9sectors_decoupling(
        benchmark_lrf=0.043
    )
    df_china = decoupling_engine.simulate_china_dual_control_pivot()

    s_abs_0, ds_dy_abs = decoupling_engine.absolute_cap_supply(1000.0, 0.043, 5.0)
    s_int_0, ds_dy_int = decoupling_engine.intensity_baseline_supply(150.0, 2.28)
    assert ds_dy_abs == 0.0, "Absolute cap dS/dY must equal 0.0"
    assert abs(ds_dy_int - 2.28) < 1e-9, "Intensity baseline dS/dY must equal iota_t^*"
    assert len(df_ind_summary) == 9, "Must evaluate all 9 notified BEE CCTS sectors"
    assert df_ind_summary["cbam_exposed"].sum() == 4, "Exactly 4 sectors must be CBAM-exposed"

    total_2026_ind = df_ind_summary["base_emissions_2026_mt"].sum()
    total_2035_int = df_ind_summary["emissions_2035_intensity_regime_mt"].sum()
    total_2035_abs = df_ind_summary["emissions_2035_abs_cap_regime_mt"].sum()
    total_wedge = df_ind_summary["cumulative_decoupling_wedge_mt"].sum()

    print(
        f"[PASS] India 9-Sector Decoupling ODE | 2026 Base: {total_2026_ind:.2f} Mt | "
        f"2035 Intensity: {total_2035_int:.2f} Mt | 2035 Abs Cap: {total_2035_abs:.2f} Mt | "
        f"Cumulative 10-Yr Wedge: +{total_wedge:.2f} Mt"
    )

    china_avoided_cum = df_china["annual_avoided_emissions_mt"].sum()
    assert df_china.loc[df_china["year"] == 2028, "dS_dY_normalized"].iloc[0] == 0.0
    print(
        f"[PASS] China 2028 Dual-Control Pivot | 2035 Pivot Cap: "
        f"{df_china.iloc[-1]['emissions_dual_control_pivot_mt']:.2f} Mt vs. "
        f"Unreformed Intensity: {df_china.iloc[-1]['emissions_unreformed_intensity_mt']:.2f} Mt | "
        f"Cumulative Avoided (2026-2035): {china_avoided_cum:.2f} Mt"
    )

    # 2. Verify Market Stability & Containment Engine (Dimension 2)
    stability_engine = MarketStabilityContainmentEngine(project_root=PROJECT_ROOT)
    df_eu_msr = stability_engine.simulate_eu_msr_trajectory_2026_2035()
    df_stress = stability_engine.run_cross_hub_stress_test()

    row_2026 = df_eu_msr[df_eu_msr["year"] == 2026].iloc[0]
    row_2027 = df_eu_msr[df_eu_msr["year"] == 2027].iloc[0]
    assert row_2026["regime_state"] == "MSR_24PCT_INTAKE"
    assert abs(row_2026["msr_intake_or_release_mt"] - (-0.24 * 1112.0)) < 1e-2
    assert row_2027["regime_state"] == "MSR_BUFFER_SMOOTHED_INTAKE"
    print(
        f"[PASS] EU-ETS MSR Control Operator   | 2026 24% Intake: {row_2026['msr_intake_or_release_mt']:.2f} Mt | "
        f"2027 Buffer Intake: {row_2027['msr_intake_or_release_mt']:.2f} Mt | "
        f"2027 Invalidated: {row_2027['msr_invalidated_mt']:.2f} Mt"
    )

    ind_spike = stability_engine.evaluate_india_ccts_penalty_cliff(
        year=2026, unconstrained_mac_inr=2150.0, rolling_avg_clearing_price_inr=980.0
    )
    assert ind_spike.regime_state == "INDIA_SECTION_26_PENALTY_CLIFF_BREACHED"
    assert ind_spike.physical_abatement_backed is False
    print(
        f"[PASS] India BEE Section 26(1A) Cliff| Shadow MAC: Rs {ind_spike.unconstrained_shadow_price:.0f}/t | "
        f"2x Penalty Trigger: Rs {ind_spike.tier2_trigger_price:.0f}/t | "
        f"Effective Cap: Rs {ind_spike.constrained_clearing_price:.0f}/t | "
        f"Backed Abatement: {ind_spike.physical_abatement_backed}"
    )
    print("=" * 88)
    print("ALL LAYER 2 MECHANICS ASSERTIONS PASSED WITH ZERO ERRORS.")
    print("=" * 88)


if __name__ == "__main__":
    run_mechanics_verification()
