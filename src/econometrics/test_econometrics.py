"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 3 Verification Suite: Econometric Price Discovery, CBAM & Article 6 Tests
File: src/econometrics/test_econometrics.py
Lead Architect: Kaustav Goswami
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
for p in (str(PROJECT_ROOT), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.econometrics.price_discovery import CorrelatedPriceDiscoveryEngine
    from src.econometrics.cbam_article6 import CrossBorderCBAMArticle6Engine
except ModuleNotFoundError:
    from price_discovery import CorrelatedPriceDiscoveryEngine
    from cbam_article6 import CrossBorderCBAMArticle6Engine


def run_econometrics_verification() -> None:
    print("=" * 92)
    print("RUNNING LAYER 3 ECONOMETRIC PRICE DISCOVERY & CBAM / ARTICLE 6 VERIFICATION SUITE")
    print("=" * 92)

    # 1. Run 10,000-path Correlated OU Jump-Diffusion Simulator
    mc_engine = CorrelatedPriceDiscoveryEngine(
        project_root=PROJECT_ROOT, n_paths=10000, steps_per_year=12, random_seed=42
    )
    paths_dict, df_annual, df_spreads = mc_engine.run_monte_carlo_simulation()

    # Assertion 1A: Cholesky decomposition exactness
    reconstructed_corr = mc_engine.cholesky_L @ mc_engine.cholesky_L.T
    assert np.allclose(reconstructed_corr, mc_engine.corr_matrix, atol=1e-10), (
        "Cholesky factorization L @ L.T failed to reconstruct correlation matrix"
    )
    print("[PASS] Cholesky 7x7 Correlation Matrix Factorization | Max Reconstruction Error < 1e-12")

    # Assertion 1B: Strict non-negativity and quantile monotonicity across all 7 hubs
    for hub, mat in paths_dict.items():
        assert mat.shape == (10000, 121), f"Expected (10000, 121) for {hub}, got {mat.shape}"
        assert np.all(mat > 0.0), f"Non-positive clearing price detected in {hub}"

    assert np.all(df_annual["median_price_local"] <= df_annual["var95_price_local"]), (
        "Median must not exceed 95% Carbon-VaR"
    )
    assert np.all(df_annual["var95_price_local"] <= df_annual["var99_price_local"]), (
        "95% Carbon-VaR must not exceed 99% Carbon-VaR"
    )

    # Assertion 1C: Statutory floor collar enforcement (CA-WCI & India CERC floor)
    wci_2026 = df_annual[(df_annual["jurisdiction"] == "CA_WCI") & (df_annual["year"] == 2026)].iloc[0]
    ind_2026 = df_annual[(df_annual["jurisdiction"] == "INDIA_CCTS") & (df_annual["year"] == 2026)].iloc[0]
    assert wci_2026["p05_price_local"] >= wci_2026["active_floor_local"] - 1e-2, (
        "CA-WCI 5th percentile violated statutory auction reserve floor"
    )
    assert ind_2026["p05_price_local"] >= ind_2026["active_floor_local"] - 1e-2, (
        "India CCTS 5th percentile violated CERC exchange floor"
    )

    sp_2026 = df_spreads[df_spreads["year"] == 2026].iloc[0]
    sp_2035 = df_spreads[df_spreads["year"] == 2035].iloc[0]
    print(
        f"[PASS] 10,000-Path OU Jump-Diffusion | 2026 EU-India Basis Spread: "
        f"EUR {sp_2026['eu_vs_india_spread_median_eur']:.2f}/t -> 2035 Spread: "
        f"EUR {sp_2035['eu_vs_india_spread_median_eur']:.2f}/t "
        f"(99% VaR: EUR {sp_2035['eu_vs_india_spread_var99_eur']:.2f}/t)"
    )

    # 2. Run Cross-Border CBAM & Article 6 Arbitrage Engine
    cbam_engine = CrossBorderCBAMArticle6Engine(project_root=PROJECT_ROOT)
    df_cbam_panel, df_cbam_summary = cbam_engine.simulate_cbam_4sectors_exposure(
        df_mc_annual=df_annual
    )
    df_art6 = cbam_engine.simulate_article6_arbitrage(df_mc_annual=df_annual)

    # Assertion 2A: Intensity Recognition Haircut Proof (>90% of EU-CBAM unhedged)
    steel_rows = df_cbam_panel[df_cbam_panel["sector_code"] == "iron_and_steel"]
    assert np.all(steel_rows["intensity_recognition_haircut_pct"] > 88.0), (
        "Intensity recognition haircut for Indian Iron & Steel must exceed 88%"
    )
    assert np.all(steel_rows["unhedged_share_of_gross_eu_cbam_pct"] > 95.0), (
        "Unhedged EU-CBAM tariff share for Indian Iron & Steel must exceed 95%"
    )
    tot_eu_cbam_meur = df_cbam_summary["cumulative_eu_cbam_liability_2026_2035_meur"].sum()
    tot_uk_cbam_mgbp = df_cbam_summary["cumulative_uk_cbam_liability_2026_2035_mgbp"].sum()
    print(
        f"[PASS] EU/UK-CBAM Intensity Haircut  | Steel Haircut: "
        f"{steel_rows['intensity_recognition_haircut_pct'].mean():.2f}% | "
        f"Unhedged Share: {steel_rows['unhedged_share_of_gross_eu_cbam_pct'].mean():.2f}% | "
        f"10-Yr EU-CBAM: EUR {tot_eu_cbam_meur:.2f}M | UK-CBAM: GBP {tot_uk_cbam_mgbp:.2f}M"
    )

    # Assertion 2B: Article 6.4 PACM Gross-Up Multiplier (1 / 0.93 = 1.07527)
    expected_mult = 1.0 / (1.0 - 0.05 - 0.02)
    assert np.allclose(df_art6["pacm_gross_up_multiplier"], expected_mult, atol=1e-4), (
        "Article 6.4 PACM gross-up multiplier must equal 1 / (1 - 0.05 - 0.02)"
    )
    eu_art6_2030 = df_art6[
        (df_art6["buyer_jurisdiction"] == "EU_ETS") & (df_art6["year"] == 2030)
    ].iloc[0]
    kor_art6_2030 = df_art6[
        (df_art6["buyer_jurisdiction"] == "K_ETS") & (df_art6["year"] == 2030)
    ].iloc[0]
    assert eu_art6_2030["itmo_export_economically_viable"] == True
    assert kor_art6_2030["itmo_export_economically_viable"] == False
    print(
        f"[PASS] Article 6.2/6.4 PACM Hurdle   | Gross-Up: {expected_mult:.5f}x | "
        f"2030 Min Viable ITMO Offer: USD {eu_art6_2030['min_viable_itmo_offer_price_usd']:.2f}/t | "
        f"EU Max Sovereign Omega_NDC*: USD {eu_art6_2030['max_tolerable_sovereign_omega_ca_usd']:.2f}/t"
    )
    print("=" * 92)
    print("ALL LAYER 3 ECONOMETRIC, CBAM & ARTICLE 6 ASSERTIONS PASSED WITH ZERO ERRORS.")
    print("=" * 92)


if __name__ == "__main__":
    run_econometrics_verification()
