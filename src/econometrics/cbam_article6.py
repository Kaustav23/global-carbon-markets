"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 3 - Module 2: EU/UK-CBAM Cross-Border Tariff & Article 6.2/6.4 Arbitrage Engine
File: src/econometrics/cbam_article6.py
Lead Architect: Kaustav Goswami

Mathematical Core:
1. EU-CBAM & UK-CBAM Specific Tariff Liability (Regulation (EU) 2023/956 Article 9 & 31):
     tau_{j -> d, s, t}^{CBAM} = max(
         0,
         [ e_{j,s,t}^{actual} - Phi_{d,t}^{CBAM} * B_{d,s,t}^{bench} ] * P_{d,t}^{ETS}
         - C_{j,s,t}^{paid,FX}
     )
2. The Intensity Recognition Haircut Proof:
   Under India's output-adjusted intensity baseline (CCTS), an obligated industrial
   facility pays domestic carbon prices ONLY on the marginal intensity gap
   max(0, iota_{s,t} - iota_{s,t}^*), whereas EU-CBAM and UK-CBAM assess total
   verified embedded emissions e_{s,t}^{actual} = iota_{s,t} less free-allocation
   benchmarks. Consequently:
     C_{IND,s,t}^{paid,EUR} = max(0, iota_{s,t} - iota_{s,t}^*) * P_{IND,t}^{CCC} * FX_{INR->EUR}
   leaving >95% of the gross CBAM tariff unhedged even at full domestic compliance.
3. UNFCCC Article 6.2 / 6.4 PACM Bilateral ITMO Arbitrage Condition:
     P_{buyer,t} >= ( MAC_{host,t} + Omega_{NDC,t}^{CA} ) / ( 1 - lambda_{SOP} - lambda_{OMGE} ) + c_{fee}
   where lambda_{SOP} = 5% (Adaptation Fund Share of Proceeds) and
   lambda_{OMGE} = 2% (Overall Mitigation in Global Emissions cancellation).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
for p in (str(PROJECT_ROOT), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.econometrics.price_discovery import CorrelatedPriceDiscoveryEngine
except ModuleNotFoundError:
    from price_discovery import CorrelatedPriceDiscoveryEngine


@dataclass
class CBAMSectorTariffRecord:
    """Annual EU-CBAM and UK-CBAM exposure metrics for an Indian export sector."""
    year: int
    sector_code: str
    sector_name: str
    india_actual_intensity_tco2_per_t: float
    india_ccts_target_intensity_tco2_per_t: float
    india_marginal_taxed_emissions_tco2_per_t: float
    eu_cbam_phi_factor: float
    uk_cbam_phi_factor: float
    eu_benchmark_tco2_per_t: float
    eu_net_liable_emissions_tco2_per_t: float
    eu_ets_median_price_eur: float
    india_ccc_median_price_eur: float
    gross_eu_cbam_cost_eur_per_t_product: float
    article9_recognized_ccts_credit_eur_per_t_product: float
    hypothetical_full_scope_indian_credit_eur_per_t_product: float
    intensity_recognition_haircut_pct: float
    net_unhedged_eu_cbam_tariff_eur_per_t_product: float
    net_unhedged_eu_cbam_var95_eur_per_t_product: float
    unhedged_share_of_gross_eu_cbam_pct: float
    net_unhedged_uk_cbam_tariff_gbp_per_t_product: float
    india_eu_export_volume_mt: float
    total_eu_cbam_sector_liability_million_eur: float
    total_uk_cbam_sector_liability_million_gbp: float


@dataclass
class Article6ArbitrageRecord:
    """Annual Article 6.2 / 6.4 PACM bilateral transfer & CA threshold analysis."""
    year: int
    buyer_jurisdiction: str
    buyer_median_price_usd: float
    buyer_var95_price_usd: float
    india_domestic_ccc_price_usd: float
    host_marginal_abatement_cost_usd: float
    configured_ndc_opportunity_cost_omega_usd: float
    pacm_gross_up_multiplier: float
    min_viable_itmo_offer_price_usd: float
    net_bilateral_arbitrage_spread_usd: float
    max_tolerable_sovereign_omega_ca_usd: float
    itmo_export_economically_viable: bool


class CrossBorderCBAMArticle6Engine:
    """
    Computes EU-CBAM and UK-CBAM tariff exposure across the 4 trade-exposed
    Indian industrial sectors (iron & steel, aluminum, cement, fertilizers)
    and evaluates UNFCCC Article 6.2 / 6.4 PACM bilateral arbitrage dynamics.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.config_path = self.project_root / "config" / "jurisdictions.json"
        self.processed_dir = self.project_root / "data" / "processed"
        self.log_file = self.project_root / "logs" / "ingestion_and_sim_audit.log"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
        self.logger = self._get_logger()

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger("CrossBorderCBAMArticle6Engine")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                )
            )
            logger.addHandler(fh)
        return logger

    def _ensure_monte_carlo_summary(self) -> pd.DataFrame:
        mc_csv = self.processed_dir / "monte_carlo_7hubs_annual_summary_2026_2035.csv"
        if mc_csv.exists():
            return pd.read_csv(mc_csv)
        mc_engine = CorrelatedPriceDiscoveryEngine(project_root=self.project_root)
        _, df_ann, _ = mc_engine.run_monte_carlo_simulation()
        return df_ann

    # --------------------------------------------------------------------------
    # 1. EU-CBAM & UK-CBAM 4-Sector Tariff & Intensity Haircut Engine
    # --------------------------------------------------------------------------
    def simulate_cbam_4sectors_exposure(
        self, df_mc_annual: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Models EU-CBAM and UK-CBAM tariff exposure from 2026 to 2035 across India's
        4 CBAM-exposed sectors (iron & steel, aluminum, cement, fertilizers) and
        quantifies the Article 9 Intensity Recognition Haircut.
        """
        if df_mc_annual is None:
            df_mc_annual = self._ensure_monte_carlo_summary()

        cbam_cfg = self.config["cbam_architecture"]
        phi_eu_sched = {int(k): float(v) for k, v in cbam_cfg["eu_cbam_free_allocation_factor_phi"].items()}
        phi_uk_sched = {int(k): float(v) for k, v in cbam_cfg["uk_cbam_free_allocation_factor_phi"].items()}
        bench_cfg = cbam_cfg["exposed_sector_benchmarks"]
        sectors_cfg = self.config["india_ccts_sectors"]

        fx_cfg = self.config["fx_spot_2026_benchmarks"]
        inr_to_eur = float(fx_cfg["INR_EUR"])
        inr_to_gbp = float(fx_cfg["INR_USD"]) / float(fx_cfg["GBP_USD"])

        records: List[CBAMSectorTariffRecord] = []

        for sector_code, b_spec in bench_cfg.items():
            s_spec = sectors_cfg[sector_code]
            sector_name = s_spec["sector_name"]

            # For aluminum, EU/UK-CBAM definitive period initially assesses direct
            # electrolysis emissions (`india_direct_cbam_scope_intensity_2026`) while
            # domestic CCTS covers total plant boundary intensity.
            actual_2026 = float(
                b_spec.get(
                    "india_direct_cbam_scope_intensity_2026",
                    b_spec["india_avg_export_intensity_2026"],
                )
            )
            # Ratio of initial actual intensity to baseline target intensity in CCTS
            ccts_actual_0 = float(s_spec["actual_initial_intensity_2026_tco2_per_t"])
            ccts_target_0 = float(s_spec["baseline_intensity_2026_tco2_per_t"])
            initial_gap_ratio = max(0.015, (ccts_actual_0 - ccts_target_0) / ccts_actual_0)

            gam_p1 = float(s_spec["annual_target_tightening_gamma_phase1"])
            gam_p2 = float(s_spec["annual_target_tightening_gamma_phase2"])
            eu_bench = float(b_spec["eu_ets_product_benchmark_tco2_per_t"])
            uk_bench = float(b_spec["uk_ets_product_benchmark_tco2_per_t"])
            eu_exp_mt = float(b_spec["india_eu_annual_export_volume_mt"])
            uk_exp_mt = float(b_spec["india_uk_annual_export_volume_mt"])

            for idx, yr in enumerate(range(2026, 2036)):
                phi_eu = phi_eu_sched[yr]
                phi_uk = phi_uk_sched[yr]

                # Plant physical decarbonization lags target tightening slightly,
                # maintaining a realistic marginal compliance obligation gap (3.5% - 6.5% of total emissions)
                active_gamma = gam_p1 if yr <= 2028 else gam_p2
                actual_abate_rate = active_gamma * 0.88
                e_actual_t = actual_2026 * np.exp(-actual_abate_rate * idx)
                marginal_gap_share = min(0.12, initial_gap_ratio + 0.004 * idx)
                e_target_t = e_actual_t * (1.0 - marginal_gap_share)
                marginal_taxed_t = max(0.0, e_actual_t - e_target_t)

                # Extract Monte Carlo price quantiles for year `yr`
                eu_row = df_mc_annual[
                    (df_mc_annual["jurisdiction"] == "EU_ETS") & (df_mc_annual["year"] == yr)
                ].iloc[0]
                uk_row = df_mc_annual[
                    (df_mc_annual["jurisdiction"] == "UK_ETS") & (df_mc_annual["year"] == yr)
                ].iloc[0]
                ind_row = df_mc_annual[
                    (df_mc_annual["jurisdiction"] == "INDIA_CCTS") & (df_mc_annual["year"] == yr)
                ].iloc[0]

                p_eu_med = float(eu_row["median_price_local"])
                p_eu_v95 = float(eu_row["var95_price_local"])
                p_uk_med = float(uk_row["median_price_local"])
                p_ind_inr_med = float(ind_row["median_price_local"])
                p_ind_eur_med = p_ind_inr_med * inr_to_eur
                p_ind_gbp_med = p_ind_inr_med * inr_to_gbp

                # EU-CBAM Liability per tonne of exported product
                eu_liable_em = max(0.0, e_actual_t - phi_eu * eu_bench)
                gross_eu_cbam_eur = eu_liable_em * p_eu_med
                gross_eu_cbam_v95_eur = eu_liable_em * p_eu_v95

                # Article 9 Recognized Credit under Intensity Baseline:
                # Exporter only pays domestic CCC price on `marginal_taxed_t` = (iota_t - iota_t^*)
                art9_credit_eur = marginal_taxed_t * p_ind_eur_med
                # Hypothetical credit if India had a full-scope cap/tax on all `e_actual_t`
                full_scope_credit_eur = e_actual_t * p_ind_eur_med

                intensity_haircut_pct = (
                    (1.0 - art9_credit_eur / full_scope_credit_eur) * 100.0
                    if full_scope_credit_eur > 0
                    else 0.0
                )

                net_eu_cbam_eur = max(0.0, gross_eu_cbam_eur - art9_credit_eur)
                net_eu_cbam_v95_eur = max(0.0, gross_eu_cbam_v95_eur - art9_credit_eur)
                unhedged_share_pct = (
                    (net_eu_cbam_eur / gross_eu_cbam_eur) * 100.0
                    if gross_eu_cbam_eur > 0
                    else 0.0
                )

                # UK-CBAM Liability per tonne of exported product (enters into force 2027; 2026 phi_uk=1.0)
                if yr == 2026:
                    net_uk_cbam_gbp = 0.0
                else:
                    uk_liable_em = max(0.0, e_actual_t - phi_uk * uk_bench)
                    gross_uk_cbam_gbp = uk_liable_em * p_uk_med
                    art9_credit_gbp = marginal_taxed_t * p_ind_gbp_med
                    net_uk_cbam_gbp = max(0.0, gross_uk_cbam_gbp - art9_credit_gbp)

                # Aggregate trade-weighted liability (export volume grows at 2.5%/yr)
                vol_scale = (1.025) ** idx
                eu_vol_t = eu_exp_mt * vol_scale
                uk_vol_t = uk_exp_mt * vol_scale

                tot_eu_liab_meur = net_eu_cbam_eur * eu_vol_t
                tot_uk_liab_mgbp = net_uk_cbam_gbp * uk_vol_t

                records.append(
                    CBAMSectorTariffRecord(
                        year=yr,
                        sector_code=sector_code,
                        sector_name=sector_name,
                        india_actual_intensity_tco2_per_t=round(e_actual_t, 4),
                        india_ccts_target_intensity_tco2_per_t=round(e_target_t, 4),
                        india_marginal_taxed_emissions_tco2_per_t=round(marginal_taxed_t, 4),
                        eu_cbam_phi_factor=phi_eu,
                        uk_cbam_phi_factor=phi_uk,
                        eu_benchmark_tco2_per_t=eu_bench,
                        eu_net_liable_emissions_tco2_per_t=round(eu_liable_em, 4),
                        eu_ets_median_price_eur=round(p_eu_med, 2),
                        india_ccc_median_price_eur=round(p_ind_eur_med, 2),
                        gross_eu_cbam_cost_eur_per_t_product=round(gross_eu_cbam_eur, 2),
                        article9_recognized_ccts_credit_eur_per_t_product=round(art9_credit_eur, 2),
                        hypothetical_full_scope_indian_credit_eur_per_t_product=round(full_scope_credit_eur, 2),
                        intensity_recognition_haircut_pct=round(intensity_haircut_pct, 2),
                        net_unhedged_eu_cbam_tariff_eur_per_t_product=round(net_eu_cbam_eur, 2),
                        net_unhedged_eu_cbam_var95_eur_per_t_product=round(net_eu_cbam_v95_eur, 2),
                        unhedged_share_of_gross_eu_cbam_pct=round(unhedged_share_pct, 2),
                        net_unhedged_uk_cbam_tariff_gbp_per_t_product=round(net_uk_cbam_gbp, 2),
                        india_eu_export_volume_mt=round(eu_vol_t, 3),
                        total_eu_cbam_sector_liability_million_eur=round(tot_eu_liab_meur, 2),
                        total_uk_cbam_sector_liability_million_gbp=round(tot_uk_liab_mgbp, 2),
                    )
                )

        df_cbam_panel = pd.DataFrame([asdict(r) for r in records])

        # Sector 10-year cumulative summary
        df_cbam_summary = (
            df_cbam_panel.groupby(["sector_code", "sector_name"], as_index=False)
            .agg(
                tariff_2026_eur_per_t=("net_unhedged_eu_cbam_tariff_eur_per_t_product", "first"),
                tariff_2030_eur_per_t=(
                    "net_unhedged_eu_cbam_tariff_eur_per_t_product",
                    lambda s: s.iloc[4],
                ),
                tariff_2035_eur_per_t=("net_unhedged_eu_cbam_tariff_eur_per_t_product", "last"),
                var95_tariff_2035_eur_per_t=("net_unhedged_eu_cbam_var95_eur_per_t_product", "last"),
                mean_intensity_haircut_pct=("intensity_recognition_haircut_pct", "mean"),
                mean_unhedged_cbam_share_pct=("unhedged_share_of_gross_eu_cbam_pct", "mean"),
                cumulative_eu_cbam_liability_2026_2035_meur=(
                    "total_eu_cbam_sector_liability_million_eur",
                    "sum",
                ),
                cumulative_uk_cbam_liability_2026_2035_mgbp=(
                    "total_uk_cbam_sector_liability_million_gbp",
                    "sum",
                ),
            )
            .round(2)
        )

        df_cbam_panel.to_csv(
            self.processed_dir / "india_cbam_4sectors_tariff_panel_2026_2035.csv",
            index=False,
        )
        df_cbam_panel.to_parquet(
            self.processed_dir / "india_cbam_4sectors_tariff_panel_2026_2035.parquet",
            index=False,
        )
        df_cbam_summary.to_csv(
            self.processed_dir / "india_cbam_4sectors_tariff_summary.csv",
            index=False,
        )

        self.logger.info(
            f"Completed EU/UK-CBAM 4-Sector Tariff simulation (2026-2035): "
            f"Cumulative EU-CBAM Liability=EUR {df_cbam_summary['cumulative_eu_cbam_liability_2026_2035_meur'].sum():.2f}M, "
            f"Cumulative UK-CBAM Liability=GBP {df_cbam_summary['cumulative_uk_cbam_liability_2026_2035_mgbp'].sum():.2f}M."
        )
        return df_cbam_panel, df_cbam_summary

    # --------------------------------------------------------------------------
    # 2. UNFCCC Article 6.2 / 6.4 PACM Arbitrage & Sovereign CA Hurdle Engine
    # --------------------------------------------------------------------------
    def simulate_article6_arbitrage(
        self, df_mc_annual: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Evaluates bilateral ITMO transfer viability from Host Country (India) to
        Buyer Compliance Jurisdictions (EU_ETS, UK_ETS, CA_WCI, NZ_ETS, K_ETS)
        under UNFCCC Article 6.2 Corresponding Adjustments (CA) and Article 6.4
        Paris Agreement Crediting Mechanism (PACM) levies:
          P_{buyer,t} >= ( MAC_{host,t} + Omega_{NDC,t}^{CA} ) / ( 1 - lambda_{SOP} - lambda_{OMGE} ) + c_{fee}
        """
        if df_mc_annual is None:
            df_mc_annual = self._ensure_monte_carlo_summary()

        art6_cfg = self.config["article_6_mechanisms"]
        sop_levy = float(art6_cfg["article_6_4_pacm"]["share_of_proceeds_adaptation_levy_pct"])     # 0.05
        omge_canc = float(art6_cfg["article_6_4_pacm"]["overall_mitigation_omge_cancellation_pct"]) # 0.02
        issuance_fee = float(art6_cfg["article_6_4_pacm"]["supervisory_body_issuance_fee_usd_per_t"]) # 0.30
        omega_base = float(art6_cfg["article_6_2_itmo"]["india_ndc_sovereign_opportunity_cost_usd_per_t"])

        # Net atmospheric delivery factor after 5% SOP + 2% OMGE = 0.93
        delivery_factor = 1.0 - sop_levy - omge_canc
        gross_up_mult = 1.0 / delivery_factor

        buyer_hubs = ["EU_ETS", "UK_ETS", "CA_WCI", "NZ_ETS", "K_ETS"]
        rows: List[Article6ArbitrageRecord] = []

        for idx, yr in enumerate(range(2026, 2036)):
            ind_row = df_mc_annual[
                (df_mc_annual["jurisdiction"] == "INDIA_CCTS") & (df_mc_annual["year"] == yr)
            ].iloc[0]
            p_ind_usd = float(ind_row["median_price_usd"])

            # Host marginal abatement cost for export-grade projects rises as low-hanging
            # abatement is absorbed domestically by CCTS Phase 1 & Phase 2
            mac_host_usd = max(p_ind_usd, 14.00 * (1.045 ** idx))
            # Sovereign NDC opportunity cost escalates as India approaches its 2030 45% NDC milestone
            omega_ndc_t = omega_base * (1.055 ** idx)

            min_viable_offer_usd = (mac_host_usd + omega_ndc_t) * gross_up_mult + issuance_fee

            for buyer in buyer_hubs:
                b_row = df_mc_annual[
                    (df_mc_annual["jurisdiction"] == buyer) & (df_mc_annual["year"] == yr)
                ].iloc[0]
                p_buyer_med_usd = float(b_row["median_price_usd"])
                p_buyer_v95_usd = float(b_row["var95_price_usd"])

                net_spread_usd = p_buyer_med_usd - min_viable_offer_usd
                # Maximum Sovereign Opportunity Cost Omega_NDC^* before arbitrage closes:
                # (P_buyer - c_fee) * (1 - lambda_SOP - lambda_OMGE) - MAC_host
                max_omega_star = max(
                    0.0, (p_buyer_med_usd - issuance_fee) * delivery_factor - mac_host_usd
                )

                rows.append(
                    Article6ArbitrageRecord(
                        year=yr,
                        buyer_jurisdiction=buyer,
                        buyer_median_price_usd=round(p_buyer_med_usd, 2),
                        buyer_var95_price_usd=round(p_buyer_v95_usd, 2),
                        india_domestic_ccc_price_usd=round(p_ind_usd, 2),
                        host_marginal_abatement_cost_usd=round(mac_host_usd, 2),
                        configured_ndc_opportunity_cost_omega_usd=round(omega_ndc_t, 2),
                        pacm_gross_up_multiplier=round(gross_up_mult, 5),
                        min_viable_itmo_offer_price_usd=round(min_viable_offer_usd, 2),
                        net_bilateral_arbitrage_spread_usd=round(net_spread_usd, 2),
                        max_tolerable_sovereign_omega_ca_usd=round(max_omega_star, 2),
                        itmo_export_economically_viable=bool(net_spread_usd >= 0.0),
                    )
                )

        df_art6 = pd.DataFrame([asdict(r) for r in rows])
        df_art6.to_csv(
            self.processed_dir / "article6_itmo_arbitrage_matrix_2026_2035.csv",
            index=False,
        )
        df_art6.to_parquet(
            self.processed_dir / "article6_itmo_arbitrage_matrix_2026_2035.parquet",
            index=False,
        )
        self.logger.info(
            f"Completed Article 6.2/6.4 PACM Arbitrage evaluation ({len(df_art6)} bilateral pairs). "
            f"PACM gross-up multiplier={gross_up_mult:.5f}x (5% SOP + 2% OMGE)."
        )
        return df_art6


if __name__ == "__main__":
    engine = CrossBorderCBAMArticle6Engine()
    df_cbam_panel, df_cbam_sum = engine.simulate_cbam_4sectors_exposure()
    df_art6 = engine.simulate_article6_arbitrage()
    print("\n=== INDIA 4-SECTOR EU-CBAM & UK-CBAM TARIFF EXPOSURE SUMMARY (2026-2035) ===")
    print(df_cbam_sum.to_string(index=False))
    print("\n=== UNFCCC ARTICLE 6.2 / 6.4 PACM ARBITRAGE HURDLE SUMMARY (2026 vs. 2030 vs. 2035) ===")
    print(
        df_art6[df_art6["year"].isin([2026, 2030, 2035])][
            [
                "year",
                "buyer_jurisdiction",
                "buyer_median_price_usd",
                "india_domestic_ccc_price_usd",
                "min_viable_itmo_offer_price_usd",
                "net_bilateral_arbitrage_spread_usd",
                "max_tolerable_sovereign_omega_ca_usd",
                "itmo_export_economically_viable",
            ]
        ].to_string(index=False)
    )
