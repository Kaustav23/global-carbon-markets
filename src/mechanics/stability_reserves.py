"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 2 - Module 2: Market Stability & Cost Containment Reserve Control Operators
File: src/mechanics/stability_reserves.py
Lead Architect: Kaustav Goswami

Implements the piecewise statutory control operators M(TNAC_t, P_t) across all hubs:
1. EU-ETS Market Stability Reserve (MSR):
   - 24% TNAC intake above 1,096 Mt; linear buffer intake (TNAC - 833 Mt) for
     833 Mt < TNAC <= 1,096 Mt; 100 Mt release if TNAC < 400 Mt; permanent
     invalidation of MSR holdings above previous year's auction volume.
2. California WCI Auction Floor, Two-Tier APCR & Hard Price Ceiling:
   - Annual floor/trigger escalation at (5% real + 2.4% CPI); Tier 1 ($59.90) &
     Tier 2 ($76.85) reserve injections (15.5 Mt each); hard ceiling ($94.80).
3. UK-ETS Auction Reserve Price (Â£22/t) & Cost Containment Mechanism (CCM):
   - Triggered when monthly UKA price exceeds 2.5x preceding 2-year rolling
     average for 3 consecutive months.
4. NZ-ETS Auction Floor (NZ$64) & Two-Tier Cost Containment Reserve (CCR):
   - Tier 1 (NZ$184, 2.8 Mt) and Tier 2 (NZ$230, 4.9 Mt) volume releases.
5. Korea K-ETS Phase 4 Market Stability Reserve (K-MSR):
   - 25 Mt reserve pool operating on +/-15% circulation corridor and
     KRW 8,500 - 28,000 intervention bounds.
6. India CCTS Section 26(1A) Statutory Penalty Cliff & CERC Exchange Bands:
   - Energy Conservation (Amendment) Act 2022 Section 26(1A) penalty at 2x
     average clearing price (without physical abatement retirement), evaluated
     against CERC floor (Rs 650/t) and forbearance ceiling (Rs 1,850/t).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import json


@dataclass
class MSRControlState:
    """Annual state transition record for the EU-ETS Market Stability Reserve."""
    year: int
    tnac_prior_mt: float
    prior_year_auction_mt: float
    msr_intake_or_release_mt: float      # Negative = absorbed from auction, Positive = released
    msr_holdings_pre_invalidation_mt: float
    msr_invalidated_mt: float
    msr_holdings_post_invalidation_mt: float
    regime_state: str


@dataclass
class PriceContainmentState:
    """State evaluation for price-collar and reserve-injection mechanisms."""
    jurisdiction: str
    year: int
    currency: str
    unconstrained_shadow_price: float
    constrained_clearing_price: float
    active_floor_price: float
    tier1_trigger_price: float
    tier2_trigger_price: float
    hard_ceiling_or_penalty_price: float
    reserve_supply_injected_mt: float
    regime_state: str
    physical_abatement_backed: bool


class MarketStabilityContainmentEngine:
    """
    Piecewise statutory supply-adjustment and price-containment operator engine
    across all 7 compliance jurisdictions (2026-2035).
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
        logger = logging.getLogger("StabilityContainmentEngine")
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

    # --------------------------------------------------------------------------
    # 1. EU-ETS Market Stability Reserve (MSR) Operator
    # --------------------------------------------------------------------------
    def evaluate_eu_msr_step(
        self,
        year: int,
        tnac_prior_mt: float,
        prior_year_auction_mt: float,
        current_msr_holdings_mt: float,
    ) -> MSRControlState:
        """
        Piecewise EU-ETS MSR intake, release, and invalidation rule:
        - If TNAC > 1096 Mt: intake = 0.24 * TNAC ( withdrawn from auctions )
        - If 833 Mt < TNAC <= 1096 Mt: buffer intake = TNAC - 833 Mt
        - If 400 Mt <= TNAC <= 833 Mt: neutral zone (intake = 0)
        - If TNAC < 400 Mt: release = min(100 Mt, current_msr_holdings_mt)
        - Invalidation: post-intake MSR holdings above `prior_year_auction_mt`
          are permanently cancelled.
        """
        msr_cfg = self.config["jurisdictions"]["EU_ETS"]["stability_mechanism"]
        upper_thresh = float(msr_cfg["tnac_upper_threshold_mt"])     # 833.0 Mt
        buffer_upper = float(msr_cfg["tnac_buffer_upper_mt"])       # 1096.0 Mt
        intake_rate = float(msr_cfg["intake_rate_standard"])        # 0.24
        lower_thresh = float(msr_cfg["tnac_lower_threshold_mt"])    # 400.0 Mt
        release_vol = float(msr_cfg["release_volume_mt"])           # 100.0 Mt

        if tnac_prior_mt > buffer_upper:
            intake = round(intake_rate * tnac_prior_mt, 3)
            supply_delta = -intake
            msr_pre = current_msr_holdings_mt + intake
            state = "MSR_24PCT_INTAKE"
        elif tnac_prior_mt > upper_thresh:
            intake = round(tnac_prior_mt - upper_thresh, 3)
            supply_delta = -intake
            msr_pre = current_msr_holdings_mt + intake
            state = "MSR_BUFFER_SMOOTHED_INTAKE"
        elif tnac_prior_mt < lower_thresh:
            actual_release = round(min(release_vol, current_msr_holdings_mt), 3)
            supply_delta = actual_release
            msr_pre = current_msr_holdings_mt - actual_release
            state = "MSR_100MT_SCARCITY_RELEASE"
        else:
            supply_delta = 0.0
            msr_pre = current_msr_holdings_mt
            state = "MSR_NEUTRAL_CORRIDOR"

        if msr_cfg.get("invalidation_above_prev_year_auction", True):
            invalidated = round(max(0.0, msr_pre - prior_year_auction_mt), 3)
            msr_post = round(msr_pre - invalidated, 3)
        else:
            invalidated = 0.0
            msr_post = round(msr_pre, 3)

        return MSRControlState(
            year=year,
            tnac_prior_mt=round(tnac_prior_mt, 2),
            prior_year_auction_mt=round(prior_year_auction_mt, 2),
            msr_intake_or_release_mt=supply_delta,
            msr_holdings_pre_invalidation_mt=round(msr_pre, 2),
            msr_invalidated_mt=invalidated,
            msr_holdings_post_invalidation_mt=msr_post,
            regime_state=state,
        )

    def simulate_eu_msr_trajectory_2026_2035(self) -> pd.DataFrame:
        """
        Simulates EU-ETS TNAC and MSR intake/invalidation dynamics from 2026 to 2035
        as the 4.3% -> 4.4% Linear Reduction Factor tightens market circulation.
        """
        eu_cfg = self.config["jurisdictions"]["EU_ETS"]
        tnac = float(eu_cfg["initial_tnac_mt"])  # 1112.0 Mt in 2026
        cap_2026 = float(eu_cfg["base_cap_2026_mt"])
        msr_holdings = 400.0
        prior_auction = round(0.57 * cap_2026, 2)

        records: List[MSRControlState] = []
        for idx, yr in enumerate(range(2026, 2036)):
            lrf = 0.043 if yr <= 2027 else 0.044
            cap_t = cap_2026 * max(0.0, 1.0 - lrf * idx)
            step_state = self.evaluate_eu_msr_step(
                year=yr,
                tnac_prior_mt=tnac,
                prior_year_auction_mt=prior_auction,
                current_msr_holdings_mt=msr_holdings,
            )
            records.append(step_state)

            # Update MSR holdings and next year's TNAC
            msr_holdings = step_state.msr_holdings_post_invalidation_mt
            # Net circulation change from MSR absorption + structural cap tightening
            tnac = max(260.0, tnac + step_state.msr_intake_or_release_mt * 0.48 - 32.0)
            prior_auction = max(200.0, round(0.57 * cap_t + step_state.msr_intake_or_release_mt, 2))

        df_msr = pd.DataFrame([asdict(r) for r in records])
        df_msr.to_csv(self.processed_dir / "eu_ets_msr_trajectory_2026_2035.csv", index=False)
        self.logger.info(
            f"Completed EU-ETS MSR trajectory (2026-2035): "
            f"Cumulative absorbed={-df_msr[df_msr['msr_intake_or_release_mt'] < 0]['msr_intake_or_release_mt'].sum():.1f} Mt, "
            f"Cumulative invalidated={df_msr['msr_invalidated_mt'].sum():.1f} Mt."
        )
        return df_msr

    # --------------------------------------------------------------------------
    # 2. California WCI Auction Floor, Two-Tier APCR & Hard Ceiling Operator
    # --------------------------------------------------------------------------
    def evaluate_california_wci_apcr(
        self,
        year: int,
        unconstrained_price_usd: float,
        t1_remaining_mt: float = 15.5,
        t2_remaining_mt: float = 15.5,
    ) -> PriceContainmentState:
        wci_cfg = self.config["jurisdictions"]["CA_WCI"]["stability_mechanism"]
        esc_rate = float(wci_cfg["annual_real_escalation_rate"]) + float(
            wci_cfg["assumed_cpi_inflation"]
        )
        t_steps = max(0, year - 2026)
        esc_factor = (1.0 + esc_rate) ** t_steps

        floor_p = round(float(wci_cfg["auction_reserve_floor_2026_local"]) * esc_factor, 2)
        t1_p = round(float(wci_cfg["apcr_tier_1_price_2026_local"]) * esc_factor, 2)
        t2_p = round(float(wci_cfg["apcr_tier_2_price_2026_local"]) * esc_factor, 2)
        ceil_p = round(float(wci_cfg["hard_price_ceiling_2026_local"]) * esc_factor, 2)

        injected_mt = 0.0
        if unconstrained_price_usd <= floor_p:
            clearing_p = floor_p
            state = "WCI_AUCTION_FLOOR_BINDING"
        elif unconstrained_price_usd >= ceil_p:
            clearing_p = ceil_p
            injected_mt = t1_remaining_mt + t2_remaining_mt
            state = "WCI_HARD_PRICE_CEILING_BINDING"
        elif unconstrained_price_usd >= t2_p:
            clearing_p = min(unconstrained_price_usd * 0.92, ceil_p)
            injected_mt = t1_remaining_mt + min(t2_remaining_mt, 10.0)
            state = "WCI_APCR_TIER2_TRIGGERED"
        elif unconstrained_price_usd >= t1_p:
            clearing_p = min(unconstrained_price_usd * 0.95, t2_p)
            injected_mt = min(t1_remaining_mt, 8.5)
            state = "WCI_APCR_TIER1_TRIGGERED"
        else:
            clearing_p = unconstrained_price_usd
            state = "WCI_INTERIOR_MARKET_CLEARING"

        return PriceContainmentState(
            jurisdiction="CA_WCI",
            year=year,
            currency="USD",
            unconstrained_shadow_price=round(unconstrained_price_usd, 2),
            constrained_clearing_price=round(clearing_p, 2),
            active_floor_price=floor_p,
            tier1_trigger_price=t1_p,
            tier2_trigger_price=t2_p,
            hard_ceiling_or_penalty_price=ceil_p,
            reserve_supply_injected_mt=round(injected_mt, 2),
            regime_state=state,
            physical_abatement_backed=True,
        )

    # --------------------------------------------------------------------------
    # 3. UK-ETS Auction Reserve Price & Cost Containment Mechanism (CCM)
    # --------------------------------------------------------------------------
    def evaluate_uk_ets_ccm(
        self,
        year: int,
        unconstrained_price_gbp: float,
        rolling_2yr_avg_gbp: float,
        consecutive_months_above: int = 3,
    ) -> PriceContainmentState:
        uk_cfg = self.config["jurisdictions"]["UK_ETS"]["stability_mechanism"]
        floor_p = float(uk_cfg["auction_reserve_price_local"])
        mult = float(uk_cfg["ccm_trigger_multiplier_2yr_avg"])
        req_months = int(uk_cfg["ccm_consecutive_months"])
        max_rel = float(uk_cfg["ccm_max_release_volume_mt"])

        ccm_trigger_p = round(rolling_2yr_avg_gbp * mult, 2)
        injected_mt = 0.0

        if unconstrained_price_gbp <= floor_p:
            clearing_p = floor_p
            state = "UK_ARP_FLOOR_BINDING"
        elif unconstrained_price_gbp >= ccm_trigger_p and consecutive_months_above >= req_months:
            injected_mt = max_rel
            clearing_p = round(unconstrained_price_gbp * 0.88, 2)
            state = "UK_CCM_SUPPLY_INTERVENTION_TRIGGERED"
        else:
            clearing_p = unconstrained_price_gbp
            state = "UK_INTERIOR_MARKET_CLEARING"

        return PriceContainmentState(
            jurisdiction="UK_ETS",
            year=year,
            currency="GBP",
            unconstrained_shadow_price=round(unconstrained_price_gbp, 2),
            constrained_clearing_price=round(clearing_p, 2),
            active_floor_price=floor_p,
            tier1_trigger_price=ccm_trigger_p,
            tier2_trigger_price=ccm_trigger_p,
            hard_ceiling_or_penalty_price=round(ccm_trigger_p * 1.25, 2),
            reserve_supply_injected_mt=injected_mt,
            regime_state=state,
            physical_abatement_backed=True,
        )

    # --------------------------------------------------------------------------
    # 4. New Zealand ETS (NZ-ETS) Auction Floor & Two-Tier CCR
    # --------------------------------------------------------------------------
    def evaluate_nz_ets_ccr(
        self, year: int, unconstrained_price_nzd: float
    ) -> PriceContainmentState:
        nz_cfg = self.config["jurisdictions"]["NZ_ETS"]["stability_mechanism"]
        esc = (1.0 + float(nz_cfg["floor_annual_escalation_rate"])) ** max(0, year - 2026)

        floor_p = round(float(nz_cfg["auction_reserve_floor_2026_local"]) * esc, 2)
        t1_p = round(float(nz_cfg["ccr_tier_1_trigger_2026_local"]) * esc, 2)
        t2_p = round(float(nz_cfg["ccr_tier_2_trigger_2026_local"]) * esc, 2)
        t1_vol = float(nz_cfg["ccr_tier_1_volume_mt"])
        t2_vol = float(nz_cfg["ccr_tier_2_volume_mt"])

        injected_mt = 0.0
        if unconstrained_price_nzd <= floor_p:
            clearing_p = floor_p
            state = "NZ_AUCTION_FLOOR_BINDING"
        elif unconstrained_price_nzd >= t2_p:
            injected_mt = t1_vol + t2_vol
            clearing_p = round(unconstrained_price_nzd * 0.90, 2)
            state = "NZ_CCR_TIER2_RELEASE"
        elif unconstrained_price_nzd >= t1_p:
            injected_mt = t1_vol
            clearing_p = round(unconstrained_price_nzd * 0.94, 2)
            state = "NZ_CCR_TIER1_RELEASE"
        else:
            clearing_p = unconstrained_price_nzd
            state = "NZ_INTERIOR_MARKET_CLEARING"

        return PriceContainmentState(
            jurisdiction="NZ_ETS",
            year=year,
            currency="NZD",
            unconstrained_shadow_price=round(unconstrained_price_nzd, 2),
            constrained_clearing_price=round(clearing_p, 2),
            active_floor_price=floor_p,
            tier1_trigger_price=t1_p,
            tier2_trigger_price=t2_p,
            hard_ceiling_or_penalty_price=t2_p,
            reserve_supply_injected_mt=injected_mt,
            regime_state=state,
            physical_abatement_backed=True,
        )

    # --------------------------------------------------------------------------
    # 5. Korea K-ETS Phase 4 Market Stability Reserve (K-MSR)
    # --------------------------------------------------------------------------
    def evaluate_korea_kmsr(
        self,
        year: int,
        unconstrained_price_krw: float,
        circulation_ratio: float = 1.0,
    ) -> PriceContainmentState:
        k_cfg = self.config["jurisdictions"]["K_ETS"]["stability_mechanism"]
        floor_p = float(k_cfg["floor_intervention_price_local"])
        ceil_p = float(k_cfg["ceiling_intervention_price_local"])
        max_rel = float(k_cfg["max_annual_release_mt"])
        max_abs = float(k_cfg["max_annual_absorption_mt"])

        injected_mt = 0.0
        if unconstrained_price_krw <= floor_p or circulation_ratio > 1.15:
            injected_mt = -max_abs
            clearing_p = max(floor_p, round(unconstrained_price_krw * 1.08, 0))
            state = "K_MSR_SUPPLY_ABSORPTION"
        elif unconstrained_price_krw >= ceil_p or circulation_ratio < 0.85:
            injected_mt = max_rel
            clearing_p = min(ceil_p, round(unconstrained_price_krw * 0.91, 0))
            state = "K_MSR_RESERVE_RELEASE"
        else:
            clearing_p = round(unconstrained_price_krw, 0)
            state = "K_ETS_INTERIOR_CORRIDOR"

        return PriceContainmentState(
            jurisdiction="K_ETS",
            year=year,
            currency="KRW",
            unconstrained_shadow_price=round(unconstrained_price_krw, 0),
            constrained_clearing_price=clearing_p,
            active_floor_price=floor_p,
            tier1_trigger_price=round(ceil_p * 0.85, 0),
            tier2_trigger_price=ceil_p,
            hard_ceiling_or_penalty_price=ceil_p,
            reserve_supply_injected_mt=injected_mt,
            regime_state=state,
            physical_abatement_backed=True,
        )

    # --------------------------------------------------------------------------
    # 6. India CCTS Section 26(1A) Penalty Cliff & CERC Exchange Collar
    # --------------------------------------------------------------------------
    def evaluate_india_ccts_penalty_cliff(
        self,
        year: int,
        unconstrained_mac_inr: float,
        rolling_avg_clearing_price_inr: float,
    ) -> PriceContainmentState:
        """
        Evaluates India's Bureau of Energy Efficiency (BEE) statutory penalty cliff
        under Section 26(1A) of the Energy Conservation (Amendment) Act, 2022:
            P_t^penalty = 2.0 * P_bar_{t,CCC}
        coupled with Central Electricity Regulatory Commission (CERC) power exchange
        floor and forbearance bands [P_t^floor, P_t^forb].

        Crucial Institutional Microstructure Insight:
        Unlike WCI's APCR (which injects backed allowances from under the cap),
        Section 26(1A) environmental compensation is a financial penalty without
        retired physical abatement backing (`physical_abatement_backed = False`
        when breached), creating a soft-cap leakage cliff whenever firm MAC
        exceeds min(CERC_Forbearance_Ceiling, 2 * Rolling_Avg_CCC_Price).
        """
        ind_cfg = self.config["jurisdictions"]["INDIA_CCTS"]["stability_mechanism"]
        mult = float(ind_cfg["statutory_penalty_multiplier"])  # 2.0x
        esc = (1.0 + float(ind_cfg["band_annual_escalation_rate"])) ** max(0, year - 2026)

        cerc_floor = round(float(ind_cfg["cerc_floor_price_2026_local"]) * esc, 2)
        cerc_forb = round(float(ind_cfg["cerc_forbearance_ceiling_2026_local"]) * esc, 2)
        sec26_penalty_cliff = round(mult * rolling_avg_clearing_price_inr, 2)
        effective_upper_bound = min(cerc_forb, sec26_penalty_cliff)

        if unconstrained_mac_inr <= cerc_floor:
            clearing_p = cerc_floor
            state = "INDIA_CERC_EXCHANGE_FLOOR_BINDING"
            backed = True
        elif unconstrained_mac_inr >= sec26_penalty_cliff:
            clearing_p = effective_upper_bound
            state = "INDIA_SECTION_26_PENALTY_CLIFF_BREACHED"
            backed = False
        elif unconstrained_mac_inr >= cerc_forb:
            clearing_p = cerc_forb
            state = "INDIA_CERC_FORBEARANCE_CEILING_BINDING"
            backed = False
        else:
            clearing_p = unconstrained_mac_inr
            state = "INDIA_CCTS_INTERIOR_EXCHANGE_CLEARING"
            backed = True

        return PriceContainmentState(
            jurisdiction="INDIA_CCTS",
            year=year,
            currency="INR",
            unconstrained_shadow_price=round(unconstrained_mac_inr, 2),
            constrained_clearing_price=round(clearing_p, 2),
            active_floor_price=cerc_floor,
            tier1_trigger_price=cerc_forb,
            tier2_trigger_price=sec26_penalty_cliff,
            hard_ceiling_or_penalty_price=effective_upper_bound,
            reserve_supply_injected_mt=0.0,
            regime_state=state,
            physical_abatement_backed=backed,
        )

    def run_cross_hub_stress_test(self) -> pd.DataFrame:
        """
        Executes a standardized containment stress-test across CA_WCI, UK_ETS,
        NZ_ETS, K_ETS, and INDIA_CCTS under Baseline, Scarcity Spike (+85% MAC),
        and Macro Recession (-45% MAC) scenarios in 2026 and 2030.
        """
        scenarios = [
            # (year, scenario_label, wci_p, uk_p, nz_p, kor_p, ind_mac, ind_avg)
            (2026, "BASELINE_2026", 39.40, 52.80, 64.50, 12400.0, 980.0, 950.0),
            (2026, "RECESSION_FLOOR_TEST", 21.00, 19.50, 55.00, 7800.0, 580.0, 820.0),
            (2026, "SCARCITY_SPIKE_TEST", 82.50, 138.00, 195.00, 29500.0, 2150.0, 980.0),
            (2030, "HIGH_DEMAND_2030", 84.00, 145.00, 235.00, 28500.0, 2680.0, 1250.0),
        ]
        rows: List[Dict] = []
        for yr, label, wci_p, uk_p, nz_p, kor_p, ind_mac, ind_avg in scenarios:
            states = [
                self.evaluate_california_wci_apcr(yr, wci_p),
                self.evaluate_uk_ets_ccm(yr, uk_p, rolling_2yr_avg_gbp=51.0, consecutive_months_above=3),
                self.evaluate_nz_ets_ccr(yr, nz_p),
                self.evaluate_korea_kmsr(yr, kor_p, circulation_ratio=0.82 if "SPIKE" in label or "HIGH" in label else 1.0),
                self.evaluate_india_ccts_penalty_cliff(yr, ind_mac, ind_avg),
            ]
            for st in states:
                d = asdict(st)
                d["scenario"] = label
                rows.append(d)

        df_stress = pd.DataFrame(rows)
        df_stress.to_csv(
            self.processed_dir / "cross_hub_containment_stress_test.csv", index=False
        )
        self.logger.info(
            f"Completed Cross-Hub Containment Stress Test ({len(df_stress)} state evaluations saved)."
        )
        return df_stress


if __name__ == "__main__":
    engine = MarketStabilityContainmentEngine()
    df_eu_msr = engine.simulate_eu_msr_trajectory_2026_2035()
    df_stress = engine.run_cross_hub_stress_test()
    print("\n=== EU-ETS MARKET STABILITY RESERVE (MSR) TRAJECTORY (2026-2035) ===")
    print(df_eu_msr.to_string(index=False))
    print("\n=== CROSS-HUB CONTAINMENT & SECTION 26 PENALTY CLIFF STRESS TEST ===")
    print(
        df_stress[
            [
                "scenario",
                "jurisdiction",
                "year",
                "unconstrained_shadow_price",
                "constrained_clearing_price",
                "active_floor_price",
                "hard_ceiling_or_penalty_price",
                "reserve_supply_injected_mt",
                "regime_state",
                "physical_abatement_backed",
            ]
        ].to_string(index=False)
    )
