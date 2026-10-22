"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 3 - Module 1: Correlated OU Mean-Reverting Jump-Diffusion Monte Carlo Engine
File: src/econometrics/price_discovery.py
Lead Architect: Kaustav Goswami

Mathematical Core:
1. Continuous-Time Stochastic Differential Equation (7 Compliance Hubs):
     dX_{k,t} = kappa_k * [ theta_k(D_{k,t}^net, S_{k,t}^eff) - X_{k,t} ] dt
                + sigma_k * X_{k,t}^{beta_k} * dW_{k,t}
                + J_{k,t} * dN_{k,t}
2. Cross-Hub Correlation Structure:
     dW_t = L @ dZ_t, where Sigma_corr = L @ L^T via Cholesky factorization
     across [EU_ETS, UK_ETS, CA_WCI, NZ_ETS, K_ETS, CHINA_ETS, INDIA_CCTS].
3. Piecewise Statutory Containment Projection Operator M(TNAC_t, P_t):
     At each monthly step dt = 1/12 (120 steps over 2026-2035 across 10,000 paths),
     unconstrained shadow marginal abatement cost X_{k,t} is projected onto each
     jurisdiction's statutory floor, tiered containment reserve (APCR/CCM/CCR/K-MSR),
     and hard ceiling or India BEE Section 26(1A) 2x average clearing price cliff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import json


@dataclass
class AnnualHubPriceMetrics:
    """Annual distribution summary for a single compliance hub across 10,000 paths."""
    jurisdiction: str
    year: int
    currency: str
    fx_to_usd: float
    fx_to_eur: float
    fundamental_theta_local: float
    active_floor_local: float
    active_ceiling_or_penalty_local: float
    mean_price_local: float
    median_price_local: float
    p05_price_local: float
    p25_price_local: float
    p75_price_local: float
    var95_price_local: float
    var99_price_local: float
    median_price_usd: float
    var95_price_usd: float
    var99_price_usd: float
    median_price_eur: float
    var95_price_eur: float
    var99_price_eur: float
    floor_binding_prob_pct: float
    containment_or_penalty_trigger_prob_pct: float
    eu_ets_basis_spread_median_eur: float


class CorrelatedPriceDiscoveryEngine:
    """
    Vectorized 10,000-path Correlated Ornstein-Uhlenbeck (OU) Jump-Diffusion
    Monte Carlo Simulator across all 7 global carbon compliance hubs (2026-2035).
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        n_paths: int = 10000,
        steps_per_year: int = 12,
        random_seed: int = 42,
    ) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.config_path = self.project_root / "config" / "jurisdictions.json"
        self.processed_dir = self.project_root / "data" / "processed"
        self.log_file = self.project_root / "logs" / "ingestion_and_sim_audit.log"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
        self.logger = self._get_logger()

        self.n_paths = int(n_paths)
        self.steps_per_year = int(steps_per_year)
        self.dt = 1.0 / float(self.steps_per_year)
        self.years = list(range(2026, 2036))
        self.n_years = len(self.years)
        self.n_steps = self.n_years * self.steps_per_year
        self.random_seed = int(random_seed)

        self.hub_order: List[str] = self.config["correlation_matrix"]["hub_order"]
        self.n_hubs = len(self.hub_order)
        self.corr_matrix = np.array(
            self.config["correlation_matrix"]["matrix"], dtype=float
        )
        self.cholesky_L = self._compute_cholesky_factor(self.corr_matrix)

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger("CorrelatedPriceDiscoveryEngine")
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

    @staticmethod
    def _compute_cholesky_factor(corr_mat: np.ndarray) -> np.ndarray:
        """
        Computes lower-triangular Cholesky factor L such that L @ L.T == corr_mat.
        Asserts symmetry and positive-definiteness.
        """
        if not np.allclose(corr_mat, corr_mat.T, atol=1e-8):
            raise ValueError("Cross-hub correlation matrix must be symmetric.")
        eigvals = np.linalg.eigvalsh(corr_mat)
        if np.any(eigvals <= 0.0):
            raise ValueError(f"Correlation matrix is not positive-definite: eigvals={eigvals}")
        return cholesky(corr_mat, lower=True)

    def _get_fx_rates(self, currency: str) -> Tuple[float, float]:
        """Returns (fx_to_usd, fx_to_eur) for a given local currency code."""
        fx_cfg = self.config["fx_spot_2026_benchmarks"]
        eur_usd = float(fx_cfg["EUR_USD"])  # 1.09 USD per EUR
        mapping_usd = {
            "EUR": eur_usd,
            "GBP": float(fx_cfg["GBP_USD"]),
            "USD": 1.0,
            "NZD": float(fx_cfg["NZD_USD"]),
            "KRW": float(fx_cfg["KRW_USD"]),
            "CNY": float(fx_cfg["CNY_USD"]),
            "INR": float(fx_cfg["INR_USD"]),
        }
        fx_usd = mapping_usd[currency]
        fx_eur = 1.0 if currency == "EUR" else fx_usd / eur_usd
        return fx_usd, fx_eur

    def _compute_statutory_bounds_for_step(
        self,
        hub: str,
        t_years: float,
        rolling_avg_price: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns vectorized statutory containment bounds across all Monte Carlo paths:
            (floor_vec, soft_tier1_vec, soft_tier2_vec, hard_ceiling_vec)
        coupled with each jurisdiction's Layer 2 containment rule.
        """
        j_cfg = self.config["jurisdictions"][hub]
        stab = j_cfg["stability_mechanism"]
        yr_int = int(np.floor(t_years))

        if hub == "EU_ETS":
            # EU-ETS has no hard floor, but MSR absorbs surplus and Article 29a
            # dampens extreme spikes above 2.4x rolling 2-year average
            floor_v = np.full(self.n_paths, 35.0)
            art29a_mult = float(stab.get("article_29a_price_spike_multiplier", 2.4))
            t1_v = art29a_mult * rolling_avg_price
            t2_v = 1.15 * t1_v
            ceil_v = 1.35 * t1_v
            return floor_v, t1_v, t2_v, ceil_v

        if hub == "UK_ETS":
            floor_p = float(stab["auction_reserve_price_local"]) * (1.02 ** yr_int)
            ccm_mult = float(stab["ccm_trigger_multiplier_2yr_avg"])
            t1_v = ccm_mult * rolling_avg_price
            t2_v = 1.12 * t1_v
            ceil_v = 1.25 * t1_v
            return np.full(self.n_paths, floor_p), t1_v, t2_v, ceil_v

        if hub == "CA_WCI":
            esc_rate = float(stab["annual_real_escalation_rate"]) + float(
                stab["assumed_cpi_inflation"]
            )
            esc = (1.0 + esc_rate) ** yr_int
            floor_p = float(stab["auction_reserve_floor_2026_local"]) * esc
            t1_p = float(stab["apcr_tier_1_price_2026_local"]) * esc
            t2_p = float(stab["apcr_tier_2_price_2026_local"]) * esc
            ceil_p = float(stab["hard_price_ceiling_2026_local"]) * esc
            return (
                np.full(self.n_paths, floor_p),
                np.full(self.n_paths, t1_p),
                np.full(self.n_paths, t2_p),
                np.full(self.n_paths, ceil_p),
            )

        if hub == "NZ_ETS":
            esc = (1.0 + float(stab["floor_annual_escalation_rate"])) ** yr_int
            floor_p = float(stab["auction_reserve_floor_2026_local"]) * esc
            t1_p = float(stab["ccr_tier_1_trigger_2026_local"]) * esc
            t2_p = float(stab["ccr_tier_2_trigger_2026_local"]) * esc
            ceil_p = 1.15 * t2_p
            return (
                np.full(self.n_paths, floor_p),
                np.full(self.n_paths, t1_p),
                np.full(self.n_paths, t2_p),
                np.full(self.n_paths, ceil_p),
            )

        if hub == "K_ETS":
            esc = (1.0 + 0.035) ** yr_int
            floor_p = float(stab["floor_intervention_price_local"]) * (1.015 ** yr_int)
            ceil_p = float(stab["ceiling_intervention_price_local"]) * esc
            t1_p = 0.85 * ceil_p
            t2_p = 0.95 * ceil_p
            return (
                np.full(self.n_paths, floor_p),
                np.full(self.n_paths, t1_p),
                np.full(self.n_paths, t2_p),
                np.full(self.n_paths, ceil_p),
            )

        if hub == "CHINA_ETS":
            esc = (1.0 + 0.04) ** yr_int
            floor_p = float(stab["soft_floor_local"]) * (1.02 ** yr_int)
            ceil_p = float(stab["soft_ceiling_local"]) * esc
            t1_p = 0.85 * ceil_p
            t2_p = 0.95 * ceil_p
            return (
                np.full(self.n_paths, floor_p),
                np.full(self.n_paths, t1_p),
                np.full(self.n_paths, t2_p),
                np.full(self.n_paths, ceil_p),
            )

        if hub == "INDIA_CCTS":
            # Statutory BEE Section 26(1A) Penalty Cliff (2.0x rolling average clearing price)
            # intersected with CERC exchange floor and forbearance ceiling
            esc = (1.0 + float(stab["band_annual_escalation_rate"])) ** yr_int
            cerc_floor = float(stab["cerc_floor_price_2026_local"]) * esc
            cerc_forb = float(stab["cerc_forbearance_ceiling_2026_local"]) * esc
            mult = float(stab["statutory_penalty_multiplier"])  # 2.0x
            sec26_cliff = mult * rolling_avg_price
            effective_ceil = np.minimum(cerc_forb, sec26_cliff)
            t1_v = 0.90 * effective_ceil
            return (
                np.full(self.n_paths, cerc_floor),
                t1_v,
                effective_ceil,
                effective_ceil,
            )

        raise KeyError(f"Unrecognized compliance hub: {hub}")

    @staticmethod
    def _project_onto_containment_manifold(
        x_shadow: np.ndarray,
        floor_v: np.ndarray,
        t1_v: np.ndarray,
        t2_v: np.ndarray,
        ceil_v: np.ndarray,
    ) -> np.ndarray:
        """
        Piecewise statutory containment projection operator M(TNAC_t, P_t):
        - Dampens excursion above Tier 1 / Tier 2 reserve release thresholds
          (reflecting APCR/CCM/CCR/K-MSR supply injection)
        - Strictly enforces [floor_v, ceil_v]:
            P_{k,t} = min( P_{k,t}^{ceiling/penalty}, max( P_{k,t}^{floor}, X_{k,t}^{dampened} ) )
        """
        p_adj = x_shadow.copy()
        above_t1 = p_adj > t1_v
        p_adj = np.where(above_t1, t1_v + 0.55 * (p_adj - t1_v), p_adj)

        above_t2 = p_adj > t2_v
        p_adj = np.where(above_t2, t2_v + 0.30 * (p_adj - t2_v), p_adj)

        p_constrained = np.minimum(ceil_v, np.maximum(floor_v, p_adj))
        return p_constrained

    # --------------------------------------------------------------------------
    # Main Vectorized 10,000-Path Monte Carlo Simulation
    # --------------------------------------------------------------------------
    def run_monte_carlo_simulation(
        self,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
        """
        Simulates 10,000 correlated monthly price trajectories across all 7 hubs
        from 2026 to 2035 (120 monthly steps).

        Returns:
        - `paths_dict`: Dict mapping hub -> ndarray of shape (n_paths, n_steps + 1)
        - `df_annual`: Annual distribution summary across all 7 hubs (70 rows)
        - `df_spreads`: Cross-hub basis spread summary in EUR/tCO2e and USD/tCO2e
        """
        rng = np.random.default_rng(self.random_seed)

        # Pre-allocate constrained price array: (n_hubs, n_paths, n_steps + 1)
        P_paths = np.zeros((self.n_hubs, self.n_paths, self.n_steps + 1), dtype=float)
        X_shadow = np.zeros((self.n_hubs, self.n_paths, self.n_steps + 1), dtype=float)
        floor_hits = np.zeros((self.n_hubs, self.n_paths, self.n_steps + 1), dtype=bool)
        ceil_hits = np.zeros((self.n_hubs, self.n_paths, self.n_steps + 1), dtype=bool)

        # Initialize t = 0 (Start of 2026)
        kappas = np.zeros(self.n_hubs)
        theta_2026 = np.zeros(self.n_hubs)
        theta_drifts = np.zeros(self.n_hubs)
        sigmas = np.zeros(self.n_hubs)
        betas = np.zeros(self.n_hubs)
        lambdas = np.zeros(self.n_hubs)
        jump_means = np.zeros(self.n_hubs)
        jump_stds = np.zeros(self.n_hubs)

        for k, hub in enumerate(self.hub_order):
            j_cfg = self.config["jurisdictions"][hub]
            sde = j_cfg["sde_parameters"]
            p0 = float(j_cfg["initial_spot_price_2026_local"])
            P_paths[k, :, 0] = p0
            X_shadow[k, :, 0] = p0

            kappas[k] = float(sde["kappa_mean_reversion"])
            theta_2026[k] = float(sde["theta_long_run_2026_local"])
            theta_drifts[k] = float(sde["theta_annual_drift_pct"])
            sigmas[k] = float(sde["sigma_volatility"])
            betas[k] = float(sde["cev_elasticity_beta"])
            lambdas[k] = float(sde["jump_intensity_lambda"])
            jump_means[k] = float(sde["jump_mean_pct"])
            jump_stds[k] = float(sde["jump_std_pct"])

        # Step forward across 120 monthly intervals (2026.0 to 2035.917)
        sqrt_dt = np.sqrt(self.dt)
        for step in range(1, self.n_steps + 1):
            t_years = (step - 1) * self.dt
            # Draw independent standard normals Z ~ N(0, I_7) of shape (n_hubs, n_paths)
            Z_indep = rng.standard_normal((self.n_hubs, self.n_paths))
            # Correlate via Cholesky factor L: dW = (L @ Z) * sqrt(dt)
            dW_corr = (self.cholesky_L @ Z_indep) * sqrt_dt

            for k, hub in enumerate(self.hub_order):
                x_prev = X_shadow[k, :, step - 1]
                p_prev = P_paths[k, :, step - 1]

                # Fundamental long-run equilibrium mean theta_k(t) driven by net scarcity
                # For China (post-2028 dual-control pivot) and India (post-2029 Phase 2),
                # accelerate fundamental drift as tightening intensifies
                drift_mult = 1.0
                if hub == "CHINA_ETS" and (2026 + t_years) >= 2028:
                    drift_mult = 1.25
                elif hub == "INDIA_CCTS" and (2026 + t_years) >= 2029:
                    drift_mult = 1.20

                theta_t = theta_2026[k] * np.exp(theta_drifts[k] * drift_mult * t_years)

                # Mean-reverting OU drift + CEV diffusion
                drift_term = kappas[k] * (theta_t - x_prev) * self.dt
                diff_term = sigmas[k] * (np.maximum(x_prev, 1e-4) ** betas[k]) * dW_corr[k]

                # Compound Poisson regulatory/energy-shock jumps J_{k,t} dN_{k,t}
                dN = rng.poisson(lam=lambdas[k] * self.dt, size=self.n_paths)
                jump_sizes = rng.normal(
                    loc=jump_means[k], scale=jump_stds[k], size=self.n_paths
                )
                jump_term = x_prev * jump_sizes * dN

                x_next = np.maximum(0.05 * theta_2026[k], x_prev + drift_term + diff_term + jump_term)
                X_shadow[k, :, step] = x_next

                # Rolling 12-month (or available history) average clearing price
                start_win = max(0, step - self.steps_per_year)
                rolling_avg = np.mean(P_paths[k, :, start_win:step], axis=1)

                floor_v, t1_v, t2_v, ceil_v = self._compute_statutory_bounds_for_step(
                    hub=hub, t_years=t_years, rolling_avg_price=rolling_avg
                )
                p_next = self._project_onto_containment_manifold(
                    x_shadow=x_next,
                    floor_v=floor_v,
                    t1_v=t1_v,
                    t2_v=t2_v,
                    ceil_v=ceil_v,
                )
                P_paths[k, :, step] = p_next
                floor_hits[k, :, step] = p_next <= (floor_v + 1e-6)
                ceil_hits[k, :, step] = x_next >= t1_v

        # Build annual summary table across 2026-2035
        annual_rows: List[AnnualHubPriceMetrics] = []
        eu_idx = self.hub_order.index("EU_ETS")

        for yr_offset, yr in enumerate(self.years):
            # Slice monthly steps belonging to year `yr` (steps 1..12 for 2026, etc.)
            s_start = yr_offset * self.steps_per_year + 1
            s_end = (yr_offset + 1) * self.steps_per_year + 1

            # EU annual path-averages in EUR for basis spread computation
            eu_yr_paths_eur = np.mean(P_paths[eu_idx, :, s_start:s_end], axis=1)

            for k, hub in enumerate(self.hub_order):
                j_cfg = self.config["jurisdictions"][hub]
                curr = j_cfg["currency"]
                fx_usd, fx_eur = self._get_fx_rates(curr)

                hub_yr_paths_local = np.mean(P_paths[k, :, s_start:s_end], axis=1)
                hub_yr_paths_usd = hub_yr_paths_local * fx_usd
                hub_yr_paths_eur = hub_yr_paths_local * fx_eur

                # Statutory reference bounds at mid-year
                rolling_ref = np.mean(P_paths[k, :, max(0, s_start - 12):s_end], axis=1)
                fl_v, t1_v, _, cl_v = self._compute_statutory_bounds_for_step(
                    hub=hub, t_years=float(yr_offset), rolling_avg_price=rolling_ref
                )

                drift_mult = 1.0
                if hub == "CHINA_ETS" and yr >= 2028:
                    drift_mult = 1.25
                elif hub == "INDIA_CCTS" and yr >= 2029:
                    drift_mult = 1.20
                theta_yr = theta_2026[k] * np.exp(theta_drifts[k] * drift_mult * yr_offset)

                fl_prob = float(np.mean(floor_hits[k, :, s_start:s_end]) * 100.0)
                cl_prob = float(np.mean(ceil_hits[k, :, s_start:s_end]) * 100.0)

                spread_vs_eu_eur = float(np.median(eu_yr_paths_eur - hub_yr_paths_eur))

                annual_rows.append(
                    AnnualHubPriceMetrics(
                        jurisdiction=hub,
                        year=yr,
                        currency=curr,
                        fx_to_usd=fx_usd,
                        fx_to_eur=round(fx_eur, 6),
                        fundamental_theta_local=round(float(theta_yr), 2),
                        active_floor_local=round(float(np.median(fl_v)), 2),
                        active_ceiling_or_penalty_local=round(float(np.median(cl_v)), 2),
                        mean_price_local=round(float(np.mean(hub_yr_paths_local)), 2),
                        median_price_local=round(float(np.median(hub_yr_paths_local)), 2),
                        p05_price_local=round(float(np.percentile(hub_yr_paths_local, 5)), 2),
                        p25_price_local=round(float(np.percentile(hub_yr_paths_local, 25)), 2),
                        p75_price_local=round(float(np.percentile(hub_yr_paths_local, 75)), 2),
                        var95_price_local=round(float(np.percentile(hub_yr_paths_local, 95)), 2),
                        var99_price_local=round(float(np.percentile(hub_yr_paths_local, 99)), 2),
                        median_price_usd=round(float(np.median(hub_yr_paths_usd)), 2),
                        var95_price_usd=round(float(np.percentile(hub_yr_paths_usd, 95)), 2),
                        var99_price_usd=round(float(np.percentile(hub_yr_paths_usd, 99)), 2),
                        median_price_eur=round(float(np.median(hub_yr_paths_eur)), 2),
                        var95_price_eur=round(float(np.percentile(hub_yr_paths_eur, 95)), 2),
                        var99_price_eur=round(float(np.percentile(hub_yr_paths_eur, 99)), 2),
                        floor_binding_prob_pct=round(fl_prob, 2),
                        containment_or_penalty_trigger_prob_pct=round(cl_prob, 2),
                        eu_ets_basis_spread_median_eur=round(spread_vs_eu_eur, 2),
                    )
                )

        df_annual = pd.DataFrame([asdict(r) for r in annual_rows])

        # Build dedicated cross-border basis spread table (EU-ETS, UK-ETS, CA-WCI vs. India CCTS & China ETS)
        spread_rows: List[Dict] = []
        ind_idx = self.hub_order.index("INDIA_CCTS")
        chn_idx = self.hub_order.index("CHINA_ETS")
        uk_idx = self.hub_order.index("UK_ETS")
        wci_idx = self.hub_order.index("CA_WCI")

        _, fx_ind_eur = self._get_fx_rates("INR")
        _, fx_chn_eur = self._get_fx_rates("CNY")
        _, fx_uk_eur = self._get_fx_rates("GBP")
        _, fx_wci_eur = self._get_fx_rates("USD")
        eur_usd = float(self.config["fx_spot_2026_benchmarks"]["EUR_USD"])

        for yr_offset, yr in enumerate(self.years):
            s_start = yr_offset * self.steps_per_year + 1
            s_end = (yr_offset + 1) * self.steps_per_year + 1

            eu_eur = np.mean(P_paths[eu_idx, :, s_start:s_end], axis=1)
            uk_eur = np.mean(P_paths[uk_idx, :, s_start:s_end], axis=1) * fx_uk_eur
            wci_eur = np.mean(P_paths[wci_idx, :, s_start:s_end], axis=1) * fx_wci_eur
            ind_eur = np.mean(P_paths[ind_idx, :, s_start:s_end], axis=1) * fx_ind_eur
            chn_eur = np.mean(P_paths[chn_idx, :, s_start:s_end], axis=1) * fx_chn_eur

            eu_ind_spread = eu_eur - ind_eur
            uk_ind_spread = uk_eur - ind_eur
            wci_ind_spread = wci_eur - ind_eur
            eu_chn_spread = eu_eur - chn_eur

            spread_rows.append(
                {
                    "year": yr,
                    "eu_ets_median_eur": round(float(np.median(eu_eur)), 2),
                    "india_ccts_median_eur": round(float(np.median(ind_eur)), 2),
                    "china_ets_median_eur": round(float(np.median(chn_eur)), 2),
                    "eu_vs_india_spread_median_eur": round(float(np.median(eu_ind_spread)), 2),
                    "eu_vs_india_spread_var95_eur": round(float(np.percentile(eu_ind_spread, 95)), 2),
                    "eu_vs_india_spread_var99_eur": round(float(np.percentile(eu_ind_spread, 99)), 2),
                    "uk_vs_india_spread_median_eur": round(float(np.median(uk_ind_spread)), 2),
                    "wci_vs_india_spread_median_usd": round(float(np.median(wci_ind_spread) * eur_usd), 2),
                    "eu_vs_china_spread_median_eur": round(float(np.median(eu_chn_spread)), 2),
                }
            )

        df_spreads = pd.DataFrame(spread_rows)

        # Save processed tables (CSV + Parquet)
        df_annual.to_csv(
            self.processed_dir / "monte_carlo_7hubs_annual_summary_2026_2035.csv",
            index=False,
        )
        df_annual.to_parquet(
            self.processed_dir / "monte_carlo_7hubs_annual_summary_2026_2035.parquet",
            index=False,
        )
        df_spreads.to_csv(
            self.processed_dir / "cross_hub_basis_spreads_2026_2035.csv",
            index=False,
        )
        df_spreads.to_parquet(
            self.processed_dir / "cross_hub_basis_spreads_2026_2035.parquet",
            index=False,
        )

        paths_dict = {hub: P_paths[k] for k, hub in enumerate(self.hub_order)}
        self.logger.info(
            f"Completed {self.n_paths}-path Correlated OU Jump-Diffusion simulation across 7 hubs "
            f"(120 monthly steps, 2026-2035). "
            f"2030 EU-ETS Median=EUR {df_spreads.loc[df_spreads['year']==2030, 'eu_ets_median_eur'].iloc[0]:.2f}/t vs. "
            f"India CCTS Median=EUR {df_spreads.loc[df_spreads['year']==2030, 'india_ccts_median_eur'].iloc[0]:.2f}/t "
            f"(Spread=EUR {df_spreads.loc[df_spreads['year']==2030, 'eu_vs_india_spread_median_eur'].iloc[0]:.2f}/t)."
        )
        return paths_dict, df_annual, df_spreads


if __name__ == "__main__":
    engine = CorrelatedPriceDiscoveryEngine(n_paths=10000, steps_per_year=12, random_seed=42)
    _, df_ann, df_sp = engine.run_monte_carlo_simulation()
    print("\n=== 2026 vs. 2030 vs. 2035 MULTI-HUB MONTE CARLO PRICE DISCOVERY SUMMARY ===")
    sub = df_ann[df_ann["year"].isin([2026, 2030, 2035])]
    print(
        sub[
            [
                "jurisdiction",
                "year",
                "currency",
                "median_price_local",
                "var95_price_local",
                "var99_price_local",
                "median_price_eur",
                "var95_price_eur",
                "containment_or_penalty_trigger_prob_pct",
            ]
        ].to_string(index=False)
    )
    print("\n=== CROSS-BORDER BASIS SPREAD TRAJECTORY (EUR / USD PER tCO2e) ===")
    print(df_sp.to_string(index=False))
