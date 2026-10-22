"""
Global Carbon Compliance Architecture: India's CCTS vs. The World (2026-2035)
Layer 2 - Module 1: Structural Mechanism Benchmarking & ODE Decoupling Engine
File: src/mechanics/cap_vs_intensity.py
Lead Architect: Kaustav Goswami

Mathematical Core:
1. Absolute Cap-and-Trade (EU-ETS, UK-ETS, CA-WCI, NZ-ETS, K-ETS):
     S_t^abs = E_0 * (1 - alpha_LRF * t),   dS_t^abs / dY_t = 0
2. Output-Adjusted Emission Intensity Baseline (India CCTS, China National ETS):
     S_t^int = Y_t * iota_t^*,              dS_t^int / dY_t = iota_t^* > 0
3. Continuous-Time Decoupling Differential:
     (dE_t^int / dt) / E_t^int = g_Y(t) - gamma_iota(t)
   Absolute emissions strictly increase under 100% compliance whenever real
   industrial output growth g_Y(t) exceeds the statutory intensity tightening
   rate gamma_iota(t).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import json


@dataclass
class SectorDecouplingMetrics:
    """Summary metrics for a single BEE CCTS sector across 2026-2035."""
    sector_code: str
    sector_name: str
    cbam_exposed: bool
    base_emissions_2026_mt: float
    emissions_2035_intensity_regime_mt: float
    emissions_2035_abs_cap_regime_mt: float
    cumulative_emissions_intensity_2026_2035_mt: float
    cumulative_emissions_abs_cap_2026_2035_mt: float
    cumulative_decoupling_wedge_mt: float
    annual_output_growth_gY: float
    gamma_phase1_2026_2028: float
    gamma_phase2_2029_2035: float
    instantaneous_peaking_threshold_gamma_star: float
    discrete_peaking_threshold_gamma_star: float
    cumulative_neutral_gamma_star_2026_2035: float
    peaks_before_2035: bool


class StructuralDecouplingEngine:
    """
    Analytical and ODE simulation engine comparing Absolute Cap-and-Trade vs.
    Output-Adjusted Emission Intensity Baselines across India's 9 CCTS sectors
    and China's 2028 dual-control transition.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.config_path = self.project_root / "config" / "jurisdictions.json"
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed"
        self.log_file = self.project_root / "logs" / "ingestion_and_sim_audit.log"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
        self.logger = self._get_logger()

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger("StructuralDecouplingEngine")
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
    # Core Mathematical Operators
    # --------------------------------------------------------------------------
    @staticmethod
    def absolute_cap_supply(e0_mt: float, alpha_lrf: float, t_years: float) -> Tuple[float, float]:
        """
        Computes fixed absolute cap supply S_t^abs and output sensitivity dS/dY:
            S_t^abs = E_0 * max(0, 1 - alpha_LRF * t)
            dS_t^abs / dY_t = 0.0
        """
        s_abs = float(e0_mt * max(0.0, 1.0 - alpha_lrf * t_years))
        ds_dy = 0.0
        return s_abs, ds_dy

    @staticmethod
    def intensity_baseline_supply(y_t: float, iota_star_t: float) -> Tuple[float, float]:
        """
        Computes endogenous intensity-based allowance supply S_t^int and dS/dY:
            S_t^int = Y_t * iota_t^*
            dS_t^int / dY_t = iota_t^* > 0
        """
        s_int = float(y_t * iota_star_t)
        ds_dy = float(iota_star_t)
        return s_int, ds_dy

    @staticmethod
    def decoupling_differential(g_y: float, gamma_iota: float) -> float:
        """
        Instantaneous logarithmic growth rate of compliance emissions:
            (dE_t / dt) / E_t = g_Y(t) - gamma_iota(t)
        """
        return float(g_y - gamma_iota)

    @staticmethod
    def solve_coupled_sector_ode(
        y0: float,
        iota0: float,
        g_y: float,
        gamma_phase1: float,
        gamma_phase2: float,
        t_eval: np.ndarray,
        phase_split_t: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrates the coupled continuous-time ODE system for t in [0, 9] (2026-2035):
            dY/dt     = g_Y * Y(t)
            diota*/dt = -gamma_iota(t) * iota*(t)
            dE/dt     = (g_Y - gamma_iota(t)) * E(t)
        using SciPy RK45 with high precision (rtol=1e-9, atol=1e-11).
        """
        e0 = y0 * iota0

        def gamma_schedule(t: float) -> float:
            return gamma_phase1 if t <= phase_split_t else gamma_phase2

        def ode_rhs(t: float, state: np.ndarray) -> List[float]:
            y_val, iota_val, e_val = state
            gam = gamma_schedule(t)
            dy_dt = g_y * y_val
            diota_dt = -gam * iota_val
            de_dt = (g_y - gam) * e_val
            return [dy_dt, diota_dt, de_dt]

        sol = solve_ivp(
            fun=ode_rhs,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=[y0, iota0, e0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )
        return sol.y[0], sol.y[1], sol.y[2]

    @staticmethod
    def compute_threshold_gamma_star(
        e0: float,
        g_y: float,
        alpha_lrf_benchmark: float = 0.043,
        horizon_years: int = 9,
    ) -> Tuple[float, float, float]:
        """
        Computes three critical tightening rate thresholds for a sector:
        1. Instantaneous peaking threshold: gamma_iota^* = g_Y (so dE/dt <= 0).
        2. Discrete annual settlement threshold: gamma_discrete^* = g_Y / (1 + g_Y)
           such that (1 + g_Y) * (1 - gamma_discrete^*) = 1.
        3. Cumulative cap-equivalence threshold (via SciPy brentq): constant rate
           gamma_cum^* such that integral_0^T E_0 exp((g_Y - gamma)*t) dt equals
           the cumulative emissions of a fixed linear reduction cap E_0*(1 - alpha*t).
        """
        gamma_inst_star = float(g_y)
        gamma_disc_star = float(g_y / (1.0 + g_y))

        t_grid = np.arange(0, horizon_years + 1, dtype=float)
        target_cum_abs = float(np.sum(e0 * np.maximum(0.0, 1.0 - alpha_lrf_benchmark * t_grid)))

        def objective(gamma_trial: float) -> float:
            sim_cum_int = float(np.sum(e0 * np.exp((g_y - gamma_trial) * t_grid)))
            return sim_cum_int - target_cum_abs

        gamma_cum_star = float(brentq(objective, 0.001, 0.35, xtol=1e-10))
        return gamma_inst_star, gamma_disc_star, gamma_cum_star

    # --------------------------------------------------------------------------
    # Multi-Sector Indian CCTS Simulation (2026-2035)
    # --------------------------------------------------------------------------
    def simulate_india_9sectors_decoupling(
        self, benchmark_lrf: float = 0.043
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the 2026-2035 ODE and discrete compliance trajectory across all 9
        notified Indian BEE CCTS sectors and compares against an Absolute Cap
        counterfactual with Linear Reduction Factor `benchmark_lrf` (4.3%/yr).
        """
        years = np.arange(2026, 2036, dtype=int)
        t_eval = (years - 2026).astype(float)

        sectors_cfg = self.config["india_ccts_sectors"]
        panel_rows: List[Dict] = []
        summary_rows: List[SectorDecouplingMetrics] = []

        for sector_code, s in sectors_cfg.items():
            y0 = float(s["base_output_2026_mt_product"])
            # Use notified 2026 baseline intensity target
            iota0 = float(s["baseline_intensity_2026_tco2_per_t"])
            g_y = float(s["annual_output_growth_gY"])
            gam_p1 = float(s["annual_target_tightening_gamma_phase1"])
            gam_p2 = float(s["annual_target_tightening_gamma_phase2"])
            cbam_flag = bool(s["cbam_exposed"])

            y_traj, iota_traj, e_int_traj = self.solve_coupled_sector_ode(
                y0=y0,
                iota0=iota0,
                g_y=g_y,
                gamma_phase1=gam_p1,
                gamma_phase2=gam_p2,
                t_eval=t_eval,
                phase_split_t=2.0,  # 2026-2028 Phase 1, 2029-2035 Phase 2
            )

            e0 = y0 * iota0
            e_abs_traj = np.array(
                [self.absolute_cap_supply(e0, benchmark_lrf, t)[0] for t in t_eval]
            )

            gam_inst, gam_disc, gam_cum = self.compute_threshold_gamma_star(
                e0=e0, g_y=g_y, alpha_lrf_benchmark=benchmark_lrf, horizon_years=len(t_eval) - 1
            )

            for idx, yr in enumerate(years):
                t_val = t_eval[idx]
                active_gamma = gam_p1 if t_val <= 2.0 else gam_p2
                net_growth_rate = self.decoupling_differential(g_y, active_gamma)
                s_int, ds_dy_int = self.intensity_baseline_supply(y_traj[idx], iota_traj[idx])
                s_abs, ds_dy_abs = self.absolute_cap_supply(e0, benchmark_lrf, t_val)

                panel_rows.append(
                    {
                        "year": int(yr),
                        "sector_code": sector_code,
                        "sector_name": s["sector_name"],
                        "cbam_exposed": cbam_flag,
                        "output_mt_product": round(float(y_traj[idx]), 3),
                        "target_intensity_tco2_per_t": round(float(iota_traj[idx]), 4),
                        "emissions_intensity_regime_mt": round(s_int, 3),
                        "emissions_abs_cap_regime_mt": round(s_abs, 3),
                        "annual_decoupling_wedge_mt": round(s_int - s_abs, 3),
                        "output_growth_gY": g_y,
                        "active_intensity_tightening_gamma": active_gamma,
                        "net_emission_growth_rate_dE_E": round(net_growth_rate, 5),
                        "dS_dY_intensity_regime": round(ds_dy_int, 4),
                        "dS_dY_abs_cap_regime": ds_dy_abs,
                    }
                )

            cum_int = float(np.sum(e_int_traj))
            cum_abs = float(np.sum(e_abs_traj))
            summary_rows.append(
                SectorDecouplingMetrics(
                    sector_code=sector_code,
                    sector_name=s["sector_name"],
                    cbam_exposed=cbam_flag,
                    base_emissions_2026_mt=round(e0, 2),
                    emissions_2035_intensity_regime_mt=round(float(e_int_traj[-1]), 2),
                    emissions_2035_abs_cap_regime_mt=round(float(e_abs_traj[-1]), 2),
                    cumulative_emissions_intensity_2026_2035_mt=round(cum_int, 2),
                    cumulative_emissions_abs_cap_2026_2035_mt=round(cum_abs, 2),
                    cumulative_decoupling_wedge_mt=round(cum_int - cum_abs, 2),
                    annual_output_growth_gY=g_y,
                    gamma_phase1_2026_2028=gam_p1,
                    gamma_phase2_2029_2035=gam_p2,
                    instantaneous_peaking_threshold_gamma_star=round(gam_inst, 4),
                    discrete_peaking_threshold_gamma_star=round(gam_disc, 4),
                    cumulative_neutral_gamma_star_2026_2035=round(gam_cum, 4),
                    peaks_before_2035=bool(gam_p2 >= g_y),
                )
            )

        df_panel = pd.DataFrame(panel_rows)
        df_summary = pd.DataFrame([asdict(r) for r in summary_rows])

        # Save outputs to data/processed/
        df_panel.to_csv(
            self.processed_dir / "india_ccts_9sectors_decoupling_panel_2026_2035.csv",
            index=False,
        )
        df_panel.to_parquet(
            self.processed_dir / "india_ccts_9sectors_decoupling_panel_2026_2035.parquet",
            index=False,
        )
        df_summary.to_csv(
            self.processed_dir / "india_ccts_9sectors_decoupling_summary.csv",
            index=False,
        )

        self.logger.info(
            f"Completed India 9-Sector Decoupling ODE simulation (2026-2035): "
            f"Total 2026 base={df_summary['base_emissions_2026_mt'].sum():.2f} Mt, "
            f"2035 Intensity Regime={df_summary['emissions_2035_intensity_regime_mt'].sum():.2f} Mt vs. "
            f"2035 Absolute Cap={df_summary['emissions_2035_abs_cap_regime_mt'].sum():.2f} Mt "
            f"(Cumulative Wedge={df_summary['cumulative_decoupling_wedge_mt'].sum():.2f} Mt)."
        )
        return df_panel, df_summary

    # --------------------------------------------------------------------------
    # China National ETS Dual-Control Pivot Simulation (2026-2035)
    # --------------------------------------------------------------------------
    def simulate_china_dual_control_pivot(self) -> pd.DataFrame:
        """
        Models China National ETS (8,000 Mt in 2026) under:
        1. Pure Output-Adjusted Intensity Baseline (g_Y = 4.5%, gamma = 3.2% throughout)
        2. Statutory 2028 Dual-Control Pivot: intensity baseline in 2026-2027
           (g_Y = 4.5%, gamma = 3.2%) pivoting in 2028 to a hard declining
           Absolute Cap with LRF = 5.0%/yr relative to the 2027 peak cap.
        """
        china_cfg = self.config["jurisdictions"]["CHINA_ETS"]
        e0_2026 = float(china_cfg["covered_emissions_2026_mt"])
        g_y = float(china_cfg["real_output_growth_rate_gY"])
        sched = china_cfg["intensity_tightening_schedule"]
        gam_pre = float(sched.get("2026_2027", sched.get(20262027)))
        lrf_post = float(sched.get("2028_2035", sched.get(20282035)))
        pivot_year = int(china_cfg["transition_to_absolute_cap_year"])

        years = list(range(2026, 2036))
        rows = []
        e_pure_int = e0_2026
        e_dual_pivot = e0_2026
        cap_at_pivot = None

        for yr in years:
            if yr == 2026:
                regime_label = "INTENSITY_BASELINE"
                ds_dy = 1.0
                net_rate_pivot = g_y - gam_pre
            elif yr < pivot_year:
                regime_label = "INTENSITY_BASELINE"
                e_pure_int *= np.exp(g_y - gam_pre)
                e_dual_pivot *= np.exp(g_y - gam_pre)
                ds_dy = round(float(np.exp(-gam_pre * (yr - 2026))), 4)
                net_rate_pivot = g_y - gam_pre
            else:
                regime_label = "ABSOLUTE_CAP_PIVOT"
                if cap_at_pivot is None:
                    cap_at_pivot = e_dual_pivot
                e_pure_int *= np.exp(g_y - gam_pre)
                years_since_pivot = yr - (pivot_year - 1)
                e_dual_pivot = cap_at_pivot * max(0.0, 1.0 - lrf_post * years_since_pivot)
                ds_dy = 0.0
                net_rate_pivot = -lrf_post

            rows.append(
                {
                    "year": yr,
                    "active_regime": regime_label,
                    "emissions_dual_control_pivot_mt": round(float(e_dual_pivot), 2),
                    "emissions_unreformed_intensity_mt": round(float(e_pure_int), 2),
                    "annual_avoided_emissions_mt": round(float(e_pure_int - e_dual_pivot), 2),
                    "dS_dY_normalized": ds_dy,
                    "net_emission_growth_rate": round(float(net_rate_pivot), 4),
                }
            )

        df_china = pd.DataFrame(rows)
        df_china.to_csv(
            self.processed_dir / "china_ets_dual_control_pivot_2026_2035.csv",
            index=False,
        )
        self.logger.info(
            f"Completed China Dual-Control Pivot simulation (pivot={pivot_year}): "
            f"2035 Dual-Control={df_china.iloc[-1]['emissions_dual_control_pivot_mt']:.1f} Mt vs. "
            f"Unreformed Intensity={df_china.iloc[-1]['emissions_unreformed_intensity_mt']:.1f} Mt "
            f"(Cumulative avoided={df_china['annual_avoided_emissions_mt'].sum():.1f} Mt)."
        )
        return df_china


if __name__ == "__main__":
    engine = StructuralDecouplingEngine()
    df_ind_panel, df_ind_summary = engine.simulate_india_9sectors_decoupling()
    df_chn = engine.simulate_china_dual_control_pivot()
    print("\n=== INDIA CCTS 9-SECTOR DECOUPLING SUMMARY (2026-2035) ===")
    print(
        df_ind_summary[
            [
                "sector_name",
                "cbam_exposed",
                "base_emissions_2026_mt",
                "emissions_2035_intensity_regime_mt",
                "emissions_2035_abs_cap_regime_mt",
                "cumulative_decoupling_wedge_mt",
                "instantaneous_peaking_threshold_gamma_star",
                "peaks_before_2035",
            ]
        ].to_string(index=False)
    )
    print("\n=== CHINA NATIONAL ETS 2028 DUAL-CONTROL PIVOT (2026-2035) ===")
    print(df_chn.to_string(index=False))
