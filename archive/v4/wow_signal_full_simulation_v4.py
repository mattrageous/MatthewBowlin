#!/usr/bin/env python3
"""
Wow! signal plausibility simulator — Version 4

Adds a persistence/non-repetition module to Version 3.

The V4 module tests:
1. Whether deep scintillation can rescue a steady source across repeated follow-ups.
2. How source duty cycle, observational coverage, sensitivity, and frequency/drift
   coverage change the probability of repeated non-detections.
3. What repeat rates remain compatible with a benchmark 192 hours of follow-up.
4. Why strong scintillation is only an escape hatch for an angularly compact source.

This remains a mechanism/likelihood sensitivity model, not a reconstruction of
every historical observing campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "wow_signal_full_simulation_v3.py"

spec = importlib.util.spec_from_file_location("wow_v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v3
spec.loader.exec_module(v3)


@dataclass(frozen=True)
class PersistenceConfig:
    followup_hours: float = 192.0
    confidence_level: float = 0.95
    nominal_independent_epochs: int = 20
    representative_coverage: float = 0.50
    representative_sensitivity: float = 0.80
    representative_frequency_drift_coverage: float = 0.80
    screen_distances_kpc: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


def fresnel_angle_microarcsec(frequency_hz: float, distance_kpc: float) -> float:
    """Illustrative Fresnel angular scale for a thin screen."""
    wavelength_m = v3.C / frequency_hz
    distance_m = distance_kpc * 3.085677581491367e19
    theta_rad = math.sqrt(wavelength_m / (2.0 * math.pi * distance_m))
    return theta_rad * (180.0 / math.pi) * 3600.0 * 1e6


def persistence_metrics(
    base_config: v3.Config,
    pconfig: PersistenceConfig,
    v3_details: dict,
) -> dict:
    p_scint_miss = float(v3_details["scintillation"]["independent_limit"])
    p_scint_detect = 1.0 - p_scint_miss

    # A. Independent re-observations of a truly steady, compact source.
    n_epochs = np.arange(1, 61)
    all_miss_steady = p_scint_miss ** n_epochs

    # B. Sensitivity to correlation. This is an effective-sample-size stress
    # test, not a claim that years-separated campaigns are correlated.
    rho = np.linspace(0.0, 0.995, 300)
    n_nominal = pconfig.nominal_independent_epochs
    n_effective = n_nominal / (1.0 + (n_nominal - 1.0) * rho)
    all_miss_correlated = p_scint_miss ** n_effective

    # C. Duty-cycle and observing-efficiency grid.
    duty_cycle = np.logspace(-4.0, 0.0, 241)
    epochs_grid = np.arange(1, 101)
    total_efficiency = (
        pconfig.representative_coverage
        * pconfig.representative_sensitivity
        * pconfig.representative_frequency_drift_coverage
        * p_scint_detect
    )
    p_detect_per_epoch = np.clip(
        duty_cycle[:, None] * total_efficiency, 0.0, 1.0
    )
    all_miss_duty = (1.0 - p_detect_per_epoch) ** epochs_grid[None, :]

    # D. Poisson stochastic-repeater benchmark using 192 hours total exposure.
    rate_per_day = np.logspace(-4.0, 2.0, 400)
    exposure_days = pconfig.followup_hours / 24.0
    coverage_values = np.array([0.25, 0.50, 0.75, 1.00])
    poisson_zero = {}
    for coverage in coverage_values:
        effective_exposure_days = (
            exposure_days
            * coverage
            * pconfig.representative_sensitivity
            * pconfig.representative_frequency_drift_coverage
            * p_scint_detect
        )
        poisson_zero[coverage] = np.exp(-rate_per_day * effective_exposure_days)

    # E. 95% upper bounds from zero detections.
    alpha = 1.0 - pconfig.confidence_level
    duty_rows = []
    for n in [3, 5, 10, 20, 50]:
        for coverage in [0.25, 0.50, 0.75, 1.00]:
            efficiency = (
                coverage
                * pconfig.representative_sensitivity
                * pconfig.representative_frequency_drift_coverage
                * p_scint_detect
            )
            d95 = (1.0 - alpha ** (1.0 / n)) / efficiency
            duty_rows.append({
                "independent_epochs": n,
                "coverage": coverage,
                "sensitivity": pconfig.representative_sensitivity,
                "frequency_drift_coverage": pconfig.representative_frequency_drift_coverage,
                "scintillation_detection_probability": p_scint_detect,
                "duty_cycle_95pct_upper": min(1.0, d95),
            })
    duty_limits = pd.DataFrame(duty_rows)

    rate_rows = []
    for coverage in coverage_values:
        efficiency = (
            coverage
            * pconfig.representative_sensitivity
            * pconfig.representative_frequency_drift_coverage
            * p_scint_detect
        )
        effective_exposure_days = exposure_days * efficiency
        lambda95 = -math.log(alpha) / effective_exposure_days
        rate_rows.append({
            "followup_hours": pconfig.followup_hours,
            "coverage": coverage,
            "sensitivity": pconfig.representative_sensitivity,
            "frequency_drift_coverage": pconfig.representative_frequency_drift_coverage,
            "scintillation_detection_probability": p_scint_detect,
            "effective_exposure_days": effective_exposure_days,
            "repeat_rate_95pct_upper_per_day": lambda95,
            "mean_interval_95pct_lower_days": 1.0 / lambda95,
        })
    rate_limits = pd.DataFrame(rate_rows)

    # F. Compactness gate. Deep diffractive scintillation is quenched when the
    # source is much larger than the relevant angular interference scale.
    compactness_rows = []
    for distance_kpc in pconfig.screen_distances_kpc:
        compactness_rows.append({
            "screen_distance_kpc": distance_kpc,
            "fresnel_angle_microarcsec": fresnel_angle_microarcsec(
                base_config.target_hz, distance_kpc
            ),
        })
    compactness = pd.DataFrame(compactness_rows)

    return {
        "p_scint_miss": p_scint_miss,
        "p_scint_detect": p_scint_detect,
        "n_epochs": n_epochs,
        "all_miss_steady": all_miss_steady,
        "rho": rho,
        "n_effective": n_effective,
        "all_miss_correlated": all_miss_correlated,
        "duty_cycle": duty_cycle,
        "epochs_grid": epochs_grid,
        "all_miss_duty": all_miss_duty,
        "total_efficiency": total_efficiency,
        "rate_per_day": rate_per_day,
        "coverage_values": coverage_values,
        "poisson_zero": poisson_zero,
        "duty_limits": duty_limits,
        "rate_limits": rate_limits,
        "compactness": compactness,
        "exposure_days": exposure_days,
    }


def updated_results(v3_results: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    results = v3_results.copy()
    p = metrics["p_scint_miss"]
    p3 = p ** 3
    p5 = p ** 5

    mask = results["Scenario"].eq(
        "Steady narrow celestial source with strong scintillation"
    )
    results.loc[mask, "Horn result"] = (
        "Can miss the second horn, but repeated independent follow-ups dominate"
    )
    results.loc[mask, "Verdict"] = "Nearly rejected as a persistent source"
    results.loc[mask, "Interpretation"] = (
        f"Toy all-miss probability is {p3:.3e} after 3 independent epochs "
        f"and {p5:.3e} after 5; scintillation only helps compact sources."
    )

    mask2 = results["Scenario"].eq(
        "Continuous narrow celestial source without propagation modulation"
    )
    results.loc[mask2, "Verdict"] = "Rejected by horn plus persistence"
    results.loc[mask2, "Interpretation"] = (
        "Without intrinsic intermittency or propagation modulation, it should "
        "have appeared in the second horn and in later observations."
    )

    extra = pd.DataFrame([
        {
            "Scenario": "Compact narrow intermittent source",
            "Bandwidth result": float(
                v3_results.loc[
                    v3_results["Scenario"].eq(
                        "Intrinsically narrow cosmic transient + propagation effects"
                    ),
                    "Bandwidth result",
                ].iloc[0]
            ),
            "Horn result": "Can evade both horn and later observations",
            "Verdict": "Conditional; duty cycle depends on coverage",
            "Interpretation": (
                "Allowed duty cycle depends strongly on independent epochs, "
                "coverage, sensitivity and drift-frequency coverage."
            ),
        },
        {
            "Scenario": "Stochastic narrowband repeater",
            "Bandwidth result": float(
                v3_results.loc[
                    v3_results["Scenario"].eq(
                        "Intrinsically narrow cosmic transient + propagation effects"
                    ),
                    "Bandwidth result",
                ].iloc[0]
            ),
            "Horn result": "Non-repetition constrains the repeat rate",
            "Verdict": "Tension, not complete exclusion",
            "Interpretation": (
                "The 192-hour benchmark strongly disfavors frequent repeaters "
                "but still permits sufficiently rare events."
            ),
        },
        {
            "Scenario": "Extended or broadband source invoking deep scintillation",
            "Bandwidth result": float(
                v3_results.loc[
                    v3_results["Scenario"].eq(
                        "Broad cosmic source + linear propagation effects"
                    ),
                    "Bandwidth result",
                ].iloc[0]
            ),
            "Horn result": "Deep scintillation is quenched by source extent",
            "Verdict": "Rejected",
            "Interpretation": (
                "The scintillation escape hatch applies only to very compact "
                "sources and does not reopen the broadband/extended class."
            ),
        },
    ])
    return pd.concat([results, extra], ignore_index=True)


def make_v4_plots(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["n_epochs"], metrics["all_miss_steady"], marker="o", markevery=5)
    plt.yscale("log")
    plt.xlabel("Independent follow-up epochs")
    plt.ylabel("Probability every epoch is below threshold")
    plt.title("A steady scintillating source collapses under repetition")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v4_steady_survival.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["rho"], metrics["all_miss_correlated"])
    plt.yscale("log")
    plt.xlabel("Assumed correlation between nominal follow-ups")
    plt.ylabel("Effective all-miss probability")
    plt.title("Correlation stress test for 20 nominal follow-ups")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v4_correlation_stress.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    image = plt.imshow(
        np.log10(np.maximum(metrics["all_miss_duty"], 1e-300)),
        origin="lower",
        aspect="auto",
        extent=[
            metrics["epochs_grid"].min(),
            metrics["epochs_grid"].max(),
            math.log10(metrics["duty_cycle"].min()),
            math.log10(metrics["duty_cycle"].max()),
        ],
    )
    plt.colorbar(image, label="log10 probability of all non-detections")
    plt.xlabel("Effective independent epochs")
    plt.ylabel("log10 source duty cycle")
    plt.title("Duty cycle needed to survive repeated non-detections")
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v4_duty_cycle_map.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for coverage in metrics["coverage_values"]:
        plt.plot(
            metrics["rate_per_day"],
            metrics["poisson_zero"][coverage],
            label=f"coverage={coverage:.2f}",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean repeat rate (events/day)")
    plt.ylabel("Probability of zero detections in 192 h")
    plt.title("Simple Poisson repeater benchmark")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v4_repeat_rate.png", dpi=180)
    plt.close()

    compact = metrics["compactness"]
    plt.figure(figsize=(8, 5))
    plt.plot(
        compact["screen_distance_kpc"],
        compact["fresnel_angle_microarcsec"],
        marker="o",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Illustrative scattering-screen distance (kpc)")
    plt.ylabel("Fresnel angular scale (microarcseconds)")
    plt.title("Deep scintillation requires an extremely compact source")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v4_compactness_gate.png", dpi=180)
    plt.close()


def write_v4_report(
    base_config: v3.Config,
    pconfig: PersistenceConfig,
    results: pd.DataFrame,
    metrics: dict,
    output_dir: Path,
) -> None:
    duty = metrics["duty_limits"]
    rates = metrics["rate_limits"]
    p = metrics["p_scint_miss"]
    representative_duty = duty[
        (duty["independent_epochs"] == 20)
        & (duty["coverage"] == 0.50)
    ].iloc[0]
    representative_rate = rates[rates["coverage"] == 0.50].iloc[0]

    report = f"""
WOW! SIGNAL COMBINED-EFFECTS SIMULATION — VERSION 4
===================================================

What Version 4 adds
-------------------
Version 4 adds the missing persistence constraint. The second horn is only one
follow-up opportunity; later re-observations are much more damaging to a steady
source. This module uses 192 hours as a published aggregate benchmark for META,
Hobart and ATA follow-up, but treats coverage and sensitivity as adjustable.

Compactness gate
----------------
The one-horn scintillation probability from V3 is {100*p:.3f}% in the fully
decorrelated toy limit. That escape hatch is not generic. Deep scintillation is
quenched when the source is extended compared with the relevant interference
angle.

At {base_config.target_hz/1e6:.3f} MHz, illustrative Fresnel angular scales are:

{metrics['compactness'].to_string(index=False)}

These microarcsecond-scale values mean that deep scintillation is naturally
associated with compact emitting regions. It does not rescue an extended
broadband collision or ordinary diffuse thermal source.

Steady compact source
---------------------
Assuming independent scintillation states with per-epoch miss probability
p={p:.6f}:

1 follow-up miss: {p:.6e}
2 follow-up misses: {p**2:.6e}
3 follow-up misses: {p**3:.6e}
5 follow-up misses: {p**5:.6e}
10 follow-up misses: {p**10:.6e}

Therefore a continuously emitting compact source is nearly rejected after only
a few genuinely independent, adequately covered observations. Correlation can
weaken this result, but years-separated campaigns should not automatically be
treated as one scintillation state. The correlation plot is a stress test, not
a fitted physical model.

Duty-cycle model
----------------
Per effective epoch:

P(detection) = duty_cycle × sky/frequency coverage × sensitivity
               × drift coverage × scintillation detection probability

The representative efficiency used in the heat map is:
coverage={pconfig.representative_coverage:.2f},
sensitivity={pconfig.representative_sensitivity:.2f},
drift/frequency coverage={pconfig.representative_frequency_drift_coverage:.2f},
scintillation detection={metrics['p_scint_detect']:.4f}.

For 20 independent epochs and 50% coverage, the 95% upper duty-cycle bound is
{representative_duty['duty_cycle_95pct_upper']:.4f}.

The full duty-cycle limits are:

{duty.to_string(index=False)}

Poisson repeating-beacon benchmark
----------------------------------
For total follow-up exposure E, a simple Poisson model gives:

P(zero detections) = exp(-repeat_rate × effective_exposure)

This does not reproduce the full Big Ear observing-log likelihood analysis; it
is a transparent sensitivity calculation.

With 192 hours and 50% coverage, the 95% upper repeat-rate bound in this
notional efficiency model is
{representative_rate['repeat_rate_95pct_upper_per_day']:.4f} events/day,
corresponding to a mean interval longer than
{representative_rate['mean_interval_95pct_lower_days']:.2f} days.

The full repeat-rate limits are:

{rates.to_string(index=False)}

Version 4 verdict
-----------------
CLEAN PASS:
An intrinsically narrow, one-time astrophysical transient. A maser-like
hydrogen-line brightening is in this class.

CLEAN LOGICAL PASS, WITHOUT AFFIRMATIVE EVIDENCE:
A genuinely non-repeating artificial transmission.

LIVE CONDITIONAL:
A narrow intermittent source. Under conservative aggregate assumptions,
non-detection does not by itself force an extremely low duty cycle.

LIVE CONDITIONAL:
Nonlinear receiver intermodulation from strong higher-frequency inputs,
especially with asymmetric sidelobe coupling.

NEARLY REJECTED:
A continuously emitting compact source rescued only by scintillation. It can
miss the second horn, but repeated independent non-detections rapidly destroy
the hypothesis.

REJECTED:
An extended/broadband source invoking deep scintillation.

REJECTED:
Broadband emission compressed into one channel by linear gravity, ionosphere,
Doppler, or ordinary radio-wave overlap.

Limitations
-----------
The 192-hour benchmark is aggregated. Historical campaigns differed in exact
coordinates, sensitivity, channel width, drift-rate coverage, and observing
cadence. The duty-cycle and Poisson calculations therefore show dependence on
explicit assumptions rather than claiming a single historical posterior.
""".strip()

    (output_dir / "wow_signal_v4_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    base_config = v3.Config()
    pconfig = PersistenceConfig()

    v3_results, v3_details = v3.build_results(base_config)
    metrics = persistence_metrics(base_config, pconfig, v3_details)
    results = updated_results(v3_results, metrics)

    results.to_csv(HERE / "wow_signal_v4_scenario_results.csv", index=False)
    metrics["duty_limits"].to_csv(
        HERE / "wow_signal_v4_duty_cycle_limits.csv", index=False
    )
    metrics["rate_limits"].to_csv(
        HERE / "wow_signal_v4_repeat_rate_limits.csv", index=False
    )
    metrics["compactness"].to_csv(
        HERE / "wow_signal_v4_compactness_gate.csv", index=False
    )
    pd.DataFrame({
        "independent_epochs": metrics["n_epochs"],
        "steady_all_miss_probability": metrics["all_miss_steady"],
    }).to_csv(HERE / "wow_signal_v4_steady_survival.csv", index=False)
    pd.DataFrame({
        "assumed_correlation": metrics["rho"],
        "effective_independent_epochs": metrics["n_effective"],
        "all_miss_probability": metrics["all_miss_correlated"],
    }).to_csv(HERE / "wow_signal_v4_correlation_stress.csv", index=False)

    make_v4_plots(metrics, HERE)
    write_v4_report(base_config, pconfig, results, metrics, HERE)

    print(results.to_string(index=False))
    print("\nV4 headline outputs:")
    print(f"  Single independent scintillation miss: {100*metrics['p_scint_miss']:.3f}%")
    print(f"  All miss after 3 independent epochs: {metrics['p_scint_miss']**3:.6e}")
    print(f"  All miss after 5 independent epochs: {metrics['p_scint_miss']**5:.6e}")

    duty20 = metrics["duty_limits"][
        (metrics["duty_limits"]["independent_epochs"] == 20)
        & (metrics["duty_limits"]["coverage"] == 0.50)
    ].iloc[0]
    rate50 = metrics["rate_limits"][
        metrics["rate_limits"]["coverage"] == 0.50
    ].iloc[0]
    print(f"  95% duty-cycle upper bound (20 epochs, 50% coverage): "
          f"{duty20['duty_cycle_95pct_upper']:.4f}")
    print(f"  95% repeat-rate upper bound (192 h, 50% coverage): "
          f"{rate50['repeat_rate_95pct_upper_per_day']:.4f}/day")
    print(f"  Corresponding mean interval lower bound: "
          f"{rate50['mean_interval_95pct_lower_days']:.2f} days")


if __name__ == "__main__":
    main()
