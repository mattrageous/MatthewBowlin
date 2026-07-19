#!/usr/bin/env python3
"""
Wow! signal simulation V5
=========================

A campaign-by-campaign persistence model using published observing details.

V5 is intentionally narrower than V1-V4. It addresses one question:

    Can a steady compact source, modulated by saturated diffractive
    scintillation, plausibly produce the original Wow! event and then evade
    the published follow-up observations?

It also includes:
- a Big Ear same-sidereal-time periodic-alias stress test;
- a constant secular frequency-drift escape test;
- explicit campaign metadata and uncertainty flags.

This is not a full historical posterior. It is a transparent likelihood
sensitivity calculation based on published campaign-level summaries.

Primary source basis
--------------------
Gray 1994, Icarus 112, 485-489, DOI 10.1006/icar.1994.1199
Gray & Marvel 2001, ApJ 546, 1171-1177, DOI 10.1086/318272
Gray & Ellingsen 2002, ApJ 578, 967-971, DOI 10.1086/342646
Harp et al. 2020, AJ 160, 162, DOI 10.3847/1538-3881/aba58f
Kipping & Gray 2022, MNRAS 515, 1122, DOI 10.1093/mnras/stac1807
Perez et al. 2022, RNAAS 6, 197, DOI 10.3847/2515-5172/ac9408
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent

WOW_CONSERVATIVE_JY = 54.0
WOW_HIGH_JY = 212.0
WOW_SNR = 30.5
ASSUMED_BIG_EAR_DETECTION_SIGMA = 5.0
SIDEREAL_DAY_S = 86164.0905
HORN_OFFSET_S = 172.37
OBS_WINDOW_S = 72.0
DISCOVERY_DATE = datetime(1977, 8, 15)


BIG_EAR_DATES = [
    "1977-08-13", "1977-08-14", "1977-08-15", "1977-08-17",
    "1977-09-16", "1977-09-17", "1977-09-18", "1977-09-19",
    "1977-09-20", "1977-09-21", "1977-09-22", "1977-09-23",
    "1977-09-24", "1977-09-25", "1977-09-26", "1977-09-27",
    "1977-09-28", "1977-09-29", "1977-09-30", "1977-10-04",
    "1977-10-30", "1978-04-10", "1978-04-11", "1978-04-17",
    "1978-04-24", "1978-04-25", "1978-04-26", "1978-04-27",
    "1978-04-28", "1978-05-01", "1978-05-02", "1978-08-28",
    "1978-08-29", "1978-08-30", "1978-08-31", "1978-09-01",
    "1978-09-02", "1978-09-03", "1978-09-04", "1983-01-02",
    "1983-01-03", "1983-01-04", "1983-01-07", "1983-01-13",
    "1983-01-15",
    "1983-01-22", "1983-01-22", "1983-01-23", "1983-01-24",
    "1983-01-25", "1983-01-28", "1983-01-29", "1983-02-02",
    "1983-02-02", "1983-02-04", "1983-02-05", "1983-02-06",
    "1983-02-07", "1983-02-12", "1983-02-14", "1983-02-15",
    "1983-02-16", "1983-02-17", "1983-02-18", "1983-02-19",
    "1983-02-20", "1983-02-21", "1983-02-22", "1983-02-23",
    "1983-02-24", "1983-02-25", "1983-02-26", "1983-02-27",
    "1983-02-28", "1983-03-01", "1983-03-07", "1983-03-08",
    "1983-03-09", "1983-03-12", "1983-03-13", "1983-03-14",
    "1983-03-20", "1983-03-21", "1983-03-23", "1983-03-26",
    "1983-03-28", "1983-03-29", "1983-04-26", "1983-05-09",
    "1984-12-06",
]


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    campaign: str
    date_label: str
    representative_date: str
    total_hours: float
    conservative_independent_states: int
    channel_width_hz: float
    usable_frequency_margin_hz: float
    one_sigma_jy: float | None
    detection_sigma: float | None
    threshold_jy: float | None
    spatial_scope: str
    linewidth_scope: str
    core_10khz_applicable: bool
    ultra_narrow_applicable: bool
    confidence: str
    notes: str
    source: str


def build_campaigns(wow_flux_jy: float = WOW_CONSERVATIVE_JY) -> list[Campaign]:
    bigear_sigma_jy = wow_flux_jy / WOW_SNR
    bigear_threshold = ASSUMED_BIG_EAR_DETECTION_SIGMA * bigear_sigma_jy

    return [
        Campaign(
            "BIGEAR",
            "Big Ear null revisits",
            "1977-1984",
            "1981-01-01",
            89 * 2 * OBS_WINDOW_S / 3600.0,
            89,
            10_000.0,
            250_000.0,
            bigear_sigma_jy,
            ASSUMED_BIG_EAR_DETECTION_SIGMA,
            bigear_threshold,
            "Original transit strip and beam geometry",
            "Signals fitting a 10 kHz channel",
            True,
            True,
            "High for dates; modeled threshold",
            "N=89 null visit-days after excluding the discovery from 90 useful visits. "
            "One independent scintillation state per visit-day is conservative. "
            "The 5-sigma threshold is a configurable model assumption.",
            "Kipping & Gray 2022; Gray 1994",
        ),
        Campaign(
            "META",
            "Harvard/Smithsonian META",
            "1987 and 1989",
            "1988-01-01",
            8.0,
            2,
            0.05,
            200_000.0,
            None,
            None,
            0.20 * wow_flux_jy,
            "Two nominal Wow positions; small primary beam",
            "Ultra-narrow, Doppler-corrected signal only",
            False,
            True,
            "Moderate",
            "Eight hours are relevant for whichever of the two possible positions is correct. "
            "Threshold is represented conservatively as 0.2 times the original OSU flux-equivalent, "
            "based on the paper's degraded-sensitivity comparison. Not applicable to a kHz-wide line.",
            "Gray 1994; Kipping & Gray 2022",
        ),
        Campaign(
            "VLA1995",
            "VLA 1995 line search",
            "1995-09-25",
            "1995-09-25",
            20.0 / 60.0,
            1,
            6_100.0,
            390_000.0,
            0.0055,
            8.0,
            0.0440,
            "Nominal positions plus offset fields",
            "Comparable to a <10 kHz Wow-like line",
            True,
            True,
            "High",
            "Uses the conservative 5.5 mJy rms at the nominal central field and an 8-sigma "
            "feature threshold. Relevant dwell is 20 minutes at the true one of two nominal positions.",
            "Gray & Marvel 2001",
        ),
        Campaign(
            "VLA1996",
            "VLA 1996 line search",
            "1996-05-07",
            "1996-05-07",
            43.0 / 60.0,
            1,
            12_200.0,
            750_000.0,
            0.0037,
            8.0,
            0.0296,
            "Two nominal positions",
            "Comparable to a <10 kHz Wow-like line",
            True,
            True,
            "High",
            "Uses the conservative 3.7 mJy rms central-field value and an 8-sigma threshold. "
            "Each nominal position was observed for 43 minutes.",
            "Gray & Marvel 2001",
        ),
        Campaign(
            "HOBART",
            "Hobart 26 m",
            "1998-1999",
            "1999-01-01",
            84.7,
            6,
            4_880.0,
            1_000_000.0,
            3.0,
            5.9,
            18.0,
            "Six pointings/runs covering nominal positions and declination offsets",
            "Wow-like kHz line",
            True,
            True,
            "High",
            "Six nearly 14-hour runs. One independent state per run is deliberately conservative. "
            "Published all-data threshold was about 18 Jy.",
            "Gray & Ellingsen 2002",
        ),
        Campaign(
            "ATA",
            "Allen Telescope Array",
            "Multi-year, through at least 2016",
            "2016-09-22",
            100.0,
            34,
            12_800.0,
            3_700_000.0,
            1.247,
            8.0,
            9.976,
            "Full 5 square-degree consistent-direction field",
            "Wow-like 12.8 kHz channel search",
            True,
            True,
            "High for totals; conservative state count",
            "More than 100 hours were obtained in sessions no longer than three hours, so at least "
            "34 separate sessions are required. The 1-minute empirical rms was 1.247 Jy and the "
            "all-search statistical threshold was 8 sigma. One state per session is conservative.",
            "Harp et al. 2020",
        ),
        Campaign(
            "BL2022",
            "GBT + ATA targeted candidate star",
            "2022-05-21",
            "2022-05-21",
            1.0,
            2,
            2.79,
            420_000_000.0,
            None,
            10.0,
            None,
            "One proposed Sun-like star only",
            "Artificial ultra-narrow drifting signals, +/-4 Hz/s",
            False,
            True,
            "High for setup; threshold omitted",
            "Two 30-minute GBT observations and six 5-minute ATA observations, with 580 seconds "
            "of simultaneous overlap. Excluded from the core likelihood because the published "
            "summary used here does not provide a directly comparable Jy threshold and the search "
            "covered only one proposed star.",
            "Perez et al. 2022",
        ),
    ]


def campaign_dataframe(campaigns: list[Campaign]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in campaigns])


def bigear_visit_dataframe() -> pd.DataFrame:
    dates = pd.to_datetime(BIG_EAR_DATES)
    offsets = (dates - pd.Timestamp(DISCOVERY_DATE)).days
    df = pd.DataFrame({
        "date": dates,
        "days_from_wow": offsets,
        "is_discovery": dates == pd.Timestamp(DISCOVERY_DATE),
    })
    df["same_sidereal_time_offset_s"] = df["days_from_wow"] * SIDEREAL_DAY_S
    return df


def log_original_density(mean_flux: np.ndarray, observed_flux: float) -> np.ndarray:
    """Exponential scintillation intensity density f(I_obs | mean)."""
    return -np.log(mean_flux) - observed_flux / mean_flux


def log_single_state_miss(mean_flux: np.ndarray, threshold: float) -> np.ndarray:
    x = threshold / mean_flux
    p = -np.expm1(-x)  # 1-exp(-x), stable for small x
    return np.log(np.clip(p, 1e-300, 1.0))


def evaluate_chain(
    mean_flux_grid: np.ndarray,
    observed_flux: float,
    campaigns: list[Campaign],
) -> dict:
    loglike = log_original_density(mean_flux_grid, observed_flux)
    for campaign in campaigns:
        if campaign.threshold_jy is None:
            continue
        loglike = loglike + (
            campaign.conservative_independent_states
            * log_single_state_miss(mean_flux_grid, campaign.threshold_jy)
        )

    idx = int(np.argmax(loglike))
    # Equal spacing in log(mean flux), with a log-uniform prior:
    weights = np.exp(loglike - np.max(loglike))
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    lo = float(mean_flux_grid[np.searchsorted(cdf, 0.025)])
    median = float(mean_flux_grid[np.searchsorted(cdf, 0.5)])
    hi = float(mean_flux_grid[np.searchsorted(cdf, 0.975)])

    return {
        "loglike": loglike,
        "map_mean_flux_jy": float(mean_flux_grid[idx]),
        "posterior_median_jy": median,
        "posterior_95_low_jy": lo,
        "posterior_95_high_jy": hi,
        "max_loglike": float(loglike[idx]),
    }


def build_steady_results(
    campaigns: list[Campaign],
    observed_flux: float,
) -> tuple[pd.DataFrame, dict[str, dict], np.ndarray]:
    mean_grid = np.logspace(-4, 3.5, 30_000)

    by_id = {c.campaign_id: c for c in campaigns}
    chains = {
        "Original event only": [],
        "+ Big Ear 89 null visits": [by_id["BIGEAR"]],
        "+ VLA 1995 and 1996": [by_id["BIGEAR"], by_id["VLA1995"], by_id["VLA1996"]],
        "+ Hobart": [
            by_id["BIGEAR"], by_id["VLA1995"], by_id["VLA1996"], by_id["HOBART"]
        ],
        "+ ATA (nominal-position chain)": [
            by_id["BIGEAR"], by_id["VLA1995"], by_id["VLA1996"],
            by_id["HOBART"], by_id["ATA"]
        ],
        "Robust full-field chain: Big Ear + ATA": [by_id["BIGEAR"], by_id["ATA"]],
        "Ultra-narrow chain including META": [
            by_id["BIGEAR"], by_id["META"], by_id["VLA1995"],
            by_id["VLA1996"], by_id["HOBART"], by_id["ATA"]
        ],
    }

    evaluated = {
        name: evaluate_chain(mean_grid, observed_flux, members)
        for name, members in chains.items()
    }
    baseline_max = evaluated["Original event only"]["max_loglike"]

    rows = []
    for name, result in evaluated.items():
        rows.append({
            "model_chain": name,
            "observed_peak_jy": observed_flux,
            "map_mean_flux_jy": result["map_mean_flux_jy"],
            "posterior_median_jy": result["posterior_median_jy"],
            "posterior_95_low_jy": result["posterior_95_low_jy"],
            "posterior_95_high_jy": result["posterior_95_high_jy"],
            "log10_max_likelihood_suppression_vs_original_only":
                (result["max_loglike"] - baseline_max) / math.log(10.0),
        })
    return pd.DataFrame(rows), evaluated, mean_grid


def periodic_alias_sweep(visits: pd.DataFrame) -> pd.DataFrame:
    """
    Big Ear-only alias stress test.

    Assumes visits occur at the same local sidereal time. The initial pulse is
    centered on the discovery window. A later detection requires a periodic
    top-hat pulse to fully cover a 72-second observing window.
    """
    periods_h = np.linspace(1.0, 48.0, 60_000)
    null_times = visits.loc[~visits["is_discovery"], "same_sidereal_time_offset_s"].to_numpy()

    rows = []
    for duration_s in [144.0, 288.0]:
        tolerance = max(0.0, (duration_s - OBS_WINDOW_S) / 2.0)
        for period_h in periods_h:
            period_s = period_h * 3600.0

            phase1 = np.mod(null_times + period_s / 2.0, period_s) - period_s / 2.0
            phase2 = np.mod(
                null_times + HORN_OFFSET_S + period_s / 2.0, period_s
            ) - period_s / 2.0

            detections = np.logical_or(
                np.abs(phase1) <= tolerance,
                np.abs(phase2) <= tolerance,
            )
            rows.append({
                "period_hours": period_h,
                "pulse_duration_s": duration_s,
                "predicted_repeat_visit_count": int(detections.sum()),
                "survives_bigear_only": int(detections.sum()) == 0,
            })
    return pd.DataFrame(rows)


def campaign_elapsed_seconds(representative_date: str) -> float:
    dt = datetime.fromisoformat(representative_date)
    return abs((dt - DISCOVERY_DATE).total_seconds())


def frequency_drift_sweep(
    visits: pd.DataFrame,
    campaigns: list[Campaign],
    observed_flux: float,
) -> pd.DataFrame:
    """
    Toy constant secular drift model.

    A campaign is included only while the line remains inside its quoted usable
    frequency margin. This does not model acceleration reversals, retuning, or
    each pipeline's short-integration drift response.
    """
    mean_grid = np.logspace(-4, 3.5, 8_000)
    original_max = float(np.max(log_original_density(mean_grid, observed_flux)))
    by_id = {c.campaign_id: c for c in campaigns}

    drift_rates = np.logspace(-8, 2, 350)
    rows = []
    null_visits = visits.loc[~visits["is_discovery"]].copy()
    elapsed_visit_s = np.abs(
        (null_visits["date"] - pd.Timestamp(DISCOVERY_DATE)).dt.total_seconds().to_numpy()
    )

    for rate in drift_rates:
        n_bigear = int(
            np.sum(rate * elapsed_visit_s <= by_id["BIGEAR"].usable_frequency_margin_hz)
        )

        active = []
        if n_bigear > 0:
            c = by_id["BIGEAR"]
            active.append((c.threshold_jy, n_bigear, "BIGEAR"))

        for campaign_id in ["VLA1995", "VLA1996", "HOBART", "ATA"]:
            c = by_id[campaign_id]
            drift_hz = rate * campaign_elapsed_seconds(c.representative_date)
            if drift_hz <= c.usable_frequency_margin_hz:
                active.append(
                    (c.threshold_jy, c.conservative_independent_states, campaign_id)
                )

        loglike = log_original_density(mean_grid, observed_flux)
        active_ids = []
        for threshold, nstates, campaign_id in active:
            if threshold is None:
                continue
            loglike += nstates * log_single_state_miss(mean_grid, threshold)
            active_ids.append(campaign_id)

        rows.append({
            "constant_secular_drift_hz_per_s": rate,
            "bigear_null_visits_still_in_band": n_bigear,
            "active_campaigns": ",".join(active_ids),
            "active_campaign_count": len(active_ids),
            "log10_max_likelihood_suppression_vs_original_only":
                (float(np.max(loglike)) - original_max) / math.log(10.0),
            "drift_during_original_72s_hz": rate * OBS_WINDOW_S,
            "fits_original_10khz_channel": rate * OBS_WINDOW_S <= 10_000.0,
        })
    return pd.DataFrame(rows)


def make_plots(
    campaigns_df: pd.DataFrame,
    steady_df: pd.DataFrame,
    evaluated: dict[str, dict],
    mean_grid: np.ndarray,
    alias_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    observed_flux: float,
) -> None:
    # Campaign depth
    plot_df = campaigns_df[
        campaigns_df["threshold_jy"].notna()
        & campaigns_df["campaign_id"].isin(
            ["BIGEAR", "META", "VLA1995", "VLA1996", "HOBART", "ATA"]
        )
    ].copy()
    plot_df["threshold_over_wow"] = plot_df["threshold_jy"] / observed_flux

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["campaign_id"], plot_df["threshold_over_wow"])
    plt.yscale("log")
    plt.ylabel("Detection threshold / 54 Jy Wow peak")
    plt.xlabel("Campaign")
    plt.title("Published follow-up depth is highly unequal")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "wow_signal_v5_campaign_depth.png", dpi=180)
    plt.close()

    # Steady-source posterior/relative likelihood
    plt.figure(figsize=(9, 5))
    selected = [
        "Original event only",
        "+ Big Ear 89 null visits",
        "+ VLA 1995 and 1996",
        "Robust full-field chain: Big Ear + ATA",
        "+ ATA (nominal-position chain)",
    ]
    for name in selected:
        loglike = evaluated[name]["loglike"]
        relative = np.exp(loglike - np.max(loglike))
        plt.plot(mean_grid, relative, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-12, 1.2)
    plt.xlabel("Underlying steady mean flux (Jy)")
    plt.ylabel("Relative likelihood within each chain")
    plt.title("Conditioning on the original event and follow-up nulls")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "wow_signal_v5_steady_source_likelihood.png", dpi=180)
    plt.close()

    # Alias map
    plt.figure(figsize=(10, 5))
    for duration in sorted(alias_df["pulse_duration_s"].unique()):
        subset = alias_df[alias_df["pulse_duration_s"] == duration]
        plt.plot(
            subset["period_hours"],
            subset["predicted_repeat_visit_count"] + 1,
            label=f"{duration:.0f} s pulse",
        )
    plt.yscale("log")
    plt.xlabel("Strict repetition period (hours)")
    plt.ylabel("Predicted Big Ear repeat visits + 1")
    plt.title("Same-sidereal-time sampling leaves substantial periodic alias space")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "wow_signal_v5_bigear_alias_map.png", dpi=180)
    plt.close()

    # Frequency drift
    plt.figure(figsize=(9, 5))
    plt.plot(
        drift_df["constant_secular_drift_hz_per_s"],
        drift_df["log10_max_likelihood_suppression_vs_original_only"],
    )
    plt.xscale("log")
    plt.xlabel("Assumed constant secular drift (Hz/s)")
    plt.ylabel("log10 maximum-likelihood suppression")
    plt.title("A monotonic chirp can escape historical frequency windows")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "wow_signal_v5_frequency_drift_escape.png", dpi=180)
    plt.close()

    # Effective-state stress test for ATA
    ata = campaigns_df.loc[campaigns_df["campaign_id"] == "ATA"].iloc[0]
    mean_fluxes = [1.0, 3.0, 10.0, 54.0]
    states = np.arange(1, 61)
    plt.figure(figsize=(9, 5))
    for mean_flux in mean_fluxes:
        p1 = -math.expm1(-float(ata["threshold_jy"]) / mean_flux)
        plt.plot(states, np.maximum(p1 ** states, 1e-300), label=f"mean={mean_flux:g} Jy")
    plt.yscale("log")
    plt.xlabel("Independent ATA scintillation states")
    plt.ylabel("Probability every state remains below threshold")
    plt.title("ATA persistence constraint versus unknown scintillation correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "wow_signal_v5_ata_state_stress.png", dpi=180)
    plt.close()


def write_report(
    campaigns_df: pd.DataFrame,
    steady54: pd.DataFrame,
    steady212: pd.DataFrame,
    alias_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> None:
    def row(df: pd.DataFrame, name: str) -> pd.Series:
        return df.loc[df["model_chain"] == name].iloc[0]

    nominal54 = row(steady54, "+ ATA (nominal-position chain)")
    full54 = row(steady54, "Robust full-field chain: Big Ear + ATA")
    vla54 = row(steady54, "+ VLA 1995 and 1996")
    nominal212 = row(steady212, "+ ATA (nominal-position chain)")

    alias_summary = (
        alias_df.groupby("pulse_duration_s")["survives_bigear_only"]
        .mean()
        .reset_index()
    )

    # Drift where suppression becomes weaker than 1e-2 and 1e-1.
    weak_2 = drift_df.loc[
        drift_df["log10_max_likelihood_suppression_vs_original_only"] > -2
    ]
    drift_weak_2 = (
        float(weak_2["constant_secular_drift_hz_per_s"].min())
        if not weak_2.empty else float("nan")
    )

    report = f"""
WOW! SIGNAL SIMULATION V5 — CAMPAIGN-BY-CAMPAIGN PERSISTENCE
=============================================================

Question
--------
Could a steady, angularly compact radio source have produced the original
Wow! peak through strong scintillation and then remained hidden during the
published follow-up observations?

Model
-----
For saturated diffractive scintillation, instantaneous intensity is modeled as
an exponential random variable with mean S.

Original-event density:
    f(I_wow | S) = exp(-I_wow/S) / S

One independent follow-up state below threshold L:
    P(miss | S) = 1 - exp(-L/S)

For N independent states:
    P(all misses | S) = P(miss | S)^N

The calculation uses the conservative 54 Jy Wow estimate. It is repeated at
212 Jy as a sensitivity check. A log-uniform prior is used only for the
illustrative credible intervals; the headline suppression values are maximum
likelihood ratios relative to fitting the original event alone.

Campaign inputs
---------------
{campaigns_df[[
    'campaign_id', 'date_label', 'total_hours',
    'conservative_independent_states', 'channel_width_hz',
    'threshold_jy', 'core_10khz_applicable', 'confidence'
]].to_string(index=False)}

Key 54 Jy results
-----------------
Big Ear plus the two VLA epochs:
    best steady mean flux = {vla54['map_mean_flux_jy']:.4f} Jy
    maximum-likelihood suppression = 10^(
        {vla54['log10_max_likelihood_suppression_vs_original_only']:.4f}
    )

Nominal-position chain through ATA:
    best steady mean flux = {nominal54['map_mean_flux_jy']:.4f} Jy
    maximum-likelihood suppression = 10^(
        {nominal54['log10_max_likelihood_suppression_vs_original_only']:.4f}
    )

Robust full-field chain using only Big Ear and ATA:
    best steady mean flux = {full54['map_mean_flux_jy']:.4f} Jy
    maximum-likelihood suppression = 10^(
        {full54['log10_max_likelihood_suppression_vs_original_only']:.4f}
    )

The robust chain deliberately discards the VLA and Hobart geometry advantage.
It still suppresses a fixed-frequency steady scintillating source by roughly
nine orders of magnitude at its own best-fitting mean flux.

High-flux calibration stress test
---------------------------------
At 212 Jy, the nominal-position chain gives:
    best steady mean flux = {nominal212['map_mean_flux_jy']:.4f} Jy
    maximum-likelihood suppression = 10^(
        {nominal212['log10_max_likelihood_suppression_vs_original_only']:.4f}
    )

A brighter original event makes the steady-source explanation less compatible,
not more.

Why the VLA matters
-------------------
The conservative modeled VLA thresholds are 0.0440 Jy in 1995 and 0.0296 Jy
in 1996, roughly 1,200-1,800 times below the conservative 54 Jy event.
At the nominal positions, those two deep epochs dominate the test of a weak
underlying source occasionally boosted by scintillation.

Why ATA matters
---------------
ATA covered the full consistent-direction field, used a comparable 12.8 kHz
channel width, and accumulated more than 100 hours. Since sessions were no
longer than three hours, at least 34 sessions were required. Treating each
whole session as only one independent scintillation state is intentionally
conservative.

Periodic alias caveat
---------------------
The Big Ear-only periodic test uses the published 90 visit dates and assumes
the telescope sampled the source at the same local sidereal time. The fraction
of periods from 1-48 hours producing zero predicted repeats is:

{alias_summary.to_string(index=False)}

These results show substantial periodic alias space for short pulses. The
surviving phase intervals can be narrow but numerous; they are not evidence
for periodicity. Dedicated tracking campaigns at other local sidereal times
are valuable because they break this same-time sampling pattern.

Constant frequency-drift caveat
-------------------------------
A toy monotonic chirp can leave each historical receiver band. In this simple
model, the combined fixed-frequency constraint begins weakening above roughly
{drift_weak_2:.6g} Hz/s. This is not a normal steady spectral line: maintaining
one-signed drift for years or decades is a different source hypothesis, and
large drift also changes how each observing pipeline would integrate the line.

Verdict
-------
EFFECTIVELY REJECTED, WITH EXPLICIT CONDITIONS:
A continuously emitting, fixed-frequency, compact source whose only
intermittency is ordinary strong scintillation, provided it lay in the
published search region and retained a Wow-like frequency and bandwidth.

STILL LIVE:
An intrinsically one-time transient.
A low-duty-cycle or stochastic repeater.
A source that changed frequency substantially between campaigns.
A source outside the partial fields of the earlier narrow-field campaigns,
although ATA covered the full reconstructed field.
A nonlinear instrumental or asymmetric-sidelobe artifact.

NOT RESCUED:
An extended or broadband source. Deep diffractive scintillation requires a
compact emitter and does not repair the bandwidth failure.

Limitations
-----------
1. The 5-sigma Big Ear follow-up threshold is a model assumption and is swept
   indirectly through the supplied script constants.
2. Independent scintillation states are counted conservatively by visit-day or
   observing session, not by every integration.
3. The VLA result applies most cleanly at the two nominal positions. The
   Big Ear+ATA chain is reported separately to avoid overstating older spatial
   coverage.
4. META is only applicable to an ultra-narrow Doppler-corrected source and is
   excluded from the 10 kHz natural-line chain.
5. The frequency-drift module assumes a constant one-signed secular chirp and
   does not reproduce every pipeline's drift-search response.
6. A complete historical posterior would require raw observation start/stop
   times, exact per-session thresholds, frequency masks, RFI excision masks,
   and the full ATA session log.
""".strip()

    (OUT / "wow_signal_v5_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    campaigns54 = build_campaigns(WOW_CONSERVATIVE_JY)
    campaigns_df = campaign_dataframe(campaigns54)
    visits = bigear_visit_dataframe()

    steady54, evaluated54, mean_grid54 = build_steady_results(
        campaigns54, WOW_CONSERVATIVE_JY
    )
    campaigns212 = build_campaigns(WOW_HIGH_JY)
    steady212, _, _ = build_steady_results(campaigns212, WOW_HIGH_JY)

    alias_df = periodic_alias_sweep(visits)
    drift_df = frequency_drift_sweep(
        visits, campaigns54, WOW_CONSERVATIVE_JY
    )

    campaigns_df.to_csv(OUT / "wow_signal_v5_campaigns.csv", index=False)
    visits.to_csv(OUT / "wow_signal_v5_bigear_visits.csv", index=False)
    steady54.to_csv(OUT / "wow_signal_v5_steady_results_54jy.csv", index=False)
    steady212.to_csv(OUT / "wow_signal_v5_steady_results_212jy.csv", index=False)
    alias_df.to_csv(OUT / "wow_signal_v5_bigear_alias_sweep.csv", index=False)
    drift_df.to_csv(OUT / "wow_signal_v5_frequency_drift_sweep.csv", index=False)

    make_plots(
        campaigns_df,
        steady54,
        evaluated54,
        mean_grid54,
        alias_df,
        drift_df,
        WOW_CONSERVATIVE_JY,
    )
    write_report(campaigns_df, steady54, steady212, alias_df, drift_df)

    package = OUT / "wow_signal_v5_package.zip"
    generated = [
        Path(__file__).resolve(),
        OUT / "wow_signal_v5_report.txt",
        OUT / "wow_signal_v5_campaigns.csv",
        OUT / "wow_signal_v5_bigear_visits.csv",
        OUT / "wow_signal_v5_steady_results_54jy.csv",
        OUT / "wow_signal_v5_steady_results_212jy.csv",
        OUT / "wow_signal_v5_bigear_alias_sweep.csv",
        OUT / "wow_signal_v5_frequency_drift_sweep.csv",
        OUT / "wow_signal_v5_campaign_depth.png",
        OUT / "wow_signal_v5_steady_source_likelihood.png",
        OUT / "wow_signal_v5_bigear_alias_map.png",
        OUT / "wow_signal_v5_frequency_drift_escape.png",
        OUT / "wow_signal_v5_ata_state_stress.png",
    ]
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in generated:
            if path.exists():
                zf.write(path, arcname=path.name)

    print("V5 completed.")
    print(steady54.to_string(index=False))
    print()
    print(f"Package: {package}")


if __name__ == "__main__":
    main()
