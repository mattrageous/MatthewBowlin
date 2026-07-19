#!/usr/bin/env python3
"""
Wow! signal plausibility simulator

Purpose
-------
Test this specific hypothesis:

    A natural cosmic radio signal was altered by the combined effects of
    Earth/Sun/Moon gravity, the ionosphere, weather, terrestrial radio
    transmissions, and the Big Ear receiver, producing a Wow!-like detection.

This is a mechanism test, not a reconstruction of August 15, 1977. Exact
ionospheric electron-density maps, complete transmitter logs, raw voltages,
receiver transfer functions, and phase information are unavailable.

Main result
-----------
Linear propagation can alter amplitude, phase, path, arrival time, and observed
frequency, but it does not compress an ordinary broadband signal into a single
10 kHz channel. A pre-existing narrowband cosmic transient can survive these
effects and look Wow-like. A human-made signal can also create an in-band
artifact through receiver nonlinearity/harmonics, but that is an instrumental
RFI explanation rather than the sky and gravity "blending" two signals.

Dependencies: numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf


# ---------------------------
# Constants and observations
# ---------------------------

C = 299_792_458.0
G = 6.67430e-11
K_B = 1.380649e-23

M_EARTH = 5.9722e24
R_EARTH = 6.371e6
M_SUN = 1.98847e30
AU = 149_597_870_700.0
M_MOON = 7.342e22
D_MOON = 384_400e3

HI_REST_HZ = 1_420_405_751.768

# Two values are retained because the 2025 archival reanalysis revised the
# older commonly used estimate.
WOW_LEGACY_HZ = 1_420_455_600.0
WOW_REVISED_HZ = 1_420_726_000.0

# Midpoints of six 10-second integrations separated by ~2 seconds processing.
WOW_TIMES_S = np.array([5, 17, 29, 41, 53, 65], dtype=float)

# Midpoints of the integer ranges encoded by 6 E Q U J 5.
WOW_SNR = np.array([6.5, 14.5, 26.5, 30.5, 19.5, 5.5], dtype=float)


@dataclass(frozen=True)
class Config:
    target_hz: float = WOW_REVISED_HZ
    channel_width_hz: float = 10_000.0
    n_channels: int = 50
    latitude_deg: float = 40.2567
    event_utc: datetime = datetime(1977, 8, 16, 3, 16, tzinfo=timezone.utc)
    seed: int = 0
    horn_peak_separation_s: float = 172.37
    wow_peak_snr: float = 30.5
    nondetection_snr: float = 1.0
    notional_system_temperature_k: float = 100.0


def gaussian(t: np.ndarray, amplitude: float, center: float,
             sigma: float, baseline: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def fit_wow_beam() -> tuple[np.ndarray, np.ndarray, float]:
    p0 = [29.0, 37.0, 15.0, 2.0]
    bounds = ([0.0, 0.0, 1.0, -10.0], [100.0, 72.0, 100.0, 10.0])
    params, _ = curve_fit(
        gaussian, WOW_TIMES_S, WOW_SNR, p0=p0, bounds=bounds
    )
    fitted = gaussian(WOW_TIMES_S, *params)
    correlation = float(np.corrcoef(WOW_SNR, fitted)[0, 1])
    return params, fitted, correlation


def ou_process(n: int, dt: float, tau: float,
               sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck process used as a toy scintillation model."""
    x = np.zeros(n)
    alpha = math.exp(-dt / tau)
    innovation_sigma = sigma * math.sqrt(1.0 - alpha**2)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + innovation_sigma * rng.normal()
    return x


def simulate_temporal_trace(
    params: np.ndarray,
    seed: int = 0,
    scintillation_s4: float = 0.03,
    receiver_gain_depth: float = 0.02,
    tropospheric_gain_depth: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    A point-like celestial source through:
      - telescope drift-scan beam
      - weak ionospheric scintillation
      - slow receiver cross-gain modulation
      - weak tropospheric amplitude variation

    These factors alter amplitude but do not create a new carrier frequency.
    """
    dt = 0.1
    t = np.arange(0.0, 72.0, dt)
    amplitude, center, sigma, baseline = params

    beam = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    rng = np.random.default_rng(seed)

    log_scint = ou_process(
        n=len(t), dt=dt, tau=20.0, sigma=scintillation_s4, rng=rng
    )
    scintillation = np.exp(log_scint - 0.5 * scintillation_s4**2)

    receiver_gain = 1.0 + receiver_gain_depth * np.sin(
        2.0 * np.pi * t / 47.0 + 0.8
    )
    troposphere = 1.0 + tropospheric_gain_depth * np.sin(
        2.0 * np.pi * t / 130.0 + 1.1
    )

    trace = baseline + amplitude * beam * scintillation * receiver_gain * troposphere

    integrated = []
    for start in [0, 12, 24, 36, 48, 60]:
        mask = (t >= start) & (t < start + 10)
        integrated.append(float(trace[mask].mean()))
    integrated = np.array(integrated)

    correlation = float(np.corrcoef(WOW_SNR, integrated)[0, 1])
    rmse = float(np.sqrt(np.mean((WOW_SNR - integrated) ** 2)))
    return t, trace, integrated, correlation, rmse


def channel_powers(
    source_fwhm_hz: float,
    channel_width_hz: float = 10_000.0,
    n_channels: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a Gaussian spectrum over receiver channels."""
    sigma = source_fwhm_hz / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    center_index = n_channels // 2
    centers = (np.arange(n_channels) - center_index) * channel_width_hz
    lower = centers - channel_width_hz / 2.0
    upper = centers + channel_width_hz / 2.0

    powers = 0.5 * (
        erf(upper / (math.sqrt(2.0) * sigma))
        - erf(lower / (math.sqrt(2.0) * sigma))
    )
    return centers, powers


def gravitational_shifts(target_hz: float) -> dict[str, float]:
    """
    Weak-field gravitational frequency terms at Earth's location.

    These are local potential-scale terms, not a complete source-to-observer
    relativistic model. Their purpose is to establish magnitude.
    """
    fractional = {
        "Earth gravity": G * M_EARTH / (R_EARTH * C**2),
        "Sun gravity at Earth": G * M_SUN / (AU * C**2),
        "Moon gravity at Earth": G * M_MOON / (D_MOON * C**2),
    }
    return {name: target_hz * value for name, value in fractional.items()}


def earth_motion_drift_72s(target_hz: float, latitude_deg: float) -> dict[str, float]:
    """
    Generous upper-scale drift estimates during the 72-second observation.

    Constant orbital/rotational velocity can shift the absolute frequency by
    kHz, but the change during 72 seconds is only on the order of Hz.
    """
    omega_earth = 7.2921159e-5
    latitude = math.radians(latitude_deg)
    rotational_speed = omega_earth * R_EARTH * math.cos(latitude)
    rotational_acceleration_scale = rotational_speed * omega_earth

    orbital_speed = 29_780.0
    orbital_angular_speed = 2.0 * math.pi / (365.256 * 86_400.0)
    orbital_acceleration_scale = orbital_speed * orbital_angular_speed

    return {
        "Earth rotation drift over 72 s":
            target_hz * rotational_acceleration_scale * 72.0 / C,
        "Earth orbit drift over 72 s":
            target_hz * orbital_acceleration_scale * 72.0 / C,
    }


def ionosphere_metrics(target_hz: float, tec_tecu: float = 10.0,
                       peak_electron_density_m3: float = 1e12) -> dict[str, float]:
    """
    Representative ionosphere values.

    The ionosphere changes phase, delay, refraction, and amplitude. In a
    time-stationary linear plasma, it does not translate an HF signal to L-band.
    """
    tec = tec_tecu * 1e16
    group_delay_s = 40.3 * tec / (C * target_hz**2)

    # Plasma frequency in Hz when electron density is in m^-3.
    plasma_frequency_hz = 8.98 * math.sqrt(peak_electron_density_m3)
    refractive_index_departure = 0.5 * (plasma_frequency_hz / target_hz) ** 2

    return {
        "Group delay at 10 TECU (ns)": group_delay_s * 1e9,
        "Peak plasma frequency (MHz)": plasma_frequency_hz / 1e6,
        "|n - 1| scale": refractive_index_departure,
    }


def julian_day(dt: datetime) -> float:
    return dt.timestamp() / 86_400.0 + 2_440_587.5


def approximate_sun_radec(dt: datetime) -> tuple[float, float]:
    """Low-precision solar position, adequate for an alignment sanity check."""
    n = julian_day(dt) - 2_451_545.0
    mean_longitude = math.radians((280.460 + 0.9856474 * n) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = (
        mean_longitude
        + math.radians(1.915) * math.sin(mean_anomaly)
        + math.radians(0.020) * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)

    ra = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_longitude),
        math.cos(ecliptic_longitude),
    ) % (2.0 * math.pi)
    dec = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))
    return math.degrees(ra) / 15.0, math.degrees(dec)


def angular_separation_deg(
    ra1_hours: float, dec1_deg: float, ra2_hours: float, dec2_deg: float
) -> float:
    ra1 = math.radians(ra1_hours * 15.0)
    ra2 = math.radians(ra2_hours * 15.0)
    dec1 = math.radians(dec1_deg)
    dec2 = math.radians(dec2_deg)
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))



def gravitational_dynamic_drift_72s(
    target_hz: float, duration_s: float = 72.0
) -> dict[str, float]:
    """Very generous upper bounds on time-varying local gravity over 72 s.

    Static gravitational potential is common to ordinary observations and does
    not create a transient. Only the tiny change in potential during the event
    can contribute to drift. The radial speeds below are deliberately generous.
    """
    static = gravitational_shifts(target_hz)
    sun_radial_speed_upper_m_s = 1_000.0
    moon_radial_speed_upper_m_s = 100.0
    return {
        "Earth local potential drift over 72 s": 0.0,
        "Sun potential drift upper bound over 72 s":
            static["Sun gravity at Earth"]
            * sun_radial_speed_upper_m_s * duration_s / AU,
        "Moon potential drift upper bound over 72 s":
            static["Moon gravity at Earth"]
            * moon_radial_speed_upper_m_s * duration_s / D_MOON,
    }


def thermal_noise_power_dbm(temperature_k: float, bandwidth_hz: float) -> float:
    watts = K_B * temperature_k * bandwidth_hz
    return 10.0 * math.log10(watts / 1e-3)


def intermodulation_metrics(config: Config) -> dict:
    """Compute low-order two-tone spur locations and notional level thresholds.

    The level calculation is only a scale test because Big Ear's exact 1977
    front-end IIP2/IIP3, filtering, system temperature, and calibration are not
    available. Equal-power tones are assumed.
    """
    noise_dbm = thermal_noise_power_dbm(
        config.notional_system_temperature_k, config.channel_width_hz
    )
    peak_excess_db = 10.0 * math.log10(config.wow_peak_snr)
    target_spur_dbm = noise_dbm + peak_excess_db

    iip3_dbm = np.linspace(-30.0, 30.0, 121)
    required_pin_im3_dbm = (target_spur_dbm + 2.0 * iip3_dbm) / 3.0

    iip2_dbm = np.linspace(-10.0, 50.0, 121)
    required_pin_im2_dbm = (target_spur_dbm + iip2_dbm) / 2.0

    # Algebraic examples only: they are not claims that these transmitters were
    # operating at Big Ear in 1977.
    f1_im2 = 700_000_000.0
    f2_im2 = config.target_hz - f1_im2

    f1_im3a = 900_000_000.0
    f2_im3a = 2.0 * f1_im3a - config.target_hz  # 2f1 - f2 = target

    f2_im3b = 800_000_000.0
    f1_im3b = 2.0 * f2_im3b - config.target_hz  # 2f2 - f1 = target

    examples = pd.DataFrame([
        {
            "Product": "Second order: f1 + f2",
            "f1_MHz": f1_im2 / 1e6,
            "f2_MHz": f2_im2 / 1e6,
            "output_MHz": (f1_im2 + f2_im2) / 1e6,
            "frequency_error_Hz": f1_im2 + f2_im2 - config.target_hz,
        },
        {
            "Product": "Third order: 2f1 - f2",
            "f1_MHz": f1_im3a / 1e6,
            "f2_MHz": f2_im3a / 1e6,
            "output_MHz": (2.0 * f1_im3a - f2_im3a) / 1e6,
            "frequency_error_Hz": 2.0 * f1_im3a - f2_im3a - config.target_hz,
        },
        {
            "Product": "Third order: 2f2 - f1",
            "f1_MHz": f1_im3b / 1e6,
            "f2_MHz": f2_im3b / 1e6,
            "output_MHz": (2.0 * f2_im3b - f1_im3b) / 1e6,
            "frequency_error_Hz": 2.0 * f2_im3b - f1_im3b - config.target_hz,
        },
    ])

    representative_iip3_dbm = -10.0
    representative_iip2_dbm = 20.0
    pin_im3_at_rep = (
        target_spur_dbm + 2.0 * representative_iip3_dbm
    ) / 3.0
    pin_im2_at_rep = (
        target_spur_dbm + representative_iip2_dbm
    ) / 2.0

    return {
        "noise_dbm": noise_dbm,
        "peak_excess_db": peak_excess_db,
        "target_spur_dbm": target_spur_dbm,
        "iip3_dbm": iip3_dbm,
        "required_pin_im3_dbm": required_pin_im3_dbm,
        "iip2_dbm": iip2_dbm,
        "required_pin_im2_dbm": required_pin_im2_dbm,
        "representative_iip3_dbm": representative_iip3_dbm,
        "representative_iip2_dbm": representative_iip2_dbm,
        "pin_im3_at_rep": pin_im3_at_rep,
        "pin_im2_at_rep": pin_im2_at_rep,
        "examples": examples,
        "max_oth_im2_hz": 60e6,
        "max_oth_im3_hz": 90e6,
    }


def horn_constraint_metrics(config: Config, beam_sigma_s: float) -> dict:
    """Quantify the one-horn constraint for transient and common-mode cases."""
    separation = config.horn_peak_separation_s
    peak_snr = config.wow_peak_snr
    threshold = config.nondetection_snr

    minimum_fade_factor = peak_snr / threshold
    minimum_fade_db = 10.0 * math.log10(minimum_fade_factor)
    max_decay_tau_s = separation / math.log(minimum_fade_factor)

    decay_tau_s = np.linspace(5.0, 300.0, 500)
    second_horn_snr = peak_snr * np.exp(-separation / decay_tau_s)

    # For a top-hat transient, require the second horn's Gaussian response at
    # switch-off to remain below 1 sigma. This gives the latest safe shutoff.
    sigma_offsets = math.sqrt(2.0 * math.log(minimum_fade_factor))
    latest_safe_shutoff_after_first_peak_s = separation - beam_sigma_s * sigma_offsets

    fitted_center = fit_wow_beam()[0][1]
    last_observed_offset_s = float(WOW_TIMES_S.max() - fitted_center)
    available_shutoff_window_s = (
        latest_safe_shutoff_after_first_peak_s - last_observed_offset_s
    )

    mismatch_fraction = np.logspace(-4.0, math.log10(0.5), 400)
    required_common_mode_snr = peak_snr / mismatch_fraction

    reference_mismatches = np.array([0.001, 0.01, 0.03, 0.10])
    reference_common_snr = peak_snr / reference_mismatches
    mismatch_table = pd.DataFrame({
        "Horn amplitude mismatch_percent": reference_mismatches * 100.0,
        "Required common-mode RFI_SNR": reference_common_snr,
    })

    return {
        "minimum_fade_factor": minimum_fade_factor,
        "minimum_fade_db": minimum_fade_db,
        "max_decay_tau_s": max_decay_tau_s,
        "decay_tau_s": decay_tau_s,
        "second_horn_snr": second_horn_snr,
        "latest_safe_shutoff_after_first_peak_s":
            latest_safe_shutoff_after_first_peak_s,
        "last_observed_offset_s": last_observed_offset_s,
        "available_shutoff_window_s": available_shutoff_window_s,
        "mismatch_fraction": mismatch_fraction,
        "required_common_mode_snr": required_common_mode_snr,
        "mismatch_table": mismatch_table,
        "continuous_second_horn_snr": peak_snr,
    }


def build_results(config: Config) -> tuple[pd.DataFrame, dict]:
    params, beam_fit, beam_corr = fit_wow_beam()
    t, trace, temporal_samples, temporal_corr, temporal_rmse = (
        simulate_temporal_trace(params, seed=config.seed)
    )

    broad_centers, broad_power = channel_powers(
        1_000_000.0, config.channel_width_hz, config.n_channels
    )
    narrow_centers, narrow_power = channel_powers(
        5_000.0, config.channel_width_hz, config.n_channels
    )
    center = config.n_channels // 2

    broad_adjacent_ratio = float(broad_power[center + 1] / broad_power[center])
    narrow_adjacent_ratio = float(narrow_power[center + 1] / narrow_power[center])

    min_oth_harmonic_order = math.ceil(config.target_hz / 30e6)
    exact_oth_subharmonic_hz = config.target_hz / min_oth_harmonic_order
    uhf_second_harmonic_hz = config.target_hz / 2.0
    uhf_third_harmonic_hz = config.target_hz / 3.0

    imd = intermodulation_metrics(config)
    horns = horn_constraint_metrics(config, beam_sigma_s=float(params[2]))

    rows = [
        {
            "Scenario": "Broad cosmic source + linear propagation effects",
            "Bandwidth result": broad_adjacent_ratio,
            "Horn result": "Continuous source should appear in both horns",
            "Verdict": "Rejected",
            "Interpretation": (
                "The time trace can look beam-shaped, but adjacent channels remain "
                "almost equally bright and a steady source repeats in the second horn."
            ),
        },
        {
            "Scenario": "Intrinsically narrow cosmic transient + propagation effects",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": (
                f"Passes if it fades by >{horns['minimum_fade_db']:.1f} dB between horns"
            ),
            "Verdict": "Physically possible",
            "Interpretation": (
                f"A narrow transient survives the path. Exponential fading requires "
                f"tau < {horns['max_decay_tau_s']:.1f} s, or a switch-off in the "
                f"available ~{horns['available_shutoff_window_s']:.0f} s timing window."
            ),
        },
        {
            "Scenario": "3-30 MHz OTH signals through linear sky/receiver path",
            "Bandwidth result": np.nan,
            "Horn result": "No in-band product",
            "Verdict": "Rejected",
            "Interpretation": (
                f"Second- and third-order products from two HF tones top out at "
                f"{imd['max_oth_im3_hz']/1e6:.0f} MHz, far below 1420.726 MHz."
            ),
        },
        {
            "Scenario": "Very-high-order harmonic from a single OTH/HF carrier",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": "Would still need one-horn geometry",
            "Verdict": "Mathematically possible, physically strained",
            "Interpretation": (
                f"At least harmonic order {min_oth_harmonic_order} is required from "
                "a 30 MHz carrier; normal low-order receiver distortion does not do this."
            ),
        },
        {
            "Scenario": "Two strong UHF tones create IM2/IM3 in nonlinear electronics",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": "Conditional on where nonlinearity occurs and horn imbalance",
            "Verdict": "Possible instrumental artifact",
            "Interpretation": (
                f"Low-order products can land exactly in-band. In the notional "
                f"{config.notional_system_temperature_k:.0f} K model, IIP3="
                f"{imd['representative_iip3_dbm']:.0f} dBm requires about "
                f"{imd['pin_im3_at_rep']:.1f} dBm per tone at the nonlinear stage."
            ),
        },
        {
            "Scenario": "Common-mode terrestrial RFI entering both horns",
            "Bandwidth result": np.nan,
            "Horn result": "Normally suppressed by differential horn switching",
            "Verdict": "Possible only with large signal plus asymmetry",
            "Interpretation": (
                f"At 1% horn/product mismatch, each horn would need a roughly "
                f"{config.wow_peak_snr/0.01:.0f}-sigma common-mode artifact to leave "
                "a 30.5-sigma difference."
            ),
        },
        {
            "Scenario": "Continuous narrow celestial source",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": f"Predicts ~{horns['continuous_second_horn_snr']:.1f} sigma again",
            "Verdict": "Rejected by one-horn result under ideal geometry",
            "Interpretation": (
                "Narrow bandwidth alone is insufficient; a persistent fixed source "
                "should have generated the second horn response 172.37 seconds later."
            ),
        },
    ]

    sun_ra_h, sun_dec_deg = approximate_sun_radec(config.event_utc)
    wow_ra_h = 19.0 + 25.0 / 60.0 + 2.0 / 3600.0
    wow_dec_deg = -(26.0 + 57.0 / 60.0)
    sun_separation_deg = angular_separation_deg(
        sun_ra_h, sun_dec_deg, wow_ra_h, wow_dec_deg
    )

    details = {
        "params": params,
        "beam_fit": beam_fit,
        "beam_corr": beam_corr,
        "time": t,
        "trace": trace,
        "temporal_samples": temporal_samples,
        "temporal_corr": temporal_corr,
        "temporal_rmse": temporal_rmse,
        "broad_centers": broad_centers,
        "broad_power": broad_power,
        "narrow_centers": narrow_centers,
        "narrow_power": narrow_power,
        "gravity_static": gravitational_shifts(config.target_hz),
        "gravity_dynamic": gravitational_dynamic_drift_72s(config.target_hz),
        "motion_drift": earth_motion_drift_72s(config.target_hz, config.latitude_deg),
        "ionosphere": ionosphere_metrics(config.target_hz),
        "min_oth_harmonic_order": min_oth_harmonic_order,
        "exact_oth_subharmonic_hz": exact_oth_subharmonic_hz,
        "uhf_second_harmonic_hz": uhf_second_harmonic_hz,
        "uhf_third_harmonic_hz": uhf_third_harmonic_hz,
        "intermod": imd,
        "horns": horns,
        "sun_ra_h": sun_ra_h,
        "sun_dec_deg": sun_dec_deg,
        "sun_separation_deg": sun_separation_deg,
    }
    return pd.DataFrame(rows), details


def make_plots(config: Config, details: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(WOW_TIMES_S, WOW_SNR, "o-", label="6EQUJ5 midpoint SNR")
    plt.plot(WOW_TIMES_S, details["temporal_samples"], "s--",
             label="Combined-effects simulation")
    plt.xlabel("Seconds into the 72-second pass")
    plt.ylabel("Signal-to-noise ratio")
    plt.title("Temporal fit: expected, but weak as a discriminator")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_temporal_fit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x_khz = details["broad_centers"] / 1e3
    broad = details["broad_power"] / details["broad_power"].max()
    narrow = details["narrow_power"] / details["narrow_power"].max()
    plt.plot(x_khz, broad, label="1 MHz-wide natural source")
    plt.plot(x_khz, narrow, label="5 kHz-wide source")
    plt.xlabel("Offset from target frequency (kHz)")
    plt.ylabel("Normalized power per 10 kHz channel")
    plt.title("Bandwidth test: linear effects cannot compress a spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_spectral_test.png", dpi=180)
    plt.close()

    effects = {}
    effects.update(details["gravity_static"])
    effects.update(details["gravity_dynamic"])
    effects.update(details["motion_drift"])
    labels = list(effects.keys())
    values = [max(abs(effects[k]), 1e-9) for k in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.axhline(config.channel_width_hz, linestyle="--",
                label="10 kHz Big Ear channel")
    plt.yscale("log")
    plt.ylabel("Frequency scale (Hz)")
    plt.title("Static gravity is small; time-varying gravity is microscopic")
    plt.xticks(rotation=27, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_frequency_effects.png", dpi=180)
    plt.close()

    imd = details["intermod"]
    plt.figure(figsize=(8, 5))
    plt.plot(imd["iip3_dbm"], imd["required_pin_im3_dbm"])
    plt.xlabel("Receiver IIP3 (dBm)")
    plt.ylabel("Required input power per tone (dBm)")
    plt.title("Notional two-tone IM3 strength needed for a Wow-level spur")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_intermod_threshold.png", dpi=180)
    plt.close()

    horns = details["horns"]
    plt.figure(figsize=(8, 5))
    plt.plot(horns["decay_tau_s"], horns["second_horn_snr"])
    plt.axhline(config.nondetection_snr, linestyle="--",
                label="1-sigma non-detection threshold")
    plt.axvline(horns["max_decay_tau_s"], linestyle=":",
                label=f"Maximum decay tau: {horns['max_decay_tau_s']:.1f} s")
    plt.yscale("log")
    plt.xlabel("Transient exponential decay time, tau (s)")
    plt.ylabel("Predicted second-horn SNR")
    plt.title("One-horn constraint: the signal must change between beam passes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_horn_decay_test.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(horns["mismatch_fraction"] * 100.0,
             horns["required_common_mode_snr"])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Horn/product amplitude mismatch (%)")
    plt.ylabel("Required common-mode RFI level (sigma)")
    plt.title("Differential horns strongly suppress common-mode RFI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v2_common_mode_test.png", dpi=180)
    plt.close()


def write_report(config: Config, results: pd.DataFrame,
                 details: dict, output_dir: Path) -> None:
    gravity_static_total = sum(details["gravity_static"].values())
    gravity_dynamic_total = sum(details["gravity_dynamic"].values())
    motion_total = sum(details["motion_drift"].values())
    imd = details["intermod"]
    horns = details["horns"]

    report = f"""
WOW! SIGNAL COMBINED-EFFECTS SIMULATION — VERSION 2
===================================================

What was added
--------------
1. Two-tone second- and third-order intermodulation products.
2. A notional IIP2/IIP3 power threshold sweep.
3. The 172.37-second dual-horn separation constraint.
4. Transient fade/switch-off requirements for a one-horn detection.
5. Differential-horn rejection of common-mode terrestrial RFI.
6. Separation of static gravity from time-varying gravity.

Temporal profile
----------------
Combined-effects correlation with 6EQUJ5: {details['temporal_corr']:.6f}
RMSE: {details['temporal_rmse']:.3f} SNR units

This remains easy to reproduce and is not the decisive discriminator. A compact
source crossing a fixed beam naturally gives a bell-shaped response.

Bandwidth test
--------------
Broad 1 MHz source adjacent/peak channel ratio:
{results.loc[0, 'Bandwidth result']:.6f}

Narrow 5 kHz source adjacent/peak channel ratio:
{results.loc[1, 'Bandwidth result']:.6f}

Result: linear propagation still cannot compress a broadband source into one
10 kHz channel. An intrinsically narrow source or a nonlinear electronic spur
is still required.

Gravity and motion
------------------
Static Earth/Sun/Moon potential-scale term: {gravity_static_total:.6f} Hz
Time-varying gravity upper bound over 72 s: {gravity_dynamic_total:.9f} Hz
Rotation/orbit drift scale over 72 s: {motion_total:.6f} Hz

The static gravitational term is common to ordinary observations; it cannot
create a one-time event. The differential gravity during the observation is
microscopic.

Two-tone intermodulation
------------------------
Notional system temperature: {config.notional_system_temperature_k:.1f} K
10 kHz thermal-noise power: {imd['noise_dbm']:.2f} dBm
30.5-sigma excess expressed as a power ratio: {imd['peak_excess_db']:.2f} dB
Notional target spur level: {imd['target_spur_dbm']:.2f} dBm

With equal tones and a representative IIP3 of
{imd['representative_iip3_dbm']:.1f} dBm, the approximate per-tone input needed
at the nonlinear stage is {imd['pin_im3_at_rep']:.2f} dBm.

With representative IIP2 of {imd['representative_iip2_dbm']:.1f} dBm, the
approximate equal-tone input needed for an IM2 sum product is
{imd['pin_im2_at_rep']:.2f} dBm per tone.

These are very strong receiver-input levels, but they are not impossible for a
nearby transmitter or severe front-end overload. The exact historical answer
cannot be computed without Big Ear's measured 1977 intercept points, filters,
and signal levels.

Crucially, two 3-30 MHz OTH signals cannot reach 1420 MHz through ordinary
second- or third-order mixing: their largest positive products are only
{imd['max_oth_im2_hz']/1e6:.0f} MHz (second order) and
{imd['max_oth_im3_hz']/1e6:.0f} MHz (third order). Low-order intermodulation
becomes credible only for much higher-frequency out-of-band tones.

Dual-horn test
--------------
Horn peak separation: {config.horn_peak_separation_s:.2f} s
Continuous-source prediction in the second horn:
{horns['continuous_second_horn_snr']:.1f} sigma
Required fade between horn peaks: factor {horns['minimum_fade_factor']:.1f}
({horns['minimum_fade_db']:.2f} dB)
Maximum exponential decay constant for <1-sigma second detection:
{horns['max_decay_tau_s']:.2f} s

For a top-hat transient that supplies the measured first-horn trace, the model
finds a roughly {horns['available_shutoff_window_s']:.1f}-second interval after
the last measured point in which it can switch off and remain below 1 sigma in
the second horn. Therefore, the one-horn result does not reject a short natural
transient. It strongly rejects a steady narrow celestial source.

Common-mode terrestrial RFI
----------------------------
The horns were differenced. A terrestrial signal entering both similarly tends
to cancel. To leave a 30.5-sigma residual:

- 0.1% mismatch requires about {config.wow_peak_snr/0.001:.0f} sigma common mode.
- 1% mismatch requires about {config.wow_peak_snr/0.01:.0f} sigma common mode.
- 3% mismatch requires about {config.wow_peak_snr/0.03:.0f} sigma common mode.
- 10% mismatch requires about {config.wow_peak_snr/0.10:.0f} sigma common mode.

This does not make RFI impossible. It means an RFI explanation needs a strong
signal, directional/asymmetric coupling, nonlinear generation before the horn
difference is formed, or some combination of those conditions.

Updated verdict
---------------
PASS AS A MECHANISM:
An intrinsically narrow, short-lived cosmic transient. Propagation can modulate
it, and the transient can disappear before the second horn.

CONDITIONAL:
Two strong UHF or microwave signals generating an in-band IM2/IM3 product in a
nonlinear receiver stage. This is a genuine missing mechanism from version 1,
but it needs strong inputs, exact frequency geometry, and horn asymmetry.

FAIL:
A broadband cosmic collision signal being compressed into a narrow Wow-like
line by gravity, ionosphere, Doppler, and ordinary linear interference.

FAIL FOR LOW-ORDER MIXING:
Two ordinary 3-30 MHz OTH radar signals creating 1420 MHz by second- or
third-order intermodulation.

FAIL IF STEADY:
A persistent narrow celestial source. It should have appeared in the second
horn 172.37 seconds later.

Limitations
-----------
This remains a mechanism-feasibility model, not a probability estimate. It does
not contain classified transmitter logs, exact local spectrum occupancy,
measured Big Ear IIP2/IIP3 and filter responses, raw voltages, or a reconstructed
1977 ionosphere. The notional absolute power calculation is therefore an order-
of-magnitude stress test, not a historical measurement.
""".strip()

    (output_dir / "wow_signal_v2_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    config = Config()
    results, details = build_results(config)
    output_dir = Path(__file__).resolve().parent

    results.to_csv(output_dir / "wow_signal_v2_scenario_results.csv", index=False)
    details["intermod"]["examples"].to_csv(
        output_dir / "wow_signal_v2_intermod_examples.csv", index=False
    )
    details["horns"]["mismatch_table"].to_csv(
        output_dir / "wow_signal_v2_horn_mismatch.csv", index=False
    )
    pd.DataFrame({
        "decay_tau_s": details["horns"]["decay_tau_s"],
        "predicted_second_horn_snr": details["horns"]["second_horn_snr"],
    }).to_csv(output_dir / "wow_signal_v2_horn_decay_sweep.csv", index=False)

    make_plots(config, details, output_dir)
    write_report(config, results, details, output_dir)

    print(results.to_string(index=False))
    print("\nKey outputs:")
    print(f"  Minimum fade between horns: {details['horns']['minimum_fade_db']:.2f} dB")
    print(f"  Maximum exponential tau: {details['horns']['max_decay_tau_s']:.2f} s")
    print(f"  Top-hat shutoff window: {details['horns']['available_shutoff_window_s']:.2f} s")
    print(f"  Notional IM3 per-tone input at IIP3=-10 dBm: {details['intermod']['pin_im3_at_rep']:.2f} dBm")
    print(f"  Dynamic gravity drift upper bound: {sum(details['gravity_dynamic'].values()):.9f} Hz")

    print("\nGenerated:")
    for filename in [
        "wow_signal_v2_scenario_results.csv",
        "wow_signal_v2_intermod_examples.csv",
        "wow_signal_v2_horn_mismatch.csv",
        "wow_signal_v2_horn_decay_sweep.csv",
        "wow_signal_v2_report.txt",
        "wow_signal_v2_temporal_fit.png",
        "wow_signal_v2_spectral_test.png",
        "wow_signal_v2_frequency_effects.png",
        "wow_signal_v2_intermod_threshold.png",
        "wow_signal_v2_horn_decay_test.png",
        "wow_signal_v2_common_mode_test.png",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
