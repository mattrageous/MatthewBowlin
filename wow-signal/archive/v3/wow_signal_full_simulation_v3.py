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
    integration_time_s: float = 10.0
    scintillation_trials: int = 250_000


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
    """Compute low-order two-tone spurs and radiometer-corrected thresholds.

    Big Ear's printed sigma was a post-detection radiometer statistic, not a
    statement that the RF signal power was 30.5 times kTB. For a total-power
    radiometer, sigma_T = Tsys / sqrt(B*tau). A 30.5-sigma excess therefore
    corresponds to DeltaT/Tsys = 30.5/sqrt(B*tau).
    """
    noise_dbm = thermal_noise_power_dbm(
        config.notional_system_temperature_k, config.channel_width_hz
    )
    radiometer_factor = math.sqrt(
        config.channel_width_hz * config.integration_time_s
    )
    sigma_temperature_k = (
        config.notional_system_temperature_k / radiometer_factor
    )
    signal_excess_temperature_k = config.wow_peak_snr * sigma_temperature_k
    signal_to_rf_noise_ratio = (
        signal_excess_temperature_k / config.notional_system_temperature_k
    )
    peak_excess_db = 10.0 * math.log10(signal_to_rf_noise_ratio)
    target_spur_dbm = noise_dbm + peak_excess_db

    # Retain the v2 mistake explicitly so the correction is auditable.
    v2_peak_excess_db = 10.0 * math.log10(config.wow_peak_snr)
    v2_target_spur_dbm = noise_dbm + v2_peak_excess_db

    iip3_dbm = np.linspace(-30.0, 30.0, 121)
    required_pin_im3_dbm = (target_spur_dbm + 2.0 * iip3_dbm) / 3.0
    old_required_pin_im3_dbm = (v2_target_spur_dbm + 2.0 * iip3_dbm) / 3.0

    iip2_dbm = np.linspace(-10.0, 50.0, 121)
    required_pin_im2_dbm = (target_spur_dbm + iip2_dbm) / 2.0
    old_required_pin_im2_dbm = (v2_target_spur_dbm + iip2_dbm) / 2.0

    f1_im2 = 700_000_000.0
    f2_im2 = config.target_hz - f1_im2
    f1_im3a = 900_000_000.0
    f2_im3a = 2.0 * f1_im3a - config.target_hz
    f2_im3b = 800_000_000.0
    f1_im3b = 2.0 * f2_im3b - config.target_hz

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
    pin_im3_at_rep = (target_spur_dbm + 2.0 * representative_iip3_dbm) / 3.0
    pin_im2_at_rep = (target_spur_dbm + representative_iip2_dbm) / 2.0
    old_pin_im3_at_rep = (
        v2_target_spur_dbm + 2.0 * representative_iip3_dbm
    ) / 3.0
    old_pin_im2_at_rep = (
        v2_target_spur_dbm + representative_iip2_dbm
    ) / 2.0

    return {
        "noise_dbm": noise_dbm,
        "radiometer_factor": radiometer_factor,
        "sigma_temperature_k": sigma_temperature_k,
        "signal_excess_temperature_k": signal_excess_temperature_k,
        "signal_to_rf_noise_ratio": signal_to_rf_noise_ratio,
        "peak_excess_db": peak_excess_db,
        "target_spur_dbm": target_spur_dbm,
        "v2_peak_excess_db": v2_peak_excess_db,
        "v2_target_spur_dbm": v2_target_spur_dbm,
        "iip3_dbm": iip3_dbm,
        "required_pin_im3_dbm": required_pin_im3_dbm,
        "old_required_pin_im3_dbm": old_required_pin_im3_dbm,
        "iip2_dbm": iip2_dbm,
        "required_pin_im2_dbm": required_pin_im2_dbm,
        "old_required_pin_im2_dbm": old_required_pin_im2_dbm,
        "representative_iip3_dbm": representative_iip3_dbm,
        "representative_iip2_dbm": representative_iip2_dbm,
        "pin_im3_at_rep": pin_im3_at_rep,
        "pin_im2_at_rep": pin_im2_at_rep,
        "old_pin_im3_at_rep": old_pin_im3_at_rep,
        "old_pin_im2_at_rep": old_pin_im2_at_rep,
        "examples": examples,
        "max_oth_im2_hz": 60e6,
        "max_oth_im3_hz": 90e6,
    }


def scintillation_metrics(config: Config) -> dict:
    """Toy strong-scintillation test for a steady compact narrowband source.

    The complex field is modeled as correlated circular Gaussian noise. This is
    an illustrative diffractive-scintillation model, not a line-of-sight claim.
    It asks whether propagation alone can create the >30.5:1 horn-to-horn fade.
    """
    rng = np.random.default_rng(config.seed + 104729)
    n = config.scintillation_trials
    separation = config.horn_peak_separation_s
    fade_ratio = config.nondetection_snr / config.wow_peak_snr

    timescales_s = np.array([10, 20, 30, 50, 75, 100, 150, 200, 300, 450, 600, 900], dtype=float)
    probabilities = []
    median_ratios = []
    p05_ratios = []

    z1 = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2.0)
    w = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2.0)
    i1 = np.abs(z1) ** 2

    for tau_s in timescales_s:
        rho = math.exp(-separation / tau_s)
        z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho**2)) * w
        i2 = np.abs(z2) ** 2
        ratio = i2 / np.maximum(i1, 1e-15)
        probabilities.append(float(np.mean(ratio < fade_ratio)))
        median_ratios.append(float(np.median(ratio)))
        p05_ratios.append(float(np.quantile(ratio, 0.05)))

    independent_limit = fade_ratio / (1.0 + fade_ratio)
    table = pd.DataFrame({
        "scintillation_timescale_s": timescales_s,
        "field_correlation": np.exp(-separation / timescales_s),
        "probability_second_below_1sigma": probabilities,
        "median_second_to_first_intensity": median_ratios,
        "p05_second_to_first_intensity": p05_ratios,
    })

    return {
        "fade_ratio": fade_ratio,
        "independent_limit": independent_limit,
        "table": table,
        "timescales_s": timescales_s,
        "probabilities": np.array(probabilities),
    }


def horn_constraint_metrics(config: Config, beam_sigma_s: float) -> dict:
    """Quantify transient fading and asymmetric horn coupling."""
    separation = config.horn_peak_separation_s
    peak_snr = config.wow_peak_snr
    threshold = config.nondetection_snr

    minimum_fade_factor = peak_snr / threshold
    minimum_fade_db = 10.0 * math.log10(minimum_fade_factor)
    max_decay_tau_s = separation / math.log(minimum_fade_factor)

    decay_tau_s = np.linspace(5.0, 300.0, 500)
    second_horn_snr = peak_snr * np.exp(-separation / decay_tau_s)

    sigma_offsets = math.sqrt(2.0 * math.log(minimum_fade_factor))
    latest_safe_shutoff_after_first_peak_s = separation - beam_sigma_s * sigma_offsets
    fitted_center = fit_wow_beam()[0][1]
    last_observed_offset_s = float(WOW_TIMES_S.max() - fitted_center)
    available_shutoff_window_s = (
        latest_safe_shutoff_after_first_peak_s - last_observed_offset_s
    )

    # r is weaker-horn / stronger-horn power coupling for an RFI source.
    coupling_ratio = np.concatenate([
        np.logspace(-4.0, -0.01, 450),
        np.linspace(0.98, 0.9999, 250),
    ])
    coupling_ratio = np.unique(np.clip(coupling_ratio, 1e-6, 0.9999))
    asymmetry_db = -10.0 * np.log10(coupling_ratio)
    required_stronger_horn_snr = peak_snr / (1.0 - coupling_ratio)

    reference_ratios = np.array([0.999, 0.99, 0.5, 0.1, 0.0316228, 0.01, 0.001])
    mismatch_table = pd.DataFrame({
        "weaker_to_stronger_power_ratio": reference_ratios,
        "sidelobe_asymmetry_dB": -10.0 * np.log10(reference_ratios),
        "required_stronger_horn_RFI_SNR": peak_snr / (1.0 - reference_ratios),
    })

    return {
        "minimum_fade_factor": minimum_fade_factor,
        "minimum_fade_db": minimum_fade_db,
        "max_decay_tau_s": max_decay_tau_s,
        "decay_tau_s": decay_tau_s,
        "second_horn_snr": second_horn_snr,
        "latest_safe_shutoff_after_first_peak_s": latest_safe_shutoff_after_first_peak_s,
        "last_observed_offset_s": last_observed_offset_s,
        "available_shutoff_window_s": available_shutoff_window_s,
        "coupling_ratio": coupling_ratio,
        "asymmetry_db": asymmetry_db,
        "required_stronger_horn_snr": required_stronger_horn_snr,
        "mismatch_table": mismatch_table,
        "continuous_second_horn_snr": peak_snr,
    }


def build_results(config: Config) -> tuple[pd.DataFrame, dict]:
    params, beam_fit, beam_corr = fit_wow_beam()
    t, trace, temporal_samples, temporal_corr, temporal_rmse = simulate_temporal_trace(
        params, seed=config.seed
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
    scint = scintillation_metrics(config)

    rows = [
        {
            "Scenario": "Broad cosmic source + linear propagation effects",
            "Bandwidth result": broad_adjacent_ratio,
            "Horn result": "Steady signal repeats unless propagation changes it",
            "Verdict": "Rejected by bandwidth",
            "Interpretation": "The time trace is easy to mimic, but adjacent channels remain almost equally bright.",
        },
        {
            "Scenario": "Intrinsically narrow cosmic transient + propagation effects",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": f"Passes if it fades by >{horns['minimum_fade_db']:.1f} dB",
            "Verdict": "Physically possible",
            "Interpretation": (
                f"Exponential fading needs tau < {horns['max_decay_tau_s']:.1f} s, "
                f"or switch-off during the ~{horns['available_shutoff_window_s']:.0f} s window."
            ),
        },
        {
            "Scenario": "Steady narrow celestial source with strong scintillation",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": "Propagation can suppress the second horn",
            "Verdict": "Possible but propagation-dependent",
            "Interpretation": (
                f"In a toy fully developed scintillation model, the decorrelated limit gives "
                f"a {100*scint['independent_limit']:.2f}% chance of a >{horns['minimum_fade_db']:.1f} dB drop."
            ),
        },
        {
            "Scenario": "3-30 MHz OTH signals through linear or low-order nonlinear path",
            "Bandwidth result": np.nan,
            "Horn result": "No 1420 MHz low-order product",
            "Verdict": "Rejected",
            "Interpretation": (
                f"Second- and third-order positive products top out at "
                f"{imd['max_oth_im3_hz']/1e6:.0f} MHz."
            ),
        },
        {
            "Scenario": "Very-high-order harmonic from a single OTH/HF carrier",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": "Still needs favorable coupling geometry",
            "Verdict": "Mathematically possible, physically strained",
            "Interpretation": f"At least harmonic order {min_oth_harmonic_order} is required from 30 MHz.",
        },
        {
            "Scenario": "Two strong UHF tones create IM2/IM3 in nonlinear electronics",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": "Conditional on nonlinear stage and sidelobe coupling",
            "Verdict": "Live instrumental-artifact pathway",
            "Interpretation": (
                f"Radiometer-corrected target is {imd['target_spur_dbm']:.1f} dBm; "
                f"IIP3={imd['representative_iip3_dbm']:.0f} dBm needs "
                f"~{imd['pin_im3_at_rep']:.1f} dBm per tone."
            ),
        },
        {
            "Scenario": "Symmetric terrestrial RFI in both horn paths",
            "Bandwidth result": np.nan,
            "Horn result": "Strongly cancelled by differencing",
            "Verdict": "Strongly constrained",
            "Interpretation": "Near-equal horn coupling requires an enormous common signal to leave 30.5 sigma residual.",
        },
        {
            "Scenario": "Asymmetric sidelobe RFI",
            "Bandwidth result": np.nan,
            "Horn result": "Can favor one feed by many dB",
            "Verdict": "Only weakly constrained by horn differencing",
            "Interpretation": "With 10-20 dB power-coupling asymmetry, the stronger path need only be about 31-34 sigma.",
        },
        {
            "Scenario": "Continuous narrow celestial source without propagation modulation",
            "Bandwidth result": narrow_adjacent_ratio,
            "Horn result": f"Predicts ~{horns['continuous_second_horn_snr']:.1f} sigma again",
            "Verdict": "Rejected only if steady and unmodulated",
            "Interpretation": "The one-horn result excludes persistence only when scintillation or other propagation fading is absent.",
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
        "scintillation": scint,
        "sun_ra_h": sun_ra_h,
        "sun_dec_deg": sun_dec_deg,
        "sun_separation_deg": sun_separation_deg,
    }
    return pd.DataFrame(rows), details


def make_plots(config: Config, details: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(WOW_TIMES_S, WOW_SNR, "o-", label="6EQUJ5 midpoint SNR")
    plt.plot(WOW_TIMES_S, details["temporal_samples"], "s--", label="Combined-effects simulation")
    plt.xlabel("Seconds into the 72-second pass")
    plt.ylabel("Signal-to-noise ratio")
    plt.title("Temporal fit: expected, but weak as a discriminator")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_temporal_fit.png", dpi=180)
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
    plt.savefig(output_dir / "wow_signal_v3_spectral_test.png", dpi=180)
    plt.close()

    effects = {}
    effects.update(details["gravity_static"])
    effects.update(details["gravity_dynamic"])
    effects.update(details["motion_drift"])
    labels = list(effects.keys())
    values = [max(abs(effects[k]), 1e-9) for k in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.axhline(config.channel_width_hz, linestyle="--", label="10 kHz channel")
    plt.yscale("log")
    plt.ylabel("Frequency scale (Hz)")
    plt.title("Static gravity is small; time-varying gravity is microscopic")
    plt.xticks(rotation=27, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_frequency_effects.png", dpi=180)
    plt.close()

    imd = details["intermod"]
    plt.figure(figsize=(8, 5))
    plt.plot(imd["iip3_dbm"], imd["old_required_pin_im3_dbm"], linestyle="--", label="v2 incorrect RF target")
    plt.plot(imd["iip3_dbm"], imd["required_pin_im3_dbm"], label="v3 radiometer-corrected target")
    plt.xlabel("Receiver IIP3 (dBm)")
    plt.ylabel("Required input power per tone (dBm)")
    plt.title("Radiometer correction makes IM3 less demanding")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_intermod_threshold.png", dpi=180)
    plt.close()

    horns = details["horns"]
    plt.figure(figsize=(8, 5))
    plt.plot(horns["decay_tau_s"], horns["second_horn_snr"])
    plt.axhline(config.nondetection_snr, linestyle="--", label="1-sigma threshold")
    plt.axvline(horns["max_decay_tau_s"], linestyle=":", label=f"Maximum tau: {horns['max_decay_tau_s']:.1f} s")
    plt.yscale("log")
    plt.xlabel("Intrinsic exponential decay time (s)")
    plt.ylabel("Predicted second-horn SNR")
    plt.title("Intrinsic transient route through the one-horn constraint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_horn_decay_test.png", dpi=180)
    plt.close()

    order = np.argsort(horns["asymmetry_db"])
    plt.figure(figsize=(8, 5))
    plt.plot(horns["asymmetry_db"][order], horns["required_stronger_horn_snr"][order])
    plt.yscale("log")
    plt.xlabel("Weaker/stronger horn power-coupling asymmetry (dB)")
    plt.ylabel("Required stronger-path RFI level (sigma)")
    plt.title("Symmetric RFI cancels; asymmetric sidelobe RFI does not")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_sidelobe_asymmetry.png", dpi=180)
    plt.close()

    scint = details["scintillation"]
    plt.figure(figsize=(8, 5))
    plt.plot(scint["timescales_s"], 100.0 * scint["probabilities"], marker="o")
    plt.axhline(100.0 * scint["independent_limit"], linestyle="--", label="Fully decorrelated limit")
    plt.xscale("log")
    plt.xlabel("Toy scintillation field-correlation timescale (s)")
    plt.ylabel("Chance second/first intensity < 1/30.5 (%)")
    plt.title("Strong scintillation can create a one-horn fade")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_v3_scintillation_test.png", dpi=180)
    plt.close()


def write_report(config: Config, results: pd.DataFrame, details: dict, output_dir: Path) -> None:
    gravity_static_total = sum(details["gravity_static"].values())
    gravity_dynamic_total = sum(details["gravity_dynamic"].values())
    motion_total = sum(details["motion_drift"].values())
    imd = details["intermod"]
    horns = details["horns"]
    scint = details["scintillation"]

    report = f"""
WOW! SIGNAL COMBINED-EFFECTS SIMULATION — VERSION 3
===================================================

Corrections from version 2
--------------------------
1. Converts Big Ear's 30.5 post-detection sigma through the radiometer equation
   instead of treating it as 30.5 times the instantaneous RF noise power.
2. Adds strong interstellar-scintillation fading as an escape from the second-
   horn expectation for a steady compact narrowband source.
3. Replaces the fragile common-mode premise with an explicit sidelobe-coupling
   asymmetry sweep.

Radiometer-corrected spur target
--------------------------------
System temperature: {config.notional_system_temperature_k:.1f} K
Bandwidth: {config.channel_width_hz:.0f} Hz
Integration time: {config.integration_time_s:.1f} s
sqrt(B*tau): {imd['radiometer_factor']:.3f}
Radiometer sigma temperature: {imd['sigma_temperature_k']:.4f} K
30.5-sigma excess temperature: {imd['signal_excess_temperature_k']:.3f} K
10 kHz RF thermal-noise power: {imd['noise_dbm']:.2f} dBm
Signal/RF-noise power ratio: {imd['signal_to_rf_noise_ratio']:.5f}
Signal relative to RF noise: {imd['peak_excess_db']:.2f} dB
Corrected target spur power: {imd['target_spur_dbm']:.2f} dBm

Version 2 incorrectly used +{imd['v2_peak_excess_db']:.2f} dB and targeted
{imd['v2_target_spur_dbm']:.2f} dBm. That was a factor-of-hundreds power error.

At IIP3={imd['representative_iip3_dbm']:.1f} dBm:
- v2 per-tone IM3 requirement: {imd['old_pin_im3_at_rep']:.2f} dBm
- corrected per-tone requirement: {imd['pin_im3_at_rep']:.2f} dBm
- correction: {imd['pin_im3_at_rep']-imd['old_pin_im3_at_rep']:.2f} dB

At IIP2={imd['representative_iip2_dbm']:.1f} dBm:
- v2 per-tone IM2 requirement: {imd['old_pin_im2_at_rep']:.2f} dBm
- corrected per-tone requirement: {imd['pin_im2_at_rep']:.2f} dBm
- correction: {imd['pin_im2_at_rep']-imd['old_pin_im2_at_rep']:.2f} dB

This materially strengthens the nonlinear-RFI pathway. The tones still need to
be strong at the nonlinear stage, but not as extreme as version 2 claimed.

Bandwidth and OTH result
------------------------
Broad 1 MHz source adjacent/peak ratio: {results.loc[0, 'Bandwidth result']:.6f}
Narrow 5 kHz source adjacent/peak ratio: {results.loc[1, 'Bandwidth result']:.6f}

Linear propagation still cannot compress broadband emission. Two 3-30 MHz OTH
signals still top out at {imd['max_oth_im3_hz']/1e6:.0f} MHz through ordinary
third-order positive products, so that specific low-order route remains closed.

One-horn result: intrinsic transient
------------------------------------
Horn separation: {config.horn_peak_separation_s:.2f} s
Required drop: {horns['minimum_fade_db']:.2f} dB
Maximum intrinsic exponential decay time: {horns['max_decay_tau_s']:.2f} s
Available top-hat shutoff window: {horns['available_shutoff_window_s']:.2f} s

A short transient passes cleanly.

One-horn result: steady source plus scintillation
-------------------------------------------------
A steady and unmodulated source would predict another {horns['continuous_second_horn_snr']:.1f}-sigma
response. That statement is not absolute once propagation fading is allowed.

The toy fully developed scintillation model asks for a horn-to-horn intensity
ratio below {scint['fade_ratio']:.5f}. In the fully decorrelated limit, the
probability is {100*scint['independent_limit']:.2f}% per paired observation.
For longer correlation times, the simulated probability falls as the two horn
measurements become more alike.

This does not establish that the Wow line of sight had the necessary scattering
regime. It does show that 'steady source' is rejected only when it is also
assumed to be unmodulated by propagation.

Horn differencing and sidelobes
-------------------------------
Near-symmetric RFI is strongly cancelled. The required stronger-path levels are:
{horns['mismatch_table'].to_string(index=False)}

The table makes the premise visible. A 1% mismatch between nearly equal horn
responses demands thousands of sigma. But 10-20 dB sidelobe power asymmetry
requires only about 31-34 sigma in the stronger path. Therefore horn differencing
strongly constrains symmetric RFI, while highly directional/asymmetric sidelobe
entry is only weakly constrained.

Gravity and motion
------------------
Static Earth/Sun/Moon potential scale: {gravity_static_total:.6f} Hz
Time-varying gravity upper bound over 72 s: {gravity_dynamic_total:.9f} Hz
Rotation/orbit drift scale over 72 s: {motion_total:.6f} Hz

Static gravity is common-mode; differential gravity is microscopic.

Version 3 verdict
-----------------
CLEAN PASS:
An intrinsically narrow short-lived cosmic transient.

LIVE CONDITIONAL:
Two high-frequency tones producing an IM2/IM3 spur in nonlinear electronics.
The corrected radiometer conversion makes this more permissive than v2.

LIVE CONDITIONAL:
A steady compact narrowband celestial source undergoing sufficiently deep
interstellar scintillation between horn passages.

STRONGLY CONSTRAINED, NOT GENERALLY EXCLUDED:
Terrestrial RFI. Symmetric coupling cancels, but asymmetric sidelobe coupling
can evade that protection.

REJECTED:
Broadband cosmic emission made narrow by linear combinations of gravity,
ionosphere, Doppler, and overlapping terrestrial waves.

REJECTED FOR ORDINARY LOW-ORDER PRODUCTS:
Two 3-30 MHz OTH signals mixing directly to 1420.726 MHz.

Limitations
-----------
This is still a mechanism-feasibility model, not a posterior probability. The
scintillation calculation is deliberately generic; it does not reconstruct the
actual scattering strength or decorrelation time toward the Wow coordinates.
The RFI calculation lacks measured 1977 filter rejection, sidelobe maps,
intercept points, local transmitter levels, and raw voltages.
""".strip()
    (output_dir / "wow_signal_v3_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    config = Config()
    results, details = build_results(config)
    output_dir = Path(__file__).resolve().parent

    results.to_csv(output_dir / "wow_signal_v3_scenario_results.csv", index=False)
    details["intermod"]["examples"].to_csv(output_dir / "wow_signal_v3_intermod_examples.csv", index=False)
    details["horns"]["mismatch_table"].to_csv(output_dir / "wow_signal_v3_sidelobe_asymmetry.csv", index=False)
    details["scintillation"]["table"].to_csv(output_dir / "wow_signal_v3_scintillation_sweep.csv", index=False)
    pd.DataFrame({
        "decay_tau_s": details["horns"]["decay_tau_s"],
        "predicted_second_horn_snr": details["horns"]["second_horn_snr"],
    }).to_csv(output_dir / "wow_signal_v3_horn_decay_sweep.csv", index=False)

    make_plots(config, details, output_dir)
    write_report(config, results, details, output_dir)

    print(results.to_string(index=False))
    print("\nCorrected outputs:")
    print(f"  Radiometer sigma temperature: {details['intermod']['sigma_temperature_k']:.4f} K")
    print(f"  Wow excess temperature: {details['intermod']['signal_excess_temperature_k']:.3f} K")
    print(f"  Corrected target spur: {details['intermod']['target_spur_dbm']:.2f} dBm")
    print(f"  IM3 per tone at IIP3=-10 dBm: {details['intermod']['pin_im3_at_rep']:.2f} dBm")
    print(f"  IM2 per tone at IIP2=20 dBm: {details['intermod']['pin_im2_at_rep']:.2f} dBm")
    print(f"  Decorrelated scintillation one-horn probability: {100*details['scintillation']['independent_limit']:.2f}%")

    print("\nGenerated:")
    for filename in [
        "wow_signal_v3_scenario_results.csv",
        "wow_signal_v3_intermod_examples.csv",
        "wow_signal_v3_sidelobe_asymmetry.csv",
        "wow_signal_v3_scintillation_sweep.csv",
        "wow_signal_v3_horn_decay_sweep.csv",
        "wow_signal_v3_report.txt",
        "wow_signal_v3_temporal_fit.png",
        "wow_signal_v3_spectral_test.png",
        "wow_signal_v3_frequency_effects.png",
        "wow_signal_v3_intermod_threshold.png",
        "wow_signal_v3_horn_decay_test.png",
        "wow_signal_v3_sidelobe_asymmetry.png",
        "wow_signal_v3_scintillation_test.png",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
