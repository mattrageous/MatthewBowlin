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

    rows = [
        {
            "Scenario": "Broad cosmic source + gravity + ionosphere + terrestrial RF (linear)",
            "Temporal correlation": temporal_corr,
            "Adjacent/peak channel power": broad_adjacent_ratio,
            "Can match one 10 kHz channel?": "No",
            "Verdict": "Fails bandwidth test",
            "Interpretation": (
                "Can reproduce the 72-second beam-shaped rise/fall, but neighboring "
                "channels become almost equally bright."
            ),
        },
        {
            "Scenario": "Narrow cosmic transient + same propagation effects",
            "Temporal correlation": temporal_corr,
            "Adjacent/peak channel power": narrow_adjacent_ratio,
            "Can match one 10 kHz channel?": "Yes",
            "Verdict": "Physically possible",
            "Interpretation": (
                "A signal already narrower than the receiver channel survives the "
                "path and can look Wow-like. The terrestrial RF is not needed."
            ),
        },
        {
            "Scenario": "3–30 MHz over-the-horizon signal, linear receiver",
            "Temporal correlation": np.nan,
            "Adjacent/peak channel power": np.nan,
            "Can match one 10 kHz channel?": "No",
            "Verdict": "No frequency conversion",
            "Interpretation": (
                "HF and 1420 MHz fields superpose; gravity and the ionosphere do not "
                "translate HF into L-band in a linear path."
            ),
        },
        {
            "Scenario": "3–30 MHz OTH signal through receiver nonlinearity",
            "Temporal correlation": np.nan,
            "Adjacent/peak channel power": np.nan,
            "Can match one 10 kHz channel?": "Only with an extreme spur",
            "Verdict": "Mathematically possible, physically strained",
            "Interpretation": (
                f"Even the highest HF fundamental needs at least harmonic order "
                f"{min_oth_harmonic_order}; that is not an ordinary mixing product."
            ),
        },
        {
            "Scenario": "Narrow UHF carrier harmonic + reflection/receiver spur",
            "Temporal correlation": beam_corr,
            "Adjacent/peak channel power": narrow_adjacent_ratio,
            "Can match one 10 kHz channel?": "Conditionally",
            "Verdict": "Possible RFI artifact with fine tuning",
            "Interpretation": (
                f"A carrier near {uhf_second_harmonic_hz/1e6:.3f} MHz (2nd) or "
                f"{uhf_third_harmonic_hz/1e6:.3f} MHz (3rd) can land in-band, but "
                "a sky-like beam trace and one-horn detection still require special geometry."
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
        "gravity": gravitational_shifts(config.target_hz),
        "motion_drift": earth_motion_drift_72s(
            config.target_hz, config.latitude_deg
        ),
        "ionosphere": ionosphere_metrics(config.target_hz),
        "min_oth_harmonic_order": min_oth_harmonic_order,
        "exact_oth_subharmonic_hz": exact_oth_subharmonic_hz,
        "uhf_second_harmonic_hz": uhf_second_harmonic_hz,
        "uhf_third_harmonic_hz": uhf_third_harmonic_hz,
        "sun_ra_h": sun_ra_h,
        "sun_dec_deg": sun_dec_deg,
        "sun_separation_deg": sun_separation_deg,
    }
    return pd.DataFrame(rows), details


def make_plots(config: Config, details: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: temporal fit
    plt.figure(figsize=(8, 5))
    plt.plot(WOW_TIMES_S, WOW_SNR, "o-", label="6EQUJ5 midpoint SNR")
    plt.plot(
        WOW_TIMES_S,
        details["temporal_samples"],
        "s--",
        label="Combined-effects simulation",
    )
    plt.xlabel("Seconds into the 72-second pass")
    plt.ylabel("Signal-to-noise ratio")
    plt.title("Wow! temporal profile: data vs simulated celestial beam")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_temporal_fit.png", dpi=180)
    plt.close()

    # Plot 2: spectral channel test
    plt.figure(figsize=(8, 5))
    x_khz = details["broad_centers"] / 1e3
    broad = details["broad_power"] / details["broad_power"].max()
    narrow = details["narrow_power"] / details["narrow_power"].max()
    plt.plot(x_khz, broad, label="1 MHz-wide natural source")
    plt.plot(x_khz, narrow, label="5 kHz-wide source")
    plt.xlabel("Offset from target frequency (kHz)")
    plt.ylabel("Normalized power per 10 kHz channel")
    plt.title("Bandwidth is the decisive test")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_spectral_test.png", dpi=180)
    plt.close()

    # Plot 3: frequency-effect magnitudes
    effects = {}
    effects.update(details["gravity"])
    effects.update(details["motion_drift"])
    labels = list(effects.keys())
    values = [max(abs(effects[k]), 1e-6) for k in labels]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.axhline(
        config.channel_width_hz,
        linestyle="--",
        label="10 kHz Big Ear channel",
    )
    plt.yscale("log")
    plt.ylabel("Frequency scale (Hz)")
    plt.title("Gravity and 72-second motion drift are tiny versus one channel")
    plt.xticks(rotation=24, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wow_signal_frequency_effects.png", dpi=180)
    plt.close()


def write_report(config: Config, results: pd.DataFrame,
                 details: dict, output_dir: Path) -> None:
    gravity_total = sum(details["gravity"].values())
    motion_total = sum(details["motion_drift"].values())

    report = f"""
WOW! SIGNAL COMBINED-EFFECTS SIMULATION
=======================================

Target frequency used
---------------------
Revised archival estimate: {config.target_hz/1e6:.6f} MHz
Legacy estimate retained in code: {WOW_LEGACY_HZ/1e6:.6f} MHz
Hydrogen rest frequency: {HI_REST_HZ/1e6:.9f} MHz

Temporal result
---------------
Gaussian fit to 6EQUJ5 correlation: {details["beam_corr"]:.6f}
Combined beam + ionosphere + receiver + troposphere correlation:
{details["temporal_corr"]:.6f}
RMSE: {details["temporal_rmse"]:.3f} SNR units

This confirms that amplitude disturbances can coexist with a Wow-like
72-second drift-scan profile.

Spectral result
---------------
For a 1 MHz-wide cosmic source, adjacent/peak channel power:
{results.loc[0, "Adjacent/peak channel power"]:.6f}

For a 5 kHz-wide cosmic source, adjacent/peak channel power:
{results.loc[1, "Adjacent/peak channel power"]:.6f}

A broadband source remains broadband. The modeled propagation effects do not
squeeze it into one 10 kHz channel. A source already narrowband can remain
confined to one channel.

Gravity and motion
------------------
Combined local Earth/Sun/Moon potential-scale frequency term:
{gravity_total:.6f} Hz

Generous combined Earth rotation/orbit drift over 72 seconds:
{motion_total:.6f} Hz

Both are far below a 10,000 Hz channel. Constant Earth motion can set a larger
absolute Doppler offset, but it does not generate a new frequency or collapse
bandwidth.

Ionosphere
----------
Representative 10-TECU group delay:
{details["ionosphere"]["Group delay at 10 TECU (ns)"]:.6f} ns

Representative peak plasma frequency:
{details["ionosphere"]["Peak plasma frequency (MHz)"]:.6f} MHz

Refractive-index departure scale:
{details["ionosphere"]["|n - 1| scale"]:.8e}

The ionosphere can refract, delay, phase-rotate, scintillate, and sometimes
reflect lower-frequency signals. At 1420 MHz it does not normally convert an
HF military signal into a new L-band carrier.

Human-made frequency routes
---------------------------
Minimum harmonic order required to turn a 3-30 MHz OTH signal into the target:
{details["min_oth_harmonic_order"]}

Exact subharmonic at that order:
{details["exact_oth_subharmonic_hz"]/1e6:.6f} MHz

More plausible low-order subharmonics:
2nd harmonic fundamental: {details["uhf_second_harmonic_hz"]/1e6:.6f} MHz
3rd harmonic fundamental: {details["uhf_third_harmonic_hz"]/1e6:.6f} MHz

A nonlinear receiver, transmitter harmonic, or reflected narrow UHF carrier
can create an in-band artifact. That is a real possible class of explanation,
but it is not the Sun, Moon, Earth, and ionosphere blending an HF signal with a
cosmic signal in free space.

Solar alignment check
---------------------
Approximate Sun coordinates at the event time:
RA {details["sun_ra_h"]:.3f} h, Dec {details["sun_dec_deg"]:.3f} deg
Approximate Sun-Wow angular separation: {details["sun_separation_deg"]:.3f} deg

That is nowhere near a solar gravitational-lensing alignment.

Bottom line
-----------
1. Starting with an ordinary broadband cosmic source:
   FAIL. Combined linear environmental effects do not make it Wow-like in
   frequency, even though the time profile can look right.

2. Starting with a naturally narrow cosmic transient:
   PASS as a possibility. The combined effects can alter its amplitude and
   exact observed frequency while preserving a Wow-like narrow line.

3. Starting with terrestrial HF/OTH transmissions:
   FAIL in a linear path. A receiver fault/nonlinearity would be required.

4. Starting with a narrow UHF carrier near an exact low-order subharmonic:
   CONDITIONAL. A harmonic or receiver spur can land near 1420 MHz, but the
   celestial beam shape, reflection geometry, and one-horn timing require
   additional fine tuning.

Important limitation
--------------------
These are mechanism and magnitude tests, not posterior probabilities.
The raw complex voltage, exact 1977 ionospheric state above the telescope,
classified transmitter logs, and measured nonlinear transfer function of the
receiver are not available in this model.
""".strip()

    (output_dir / "wow_signal_simulation_report.txt").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    config = Config()
    results, details = build_results(config)
    output_dir = Path(__file__).resolve().parent

    results.to_csv(output_dir / "wow_signal_scenario_results.csv", index=False)
    make_plots(config, details, output_dir)
    write_report(config, results, details, output_dir)

    print(results.to_string(index=False))
    print()
    print("Generated:")
    for filename in [
        "wow_signal_scenario_results.csv",
        "wow_signal_simulation_report.txt",
        "wow_signal_temporal_fit.png",
        "wow_signal_spectral_test.png",
        "wow_signal_frequency_effects.png",
    ]:
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
