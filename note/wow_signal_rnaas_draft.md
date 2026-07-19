# A Reproducible Mechanism-Elimination Analysis of the Wow! Signal

**Matthew Bowlin** — Independent researcher, Ravena, NY
*Draft for submission as a Research Note of the AAS (~1,000 words, one figure)*

---

## Introduction

The Wow! signal of 1977 August 15 remains unexplained: a 72-second, ~10 kHz-wide feature near the hydrogen line, detected in one horn of the Big Ear transit telescope at ≥30σ and never re-detected (Kraus 1979; Gray 1994). Recent archival re-analysis revises the frequency to 1420.726 ± 0.005 MHz and the peak flux above 250 Jy, and proposes an astrophysical mechanism: transient maser-like amplification of a cold H I cloud, plausibly triggered by a magnetar flare (Méndez et al. 2024, arXiv:2408.08513; Méndez et al. 2025, arXiv:2508.10657). This note presents a single reproducible model that tests the major candidate explanation classes against the event's constraints and the published follow-up record. The result is not a new explanation but an explicit, auditable elimination argument, with every assumption exposed in released code.

## Method

The model (Python; code and inputs at [REPO LINK]) implements independent mechanism-feasibility tests: (1) spectral response of broadband versus narrowband sources through Big Ear's 10 kHz channelization under linear propagation (gravitational, ionospheric, Doppler terms evaluated explicitly); (2) the dual-horn timing constraint (172.4 s separation) for intrinsic transients and for steady sources modulated by strong diffractive scintillation; (3) notional receiver intermodulation budgets converted through the radiometer equation; and (4) a campaign-by-campaign persistence likelihood over the published follow-ups: 89 Big Ear null revisit-days (Kipping & Gray 2022), META (Gray 1994), the VLA searches (Gray & Marvel 2001), Hobart (Gray & Ellingsen 2002), and the ATA campaign (Harp et al. 2020), with per-campaign thresholds, channel widths, and conservatively counted independent scintillation states tabulated with sources.

## Results

**Bandwidth.** Linear propagation cannot compress broadband emission into one channel: a 1 MHz-wide source yields an adjacent-to-peak channel ratio of 0.9997, versus 0.0094 for a 5 kHz source. Combined static gravitational potential terms (~15 Hz), differential gravity over 72 s (<10⁻⁵ Hz), and Earth-motion drift (~11 Hz) are negligible against a 10⁴ Hz channel. Any viable source is intrinsically narrowband.

**Persistence.** For a steady compact source of mean flux S under saturated scintillation (exponential intensity statistics — the assumption most favorable to hiding), the joint likelihood of the 54 Jy original event and all follow-up nulls peaks at S ≈ 2.8 Jy but is suppressed by 10⁻¹²·⁵ relative to fitting the event alone (10⁻⁸·⁷ using only Big Ear and ATA, which covered the full reconstructed field). The mechanism is transparent: the best-fit S requires a 19× scintillation boost for the original event (probability ~5×10⁻⁹), and no S escapes both that penalty and 130 conservative nulls. The revised ≥250 Jy flux strengthens this rejection (suppression ~10⁻²⁰ at 212 Jy). Deep scintillation additionally requires microarcsecond compactness (Fresnel scales 2–22 μas for screens at 0.1–10 kpc), so extended sources cannot invoke it. Sampling caveats are quantified: ~80% of 1–48 hr periods survive the Big Ear-only same-sidereal-time visit pattern for 144 s pulses, so periodic or stochastic repeaters are constrained far more weakly than steady sources; a sustained monotonic chirp above ~0.09 Hz s⁻¹ (~2.9 MHz yr⁻¹) could also evade the fixed-frequency chain, at the cost of a much stranger source hypothesis.

**Instrumental pathways.** Horn differencing strongly cancels symmetric terrestrial interference (a 1% path mismatch requires ~3,000σ common-mode power) but constrains asymmetric sidelobe entry only weakly (10–20 dB asymmetry requires just 31–34σ in the stronger path). Radiometer conversion of the 30σ excess gives a signal ~10 dB below the 10 kHz RF noise floor (~−149 dBm for Tsys = 100 K), so two-tone intermodulation at representative intercept points needs per-tone inputs near −56 dBm (IIP3 = −10 dBm) — strong but not excludable without the 1977 receiver's measured nonlinearity. Direct low-order mixing of HF/OTH signals to 1420 MHz is arithmetically impossible.

## Conclusion

Rejected: broadband or extended emission narrowed by linear propagation; a steady narrowband source, with or without scintillation. Surviving: an intrinsically narrow one-time transient (requiring only beam-scale, arcminute localization); a low-duty-cycle or stochastic repeater; a substantially frequency-mobile source; or a nonlinear instrumental artifact with asymmetric coupling. Among natural candidates, transiently amplified H I emission (Méndez et al. 2024) is the most concrete, possessing both a physical mechanism and weaker archival analogues; the present analysis independently arrives at the explanation class this hypothesis occupies without assuming it, while the evidence does not yet identify it as the cause.

## Figure

*Figure 1: Maximum-likelihood suppression of the steady scintillating-source hypothesis as follow-up campaigns are added sequentially (Big Ear null revisits, VLA 1995/1996, Hobart, ATA), for 54 Jy and 212 Jy original-event calibrations. The robust Big Ear+ATA-only chain is shown separately.*
[Use: wow_signal_v5_steady_source_likelihood.png]

## References

- Gray, R. H. 1994, Icarus, 112, 485
- Gray, R. H., & Marvel, K. B. 2001, ApJ, 546, 1171
- Gray, R. H., & Ellingsen, S. 2002, ApJ, 578, 967
- Harp, G. R., et al. 2020, AJ, 160, 162
- Kipping, D., & Gray, R. H. 2022, MNRAS, 515, 1122
- Kraus, J. D. 1979, Cosmic Search, 1, 31
- Méndez, A., et al. 2024, arXiv:2408.08513
- Méndez, A., et al. 2025, arXiv:2508.10657

