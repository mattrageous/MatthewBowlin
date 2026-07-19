# Wow! Signal Mechanism Tests

A reproducible, single-script mechanism-elimination analysis of the 1977 Wow! signal.
Every assumption is exposed in code and in per-campaign input tables with sources.

**Author:** Matthew Bowlin ([matthewbowlin.com](https://matthewbowlin.com))
**Status:** Analysis complete (v5). A condensed research note is in [`note/`](note/).

## The result, in one paragraph

The surviving explanation for the Wow! signal must be intrinsically narrowband and
non-persistent. Linear propagation (gravity, ionosphere, Doppler) cannot compress
broadband emission into one 10 kHz channel, so any viable source is narrow to begin
with. A campaign-by-campaign maximum-likelihood chain over the published follow-ups
(Big Ear's 89 null revisit-days, META, VLA 1995/1996, Hobart, ATA) suppresses a
steady fixed-frequency compact source — even one hidden by the most generous
scintillation statistics — by ~10^-12.5 (nominal-position chain) or ~10^-8.7
(robust Big Ear+ATA-only chain). What remains: an intrinsically narrow one-time
transient, a low-duty-cycle or stochastic repeater, a substantially frequency-mobile
source, or a nonlinear instrumental artifact. Among natural candidates, transiently
amplified hydrogen-line emission (Méndez et al. 2024, arXiv:2408.08513; 2025,
arXiv:2508.10657) is the most concrete — a proposed mechanism plus weaker archival
analogues — but the evidence supports the class, not the identification.

## The elimination ladder

| Hypothesis class | Status | Killed / constrained by |
|---|---|---|
| Broadband or extended source narrowed by linear propagation | **Rejected** | Bandwidth physics: adjacent/peak channel ratio 0.9997 (1 MHz source) vs 0.0094 (5 kHz source) |
| Two 3–30 MHz OTH signals mixing to 1420 MHz at low order | **Rejected** | Arithmetic: largest 2nd/3rd-order sums reach 60/90 MHz |
| Steady narrow celestial source | **Rejected** | Second-horn null + follow-up likelihood |
| Steady compact source hidden by deep scintillation | **Effectively rejected** | Requires µas compactness (Fresnel 2–22 µas) *and* survives 130 conservative nulls at ~10^-9 to 10^-12.5 likelihood; brighter revised flux (≥250 Jy) makes it worse |
| Intrinsically narrow one-time transient | **Pass** | Needs only arcminute (beam-scale) localization; max intrinsic decay τ ≈ 50 s clears the 172.4 s second-horn window |
| Low-duty-cycle / stochastic / periodic repeater | **Live** | ~80% of 1–48 h periods survive Big Ear's same-LST sampling for 144 s pulses; duty-cycle 95% bounds tabulated vs epoch count |
| Frequency-mobile source (sustained chirp > ~0.09 Hz/s) | **Live, contrived** | Escapes the fixed-frequency chain at the cost of ~2.9 MHz/yr one-signed drift for decades |
| Nonlinear receiver intermod / asymmetric-sidelobe RFI | **Live** | Radiometer-corrected spur target −148.8 dBm; needs ~−56 dBm per tone at IIP3 = −10 dBm and 10–20 dB horn-coupling asymmetry |

Note which constraint applies to which class: the microarcsecond compactness gate
belongs **only** to the steady-source-plus-scintillation hypothesis. An intrinsically
transient event never invokes propagation fading and needs only to be unresolved by
Big Ear's arcminute-scale beam.

## Running it

```
pip install numpy scipy matplotlib pandas
python wow_signal_full_simulation_v5.py
```

Outputs regenerate the CSVs in [`data/`](data/) and figures in [`figures/`](figures/).
The full narrative of the v5 results is in [`report.txt`](report.txt).
Campaign inputs, thresholds, independent-state counts, and their literature sources
are in [`data/wow_signal_v5_campaigns.csv`](data/wow_signal_v5_campaigns.csv).

## Version history (including corrected errors)

Earlier versions are preserved unmodified in [`archive/`](archive/), errors included.

- **v1** — Linear combined-effects tests: bandwidth, gravity/Doppler magnitudes,
  ionosphere, harmonic routes, solar-alignment check.
- **v2** — Added two-tone IM2/IM3 thresholds, the 172.37 s dual-horn constraint,
  common-mode RFI rejection, static vs time-varying gravity.
  **Contains a known error:** the 30.5σ excess was treated as an RF-domain power
  ratio (+14.8 dB) instead of a post-detection radiometer sigma.
- **v3** — **Corrects the v2 error** via the radiometer equation: the true signal
  level is ~10 dB *below* the RF noise floor (−148.8 dBm vs v2's −123.8 dBm),
  a ~25 dB shift that makes the intermod pathway materially more permissive.
  Also adds the scintillation escape for steady sources and replaces the fragile
  common-mode premise with an explicit sidelobe-asymmetry sweep.
- **v4** — Persistence constraints: compactness gate, steady-source survival,
  duty-cycle bounds vs epoch count, Poisson repeat-rate limits.
- **v5** — Campaign-by-campaign likelihood with sourced per-campaign inputs;
  periodic-alias and frequency-drift caveats quantified; dual nominal-position
  and robust full-field chains reported separately.

The v2→v3 correction is documented deliberately: conflating post-detection sigma
with RF power is an easy mistake, and its 25 dB consequence is a useful worked
example for anyone attempting similar budgets.

## What this is and is not

These are mechanism-feasibility tests and likelihood comparisons under stated
assumptions — not posterior probabilities. Missing inputs that would be required
for a true historical posterior are listed in the Limitations section of
[`report.txt`](report.txt).

## Key references

Kraus (1979); Gray (1994); Gray & Marvel (2001); Gray & Ellingsen (2002);
Harp et al. (2020); Kipping & Gray (2022); Méndez et al. (2024, arXiv:2408.08513);
Méndez et al. (2025, arXiv:2508.10657).
