# Doppler measurement-noise robustness

Completed 21200 / 21200 planned trials.
Additional Gaussian noise is in Hz; zero means no ADDED noise, not perfect measurements.
Intervals resample encounters, keeping noise repetitions together; constant outcomes use bounded-mean fallbacks.
Errors count as safety failures; collision outcomes on errors remain unknown.
Fixed-time error metrics are conditional on an available estimate; read availability too.
Timing screens exclude acoustic processing and networking; compute delays are not simulated.

| UAVs | Class | Method/policy | Added sigma (Hz) | Trials | Geometries | Safe | Errors |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | direct_threat | estimate/baseline | 0.0 | 50 | 50 | 90.0% | 0 |
| 4 | direct_threat | estimate/fixed | 0.5 | 250 | 50 | 52.0% | 0 |
| 4 | direct_threat | estimate/variance_aware | 0.5 | 250 | 50 | 56.4% | 0 |
| 4 | direct_threat | estimate/fixed | 1.0 | 250 | 50 | 29.2% | 0 |
| 4 | direct_threat | estimate/variance_aware | 1.0 | 250 | 50 | 32.4% | 0 |
| 4 | direct_threat | estimate/fixed | 2.0 | 250 | 50 | 11.2% | 0 |
| 4 | direct_threat | estimate/variance_aware | 2.0 | 250 | 50 | 18.8% | 0 |
| 4 | direct_threat | estimate/fixed | 4.0 | 250 | 50 | 6.4% | 0 |
| 4 | direct_threat | estimate/variance_aware | 4.0 | 250 | 50 | 8.4% | 0 |
| 4 | direct_threat | estimate/fixed | 8.0 | 250 | 50 | 2.0% | 0 |
| 4 | direct_threat | estimate/variance_aware | 8.0 | 250 | 50 | 4.0% | 0 |
| 4 | direct_threat | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 4 | direct_threat | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 4 | grazing_threat | estimate/baseline | 0.0 | 50 | 50 | 90.0% | 0 |
| 4 | grazing_threat | estimate/fixed | 0.5 | 250 | 50 | 60.0% | 0 |
| 4 | grazing_threat | estimate/variance_aware | 0.5 | 250 | 50 | 59.6% | 0 |
| 4 | grazing_threat | estimate/fixed | 1.0 | 250 | 50 | 36.8% | 0 |
| 4 | grazing_threat | estimate/variance_aware | 1.0 | 250 | 50 | 40.0% | 0 |
| 4 | grazing_threat | estimate/fixed | 2.0 | 250 | 50 | 19.6% | 0 |
| 4 | grazing_threat | estimate/variance_aware | 2.0 | 250 | 50 | 26.8% | 0 |
| 4 | grazing_threat | estimate/fixed | 4.0 | 250 | 50 | 9.2% | 0 |
| 4 | grazing_threat | estimate/variance_aware | 4.0 | 250 | 50 | 14.4% | 0 |
| 4 | grazing_threat | estimate/fixed | 8.0 | 250 | 50 | 6.0% | 0 |
| 4 | grazing_threat | estimate/variance_aware | 8.0 | 250 | 50 | 8.0% | 0 |
| 4 | grazing_threat | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 4 | grazing_threat | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 4 | multiple_threats | estimate/baseline | 0.0 | 50 | 50 | 52.0% | 0 |
| 4 | multiple_threats | estimate/fixed | 0.5 | 250 | 50 | 12.0% | 0 |
| 4 | multiple_threats | estimate/variance_aware | 0.5 | 250 | 50 | 15.2% | 0 |
| 4 | multiple_threats | estimate/fixed | 1.0 | 250 | 50 | 4.4% | 0 |
| 4 | multiple_threats | estimate/variance_aware | 1.0 | 250 | 50 | 5.6% | 0 |
| 4 | multiple_threats | estimate/fixed | 2.0 | 250 | 50 | 0.8% | 0 |
| 4 | multiple_threats | estimate/variance_aware | 2.0 | 250 | 50 | 2.8% | 0 |
| 4 | multiple_threats | estimate/fixed | 4.0 | 250 | 50 | 0.4% | 0 |
| 4 | multiple_threats | estimate/variance_aware | 4.0 | 250 | 50 | 0.8% | 0 |
| 4 | multiple_threats | estimate/fixed | 8.0 | 250 | 50 | 0.0% | 0 |
| 4 | multiple_threats | estimate/variance_aware | 8.0 | 250 | 50 | 0.0% | 0 |
| 4 | multiple_threats | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 4 | multiple_threats | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 4 | safe_miss | estimate/baseline | 0.0 | 50 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/fixed | 0.5 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/variance_aware | 0.5 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/fixed | 1.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/variance_aware | 1.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/fixed | 2.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/variance_aware | 2.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/fixed | 4.0 | 250 | 50 | 99.6% | 0 |
| 4 | safe_miss | estimate/variance_aware | 4.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/fixed | 8.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | estimate/variance_aware | 8.0 | 250 | 50 | 100.0% | 0 |
| 4 | safe_miss | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 4 | safe_miss | static/reference | N/A | 50 | 50 | 100.0% | 0 |
| 8 | direct_threat | estimate/baseline | 0.0 | 50 | 50 | 100.0% | 0 |
| 8 | direct_threat | estimate/fixed | 0.5 | 250 | 50 | 98.4% | 0 |
| 8 | direct_threat | estimate/variance_aware | 0.5 | 250 | 50 | 98.4% | 0 |
| 8 | direct_threat | estimate/fixed | 1.0 | 250 | 50 | 88.4% | 0 |
| 8 | direct_threat | estimate/variance_aware | 1.0 | 250 | 50 | 91.6% | 0 |
| 8 | direct_threat | estimate/fixed | 2.0 | 250 | 50 | 66.4% | 0 |
| 8 | direct_threat | estimate/variance_aware | 2.0 | 250 | 50 | 72.0% | 0 |
| 8 | direct_threat | estimate/fixed | 4.0 | 250 | 50 | 36.0% | 0 |
| 8 | direct_threat | estimate/variance_aware | 4.0 | 250 | 50 | 38.8% | 0 |
| 8 | direct_threat | estimate/fixed | 8.0 | 250 | 50 | 15.2% | 0 |
| 8 | direct_threat | estimate/variance_aware | 8.0 | 250 | 50 | 18.4% | 0 |
| 8 | direct_threat | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 8 | direct_threat | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 8 | grazing_threat | estimate/baseline | 0.0 | 50 | 50 | 100.0% | 0 |
| 8 | grazing_threat | estimate/fixed | 0.5 | 250 | 50 | 98.8% | 0 |
| 8 | grazing_threat | estimate/variance_aware | 0.5 | 250 | 50 | 99.2% | 0 |
| 8 | grazing_threat | estimate/fixed | 1.0 | 250 | 50 | 93.2% | 0 |
| 8 | grazing_threat | estimate/variance_aware | 1.0 | 250 | 50 | 94.4% | 0 |
| 8 | grazing_threat | estimate/fixed | 2.0 | 250 | 50 | 72.0% | 0 |
| 8 | grazing_threat | estimate/variance_aware | 2.0 | 250 | 50 | 76.0% | 0 |
| 8 | grazing_threat | estimate/fixed | 4.0 | 250 | 50 | 45.6% | 0 |
| 8 | grazing_threat | estimate/variance_aware | 4.0 | 250 | 50 | 46.4% | 0 |
| 8 | grazing_threat | estimate/fixed | 8.0 | 250 | 50 | 22.4% | 0 |
| 8 | grazing_threat | estimate/variance_aware | 8.0 | 250 | 50 | 28.0% | 0 |
| 8 | grazing_threat | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 8 | grazing_threat | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 8 | multiple_threats | estimate/baseline | 0.0 | 50 | 50 | 100.0% | 0 |
| 8 | multiple_threats | estimate/fixed | 0.5 | 250 | 50 | 96.0% | 0 |
| 8 | multiple_threats | estimate/variance_aware | 0.5 | 250 | 50 | 95.6% | 0 |
| 8 | multiple_threats | estimate/fixed | 1.0 | 250 | 50 | 83.6% | 0 |
| 8 | multiple_threats | estimate/variance_aware | 1.0 | 250 | 50 | 84.0% | 0 |
| 8 | multiple_threats | estimate/fixed | 2.0 | 250 | 50 | 40.4% | 0 |
| 8 | multiple_threats | estimate/variance_aware | 2.0 | 250 | 50 | 46.8% | 0 |
| 8 | multiple_threats | estimate/fixed | 4.0 | 250 | 50 | 8.0% | 0 |
| 8 | multiple_threats | estimate/variance_aware | 4.0 | 250 | 50 | 14.8% | 0 |
| 8 | multiple_threats | estimate/fixed | 8.0 | 250 | 50 | 1.6% | 0 |
| 8 | multiple_threats | estimate/variance_aware | 8.0 | 250 | 50 | 2.4% | 0 |
| 8 | multiple_threats | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 8 | multiple_threats | static/reference | N/A | 50 | 50 | 0.0% | 0 |
| 8 | safe_miss | estimate/baseline | 0.0 | 50 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/fixed | 0.5 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/variance_aware | 0.5 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/fixed | 1.0 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/variance_aware | 1.0 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/fixed | 2.0 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/variance_aware | 2.0 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/fixed | 4.0 | 250 | 50 | 99.6% | 0 |
| 8 | safe_miss | estimate/variance_aware | 4.0 | 250 | 50 | 100.0% | 0 |
| 8 | safe_miss | estimate/fixed | 8.0 | 250 | 50 | 99.2% | 0 |
| 8 | safe_miss | estimate/variance_aware | 8.0 | 250 | 50 | 99.6% | 0 |
| 8 | safe_miss | oracle/reference | N/A | 50 | 50 | 100.0% | 0 |
| 8 | safe_miss | static/reference | N/A | 50 | 50 | 100.0% | 0 |
