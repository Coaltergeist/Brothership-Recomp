/* Force-included into FPX builds: counts drippu's native-FP fast-path hits (1)
   against fallbacks to the exact body (2), so a test run can show the fast path
   was actually exercised. */
#pragma once
extern unsigned long long dt_fpx_probe_counts[3];
#define RECOMP_FPX_PROBE(k) ((void)++dt_fpx_probe_counts[k])
