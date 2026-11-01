/* FPX probe: force-included into the generated code by -DRT_FPX_PROBE=ON (its own
   build directory). drippu's FP helper bodies call RECOMP_FPX_PROBE(1) when its host fast path was taken and
   RECOMP_FPX_PROBE(2) when not; this counts both, with the guest's FPSR/FPCR, in the runtime (fpx_probe.cpp).
   drippu defines the macro as nothing unless it is defined first. */
#ifndef RT_FPX_PROBE_H
#define RT_FPX_PROBE_H
#ifdef __cplusplus
extern "C" {
#endif
void rt_fpx_probe(int kind, unsigned long long fpsr, unsigned long long fpcr);
#ifdef __cplusplus
}
#endif
#define RECOMP_FPX_PROBE(k) rt_fpx_probe((k), (c)->fpsr, (c)->fpcr)
#endif
