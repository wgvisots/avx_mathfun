# avx_mathfun
Yes, this is yet another fork of the great work from Giovanni Garberoglio (based on the earlier work by (Julien Pommier).

But with some added tools and examples of doing vectorized math with AVX2.

## AVX-512

`src/avx512_mathfun.h` provides 16-wide (`__m512`) versions of the functions in
`src/avx_mathfun.h`, in namespace `avx512_mathfun`:

| AVX/AVX2 (`avx_mathfun`) | AVX-512 (`avx512_mathfun`) |
|---|---|
| `log256_ps`    | `log512_ps`    |
| `exp256_ps`    | `exp512_ps`    |
| `sin256_ps`    | `sin512_ps`    |
| `cos256_ps`    | `cos512_ps`    |
| `sincos256_ps` | `sincos512_ps` |
| `atan(__m256)` | `atan(__m512)` |
| `atan2(__m256, __m256)` | `atan2(__m512, __m512)` |

Only AVX-512F is required, so it runs on any AVX-512 or AVX10 CPU (Skylake-SP and
later Xeons, Ice Lake, Sapphire Rapids, AMD Zen 4/5, ...). Build with `-mavx512f`,
`-march=x86-64-v4`, or `-march=native` on a capable machine.

The algorithms are the same Cephes polynomials, but edge cases (subnormals,
`log(0)`, overflow and underflow in `exp`, signed zeros, infinities and NaN in
`atan`/`atan2`) follow the C library. See the header comment for details.

### Testing without AVX-512 hardware

`runUnitTests512` runs natively if the build machine supports AVX-512F. If it
doesn't, the test runs under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
when one can be found:

```
cmake --preset gcc_release_deb -DSDE_EXECUTABLE=/path/to/sde64   # or: export SDE_PATH=/path/to/sde
cmake --build build && (cd build && ctest -R runUnitTests512 --output-on-failure)
```

If neither is available, the test is reported as disabled rather than failing.
