# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use GitHub's private reporting:
on the repository's **Security** tab, choose **Report a vulnerability**.

Please include:
- the version or commit you tested,
- the function involved (for example `log256_ps` or `atan2(__m512, __m512)`) and the input values
  that trigger the problem,
- the compiler, its flags (`-mavx2`, `-mavx512f`, `-march=...`) and the CPU, and
- what happens: a crash, undefined behavior reported by a sanitizer, a wrong result, and so on.

You can expect a reply within two weeks. Once a fix is ready, it is released and the report is
published as a GitHub security advisory, crediting you unless you'd rather not be named.

## Supported versions

Fixes are made on the latest commit of `main` only.

## What counts as a vulnerability

- Undefined behavior, crashes or hangs caused by input **values**, such as NaN, infinity,
  subnormals or extreme numbers.
- Instructions beyond the documented requirements being executed: `avx_mathfun.h` must work with
  AVX alone, and `avx512_mathfun.h` with AVX-512F alone.

Not vulnerabilities: results within the documented accuracy limits, and running the headers on a
CPU without the instruction set they were compiled for (choosing the right code for the CPU is the
caller's job).
