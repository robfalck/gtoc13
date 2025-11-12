# GTOC13 - Global Trajectory Optimization Competition Round 13

[![Tests](https://github.com/robfalck/gtoc13/actions/workflows/test.yml/badge.svg)](https://github.com/robfalck/gtoc13/actions/workflows/test.yml)

A trajectory optimization framework for GTOC13 using JAX for high-performance computation.

## Installation

Install the package in development mode:

```bash
python -m pip install -e .
```

Install a modified version of dymos

```bash
python -m pip install git+https://github.com/robfalck/dymos.git@gtoc13
```

Install a jax lambert solver, available as `lamberthub.vallado2013_jax`

```bash
python -m pip install git+https://github.com/robfalck/lamberthub.git@jax
```
