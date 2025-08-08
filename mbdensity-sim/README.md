# mbdensity-sim (v0.1.0)

Minimal reference repo to reproduce the figures with a **fixed seed** and **default parameters**:
- Tag/Version: **v0.1.0**
- Default params: `r1=3`, `r2=10`, `k=5`, `epsilon=1e-6`, `delta=0.05`
- Seed: **42** (applied to NumPy + Python `random`)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run a small demo simulation and dump outputs to ./outputs/
python scripts/run_sim.py --out ./outputs
```

## Parameters
- **r1, r2**: neighborhood radii/levels for local MI/CMI estimation windows (used to subsample neighbors prior to KSG).
- **k**: k-nearest neighbors for KSG MI/CMI.
- **epsilon (ε)**: small positive regularizer to stabilize divisions when MI is near zero.
- **delta (δ)**: confidence level for concentration-style intervals in summary stats (1-δ is the confidence).

You can override any default via CLI flags. See `python scripts/run_sim.py -h`.

## Notes
- This is **not** a full reproduction of every experiment; it’s a minimal, deterministic scaffold you can extend.
- The structure is deliberately simple and explicit to support review and tagging.
