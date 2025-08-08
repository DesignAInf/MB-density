# MB Density — Paper Simulations (v0.2.0)

This repository recreates the **simulation/visualization code for each figure** in the manuscript,
with a **fixed seed** and **documented defaults**.

- Version/Tag: **v0.2.0**
- Global seed: **42**
- KSG defaults: `k=5`
- Confidence/hyper-parameters (where used): `epsilon=1e-6`, `delta=0.05`
- Per-figure parameters are set from the paper (Appendix A for Figs. 6–8 and main text for 2–5,10).

> These scripts are faithful implementations based on the descriptions in the manuscript.  
> Figures 6–8 follow **Appendix A** specs closely. Figure 9 demonstrates the online KSG + joint-learning
> pipeline in a simplified but consistent manner.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Example: reproduce Fig. 6
python scripts/fig6_dynamic_3d.py --out outputs/fig6

# Reproduce all
python scripts/run_all.py --out outputs
```
Outputs are written as `.png` images and `.npy` arrays where appropriate.

## Scripts mapping
- `scripts/fig2_trajectories_2d.py` — Two trajectories (low vs high ρ) on F(x,y)=x^2+y^2.  (Sec. text)  
- `scripts/fig3_topology_surface.py` — 3D surface of high-density ρ(x).  (Sec. text)  
- `scripts/fig4_blocked_minimization.py` — Free-energy landscape blocked by high-ρ barrier.  (Sec. text)  
- `scripts/fig5_modulation_panels.py` — Side-by-side 3D surfaces: low vs high ρ.  (Fig. 5; main)  
- `scripts/fig6_dynamic_3d.py` — Algorithm agent in dynamic 3D ρ with two moving Gaussians.  (App. A.1)  
- `scripts/fig7_advanced_3d.py` — Four blobs + micro obstacles + random Fourier field.  (App. A.2)  
- `scripts/fig8_ultra_complex_3d.py` — Three blobs + micro obstacles + random Fourier field. (App. A.3)  
- `scripts/fig9_agent_pathways_ksg.py` — Online KSG + joint learning pathways map. (App. A.4)  
- `scripts/fig10_streamlines_modulated.py` — Streamlines of -(1-ρ)∇F + trajectories. (Main text)

Use `--help` on each script to see figure-specific options.

## License
MIT
