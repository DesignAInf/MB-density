import argparse, os, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scripts = [
        "fig2_trajectories_2d.py",
        "fig3_topology_surface.py",
        "fig4_blocked_minimization.py",
        "fig5_modulation_panels.py",
        "fig6_dynamic_3d.py",
        "fig7_advanced_3d.py",
        "fig8_ultra_complex_3d.py",
        "fig9_agent_pathways_ksg.py",
        "fig10_streamlines_modulated.py",
    ]
    for s in scripts:
        print("Running", s); sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), s), "--out", os.path.join(args.out, s.replace(".py",""))], check=True)

if __name__=="__main__":
    main()
