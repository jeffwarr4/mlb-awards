from pathlib import Path
from datetime import datetime
import sys, subprocess

# Make "src" importable
sys.path.append(str(Path(__file__).parent / "src"))

YEAR = 2025

def run():
    outdir = Path("predictions") / str(YEAR)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        from predict_current_awards import main as predict_main  # expects: main(YEAR, outdir, timestamp=None) -> (mvp_csv, cy_csv)
        mvp_csv, cy_csv = predict_main(YEAR=YEAR, outdir=outdir, timestamp=ts)
        print("✅ Done (import):", mvp_csv, cy_csv)
    except Exception as e:
        print("Falling back to CLI mode:", e)
        # expects your script to accept --year and --outdir
        cmd = [sys.executable, "src/predict_current_awards.py", "--year", str(YEAR), "--outdir", str(outdir)]
        res = subprocess.run(cmd, check=True)
        print("✅ Done (CLI fallback):", res.returncode)

if __name__ == "__main__":
    run()
