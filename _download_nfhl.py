import sys, time
sys.path.insert(0, "c:/Users/chaitanya/Documents/ML Project")
from src.data_acquisition import download_nfhl_state
import json
from pathlib import Path

RAW = Path("c:/Users/chaitanya/Documents/ML Project/data/raw")
states = ["48","22","12","37","45","13","01","28"]
MAX_ATTEMPTS = 10

for attempt in range(1, MAX_ATTEMPTS + 1):
    pending = []
    for fips in states:
        p = RAW / f"nfhl_sfha_{fips}.geojson"
        if p.exists():
            try:
                data = json.loads(p.read_bytes())
                if data.get("complete"):
                    continue  # done
            except Exception:
                pass
        pending.append(fips)

    if not pending:
        print("All states complete!")
        break

    print(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} | pending: {pending} ---")
    for fips in pending:
        download_nfhl_state(fips, force=False)
        time.sleep(2)  # brief pause between states

# Final summary
print("\n=== Summary ===")
for fips in states:
    p = RAW / f"nfhl_sfha_{fips}.geojson"
    if p.exists():
        data = json.loads(p.read_bytes())
        n = len(data.get("features", []))
        status = "COMPLETE" if data.get("complete") else f"partial ({n} features)"
        print(f"  state {fips}: {status}")
    else:
        print(f"  state {fips}: MISSING")
