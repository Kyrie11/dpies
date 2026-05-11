from pathlib import Path
import argparse
import numpy as np
import zipfile

def check(path: Path):
	try:
		if not zipfile.is_zipfile(path):
			return False, "not a zip/npz file"
		with np.load(path, allow_pickle=False) as z:
			for k in z.files:
				_ = z[k]
		return True, ""
	except Exception as e:
		return False, repr(e)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cache-dir", required=True)
	ap.add_argument("--delete", action="store_true")
	args = ap.parse_args()

	root = Path(args.cache_dir)
	files = sorted(root.rglob("*.npz"))
	print("total npz:", len(files))

	bad = []
	for i, f in enumerate(files):
		ok, err = check(f)
		if not ok:
			bad.append((f, err))
			print("[BAD]", f, err, flush=True)
			if args.delete:
				f.unlink(missing_ok=True)
		if (i + 1) % 10000 == 0:
			print("checked", i + 1, "bad", len(bad), flush=True)

	print("bad_count:", len(bad))
	if bad:
		out = root / "bad_npz_files.txt"
		out.write_text("\n".join(f"{p}\t{e}" for p, e in bad), encoding="utf-8")
		print("written:", out)

if __name__ == "__main__":
	main()