import os
import shutil
import recbole

# get recbole library path
recbole_root = os.path.dirname(recbole.__file__)

# path to our patched files
patch_dir = os.path.join(os.path.dirname(__file__), "patches")

# map from patch filename → target subpath under recbole_root
PATCH_MAP = {
    "case_study.py": ["utils", "case_study.py"],
    "slimelastic.py": ["model", "general_recommender", "slimelastic.py"],
}

for fname, rel_parts in PATCH_MAP.items():
    src = os.path.join(patch_dir, fname)
    dst = os.path.join(recbole_root, *rel_parts)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"patch file not found: {src}")
    if not os.path.isfile(dst):
        raise FileNotFoundError(f"target file not found: {dst}")
    print(f"Patching {dst} ← {src}")
    shutil.copy(src, dst)

print("RecBole patched successfully.")
