from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/content/drive/MyDrive/learning-cifar10-from-scratch")
COLAB_ROOT = Path("/content/learning-cifar10-from-scratch")
REPO_URL = "https://github.com/Appolinairee/-research-scratch-deep-learning-vision.git"
BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "shared")

DATA_ROOT = PROJECT_ROOT / "data" / BENCHMARK_NAME
MODEL_ROOT = PROJECT_ROOT / "models" / BENCHMARK_NAME
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

if not (PROJECT_ROOT / ".git").exists():
    temporary_root = Path("/content/research_scratch_repo")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    subprocess.run(["git", "clone", REPO_URL, str(temporary_root)], check=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=data/",
            "--exclude=models/",
            f"{temporary_root}/",
            f"{PROJECT_ROOT}/",
        ],
        check=True,
    )
    shutil.rmtree(temporary_root)
else:
    subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "remote", "set-url", "origin", REPO_URL],
        check=True,
    )
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", "main"], check=True)
    subprocess.run(
        [
            "find",
            str(PROJECT_ROOT),
            "-type",
            "d",
            "-name",
            "__pycache__",
            "-prune",
            "-exec",
            "rm",
            "-rf",
            "{}",
            "+",
        ],
        check=True,
    )
    sync = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only", "origin", "main"],
        capture_output=True,
        text=True,
    )
    if sync.returncode != 0:
        raise RuntimeError(
            "Synchronisation Git impossible.\n"
            f"{sync.stderr}\n"
            "Vérifie les changements locaux et la divergence de la branche main."
        )
    if sync.stdout.strip():
        print(sync.stdout.strip())

for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]
sys.path[:] = [entry for entry in sys.path if entry != str(PROJECT_ROOT)]
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Projet synchronisé : {PROJECT_ROOT}")
print(f"Données : {DATA_ROOT}")
print(f"Modèles : {MODEL_ROOT}")
