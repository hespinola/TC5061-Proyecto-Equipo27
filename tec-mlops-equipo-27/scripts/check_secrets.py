"""Revisa archivos versionados y pendientes; la baseline contiene falsos positivos auditados."""

from contextlib import chdir
from pathlib import Path
import shutil

# Solo se consulta Git con argumentos fijos y sin shell.
import subprocess  # nosec B404
import sys

from detect_secrets.pre_commit_hook import main as detect_secrets

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "tec-mlops-equipo-27" / ".secrets.baseline"


def main() -> int:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError(
            "Git debe estar instalado para revisar los archivos del repositorio."
        )
    # Ruta del ejecutable resuelta y argumentos fijos, sin shell.
    output = subprocess.check_output(  # nosec B603
        [git_executable, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        shell=False,
    )
    # La baseline solo almacena hashes de hallazgos revisados y no se escanea a sí misma.
    filenames = list(
        dict.fromkeys(
            name for name in output.decode().split("\0") if name and REPO_ROOT / name != BASELINE
        )
    )
    with chdir(REPO_ROOT):
        result = detect_secrets(
            ["--no-verify", "--baseline", str(BASELINE.relative_to(REPO_ROOT)), "--", *filenames]
        )
    if result == 0:
        print(f"Secretos: {len(filenames)} archivos revisados, sin hallazgos nuevos.")
    return result


if __name__ == "__main__":
    sys.exit(main())
