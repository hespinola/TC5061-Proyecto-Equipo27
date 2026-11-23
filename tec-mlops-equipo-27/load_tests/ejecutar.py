"""Locust + muestreo de contenedores; no construye ni detiene los servicios."""

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess  # nosec B404 -- comandos sin shell
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def ejecutar(command):
    return subprocess.check_output(  # nosec B603 -- argumentos sin shell
        command, text=True, encoding="utf-8", timeout=30, cwd=ROOT
    )


def memoria_bytes(value):
    units = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    for unit in sorted(units, key=len, reverse=True):
        if value.strip().endswith(unit):
            return float(value.strip()[: -len(unit)]) * units[unit]
    raise ValueError(f"Unidad de memoria desconocida: {value}")


def normalizar(raw):
    used, limit = raw["MemUsage"].split("/")
    return {
        "contenedor": raw.get("Name") or raw.get("Container"),
        "cpu_pct": float(raw["CPUPerc"].rstrip("%")),
        "memoria_bytes": memoria_bytes(used),
        "limite_bytes": memoria_bytes(limit),
    }


def graficar(destino, samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (destino / "locust_stats_history.csv").open(encoding="utf-8") as f:
        history = [r for r in csv.DictReader(f) if r["Name"] == "Aggregated"]
    if not history:
        raise ValueError("Locust no produjo historia de solicitudes")
    start = float(history[0]["Timestamp"])
    x = [float(r["Timestamp"]) - start for r in history]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for key in ["50%", "95%", "99%"]:
        axes[0].plot(
            x,
            [float(r[key]) if r[key] not in ("N/A", "") else float("nan") for r in history],
            label=key,
        )
    axes[0].set_ylabel("Latencia (ms)")
    axes[0].legend()
    axes[1].plot(x, [float(r["Requests/s"]) for r in history])
    axes[1].set_ylabel("Peticiones/s")
    axes[2].plot(
        x,
        [
            100 * float(r["Failures/s"]) / float(r["Requests/s"]) if float(r["Requests/s"]) else 0
            for r in history
        ],
    )
    axes[2].set_ylabel("Errores (%)")
    axes[2].set_xlabel("Segundos desde primera muestra (ventanas de Locust)")
    fig.suptitle("Carga HTTP: una inferencia por petición")
    fig.tight_layout()
    fig.savefig(destino / "desempeno.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in sorted({r["contenedor"] for r in samples}):
        rows = [r for r in samples if r["contenedor"] == name]
        axes[0].plot([r["segundos"] for r in rows], [r["cpu_pct"] for r in rows], label=name)
        axes[1].plot(
            [r["segundos"] for r in rows], [r["memoria_bytes"] / 1024**2 for r in rows], label=name
        )
    axes[0].set_ylabel("CPU (%) según motor")
    axes[1].set_ylabel("Memoria (MiB)")
    axes[1].set_xlabel("Segundos desde inicio de Locust")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "recursos.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", choices=["docker", "podman"], default="docker")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--usuarios", type=int, default=10)
    parser.add_argument("--tasa", type=float, default=2)
    parser.add_argument("--segundos", type=int, default=120)
    parser.add_argument("--intervalo", type=float, default=2)
    parser.add_argument("--destino", type=Path, required=True)
    args = parser.parse_args()
    if min(args.usuarios, args.tasa, args.segundos, args.intervalo) <= 0:
        parser.error("Los parámetros de carga y muestreo deben ser positivos")
    if args.usuarios / args.tasa >= args.segundos:
        parser.error("La duración debe superar el tiempo de subida de usuarios")
    if not args.host.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Esta prueba está limitada a servicios locales")
    motor = shutil.which(args.motor)
    if motor is None:
        parser.error(f"No se encontró {args.motor}")
    destination = args.destino.resolve()
    if destination.exists():
        parser.error("El destino ya existe; usa una carpeta nueva para conservar evidencia")
    # Verificar ambos contenedores y servicio antes de crear evidencia.
    containers = ["mibici-api", "mibici-mlflow"]
    inspected = json.loads(ejecutar([motor, "inspect", *containers]))
    if not all(c["State"]["Running"] for c in inspected):
        raise RuntimeError("Los dos contenedores deben estar ejecutándose")
    with urlopen(args.host + "/health", timeout=15) as response:  # nosec B310 -- localhost validado
        health = json.load(response)
    if health.get("estado") != "ok":
        raise RuntimeError("El modelo no está listo")
    destination.mkdir(parents=True)
    metadata = {
        "inicio_utc": datetime.now(UTC).isoformat(),
        "parametros": vars(args) | {"destino": str(destination)},
        "health": health,
        "python": sys.version,
        "locust": ejecutar([sys.executable, "-m", "locust", "--version"]).strip(),
        "motor": ejecutar([motor, "version", "--format", "{{json .Server}}"]),
        "recursos_motor": ejecutar(
            [motor, "info", "--format", "{{.NCPU}} CPU; {{.MemTotal}} bytes RAM"]
        )
        if args.motor == "docker"
        else "Consultar límites de la máquina Podman",
        "commit": ejecutar([shutil.which("git"), "rev-parse", "HEAD"]).strip(),
        "codigo_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "contenedores": [
            {
                "nombre": c["Name"],
                "imagen": c["Image"],
                "limites": {
                    k: c["HostConfig"].get(k)
                    for k in ["Memory", "NanoCpus", "CpuQuota", "CpuPeriod"]
                },
            }
            for c in inspected
        ],
        "alcance": "Modelo cerrado; payload fijo; incluye rampa; sin calentamiento excluido. "
        "Generador y contenedores comparten anfitrión. No determina capacidad máxima.",
    }
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(ROOT / "load_tests/locustfile.py"),
        "--headless",
        "--host",
        args.host,
        "-u",
        str(args.usuarios),
        "-r",
        str(args.tasa),
        "-t",
        str(args.segundos),
        "--stop-timeout",
        "35",
        "--csv-full-history",
        "--csv",
        str(destination / "locust"),
        "--html",
        str(destination / "locust.html"),
    ]
    samples, errors = [], []
    start = time.monotonic()
    with (
        (destination / "locust.log").open("w", encoding="utf-8") as log,
        (destination / "stats_raw.jsonl").open("w", encoding="utf-8") as rawfile,
    ):
        process = subprocess.Popen(command, stdout=log, stderr=log, cwd=ROOT)  # nosec B603
        try:
            while process.poll() is None:
                try:
                    output = ejecutar(
                        [motor, "stats", "--no-stream", "--format", "{{json .}}", *containers]
                    )
                    timestamp = datetime.now(UTC).isoformat()
                    elapsed = time.monotonic() - start
                    for line in output.splitlines():
                        raw = json.loads(line)
                        rawfile.write(
                            json.dumps({"utc": timestamp, "segundos": elapsed, "raw": raw}) + "\n"
                        )
                        rawfile.flush()
                        samples.append({"utc": timestamp, "segundos": elapsed, **normalizar(raw)})
                except (subprocess.SubprocessError, ValueError, KeyError) as exc:
                    errors.append(str(exc))
                time.sleep(args.intervalo)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=40)
            metadata.update(
                fin_utc=datetime.now(UTC).isoformat(),
                duracion_observada_s=time.monotonic() - start,
                locust_exit_code=process.returncode,
                errores_muestreo=errors,
            )
            (destination / "ejecucion.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            if samples:
                with (destination / "recursos.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(samples[0]))
                    writer.writeheader()
                    writer.writerows(samples)
    graficar(destination, samples)
    if errors or len({r["contenedor"] for r in samples}) != 2:
        raise RuntimeError("Captura de recursos incompleta; revisar ejecucion.json")
    print(f"Evidencia guardada en {destination}; Locust exit={process.returncode}")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
