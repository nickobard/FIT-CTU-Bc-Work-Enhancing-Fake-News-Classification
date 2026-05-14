from pathlib import Path
import argparse
import yaml
import shutil


def remove_pycache_dirs(root: Path) -> None:
    for pycache_dir in root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            print(f"Removed: {pycache_dir}")

def ensure_artifact_dir_exists(artifact_uri: str, mlruns_root: Path) -> None:
    path = normalize_path(artifact_uri)

    if path.startswith("./mlruns/"):
        relative_part = path[len("./mlruns/"):]
        artifact_path = mlruns_root / relative_part
    else:
        artifact_path = Path(path)

    artifact_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured artifact directory exists: {artifact_path}")

def normalize_path(value: str) -> str:
    value = value.strip()

    if value.startswith("file:"):
        value = value[len("file:"):]

    return value.replace("\\", "/")


def to_relative_mlruns_path(value: str) -> str:
    """
    Converts:
    file:C:/.../mlruns/123/run/artifacts

    to:
    file:./mlruns/123/run/artifacts
    """
    normalized = normalize_path(value)

    marker = "/mlruns/"
    if marker not in normalized:
        return value

    relative_part = normalized.split(marker, 1)[1]

    return f"./mlruns/{relative_part}"


def to_absolute_mlruns_path(value: str, mlruns_root: Path) -> str:
    """
    Converts:
    file:./mlruns/123/run/artifacts

    to:
    file:/absolute/path/to/mlruns/123/run/artifacts
    """
    normalized = normalize_path(value)

    marker = "./mlruns/"
    if marker not in normalized:
        return value

    relative_part = normalized.split(marker, 1)[1]

    absolute_path = (mlruns_root / relative_part).resolve()

    return f"{absolute_path.as_posix()}"


def convert_path(value: str, mode: str, mlruns_root: Path) -> str:
    if mode == "relative":
        return to_relative_mlruns_path(value)

    if mode == "absolute":
        return to_absolute_mlruns_path(value, mlruns_root)

    raise ValueError(f"Unknown mode: {mode}")


def fix_yaml_file(
    path: Path,
    key: str,
    mode: str,
    mlruns_root: Path,
) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return False

    if key not in data or not data[key]:
        return False

    old_value = str(data[key])
    new_value = convert_path(old_value, mode, mlruns_root)

    if key == "artifact_uri":
        ensure_artifact_dir_exists(new_value, mlruns_root)

    if old_value == new_value:
        return True

    data[key] = new_value

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Fixed {key}: {path}")
    print(f"  {old_value}")
    print(f"  -> {new_value}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--relative",
        action="store_true",
        help="Convert artifact paths to relative paths",
    )

    group.add_argument(
        "--absolute",
        action="store_true",
        help="Convert artifact paths to absolute paths",
    )

    args = parser.parse_args()

    mode = "relative" if args.relative else "absolute"

    mlruns_root = Path.cwd().resolve()

    if mlruns_root.name != "mlruns":
        raise RuntimeError(
            f"Current directory must be mlruns, got: {mlruns_root}"
        )

    for experiment_dir in mlruns_root.iterdir():
        if not experiment_dir.is_dir():
            continue

        experiment_meta = experiment_dir / "meta.yaml"

        if not experiment_meta.exists():
            continue

        valid_experiment = fix_yaml_file(
            experiment_meta,
            "artifact_location",
            mode,
            mlruns_root,
        )

        if not valid_experiment:
            continue

        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_meta = run_dir / "meta.yaml"

            if not run_meta.exists():
                continue

            fix_yaml_file(
                run_meta,
                "artifact_uri",
                mode,
                mlruns_root,
            )

    remove_pycache_dirs(mlruns_root)


if __name__ == "__main__":
    main()