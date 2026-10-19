#!/usr/bin/env python3
"""Predict the IsPETase S185H ddG using the released XuanWu ESM2-t30-150M checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchdrug import core, data


EXAMPLE_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    """Locate the XuanWu repository root containing the modelcomb package."""
    for candidate in (EXAMPLE_DIR, *EXAMPLE_DIR.parents):
        if (candidate / "modelcomb" / "code").is_dir():
            return candidate
    raise RuntimeError(
        "Unable to locate the XuanWu repository root. Place this example "
        "inside the original source repository before running it."
    )


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import these modules to register their classes with TorchDrug. The underscore
# aliases indicate that the modules do not need to be called directly here.
from modelcomb.code import model as _model_registry  # noqa: E402,F401
from modelcomb.code import residue_constants  # noqa: E402
from modelcomb.code import task as _task_registry  # noqa: E402,F401
from modelcomb.newcode import dataset_s669_esm2 as _dataset_registry  # noqa: E402,F401
from modelcomb.newcode import model_local as _local_model_registry  # noqa: E402,F401
from modelcomb.newcode import task_weighted as _weighted_task_registry  # noqa: E402,F401


AUXILIARY_CHECKPOINT_KEYS = {"mean", "std", "weight", "bin_edges", "bin_weights"}


def apply_placeholders(config: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    text = yaml.safe_dump(config)
    for key, value in mapping.items():
        text = text.replace(f"${{{key}}}", str(value))
    return yaml.safe_load(text)


def set_input_dim(config: dict[str, Any], input_dim: int) -> None:
    model_cfg = config["task"]["model"]
    if model_cfg.get("class") in {
        "models.BindModel",
        "models.BindModelLocal",
        "models.BindModelLocalAttnDiff",
    }:
        model_cfg = model_cfg["model"]
    model_cfg["input_dim"] = input_dim


def read_and_validate_input(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            f"This example must contain exactly one record; found {len(rows)}"
        )

    required = {"pdbid", "wt", "mut", "substitution", "mutation", "ddg"}
    missing_columns = required.difference(rows[0])
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")
    if rows[0]["substitution"] != "S185H" or rows[0]["mutation"] != "A185H":
        raise ValueError(
            "This example requires substitution=S185H and mutation=A185H"
        )
    return rows


def validate_files(rows: list[dict[str, str]], input_dir: Path, esm_dir: Path) -> None:
    missing: list[Path] = []
    for row in rows:
        for key, subdir in (("wt", "wt"), ("mut", "mut")):
            stem = row[key]
            pdb_path = input_dir / f"{stem}.pdb"
            embedding_path = esm_dir / subdir / f"{stem}.pt"
            if not pdb_path.is_file():
                missing.append(pdb_path)
            if not embedding_path.is_file():
                missing.append(embedding_path)
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(map(str, missing)))


def predict(task: torch.nn.Module, dataset, device: torch.device) -> list[float]:
    task = task.to(device)
    task.eval()
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            prediction = task.predict(batch, all_loss=None, metric=None).view(-1)
            values.extend(float(item) for item in prediction.detach().cpu().tolist())
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the ddG of IsPETase S185H")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; by default, automatically select CUDA or CPU",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "output" / "prediction_output.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = EXAMPLE_DIR / "config" / "model_config.yaml"
    checkpoint_path = EXAMPLE_DIR / "checkpoint" / "best_model.pth"
    input_dir = EXAMPLE_DIR / "input"
    input_csv = input_dir / "data.csv"
    esm_dir = EXAMPLE_DIR / "esm2_cache"
    runtime_cache = EXAMPLE_DIR / "runtime_cache"

    for path in (config_path, checkpoint_path, input_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_and_validate_input(input_csv)
    validate_files(rows, input_dir, esm_dir)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no available GPU was detected")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda:0" if use_cuda else "cpu")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    input_dim = (
        residue_constants.atom_type_num
        + len(data.Protein.residue2id)
        + 1
        + int(raw_config["dataset"].get("esm_dim", 0))
    )
    config = apply_placeholders(
        raw_config,
        {
            "DATA_DIR": input_dir,
            "SMEGA_DIR": input_dir,
            "S4126_DIR": input_dir,
            "TEST_FOLD": "unused",
            "INPUT_DIM": input_dim,
            "CACHE_DIR": runtime_cache,
        },
    )
    set_input_dim(config, input_dim)

    dataset_config = dict(config["dataset"])
    dataset_config.update(
        {
            "class": "datasets.S669SingleChainESM2",
            "path": str(input_dir),
            "csv_file": input_csv.name,
            "esm_cache_dir": str(esm_dir),
            "esm_wt_subdir": "wt",
            "esm_mut_subdir": "mut",
            "cache_dir": str(runtime_cache),
            "cache_write": False,
            "validate_pdbs": True,
        }
    )
    dataset_config.pop("split", None)
    dataset = core.Configurable.load_config_dict(dataset_config)
    _, _, test_set = dataset.split()
    if len(test_set) != 1:
        raise RuntimeError(
            f"Exactly one record should remain after PDB validation; "
            f"found {len(test_set)}"
        )

    task = core.Configurable.load_config_dict(config["task"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = task.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key not in AUXILIARY_CHECKPOINT_KEYS
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )

    predictions = predict(task, test_set, device)
    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction; found {len(predictions)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pdbid",
                "substitution",
                "model_locator",
                "ddg_pred_kcal_mol",
                "execution_device",
            ]
        )
        writer.writerow(
            [
                rows[0]["pdbid"],
                rows[0]["substitution"],
                rows[0]["mutation"],
                predictions[0],
                str(device),
            ]
        )

    print(f"Device: {device}")
    print(
        f"Substitution: {rows[0]['substitution']} "
        f"(model locator: {rows[0]['mutation']})"
    )
    print(f"Predicted ddG: {predictions[0]:.6f} kcal/mol")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
