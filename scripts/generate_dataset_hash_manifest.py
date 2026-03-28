from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[1]
RUN_METADATA_PATH = ROOT / "results" / "run_metadata.json"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "dataset_hash_manifest.csv"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_uint64_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.uint64).tobytes()


def _dataframe_digest_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    cols = [str(c) for c in df.columns]
    dtypes = [str(t) for t in df.dtypes]
    h.update(json.dumps(cols, ensure_ascii=True).encode("utf-8"))
    h.update(json.dumps(dtypes, ensure_ascii=True).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    h.update(_to_uint64_bytes(row_hashes))
    return h.hexdigest()


def _series_digest_sha256(s: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(str(s.name).encode("utf-8"))
    h.update(str(s.dtype).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(s, index=True).to_numpy(dtype=np.uint64)
    h.update(_to_uint64_bytes(row_hashes))
    return h.hexdigest()


def _combined_digest_sha256(
    dataset_alias: str,
    data_id: int,
    n_rows: int,
    n_features: int,
    feature_digest: str,
    target_digest: str,
) -> str:
    h = hashlib.sha256()
    h.update(str(dataset_alias).encode("utf-8"))
    h.update(str(data_id).encode("utf-8"))
    h.update(str(n_rows).encode("utf-8"))
    h.update(str(n_features).encode("utf-8"))
    h.update(feature_digest.encode("utf-8"))
    h.update(target_digest.encode("utf-8"))
    return h.hexdigest()


def _targets_from_run_metadata(path: Path) -> List[Tuple[str, int]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    profiles: Dict[str, Dict] = payload.get("dataset_profiles", {})
    out: List[Tuple[str, int]] = []
    for alias, profile in profiles.items():
        data_id = profile.get("openml_data_id")
        if data_id is None:
            continue
        try:
            out.append((str(alias), int(data_id)))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[0].lower())


def _compute_row(dataset_alias: str, data_id: int) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset_alias,
        "openml_data_id": data_id,
        "source_url": f"https://www.openml.org/d/{data_id}",
        "status": "ok",
        "error": "",
        "generated_utc": _now_utc_iso(),
        "hash_method": "pandas_hash_rows_sha256_v1",
        "n_rows": np.nan,
        "n_features": np.nan,
        "feature_digest_sha256": "",
        "target_digest_sha256": "",
        "combined_digest_sha256": "",
    }

    try:
        bunch = fetch_openml(data_id=data_id, as_frame=True)
        X = pd.DataFrame(bunch.data)
        y = pd.Series(bunch.target, name="target")
        n_rows = int(len(X))
        n_features = int(X.shape[1])
        feat_digest = _dataframe_digest_sha256(X)
        target_digest = _series_digest_sha256(y)
        combo_digest = _combined_digest_sha256(
            dataset_alias=dataset_alias,
            data_id=data_id,
            n_rows=n_rows,
            n_features=n_features,
            feature_digest=feat_digest,
            target_digest=target_digest,
        )
        row.update(
            {
                "n_rows": n_rows,
                "n_features": n_features,
                "feature_digest_sha256": feat_digest,
                "target_digest_sha256": target_digest,
                "combined_digest_sha256": combo_digest,
            }
        )
    except Exception as exc:
        row["status"] = "fetch_or_hash_failed"
        row["error"] = str(exc).replace("\n", " ").strip()
    return row


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "dataset",
        "openml_data_id",
        "source_url",
        "status",
        "error",
        "generated_utc",
        "hash_method",
        "n_rows",
        "n_features",
        "feature_digest_sha256",
        "target_digest_sha256",
        "combined_digest_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic dataset content hashes for pinned OpenML dataset IDs "
            "using existing metadata targets."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path (default: results/dataset_hash_manifest.csv).",
    )
    args = parser.parse_args()

    targets = _targets_from_run_metadata(RUN_METADATA_PATH)
    if not targets:
        raise RuntimeError(
            "No dataset targets found in results/run_metadata.json (dataset_profiles)."
        )

    rows = [_compute_row(alias, data_id) for alias, data_id in targets]
    _write_rows(Path(args.output), rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    total = len(rows)
    print(f"[dataset-hash] wrote {args.output} ({ok}/{total} successful rows)")


if __name__ == "__main__":
    main()

