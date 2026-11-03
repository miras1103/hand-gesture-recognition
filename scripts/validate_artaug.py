from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_ORIGINAL_DATA_DIR = PROJECT_DIR.parent / "dataset"
REVIEW_FIELDS = (
    "filename",
    "class",
    "technical_status",
    "manual_review_status",
    "accepted",
    "notes",
)
MANIFEST_FIELDS = (
    "filename",
    "class",
    "source",
    "original_filename",
    "split",
    "approval_status",
)
CLASSES = ("peace", "stop", "thumbs_up")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORMAT_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "BMP": {".bmp"},
    "WEBP": {".webp"},
}
TECHNICAL_NOTE_PREFIX = "Technical checks: "


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_fingerprints(path: Path) -> tuple[str, str, str, int, int]:
    file_digest = _file_sha256(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image_format = image.format or ""
            image.verify()
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            width, height = oriented.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
            pixel_digest = hashlib.sha256(
                f"{width}x{height}:".encode("ascii") + oriented.tobytes()
            ).hexdigest()
    return file_digest, pixel_digest, image_format, width, height


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(MANIFEST_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Manifest must contain these columns: {', '.join(MANIFEST_FIELDS)}"
            )
        return [
            {key: (value or "") for key, value in row.items()}
            for row in reader
        ]


def _original_training_sources(
    data_dir: Path, manifest_rows: list[dict[str, str]]
) -> dict[tuple[str, str], tuple[Path, str]]:
    sources: dict[tuple[str, str], tuple[Path, str]] = {}
    resolved_data_dir = data_dir.resolve()
    for row in manifest_rows:
        if (
            row.get("source", "").lower() != "original"
            or row.get("split", "").lower() != "train"
            or row.get("approval_status", "").lower() != "approved"
            or row.get("class") not in CLASSES
        ):
            continue
        relative = PurePosixPath(row["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if (
            len(relative.parts) != 3
            or relative.parts[0] != "artaug_input"
            or relative.parts[1] != row["class"]
        ):
            continue
        source_path = (data_dir / Path(*relative.parts)).resolve()
        if not source_path.is_relative_to(resolved_data_dir):
            continue
        if (
            source_path.parent.name != row["class"]
            or not source_path.is_file()
        ):
            continue
        original_filename = row.get("original_filename") or source_path.name
        sources[(row["class"], source_path.stem.casefold())] = (
            source_path,
            original_filename,
        )
    return sources


def _scan_originals(original_data_dir: Path) -> tuple[dict[str, set[str]], list[str]]:
    if not original_data_dir.is_dir():
        raise FileNotFoundError(
            f"Original dataset directory not found: {original_data_dir}. "
            "Pass --original-data-dir to specify it."
        )
    fingerprints: dict[str, set[str]] = {"file": set(), "pixels": set()}
    errors: list[str] = []
    for class_name in CLASSES:
        class_dir = original_data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Original dataset is missing class folder: {class_dir}"
            )
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                file_hash, pixel_hash, _, _, _ = _image_fingerprints(path)
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as exc:
                errors.append(f"{path}: {exc}")
                continue
            fingerprints["file"].add(file_hash)
            fingerprints["pixels"].add(pixel_hash)
    return fingerprints, errors


def _existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(REVIEW_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Existing review report must contain: {', '.join(REVIEW_FIELDS)}"
            )
        return {
            row["filename"]: {key: (value or "") for key, value in row.items()}
            for row in reader
            if row.get("filename")
        }


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in REVIEW_FIELDS} for row in rows)
    os.replace(temporary_path, path)


def _candidate_paths(generated_dir: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"ArtAug output folder not found: {generated_dir}")
    for path in sorted(generated_dir.rglob("*")):
        if path.is_file():
            candidates.append((path.parent.name, path))
    return candidates


def validate_artaug_images(
    data_dir: Path,
    original_data_dir: Path,
    review_csv: Path,
) -> list[dict[str, str]]:
    data_dir = data_dir.resolve()
    original_data_dir = original_data_dir.resolve()
    generated_dir = data_dir / "artaug_generated"
    manifest_rows = _read_manifest(data_dir / "manifest.csv")
    training_sources = _original_training_sources(data_dir, manifest_rows)
    original_fingerprints, original_errors = _scan_originals(original_data_dir)
    prior_reviews = _existing_review(review_csv)
    raw_candidates = _candidate_paths(generated_dir)
    rows: list[dict[str, str]] = []
    generated_fingerprints: list[dict[str, Any]] = []

    for folder_class, path in raw_candidates:
        filename = _relative_posix(path, data_dir)
        issues: list[str] = []
        if folder_class not in CLASSES or path.parent.name != folder_class:
            issues.append(f"unexpected class folder {folder_class!r}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            issues.append(f"unsupported image extension {path.suffix or '(none)'}")
        if not path.stem.casefold().endswith("_artaug"):
            issues.append("filename must end with _artaug before its extension")

        source_entry: tuple[Path, str] | None = None
        original_stem = (
            path.stem[: -len("_artaug")]
            if path.stem.casefold().endswith("_artaug")
            else ""
        )
        if folder_class in CLASSES and original_stem:
            source_entry = training_sources.get(
                (folder_class, original_stem.casefold())
            )
        if source_entry is None:
            issues.append(
                "no approved original training image with this class/name is in the manifest"
            )

        file_hash: str | None = None
        pixel_hash: str | None = None
        try:
            file_hash, pixel_hash, image_format, width, height = _image_fingerprints(path)
            if path.suffix.lower() not in FORMAT_EXTENSIONS.get(image_format, set()):
                issues.append(
                    f"extension {path.suffix.lower()} does not match decoded format {image_format}"
                )
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            issues.append(f"cannot decode a valid image ({exc})")
            image_format = ""
            width = 0
            height = 0

        if file_hash in original_fingerprints["file"] or pixel_hash in original_fingerprints["pixels"]:
            issues.append("exact image duplicate exists in the original dataset")
        if file_hash is not None:
            generated_fingerprints.append(
                {
                    "path": path,
                    "folder_class": folder_class,
                    "file_hash": file_hash,
                    "pixel_hash": pixel_hash,
                }
            )
        prior = prior_reviews.get(filename, {})
        manual_status = prior.get("manual_review_status") or "pending"
        accepted = prior.get("accepted", "")
        notes = prior.get("notes", "")
        if notes.startswith(TECHNICAL_NOTE_PREFIX):
            notes = ""
        rows.append(
            {
                "filename": filename,
                "class": folder_class,
                "technical_status": "passed" if not issues else "failed",
                "manual_review_status": manual_status,
                "accepted": accepted,
                "notes": notes,
                "_issues": issues,
                "_original_filename": source_entry[1] if source_entry else "",
                "_path": str(path),
                "_image_format": image_format,
                "_width": str(width),
                "_height": str(height),
                "_file_hash": file_hash or "",
                "_pixel_hash": pixel_hash or "",
            }
        )

    by_name: dict[str, list[dict[str, Any]]] = {}
    by_file_hash: dict[str, list[dict[str, Any]]] = {}
    by_pixel_hash: dict[str, list[dict[str, Any]]] = {}
    for item in generated_fingerprints:
        by_name.setdefault(item["path"].name.casefold(), []).append(item)
        by_file_hash.setdefault(item["file_hash"], []).append(item)
        by_pixel_hash.setdefault(item["pixel_hash"], []).append(item)

    for items in by_name.values():
        if len(items) > 1:
            for item in items:
                for row in rows:
                    if row["_path"] == str(item["path"]):
                        row["_issues"].append("duplicate filename occurs in another output folder")
    for items in by_file_hash.values():
        if len(items) > 1:
            for item in items:
                for row in rows:
                    if row["_path"] == str(item["path"]):
                        row["_issues"].append("duplicate file content exists in ArtAug outputs")
    for items in by_pixel_hash.values():
        if len(items) > 1:
            for item in items:
                for row in rows:
                    if row["_path"] == str(item["path"]):
                        row["_issues"].append("exact duplicate image exists in ArtAug outputs")

    for row in rows:
        issues = row.pop("_issues")
        row["technical_status"] = "passed" if not issues else "failed"
        technical_note = (
            f"{TECHNICAL_NOTE_PREFIX}{'; '.join(dict.fromkeys(issues))}"
            if issues
            else ""
        )
        manual_note = row["notes"].strip()
        row["notes"] = "; ".join(note for note in (manual_note, technical_note) if note)
        for key in (
            "_original_filename",
            "_path",
            "_image_format",
            "_width",
            "_height",
            "_file_hash",
            "_pixel_hash",
        ):
            row.pop(key)

    _write_review(review_csv, rows)
    if original_errors:
        print(
            f"Warning: {len(original_errors)} original image(s) could not be decoded; "
            "those files were excluded from exact-duplicate matching."
        )
        for error in original_errors:
            print(f"  {error}")
    return rows


def sync_approved_to_manifest(data_dir: Path, review_rows: list[dict[str, str]]) -> int:
    manifest_path = data_dir / "manifest.csv"
    manifest_rows = _read_manifest(manifest_path)
    existing_names = {row["filename"] for row in manifest_rows}
    additions: list[dict[str, str]] = []
    for review in review_rows:
        if (
            review["technical_status"] != "passed"
            or review["manual_review_status"].strip().lower() != "approved"
            or review["accepted"].strip().lower() not in {"true", "yes", "1"}
        ):
            continue
        generated_relative = PurePosixPath(review["filename"])
        generated_path = data_dir / Path(*generated_relative.parts)
        if (
            generated_relative.is_absolute()
            or ".." in generated_relative.parts
            or not generated_path.is_file()
            or generated_path.parent.name != review["class"]
        ):
            raise ValueError(f"Approved review row has an invalid image path: {review['filename']}")
        original_stem = generated_path.stem[: -len("_artaug")]
        source_candidates = [
            row
            for row in manifest_rows
            if row.get("class") == review["class"]
            and row.get("source", "").lower() == "original"
            and row.get("split", "").lower() == "train"
            and row.get("approval_status", "").lower() == "approved"
            and Path(row["filename"].replace("/", "\\")).stem.casefold()
            == original_stem.casefold()
        ]
        if len(source_candidates) != 1:
            raise ValueError(
                f"Approved output {review['filename']} must map to exactly one "
                "approved original training manifest row."
            )
        if review["filename"] in existing_names:
            continue
        additions.append(
            {
                "filename": review["filename"],
                "class": review["class"],
                "source": "ArtAug",
                "original_filename": source_candidates[0]["original_filename"],
                "split": "train",
                "approval_status": "approved",
            }
        )
        existing_names.add(review["filename"])

    if not additions:
        return 0

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=manifest_path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(
                {key: row.get(key, "") for key in MANIFEST_FIELDS}
                for row in [*manifest_rows, *additions]
            )
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return len(additions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ArtAug outputs and preserve a separate human review report."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--original-data-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DATA_DIR,
        help="Original dataset folder with peace/stop/thumbs_up subfolders.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Review report path (default: <data-dir>/artaug_review.csv).",
    )
    parser.add_argument(
        "--sync-approved-manifest",
        action="store_true",
        help=(
            "Append only technically passed images explicitly marked "
            "manual_review_status=approved and accepted=true."
        ),
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    review_csv = (
        args.review_csv.expanduser().resolve()
        if args.review_csv
        else data_dir / "artaug_review.csv"
    )
    rows = validate_artaug_images(
        data_dir=data_dir,
        original_data_dir=args.original_data_dir.expanduser().resolve(),
        review_csv=review_csv,
    )
    if args.sync_approved_manifest:
        added = sync_approved_to_manifest(data_dir, rows)
        print(f"Manifest updated with {added} explicitly approved ArtAug image(s).")
    technical_pass = sum(row["technical_status"] == "passed" for row in rows)
    human_pending = sum(row["manual_review_status"].lower() == "pending" for row in rows)
    human_accepted = sum(
        row["technical_status"] == "passed"
        and row["manual_review_status"].lower() == "approved"
        and row["accepted"].strip().lower() in {"true", "yes", "1"}
        for row in rows
    )
    print(f"ArtAug images scanned: {len(rows)}")
    print(f"Passed technical checks: {technical_pass}")
    print(f"Still requiring manual review: {human_pending}")
    print(f"Manually accepted and technically valid: {human_accepted}")
    print(f"Review report: {review_csv}")
    print("Technical checks do not confirm gesture identity or image quality.")


if __name__ == "__main__":
    main()
