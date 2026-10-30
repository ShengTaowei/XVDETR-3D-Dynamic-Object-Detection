"""Convert a nuScenes KITTI export into the layouts used by MonoCoP.

The nuScenes dev-kit exporter writes files using the 32-character sample token
as the filename (``image_2/<token>.png``).  MonoCoP's loaders intentionally use
the conventional KITTI six-digit frame id instead.  This utility makes that
mapping explicit and writes split files together with the images, labels and
calibration files.

Two input forms are supported:

* an existing dev-kit export (the form currently in
  ``Dual-branch-detection/NuScenes``), and
* official nuScenes data, by forwarding to the dev-kit ``KittiConverter``.

Examples (PowerShell)::

    python tools/convert_nuscenes_to_monocop.py \
      --source D:/PythonProject/Dual-branch-detection/NuScenes \
      --output data/NuScenesKITTI --layout nusc --split train

    python tools/convert_nuscenes_to_monocop.py \
      --raw-dataroot D:/datasets/nuScenes --version v1.0-trainval \
      --devkit D:/PythonProject/nuscenes-devkit-master \
      --output data/NuScenesKITTI --layout nusc --split train --max-samples 6000

The conversion is deliberately split into two stages so that the slow raw
export can be resumed after an interruption:

* ``--stage export`` runs step one only: it writes a token-based KITTI export
  (``image_2``/``calib``/``label_2``/``velodyne``/``oxts``) into
  ``--export-dir``.  Samples whose files are already present are skipped, so
  re-running the command only converts what is missing (use ``--overwrite`` to
  force a full re-export).
* ``--stage convert`` runs step two only: it reads that export (``--source``,
  defaulting to ``--export-dir``) and writes the six-digit MonoCoP layout into
  ``--output``.
* ``--stage all`` (default) runs both, which is the original behaviour.

Examples (PowerShell)::

    python tools/convert_nuscenes_to_monocop.py \
      --raw-dataroot D:/datasets/nuScenes --devkit D:/nuscenes-devkit-master \
      --output data/NuScenesKITTI --split train --stage export

    python tools/convert_nuscenes_to_monocop.py \
      --output data/NuScenesKITTI --layout monocop --taxonomy kitti --split train --stage convert

The output is deliberately copy-based (never a symlink), so it can be moved to
another machine.  A ``token_mapping.json`` file records the original sample
token for every numeric id.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NUSC_SPLITS = ("train", "val")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# This is intentionally conservative.  It gives a useful 3-class KITTI view
# while making the lossy nature of the conversion obvious in the report.
KITTI_CLASS_MAP = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}


def _die(message: str) -> "NoReturn":
    raise SystemExit(f"error: {message}")


class _ThrottledWriter(io.TextIOBase):
    """Sink for the dev-kit's per-sample prints that emits coarse progress.

    The sink is constructed *before* ``redirect_stdout`` swaps ``sys.stdout``,
    so the captured stream is the real terminal and progress cannot recurse.
    """

    def __init__(self, prefix: str, total: int, every: int = 500) -> None:
        self.prefix = prefix
        self.total = total
        self.every = max(int(every or 0), 1)
        self.count = 0
        self._next = self.every
        self.stream = sys.stdout

    def write(self, text: str) -> int:
        self.count += text.count("\n")
        if self.count >= self._next:
            self.stream.write(f"{self.prefix}: {self.count}/{self.total}\n")
            self.stream.flush()
            while self.count >= self._next:
                self._next += self.every
        return len(text)

    def flush(self) -> None:
        return None

    def writable(self) -> bool:
        return True


def _exported_sample_files(split_dir: Path, token: str, with_oxts: bool) -> List[Path]:
    """Files the dev-kit writes for one sample (label_2 last, image first)."""
    paths = [
        split_dir / "image_2" / f"{token}.png",
        split_dir / "calib" / f"{token}.txt",
        split_dir / "label_2" / f"{token}.txt",
    ]
    if with_oxts:
        paths.append(split_dir / "oxts" / f"{token}.txt")
    return paths


def _pending_export_samples(
    split_dir: Path,
    samples: Sequence[dict],
    with_oxts: bool,
) -> List[dict]:
    """Keep only samples whose KITTI export is incomplete or missing."""
    pending = []
    for sample in samples:
        token = sample["token"]
        if all(path.is_file() for path in _exported_sample_files(split_dir, token, with_oxts)):
            continue
        pending.append(sample)
    return pending


def _find_table_dir(root: Path, table: str) -> Optional[Path]:
    """Find one table directory in a dev-kit export."""
    candidates = [
        root / table,
        root / f"{table}_2" if table == "image" else root / table,
    ]
    if table == "image":
        candidates += [root / "image_2"]
    if table == "label":
        candidates += [root / "label_2"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _discover_source(source: Path, requested_split: str) -> List[Tuple[str, Path]]:
    """Return ``[(split, directory), ...]`` for flat or split exports."""
    # Official export: root/{train,val}/{image_2,calib,label_2}.
    split_dirs: List[Tuple[str, Path]] = []
    for split, dirname in (("train", "train"), ("val", "val"),
                           ("train", "training"), ("val", "validation")):
        if (source / dirname).is_dir() and _find_table_dir(source / dirname, "image"):
            split_dirs.append((split, source / dirname))
    if split_dirs:
        return [(s, d) for s, d in split_dirs if requested_split == "all" or s == requested_split]

    # MonoCoP/NuScenes exports are usually flat.  Treat them as the requested
    # split; the caller can run the command twice if a train/val partition is
    # desired.
    if _find_table_dir(source, "image"):
        split = "train" if requested_split == "all" else requested_split
        return [(split, source)]
    _die(f"{source} has no image_2/image directory (or train/val subdirectories)")


def _token_from_path(path: Path) -> str:
    return path.stem


def _image_paths(table_dir: Path) -> Dict[str, Path]:
    image_dir = _find_table_dir(table_dir, "image")
    assert image_dir is not None
    paths: Dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{ext}"):
            paths[_token_from_path(path)] = path
    return paths


def _label_path(table_dir: Path, token: str) -> Optional[Path]:
    for dirname in ("label", "label_2"):
        path = table_dir / dirname / f"{token}.txt"
        if path.exists():
            return path
    return None


def _calib_path(table_dir: Path, token: str) -> Path:
    path = table_dir / "calib" / f"{token}.txt"
    if not path.exists():
        _die(f"missing calibration for sample {token}: {path}")
    return path


def _rewrite_labels(path: Optional[Path], taxonomy: str) -> str:
    if path is None:
        return ""
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split()
        if not fields:
            continue
        if taxonomy == "kitti":
            mapped = KITTI_CLASS_MAP.get(fields[0].lower())
            if mapped is None:
                continue
            fields[0] = mapped
        lines.append(" ".join(fields))
    return "\n".join(lines) + ("\n" if lines else "")


def _copy_sample(
    table_dir: Path,
    token: str,
    image: Path,
    numeric_id: int,
    output_split_dir: Path,
    layout: str,
    taxonomy: str,
) -> None:
    calib = _calib_path(table_dir, token)
    label = _label_path(table_dir, token)
    stem = f"{numeric_id:06d}"

    if layout == "monocop":
        image_out = output_split_dir / "ground_image" / f"{stem}.png"
        label_out = output_split_dir / "ground_label" / f"{stem}.txt"
        calib_out = output_split_dir / "calib" / f"{stem}.txt"
    else:
        image_name = "image" if layout == "nusc" else "image_2"
        label_name = "label" if layout == "nusc" else "label_2"
        image_out = output_split_dir / image_name / f"{stem}.png"
        label_out = output_split_dir / label_name / f"{stem}.txt"
        calib_out = output_split_dir / "calib" / f"{stem}.txt"

    for path in (image_out, label_out, calib_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    # PNG is the on-disk contract of all current MonoCoP loaders.  Avoid a
    # PIL re-encode (and its possible colour/profile changes) when already PNG.
    if image.suffix.lower() == ".png":
        shutil.copy2(image, image_out)
    else:
        try:
            from PIL import Image
            Image.open(image).convert("RGB").save(image_out, "PNG")
        except ImportError as exc:
            _die(f"Pillow is required to convert {image.suffix} images: {exc}")
    label_out.write_text(_rewrite_labels(label, taxonomy), encoding="utf-8")
    shutil.copy2(calib, calib_out)

    # The monocular loaders do not consume lidar, but retaining an optional
    # dev-kit velodyne file makes the result a complete KITTI-style sample when
    # the raw conversion produced one.
    lidar = table_dir / "velodyne" / f"{token}.bin"
    if lidar.exists():
        lidar_out = output_split_dir / "velodyne" / f"{stem}.bin"
        lidar_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lidar, lidar_out)

    # KITTI raw readers often expect one OXTS record per frame.  NuScenes does
    # not have GPS/IMU OXTS files, but raw exports created from the official
    # metadata can contain a KITTI-compatible pose record generated below.
    oxts = table_dir / "oxts" / f"{token}.txt"
    if oxts.exists():
        oxts_out = output_split_dir / "oxts" / f"{stem}.txt"
        oxts_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(oxts, oxts_out)

    # Keep a label_2 alias in the native layout.  It costs little and lets
    # both the NUSC_Dataset and KITTI-style inspection tools consume the output.
    if layout == "nusc":
        alias = output_split_dir / "label_2" / f"{stem}.txt"
        alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_out, alias)


def _write_split_index(path: Path, ids: Sequence[str]) -> None:
    """Write a split index, never clobbering one this run has no data for.

    Running the conversion once per split would otherwise overwrite the other
    split's index with an empty file.
    """
    if not ids and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")


def _write_monocop_splits(output: Path, split_to_ids: Dict[str, Sequence[str]]) -> None:
    split_dir = output / "chen_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in split_to_ids.items():
        _write_split_index(split_dir / f"{split}.txt", ids)


def _next_free_id(output: Path) -> int:
    """Continue numbering after the frames the MonoCoP output already holds.

    The monocop layout keeps every split in one flat directory, so a second
    invocation must not restart at 000000 and overwrite earlier frames.
    """
    image_dir = output / "ground_image"
    if not image_dir.is_dir():
        return 0
    highest = -1
    for path in image_dir.glob("*.png"):
        try:
            highest = max(highest, int(path.stem))
        except ValueError:
            continue
    return highest + 1


def convert_export(
    source: Path,
    output: Path,
    layout: str,
    taxonomy: str,
    split: str,
    max_samples: Optional[int],
    progress_every: int = 0,
) -> Dict[str, List[str]]:
    source_entries = _discover_source(source, split)
    split_to_ids: Dict[str, List[str]] = {"train": [], "val": []}
    # Monocop keeps all splits in one flat directory: never restart the ids.
    global_id = _next_free_id(output) if layout == "monocop" else 0
    mapping: Dict[str, Dict[str, str]] = {}

    for source_split, table_dir in source_entries:
        images = _image_paths(table_dir)
        tokens = sorted(images)
        if max_samples is not None:
            tokens = tokens[:max_samples]
        if source_split not in split_to_ids:
            split_to_ids[source_split] = []
        out_split_dir = output if layout == "monocop" else output / (
            "training" if source_split == "train" else "validation"
        )
        total = len(tokens)
        for index, token in enumerate(tokens, start=1):
            stem = f"{global_id:06d}"
            _copy_sample(table_dir, token, images[token], global_id, out_split_dir, layout, taxonomy)
            split_to_ids[source_split].append(stem)
            if progress_every and (index % progress_every == 0 or index == total):
                print(f"convert[{source_split}]: {index}/{total}", flush=True)
            # The same six-digit id may legitimately occur in separate
            # ``training`` and ``validation`` directories when the two splits
            # are exported in separate invocations, so qualify the index key.
            mapping[f"{source_split}/{stem}"] = {"id": stem, "token": token, "split": source_split}
            global_id += 1
        pose_sidecar = table_dir / "oxts_pose.json"
        if pose_sidecar.exists():
            shutil.copy2(pose_sidecar, out_split_dir / "oxts_pose.json")

    if layout == "monocop":
        _write_monocop_splits(output, split_to_ids)
    else:
        image_sets = output / "ImageSets"
        image_sets.mkdir(parents=True, exist_ok=True)
        for name in NUSC_SPLITS:
            _write_split_index(image_sets / f"{name}.txt", split_to_ids[name])

    output.mkdir(parents=True, exist_ok=True)
    mapping_path = output / "token_mapping.json"
    if mapping_path.exists():
        try:
            previous = json.loads(mapping_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                previous.update(mapping)
                mapping = previous
        except (OSError, ValueError):
            # A partial/invalid old index must not prevent rebuilding the data;
            # the newly generated mapping is still complete for this run.
            pass
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return split_to_ids


def _export_raw(args: argparse.Namespace) -> Path:
    """Stage one: write a resumable token-based KITTI export under --export-dir.

    The export lives in a persistent directory (never a temporary one) so the
    command can be interrupted and re-run: samples whose files already exist
    are skipped unless ``--overwrite`` is given.
    """
    if not args.raw_dataroot:
        _die("either --source or --raw-dataroot is required")
    if not args.devkit:
        _die("--devkit is required with --raw-dataroot")
    if args.split == "all":
        _die("raw export must select one dev-kit split (--split train or --split val)")
    sdk = Path(args.devkit) / "python-sdk"
    if not sdk.is_dir():
        _die(f"cannot find nuScenes python-sdk under {sdk}")
    sys.path.insert(0, str(sdk))
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.scripts.export_kitti import KittiConverter
    except ImportError as exc:
        _die(f"unable to import nuScenes dev-kit: {exc}")

    export_root = Path(args.export_dir)
    split_dir = export_root / args.split
    # Construct without calling KittiConverter.__init__: that constructor
    # eagerly opens the default ``/data/sets/nuscenes`` root before it accepts
    # a custom dataroot.  These are the fields used by nuscenes_gt_to_kitti().
    converter = KittiConverter.__new__(KittiConverter)
    converter.nusc_kitti_dir = str(Path(export_root).expanduser())
    converter.cam_name_list = {
        "CAM_FRONT": "0", "CAM_FRONT_LEFT": "1", "CAM_FRONT_RIGHT": "2",
        "CAM_BACK": "3", "CAM_BACK_LEFT": "4", "CAM_BACK_RIGHT": "5",
    }
    converter.cam_name = args.cam_name
    converter.lidar_name = args.lidar_name
    converter.image_count = args.max_samples or 10**9
    converter.nusc_version = args.version
    converter.split = args.split
    converter.output_dir = None
    converter.remove_zero_lidar_radar = True
    converter.nusc = NuScenes(version=args.version, dataroot=args.raw_dataroot, verbose=False)
    _complete_nuscenes_sample_links(converter.nusc)
    available = _filter_samples_with_available_camera(converter.nusc, args.cam_name)
    if not available:
        _die(f"no {args.cam_name} images referenced by metadata exist under {args.raw_dataroot}")
    # The dev-kit exporter silently drops samples from other splits; mirror that
    # here so oxts records are only written for samples that are really exported.
    split_logs = _split_logfiles(converter.nusc, args.split)
    in_split = _filter_samples_in_split(converter.nusc, available, split_logs)
    out_of_split = len(available) - len(in_split)
    available = in_split
    if not available:
        _die(f"no {args.cam_name} image of the '{args.split}' split exists under {args.raw_dataroot}")
    if args.max_samples is not None:
        available = available[: args.max_samples]
    # Resume: skip samples whose export is already on disk.
    pending = available if args.overwrite else _pending_export_samples(
        split_dir, available, args.write_oxts
    )
    print(
        json.dumps(
            {
                "stage": "export",
                "export_dir": str(split_dir),
                "split": args.split,
                "candidates": len(available),
                "out_of_split_skipped": out_of_split,
                "already_exported": len(available) - len(pending),
                "todo": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        return split_dir

    converter.nusc.sample = pending
    # NuScenes caches token -> row indices at construction time; rebuild the
    # sample index after filtering the table.
    converter.nusc._token2ind["sample"] = {
        sample["token"]: index for index, sample in enumerate(pending)
    }
    converter.image_count = len(pending)
    split_dir.mkdir(parents=True, exist_ok=True)
    # The dev-kit prints one line per sample; swallow that chatter and emit a
    # coarse progress line instead so large exports are not bottlenecked by the
    # terminal/PTY.
    with contextlib.redirect_stdout(
        _ThrottledWriter("export", len(pending), args.progress_every)
    ):
        converter.nuscenes_gt_to_kitti()
    if args.write_oxts:
        _write_nuscenes_oxts(converter.nusc, split_dir, args.cam_name)
    print(
        json.dumps({"stage": "export", "export_dir": str(split_dir), "exported": len(pending)}),
        flush=True,
    )
    return split_dir


def _complete_nuscenes_sample_links(nusc) -> None:
    """Rebuild ``sample.data``/``sample.anns`` when metadata is compact.

    Official NuScenes tables normally contain these two derived fields in
    ``sample.json``.  The copy bundled with this workstation contains the
    normalized relational tables but omits the fields, so the stock dev-kit
    exporter would otherwise fail with ``KeyError: data``.  The information is
    fully recoverable from ``sample_data.json`` and ``sample_annotation.json``.
    """
    sensor_channel = {}
    for calibrated in nusc.calibrated_sensor:
        sensor = nusc.get("sensor", calibrated["sensor_token"])
        sensor_channel[calibrated["token"]] = sensor["channel"]

    data_by_sample = {}
    for sample_data in nusc.sample_data:
        sample_token = sample_data["sample_token"]
        channel = sensor_channel.get(sample_data["calibrated_sensor_token"])
        if channel is None:
            continue
        data_by_sample.setdefault(sample_token, {})
        # Prefer keyframes, but retain a non-keyframe record as a fallback for
        # incomplete downloads.
        current_token = data_by_sample[sample_token].get(channel)
        if current_token is None or sample_data.get("is_key_frame", False):
            data_by_sample[sample_token][channel] = sample_data["token"]

    anns_by_sample = {}
    for annotation in nusc.sample_annotation:
        anns_by_sample.setdefault(annotation["sample_token"], []).append(annotation["token"])

    for sample in nusc.sample:
        token = sample["token"]
        sample["data"] = data_by_sample.get(token, {})
        sample["anns"] = anns_by_sample.get(token, [])


def _filter_samples_with_available_camera(nusc, cam_name: str):
    """Keep only samples whose requested camera file is present on disk."""
    available = []
    for sample in nusc.sample:
        sample_data_token = sample.get("data", {}).get(cam_name)
        if sample_data_token is None:
            continue
        sample_data = nusc.get("sample_data", sample_data_token)
        image_path = Path(nusc.dataroot) / sample_data["filename"]
        if image_path.is_file():
            available.append(sample)
    return available


def _split_logfiles(nusc, split: str) -> set:
    """Log files belonging to ``split``, using the dev-kit's own scene lists."""
    from nuscenes.utils.splits import create_splits_scenes

    scene_names = set(create_splits_scenes(verbose=False).get(split, []))
    if not scene_names:
        return set()
    logfile_by_scene = {
        scene["name"]: nusc.get("log", scene["log_token"])["logfile"] for scene in nusc.scene
    }
    return {logfile_by_scene[name] for name in scene_names if name in logfile_by_scene}


def _filter_samples_in_split(nusc, samples: Sequence[dict], split_logs: set) -> List[dict]:
    """Drop samples whose scene is outside the requested split.

    ``KittiConverter.nuscenes_gt_to_kitti`` applies exactly this filter
    internally (``_split_to_samples`` + ``create_splits_logs``), so samples
    from other splits would never be exported while still receiving an oxts
    record written by this script.  Filtering up front keeps every output
    folder the same size.
    """
    if not split_logs:
        return list(samples)
    logfile_by_scene: Dict[str, str] = {}
    kept = []
    for sample in samples:
        scene_token = sample.get("scene_token")
        if scene_token is None:
            continue
        if scene_token not in logfile_by_scene:
            scene = nusc.get("scene", scene_token)
            logfile_by_scene[scene_token] = nusc.get("log", scene["log_token"])["logfile"]
        if logfile_by_scene[scene_token] in split_logs:
            kept.append(sample)
    return kept


def _write_nuscenes_oxts(nusc, split_dir: Path, cam_name: str) -> None:
    """Write KITTI-style OXTS records from NuScenes ego poses.

    NuScenes has no latitude/longitude or raw IMU channels in these tables.
    The first six OXTS fields are therefore set to zero rather than inventing
    GPS coordinates.  Velocity is a finite difference of the NuScenes global
    translation, while acceleration/angular-rate and navigation status fields
    are conservative zero placeholders.  Exact pose data is written beside
    the records as ``oxts_pose.json`` for consumers that understand NuScenes.
    """
    try:
        from pyquaternion import Quaternion
    except ImportError as exc:
        _die(f"pyquaternion is required to write OXTS records: {exc}")

    records = []
    for sample in nusc.sample:
        sd_token = sample.get("data", {}).get(cam_name)
        if sd_token is None:
            continue
        sample_data = nusc.get("sample_data", sd_token)
        pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        records.append((sample["token"], float(pose["timestamp"]) / 1e6, pose))
    records.sort(key=lambda item: item[1])
    if not records:
        return

    pose_payload = {}
    translations = [tuple(float(v) for v in rec[2]["translation"]) for rec in records]
    for index, (token, timestamp, pose) in enumerate(records):
        if len(records) == 1:
            velocity = (0.0, 0.0, 0.0)
        elif index == 0:
            dt = max(records[1][1] - timestamp, 1e-6)
            velocity = tuple((translations[1][axis] - translations[0][axis]) / dt for axis in range(3))
        elif index == len(records) - 1:
            dt = max(timestamp - records[index - 1][1], 1e-6)
            velocity = tuple((translations[index][axis] - translations[index - 1][axis]) / dt for axis in range(3))
        else:
            dt = max(records[index + 1][1] - records[index - 1][1], 1e-6)
            velocity = tuple((translations[index + 1][axis] - translations[index - 1][axis]) / dt for axis in range(3))

        # pyquaternion exposes yaw, pitch, roll in that order.  KITTI OXTS
        # stores roll, pitch, yaw.
        yaw, pitch, roll = Quaternion(pose["rotation"]).yaw_pitch_roll
        vx, vy, vz = velocity
        horizontal_speed = math.sqrt(vx * vx + vy * vy)
        fields = [
            0.0, 0.0, float(pose["translation"][2]),
            float(roll), float(pitch), float(yaw),
            vx, vy, horizontal_speed, 0.0, vz,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        (split_dir / "oxts").mkdir(parents=True, exist_ok=True)
        (split_dir / "oxts" / f"{token}.txt").write_text(
            " ".join(f"{value:.12g}" for value in fields) + "\n", encoding="utf-8"
        )
        pose_payload[token] = {
            "timestamp": timestamp,
            "translation": [float(v) for v in pose["translation"]],
            "rotation_wxyz": [float(v) for v in pose["rotation"]],
            "coordinate_frame": "nuscenes_global",
        }
    (split_dir / "oxts_pose.json").write_text(
        json.dumps(pose_payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _attach_oxts_to_existing_export(
    source: Path,
    pose_dataroot: Path,
    devkit: Path,
    version: str,
    cam_name: str,
) -> None:
    """Add pose records to an existing token-based export without raw JPGs."""
    sdk = devkit / "python-sdk"
    if not sdk.is_dir():
        _die(f"cannot find nuScenes python-sdk under {sdk}")
    sys.path.insert(0, str(sdk))
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        _die(f"unable to import nuScenes dev-kit: {exc}")
    nusc = NuScenes(version=version, dataroot=pose_dataroot, verbose=False)
    _complete_nuscenes_sample_links(nusc)

    entries = _discover_source(source, "all")
    if not entries:
        _die(f"cannot find image_2/image directory under {source}")
    found = False
    all_samples = list(nusc.sample)
    token_mapping = {}
    mapping_path = source / "token_mapping.json"
    if mapping_path.exists():
        try:
            raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            token_mapping = {
                str(value["id"]): str(value["token"])
                for value in raw_mapping.values()
                if isinstance(value, dict) and value.get("id") and value.get("token")
            }
        except (OSError, ValueError, KeyError, TypeError):
            token_mapping = {}
    for _, table_dir in entries:
        image_tokens = set(_image_paths(table_dir))
        original_tokens = {token_mapping.get(token, token) for token in image_tokens}
        selected = [sample for sample in all_samples if sample["token"] in original_tokens]
        if not selected:
            continue
        found = True
        nusc.sample = selected
        nusc._token2ind["sample"] = {
            sample["token"]: index for index, sample in enumerate(selected)
        }
        _write_nuscenes_oxts(nusc, table_dir, cam_name)
        for numeric_id, original_token in token_mapping.items():
            token_oxts = table_dir / "oxts" / f"{original_token}.txt"
            if original_token in original_tokens and token_oxts.exists():
                shutil.copy2(token_oxts, table_dir / "oxts" / f"{numeric_id}.txt")
    if not found:
        _die("no exported image token was found in the NuScenes metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing token-based KITTI export")
    parser.add_argument("--raw-dataroot", type=Path, help="official nuScenes dataroot")
    parser.add_argument(
        "--pose-dataroot",
        type=Path,
        help="NuScenes dataroot used only to add oxts to an existing token export",
    )
    parser.add_argument("--devkit", type=Path, help="nuscenes-devkit-master directory")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--cam-name", default="CAM_FRONT")
    parser.add_argument("--lidar-name", default="LIDAR_TOP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("nusc", "kitti", "monocop"), default="nusc")
    parser.add_argument("--taxonomy", choices=("native", "kitti"), default="native")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--stage",
        choices=("export", "convert", "all"),
        default="all",
        help="export=step one only (raw nuScenes -> token KITTI export), "
             "convert=step two only (token export -> MonoCoP layout), all=both",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="where step one writes its token-based KITTI export "
             "(default: <output>_kitti_export); also the default --source of step two",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-export samples that already exist in --export-dir instead of skipping them",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print one progress line every N samples (0 disables progress output)",
    )
    parser.add_argument(
        "--no-oxts",
        dest="write_oxts",
        action="store_false",
        help="do not generate KITTI-style oxts records for raw exports",
    )
    parser.set_defaults(write_oxts=True)

    args = parser.parse_args()
    if args.export_dir is None:
        args.export_dir = args.output.parent / f"{args.output.name}_kitti_export"
    return args


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        _die("--max-samples must be positive")
    if args.source and args.raw_dataroot:
        _die("--source and --raw-dataroot are mutually exclusive")
    if args.pose_dataroot and not args.source:
        _die("--pose-dataroot requires --source")

    # Step two only: consume an existing token-based export.
    if args.stage == "convert":
        source = args.source or args.export_dir
        if not source:
            _die("--source (or --export-dir) is required for --stage convert")
        source = Path(source)
        if args.pose_dataroot:
            if not args.devkit:
                _die("--devkit is required with --pose-dataroot")
            _attach_oxts_to_existing_export(
                source=source,
                pose_dataroot=args.pose_dataroot,
                devkit=args.devkit,
                version=args.version,
                cam_name=args.cam_name,
            )
        args.output.mkdir(parents=True, exist_ok=True)
        split_to_ids = convert_export(
            source=source,
            output=args.output,
            layout=args.layout,
            taxonomy=args.taxonomy,
            split=args.split,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )
        summary = {key: len(value) for key, value in split_to_ids.items()}
        print(json.dumps({"stage": "convert", "output": str(args.output), "frames": summary}, indent=2))
        return

    # Step one: raw nuScenes -> token-based KITTI export (resumable).
    if not args.raw_dataroot:
        _die("--raw-dataroot is required for --stage export")
    source = _export_raw(args)
    if args.stage == "export":
        print(
            json.dumps(
                {
                    "stage": "export",
                    "export_dir": str(source),
                    "next": (
                        f"--source {source} --output {args.output} "
                        f"--layout {args.layout} --taxonomy {args.taxonomy} "
                        f"--split {args.split} --stage convert"
                    ),
                },
                indent=2,
            )
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    split_to_ids = convert_export(
        source=source,
        output=args.output,
        layout=args.layout,
        taxonomy=args.taxonomy,
        split=args.split,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    summary = {key: len(value) for key, value in split_to_ids.items()}
    print(json.dumps({"stage": "all", "output": str(args.output), "frames": summary}, indent=2))


if __name__ == "__main__":
    main()
