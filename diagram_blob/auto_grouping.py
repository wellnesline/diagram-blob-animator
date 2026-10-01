"""auto_grouping — v5d 機械的ルールによる order JSON 自動生成

解説動画系の subagent (orderer) を置換するモジュール。
セミナー型の subagent (orderer) には適用しない。

ルール (v5d、 holdout 60% 達成):
  if max_blob_area_ratio > 0.30 or < 0.04: → B (container 検出 + line cluster)
  else: → A (geometric line cluster)

A: y-band overlap >= 0.4 で同一行クラスタ → 行内 x-gap で word merge
B: 大 bbox を container として検出 → frame 単体 group + 内側 blob を line cluster

入力:
  raw_png:  diagrams/<diag_name>_raw.png
  blobs:    work/blobs/<diag_name>/blobs.json (blob_split.py v1 schema)

出力:
  work/orders/<diag_name>_order.json
  (schema: orderer の出力と互換)

依存: blob_split / blob_animate のスキーマを尊重
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# v5d ルーター
# ---------------------------------------------------------------------------

def compute_max_area_ratio(blobs: list[dict], slide_w: int, slide_h: int) -> float:
    """v5d 軸: 最大 blob bbox 面積 / slide 面積"""
    if not blobs:
        return 0.0
    slide_area = slide_w * slide_h
    max_area = max(
        (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in blobs
    )
    return max_area / slide_area


def route_v5d(blobs: list[dict], slide_w: int, slide_h: int) -> str:
    """v5d ルール: 'A' or 'B'"""
    ma = compute_max_area_ratio(blobs, slide_w, slide_h)
    if ma > 0.30 or ma < 0.04:
        return "B"
    return "A"


# ---------------------------------------------------------------------------
# 共通ヘルパー (blob_split v1 schema → 内部 bbox 形式に正規化)
# ---------------------------------------------------------------------------

def _v1_to_internal(blobs: list[dict]) -> list[dict]:
    """blob_split.py v1 schema [{id,x0,y0,x1,y1,area}] → 内部 [{bbox:[x0,y0,x1,y1], size, _id}] に変換"""
    out = []
    for b in blobs:
        out.append({
            "_id": b["id"],
            "bbox": [b["x0"], b["y0"], b["x1"], b["y1"]],
            "size": b["area"],
        })
    return out


def _y_overlap_ratio(a, b) -> float:
    _, ay0, _, ay1 = a
    _, by0, _, by1 = b
    inter = max(0, min(ay1, by1) - max(ay0, by0))
    short = min(ay1 - ay0, by1 - by0)
    return inter / short if short > 0 else 0.0


def _x_gap(a, b) -> int:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    if ax1 < bx0:
        return bx0 - ax1
    if bx1 < ax0:
        return ax0 - bx1
    return 0


def _union_bbox(bboxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


# ---------------------------------------------------------------------------
# A: Geometric line clustering
# ---------------------------------------------------------------------------

def _cluster_A(blobs_internal: list[dict],
               y_overlap_thresh: float = 0.4,
               x_gap_factor: float = 1.5) -> list[dict]:
    """A: y-band line clustering + within-line x-gap word merge

    Returns:
        [{"members": [_id, ...], "bbox": [x0,y0,x1,y1]}, ...]
    """
    if not blobs_internal:
        return []
    import statistics
    indexed = list(enumerate(blobs_internal))
    indexed.sort(key=lambda t: (t[1]["bbox"][1] + t[1]["bbox"][3]) / 2)
    lines: list[list[tuple[int, dict]]] = []
    for _, b in indexed:
        bx = b["bbox"]
        placed = False
        for line in lines:
            ref = line[-1][1]["bbox"]
            if _y_overlap_ratio(tuple(bx), tuple(ref)) >= y_overlap_thresh:
                line.append((b["_id"], b))
                placed = True
                break
        if not placed:
            lines.append([(b["_id"], b)])

    word_groups: list[dict] = []
    for line in lines:
        line.sort(key=lambda t: t[1]["bbox"][0])
        median_h = float(statistics.median(b["bbox"][3] - b["bbox"][1] for _, b in line))
        gap_thresh = max(median_h * x_gap_factor, 20.0)
        current: list[tuple[Any, dict]] = []
        for _id, b in line:
            if not current:
                current.append((_id, b))
                continue
            prev_bb = current[-1][1]["bbox"]
            if _x_gap(tuple(prev_bb), tuple(b["bbox"])) <= gap_thresh:
                current.append((_id, b))
            else:
                word_groups.append({
                    "members": [m[0] for m in current],
                    "bbox": _union_bbox([m[1]["bbox"] for m in current]),
                })
                current = [(_id, b)]
        if current:
            word_groups.append({
                "members": [m[0] for m in current],
                "bbox": _union_bbox([m[1]["bbox"] for m in current]),
            })
    word_groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
    return word_groups


# ---------------------------------------------------------------------------
# B: Container + line hybrid
# ---------------------------------------------------------------------------

def _contains_center(container_bbox, center) -> bool:
    cx, cy = center
    x0, y0, x1, y1 = container_bbox
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _find_containers(blobs_internal: list[dict], slide_area: int) -> list[int]:
    """container blob の (blobs_internal における) index list を返す"""
    centers = [
        ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
        for b in blobs_internal
    ]
    containers: list[int] = []
    for i, b in enumerate(blobs_internal):
        bx = b["bbox"]
        area = (bx[2] - bx[0]) * (bx[3] - bx[1])
        if area < slide_area * 0.05:
            continue
        n_inside = sum(1 for j, c in enumerate(centers) if j != i and _contains_center(bx, c))
        if n_inside >= 2:
            containers.append(i)
    # outer container は drop (innermost のみ)
    final = []
    for c in containers:
        bx = blobs_internal[c]["bbox"]
        is_outer = False
        for c2 in containers:
            if c == c2:
                continue
            bx2 = blobs_internal[c2]["bbox"]
            if (bx[0] <= bx2[0] and bx[1] <= bx2[1] and bx[2] >= bx2[2] and bx[3] >= bx2[3]
                    and (bx[2] - bx[0]) * (bx[3] - bx[1]) > (bx2[2] - bx2[0]) * (bx2[3] - bx2[1])):
                is_outer = True
                break
        if not is_outer:
            final.append(c)
    return final


def _group_B(blobs_internal: list[dict], slide_w: int, slide_h: int) -> list[dict]:
    """B: container 検出 + 内側 line cluster + free line cluster

    Returns:
        [{"members": [_id, ...], "bbox": [...], "_kind": "frame"|"inner"|"free"}, ...]
    """
    slide_area = slide_w * slide_h
    centers = [
        ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
        for b in blobs_internal
    ]
    containers = _find_containers(blobs_internal, slide_area)
    assigned = [-1] * len(blobs_internal)
    for i in range(len(blobs_internal)):
        if i in containers:
            assigned[i] = i
            continue
        best_c, best_area = -1, float("inf")
        for c in containers:
            bx = blobs_internal[c]["bbox"]
            if _contains_center(bx, centers[i]):
                area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                if area < best_area:
                    best_area, best_c = area, c
        assigned[i] = best_c
    container_inner: dict[int, list[int]] = {c: [] for c in containers}
    free_idx: list[int] = []
    for i, c in enumerate(assigned):
        if c == -1:
            free_idx.append(i)
        elif c != i:
            container_inner[c].append(i)

    final_groups: list[dict] = []
    for c in containers:
        final_groups.append({
            "members": [blobs_internal[c]["_id"]],
            "bbox": list(blobs_internal[c]["bbox"]),
            "_kind": "frame",
        })
        inner_blobs = [blobs_internal[i] for i in container_inner[c]]
        for lg in _cluster_A(inner_blobs):
            final_groups.append({**lg, "_kind": "inner"})
    free_blobs = [blobs_internal[i] for i in free_idx]
    for lg in _cluster_A(free_blobs):
        final_groups.append({**lg, "_kind": "free"})
    final_groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
    return final_groups


# ---------------------------------------------------------------------------
# anim 種類割り当て (group kind ベース)
# ---------------------------------------------------------------------------

def _assign_anim(group: dict) -> str:
    """group の種類から anim タイプを決定"""
    kind = group.get("_kind", "line")
    if kind == "frame":
        return "scale_pop"
    return "wipe_lr"


# ---------------------------------------------------------------------------
# order JSON 出力 (blob_animate.py の入力 schema 互換)
# ---------------------------------------------------------------------------

def _format_blob_filter(members: list[str]) -> str:
    """member id list → "manual:00,01,02" 形式"""
    sorted_ids = sorted(members, key=lambda x: int(x))
    return "manual:" + ",".join(sorted_ids)


def build_order(raw_png: Path, blobs: list[dict], slide_w: int, slide_h: int) -> dict:
    """raw_png + blobs から v5d ルールで order dict を作る"""
    diag_name = raw_png.name.replace("_raw.png", "").replace(".png", "")
    blobs_internal = _v1_to_internal(blobs)
    branch = route_v5d(blobs, slide_w, slide_h)
    if branch == "A":
        word_groups = _cluster_A(blobs_internal)
        groups = [{**g, "_kind": "line"} for g in word_groups]
    else:
        groups = _group_B(blobs_internal, slide_w, slide_h)
    order_groups = []
    for i, g in enumerate(groups):
        label_prefix = g.get("_kind", "g")
        order_groups.append({
            "label": f"{label_prefix}_{i:02d}",
            "anim": _assign_anim(g),
            "blob_filter": _format_blob_filter(g["members"]),
        })
    return {
        "diagram_id": diag_name,
        "_generated_by": "auto_grouping_v5d",
        "_route": branch,
        "groups": order_groups,
    }


def generate_for_topic(topic_dir: Path, overwrite: bool = False) -> int:
    """topic_dir の全 diagram に対して order JSON を生成

    Returns: 生成件数
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"auto_grouping には Pillow が必要: {e}") from e

    diagrams_dir = topic_dir / "diagrams"
    blobs_dir = topic_dir / "work" / "blobs"
    orders_dir = topic_dir / "work" / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)

    raw_pngs = sorted(diagrams_dir.glob("*_raw.png"))
    n_gen = 0
    for raw_path in raw_pngs:
        diag_name = raw_path.name.replace("_raw.png", "")
        if diag_name.startswith("_"):
            continue  # _outro_thumb / _short_title 等は対象外
        order_path = orders_dir / f"{diag_name}_order.json"
        if order_path.exists() and not overwrite:
            print(f"  [auto_grouping] skip (exists): {order_path.name}")
            continue
        blobs_json = blobs_dir / diag_name / "blobs.json"
        if not blobs_json.exists():
            # blob_split_v2 の出力 blobs_v2.json も試す
            blobs_v2_json = blobs_dir / diag_name / "blobs_v2.json"
            if blobs_v2_json.exists():
                v2 = json.loads(blobs_v2_json.read_text())["blobs"]
                # v2 schema [{name, bbox:[x0,y0,x1,y1], size, center}] → v1 schema
                blobs = []
                for i, b in enumerate(v2):
                    x0, y0, x1, y1 = b["bbox"]
                    blobs.append({
                        "id": f"{i:02d}",
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "area": b["size"],
                    })
            else:
                print(f"  [auto_grouping] skip (no blobs.json): {diag_name}")
                continue
        else:
            blobs = json.loads(blobs_json.read_text())["blobs"]
        with Image.open(raw_path) as im:
            W, H = im.size
        order = build_order(raw_path, blobs, W, H)
        order_path.write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n")
        n_gen += 1
        print(f"  [auto_grouping] {diag_name}: route={order['_route']} groups={len(order['groups'])}")
    return n_gen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="v5d 機械的ルールで diagram-blob 用 order JSON を自動生成")
    ap.add_argument("--topic-dir", required=True, help="topic ディレクトリ (channels/<ch>/videos/<topic>)")
    ap.add_argument("--overwrite", action="store_true", help="既存 order JSON を上書きする")
    args = ap.parse_args()
    topic_dir = Path(args.topic_dir)
    if not topic_dir.exists():
        raise SystemExit(f"topic_dir が無い: {topic_dir}")
    n = generate_for_topic(topic_dir, overwrite=args.overwrite)
    print(f"\n{n} order JSONs generated")


if __name__ == "__main__":
    _main()
