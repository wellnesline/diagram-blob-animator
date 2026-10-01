"""blob_split_v2 — 連結成分分離 v2(alpha シャープ抽出版)

blob_split.py との違い:
  - ラベリング用 mask は dilate(iterations=2) で連結性を確保し、blob 番号体系を維持
  - 出力 PNG の alpha は dilate なしの strict mask で抽出 → 文字縁がシャープ
  - CLI / API シグネチャは blob_split.py と互換

出力先:
  <blobs_dir>/<diag_name>/blob_NN.png      RGBA PNG(alpha シャープ版)
  <blobs_dir>/<diag_name>/blobs_v2.json    bbox / area / center の JSON
  <blobs_dir>/all_blobs.json               全 diagram 集約 JSON(既存と同名、上書き)

使い方(スクリプトから):
  from src.diagram_blob.blob_split_v2 import split_diagram_v2, write_all_blobs_json_v2

使い方(CLI):
  python -m src.diagram_blob.blob_split_v2 \\
    --raw channels/<channel>/videos/<topic>/diagrams/diag_01_question_raw.png \\
    --blobs-dir channels/<channel>/videos/<topic>/work/blobs

依存:
  Pillow, numpy, scipy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# 白からの輝度閾値: これ以下を「描画あり」とみなす
_WHITE_THRESH = 240
# ラベリング用 dilation イテレーション(連結性向上)
_DILATE_ITER = 2
# 最小 blob 面積(px²)
_MIN_PIXELS = 200
# ソート用グリッド幅(px)
_SORT_GRID = 80
# 弱描画閾値: diff > 5 を「視認可能だが strict mask に入らない faint pixel」とみなす
# (cream / pale yellow fill 等で diff 5~20 の領域が大量にあるケース対策)
_WEAK_THRESH = 5
# faint pixel 吸収距離キャップ(px): 最近接の labeled pixel までこの距離内のみ吸収
_WEAK_ABSORB_MAX_DIST = 100


# ---------------------------------------------------------------------------
# コア処理
# ---------------------------------------------------------------------------

def split_diagram_v2(
    raw_path: Path,
    blobs_dir: Path,
    min_pixels: int = _MIN_PIXELS,) -> list[dict[str, Any]]:
    """raw PNG から blob を分離し、RGBA PNG と blobs_v2.json を書き出す。

    Args:
        raw_path:   {diag_name}_raw.png の絶対/相対パス(RGB 前提)
        blobs_dir:  出力先ディレクトリ(work/blobs/ 等)
        min_pixels: 最小 blob 面積(px²)。これ未満の成分はノイズとして除外。

    Returns:
        blob エントリのリスト:
          [{"name": "blob_NN", "bbox": [x0,y0,x1,y1], "size": int,
            "center": [cx, cy]}, ...]
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as e:
        raise ImportError(
            f"blob_split_v2 には Pillow / numpy / scipy が必要です: {e}") from e

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw PNG が見つかりません: {raw_path}")

    diag_name = raw_path.name.replace("_raw.png", "").replace(".png", "")
    out_dir = Path(blobs_dir) / diag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(raw_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    gray = arr.mean(axis=2)

    # strict mask: 「白から離れたピクセル」を全部 catch(旧版 v1 互換)
    # 旧版は gray < 240 だったが、一部配色の light yellow fill (#fff9c4=233-254)
    # の一部(244-254)が 240 閾値で漏れて ghost 事故に。
    # v1 ロジック(max(255-arr) > 20)はチャンネル単位最大乖離なので fill も確実に catch。
    # 過去の事故由来。
    diff = np.max(255 - arr, axis=2)
    mask_strict = diff > 20
    # ラベリング用 mask: dilate で連結性向上
    mask_label = ndimage.binary_dilation(mask_strict, iterations=_DILATE_ITER)
    labels, num = ndimage.label(mask_label, structure=np.ones((3, 3)))

    # 小成分 drop → 近接マージ に変更
    # 旧版: size < min_pixels の連結成分を「ノイズ」として drop していた。
    # 結果として元描画ピクセルが labels に残らず、render_diagram_anim の
    # static_layer 構築で消されず、フレーム 0 から visible になっていた
    # (点線アンダーライン、装飾ドット、anti-aliasing halo 等が leak)。
    # 新版: 小成分は破棄せず、bbox 中心から最近接の大成分に label を付け替えて
    # 吸収する。これにより全 strict 描画ピクセルがどこかの blob に帰属し、
    # static_layer で確実に消去できる。
    comp_info: dict[int, dict[str, Any]] = {}
    large_labels: list[int] = []
    small_labels: list[int] = []
    for i in range(1, num + 1):
        bm = labels == i
        size = int(bm.sum())
        if size == 0:
            continue
        ys, xs = np.where(bm)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        comp_info[i] = {
            "mask": bm,
            "size": size,
            "bbox": (x0, y0, x1, y1),
            "center": (cx, cy),
        }
        if size >= min_pixels:
            large_labels.append(i)
        else:
            small_labels.append(i)

    def _bbox_dist_sq(cx: float, cy: float, bbox: tuple[int, int, int, int]) -> float:
        x0, y0, x1, y1 = bbox
        dx = max(x0 - cx, 0.0, cx - x1)
        dy = max(y0 - cy, 0.0, cy - y1)
        return dx * dx + dy * dy

    # 大成分が 1 件以上ある時のみマージ実行。 0 件なら小成分は単独で生かす
    if large_labels and small_labels:
        for si in small_labels:
            scx, scy = comp_info[si]["center"]
            best_li = min(
                large_labels,
                key=lambda li: _bbox_dist_sq(scx, scy, comp_info[li]["bbox"]),
            )
            sm = comp_info[si]
            lg = comp_info[best_li]
            labels[sm["mask"]] = best_li
            lg["mask"] = lg["mask"] | sm["mask"]
            lx0, ly0, lx1, ly1 = lg["bbox"]
            sx0, sy0, sx1, sy1 = sm["bbox"]
            lg["bbox"] = (
                min(lx0, sx0), min(ly0, sy0),
                max(lx1, sx1), max(ly1, sy1),
            )
            lg["size"] += sm["size"]
    elif not large_labels:
        # 全成分が小さい(極端ケース): 小も大として扱う
        large_labels = small_labels
        small_labels = []

    blobs: list[dict[str, Any]] = []
    for li in large_labels:
        info = comp_info[li]
        x0, y0, x1, y1 = info["bbox"]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        blobs.append({
            "_label": li,
            "_mask": info["mask"],
            "size": info["size"],
            "bbox": [x0, y0, x1, y1],
            "center": [round(cx, 1), round(cy, 1)],
        })

    # 視覚的左上→右下順: x グリッド列優先、次に y
    blobs.sort(key=lambda b: (b["center"][0] // _SORT_GRID, b["center"][1]))

    info: list[dict[str, Any]] = []
    # labels.npz は v2 sort 順の blob index に基づいて再構築する。
    # (旧 labels.npz が v1 sort 順だと render_diagram_anim で blob mismatch
    #  事故が起きるため必ず上書き保存する。過去の事故由来)
    #
    # 重要: dilated mask (b["_mask"]) で labels を構築する(strict mask ではない)。
    # 理由: render_diagram_anim の static_layer は labels==N+1 の pixel を背景色で
    # 消すが、strict mask だけだと anti-aliasing halo が消えず blob 周辺に
    # ghost 状の薄い色付きが残る(ghost 事故由来)。dilated 領域で
    # 消すことで halo も消える。重複領域は後勝ちだが halo 部のみで視覚影響少。
    final_labels = np.zeros((H, W), dtype=np.uint16)
    for idx, b in enumerate(blobs):
        # alpha は strict mask & ラベル領域の積でシャープに抽出(blob PNG)
        bm_strict = mask_strict & b["_mask"]
        layer = np.zeros((H, W, 4), dtype=np.uint8)
        layer[..., :3] = arr
        layer[..., 3] = (bm_strict * 255).astype(np.uint8)
        blob_name = f"blob_{idx:02d}"
        Image.fromarray(layer).save(out_dir / f"{blob_name}.png")
        info.append({
            "name": blob_name,
            "bbox": b["bbox"],
            "size": b["size"],
            "center": b["center"],
        })
        # final_labels[y,x] = idx+1 (1-based、0=背景)
        # dilated 領域(b["_mask"])で書く: halo もこの blob に帰属
        final_labels[b["_mask"]] = idx + 1

    # faint pixel 吸収(diff > _WEAK_THRESH but <= 20 で blob 化されない領域)
    # 旧版: cream / pale yellow fill (RGB 245,247,240 等で diff 5-20) が strict mask に
    # 入らず labels.npz に記録されない → static_layer 構築で消されず、frame 0 から
    # raw image のまま leak 表示される事故。 過去の事故由来。
    # 新版: faint pixel を最近接の labeled blob に label 付け替えで吸収。
    # 距離キャップ (_WEAK_ABSORB_MAX_DIST) を超える場合は吸収しない(遠方の JPEG noise 等)。
    mask_weak = diff > _WEAK_THRESH
    unlabeled_weak = mask_weak & (final_labels == 0)
    if unlabeled_weak.any() and final_labels.max() > 0:
        distances, indices = ndimage.distance_transform_edt(
            final_labels == 0,
            return_indices=True,)
        yi, xi = indices
        cap_mask = unlabeled_weak & (distances < _WEAK_ABSORB_MAX_DIST)
        if cap_mask.any():
            final_labels[cap_mask] = final_labels[yi[cap_mask], xi[cap_mask]]
            # info の size を吸収分で更新(orderer 等の参考値、 機能には影響しない)
            absorbed_per_label = np.bincount(
                final_labels[cap_mask].ravel(),
                minlength=int(final_labels.max()) + 1,
            )
            for idx in range(len(info)):
                info[idx]["size"] = int(info[idx]["size"] + absorbed_per_label[idx + 1])

    np.savez_compressed(out_dir / "labels.npz", labels=final_labels)

    # blobs_v2.json
    (out_dir / "blobs_v2.json").write_text(
        json.dumps({"diag_name": diag_name, "blobs": info}, ensure_ascii=False, indent=2) + "\n"
    )
    return info


def write_all_blobs_json_v2(
    blobs_dir: Path,
    results: dict[str, list[dict[str, Any]]],) -> Path:
    """全 diagram の blob 結果を blobs_dir/all_blobs.json に書き出す。

    Args:
        blobs_dir: 出力先ディレクトリ(work/blobs/ 等)
        results:   {diag_name: [blob_entry, ...], ...}

    Returns:
        all_blobs.json のパス
    """
    out_path = Path(blobs_dir) / "all_blobs.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="解説画像の連結成分を alpha シャープ版 PNG として分離する(v2)"
    )
    ap.add_argument("--raw", required=True, help="raw PNG のパス")
    ap.add_argument(
        "--blobs-dir", required=True,
        help="出力先ディレクトリ(例: channels/<channel>/videos/<topic>/work/blobs)",
    )
    ap.add_argument(
        "--min-pixels", type=int, default=_MIN_PIXELS,
        help=f"最小 blob 面積(px²) (default {_MIN_PIXELS})",
    )
    args = ap.parse_args()

    info = split_diagram_v2(
        raw_path=Path(args.raw),
        blobs_dir=Path(args.blobs_dir),
        min_pixels=args.min_pixels,
    )
    print(f"blob 数: {len(info)}")
    for b in info:
        bbox = b["bbox"]
        print(f"  {b['name']}: bbox={bbox} size={b['size']} center={b['center']}")


if __name__ == "__main__":
    _main()
