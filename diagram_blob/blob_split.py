"""blob_split — 解説画像の連結成分(blob)分離モジュール

解説動画系統の raw PNG(1536×1024)を受け取り、
連結成分(白背景から分離された描画要素)を blob として列挙し JSON に書き出す。

出力先:
  <blobs_dir>/<diag_name>/blobs.json      各 blob の bbox・面積
  <blobs_dir>/<diag_name>/blob_*.png      切り出し画像(デバッグ用)
  <blobs_dir>/all_blobs.json              全 diagram の blob 集約 JSON

blobs_dir のデフォルトは topic ディレクトリ配下の work/blobs/ を推奨。
/tmp 等の一時ディレクトリへのハードコードは禁止。

依存:
  Pillow, numpy, scipy

使い方(スクリプトから):
  from src.diagram_blob.blob_split import split_diagram, write_all_blobs_json

使い方(CLI):
  python -m src.diagram_blob.blob_split \\
    --raw channels/<channel>/videos/<topic>/diagrams/diag_01_question_raw.png \\
    --blobs-dir channels/<channel>/videos/<topic>/work/blobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# 白背景からの乖離閾値(0-255): これ以上の輝度差を「描画あり」とみなす
_THRESH = 20
# faint pixel absorption(opt-in via --absorb-faint)
# 「視認可能だが strict mask (>20) に入らない faint pixel」(diff 5-20、軽い水彩塗りの fill 等)
# を最近接 labeled blob に距離変換で吸収する。セミナー型で leak 修正用。
_WEAK_THRESH = 5
_WEAK_ABSORB_MAX_DIST = 100
# 最小 blob 面積(px²): ノイズ・JPEGアーチファクト除去
_MIN_AREA = 50


# ---------------------------------------------------------------------------
# コア処理
# ---------------------------------------------------------------------------

def split_diagram(
    raw_path: Path,
    blobs_dir: Path,
    save_crops: bool = False,
    dilate: int = 0,
    absorb_faint: bool = False,) -> list[dict[str, Any]]:
    """raw PNG から blob を分離し、blobs_dir/<diag_name>/ に保存する。

    Args:
        raw_path:     {diag_name}_raw.png の絶対/相対パス
        blobs_dir:    出力先ディレクトリ(work/blobs/ 等)
        save_crops:   True のとき各 blob の切り出し PNG も保存する
        dilate:       blob 検出前に binary_dilation を N px 適用(>0 で有効)
        absorb_faint: True のとき strict mask 構築後、`diff > _WEAK_THRESH` 但し strict 未満の
                      faint pixel(軽い水彩塗りの fill 等)を距離変換で最近接 blob に吸収する。
                      セミナー型の cell fill leak 修正用(追加、opt-in)。

    Returns:
        blob エントリのリスト:
          [{"id": "00", "x0": int, "y0": int, "x1": int, "y1": int, "area": int}, ...]
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import label, binary_dilation, distance_transform_edt
    except ImportError as e:
        raise ImportError(
            f"blob_split には Pillow / numpy / scipy が必要です: {e}") from e

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw PNG が見つかりません: {raw_path}")

    diag_name = raw_path.name.replace("_raw.png", "").replace(".png", "")
    out_dir = Path(blobs_dir) / diag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(raw_path).convert("RGB")
    arr = np.array(img)

    # 白(255,255,255)からの最大チャンネル乖離でマスク作成
    diff = np.max(255 - arr, axis=2)
    mask = diff > _THRESH

    # ラベリング用 mask(dilation で粒度調整、bbox は元 mask から取る)
    label_mask = mask
    if dilate > 0:
        label_mask = binary_dilation(mask, iterations=int(dilate))

    # 連結成分ラベリング(label_mask ベース)
    labeled, n_features = label(label_mask)

    # 小成分 drop → 近接マージ に変更
    # 旧版は area < _MIN_AREA を「ノイズ」として捨てていたが、結果として
    # 元描画ピクセルが labels.npz に記録されず、render_diagram_anim の
    # static_layer 構築で消去対象から外れて、フレーム 0 から visible に
    # leak する事故が発生していた(点線アンダーライン等)。
    # 新版は小成分を近接大成分に label 付け替えで吸収する。
    _large_comp: list[tuple[int, tuple[int, int, int, int], tuple[float, float]]] = []
    _small_comp: list[tuple[int, tuple[int, int, int, int], tuple[float, float]]] = []
    for cid in range(1, n_features + 1):
        in_orig_pre = mask & (labeled == cid)
        ys, xs = np.where(in_orig_pre)
        if len(xs) == 0:
            continue
        area_pre = int(len(xs))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = (x0, y0, x1, y1)
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if area_pre >= _MIN_AREA:
            _large_comp.append((cid, bbox, center))
        else:
            _small_comp.append((cid, bbox, center))

    if _large_comp and _small_comp:
        def _bbox_dist_sq(cx: float, cy: float, bbox: tuple[int, int, int, int]) -> float:
            x0, y0, x1, y1 = bbox
            dx = max(x0 - cx, 0.0, cx - x1)
            dy = max(y0 - cy, 0.0, cy - y1)
            return dx * dx + dy * dy
        for sid, _sbbox, (scx, scy) in _small_comp:
            best_lid, _, _ = min(
                _large_comp,
                key=lambda t: _bbox_dist_sq(scx, scy, t[1]),
            )
            labeled[labeled == sid] = best_lid

    blobs: list[dict[str, Any]] = []
    comp_id_for_blob: list[int] = []   # 各 blob 出力に対応する元 comp_id を保持
    for comp_id in range(1, n_features + 1):
        # ラベリング領域内の元描画ピクセル(mask)のみで bbox/area を計算
        in_label = labeled == comp_id
        in_orig = mask & in_label
        ys, xs = np.where(in_orig)
        if len(xs) == 0:
            continue
        area = int(len(xs))
        if area < _MIN_AREA:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        blob_id = f"{len(blobs):02d}"
        entry: dict[str, Any] = {
            "id": blob_id,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "area": area,
        }
        if save_crops:
            crop = img.crop((x0, y0, x1, y1))
            crop.save(out_dir / f"blob_{blob_id}.png")
        blobs.append(entry)
        comp_id_for_blob.append(comp_id)

    # bbox を y0 昇順 → x0 昇順でソート(視覚的に読みやすい順)
    sort_indices = sorted(range(len(blobs)), key=lambda i: (blobs[i]["y0"], blobs[i]["x0"]))
    blobs = [blobs[i] for i in sort_indices]
    comp_id_for_blob = [comp_id_for_blob[i] for i in sort_indices]
    # id を再付番
    for i, b in enumerate(blobs):
        b["id"] = f"{i:02d}"

    # === 正確な blob ピクセルマスクを labels.npz に保存 ===
    # final_labels[y, x] = blob_index + 1(1-based、0 = 背景)
    # 元描画ピクセル(mask)のみを記録(dilation 領域は含まない=正確マスク)
    final_labels = np.zeros(labeled.shape, dtype=np.uint16)
    for i, comp_id in enumerate(comp_id_for_blob):
        in_label = labeled == comp_id
        in_orig = mask & in_label
        final_labels[in_orig] = i + 1

    # faint pixel absorption(opt-in)
    # diff > _WEAK_THRESH かつ strict 未満のピクセル(軽い水彩塗りの fill 等)を
    # 最近接 labeled blob に距離変換で吸収する。`render_diagram_anim` の static_layer
    # 構築でこれらが消える=フレーム0 leak が解消する。セミナー型で必要。
    if absorb_faint:
        diff = np.max(255 - arr, axis=2)
        mask_weak = diff > _WEAK_THRESH
        unlabeled_weak = mask_weak & (final_labels == 0)
        if unlabeled_weak.any() and final_labels.max() > 0:
            distances, indices = distance_transform_edt(
                final_labels == 0, return_indices=True,)
            yi, xi = indices
            cap_mask = unlabeled_weak & (distances < _WEAK_ABSORB_MAX_DIST)
            if cap_mask.any():
                final_labels[cap_mask] = final_labels[yi[cap_mask], xi[cap_mask]]
                # blob area を吸収分で更新(参考値、機能には影響しない)
                absorbed_per_label = np.bincount(
                    final_labels[cap_mask].ravel(),
                    minlength=len(blobs) + 1,
                )
                for idx in range(len(blobs)):
                    blobs[idx]["area"] = int(blobs[idx]["area"] + absorbed_per_label[idx + 1])

    np.savez_compressed(out_dir / "labels.npz", labels=final_labels)

    # blobs.json に書き出し
    blobs_json_path = out_dir / "blobs.json"
    blobs_json_path.write_text(
        json.dumps({"diag_name": diag_name, "blobs": blobs}, ensure_ascii=False, indent=2) + "\n"
    )
    return blobs


def write_all_blobs_json(
    blobs_dir: Path,
    results: dict[str, list[dict[str, Any]]],) -> Path:
    """全 diagram の blob 結果を blobs_dir/all_blobs.json にまとめる。

    Args:
        blobs_dir: 出力先ディレクトリ
        results:   {diag_name: [blob, ...], ...}

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
        description="解説画像の連結成分(blob)を分離して JSON に書き出す"
    )
    ap.add_argument(
        "--raw", required=True,
        help="raw PNG のパス ({diag_name}_raw.png)",
    )
    ap.add_argument(
        "--blobs-dir", required=True,
        help="出力先ディレクトリ(例: channels/<channel>/videos/<topic>/work/blobs)",
    )
    ap.add_argument(
        "--save-crops", action="store_true",
        help="各 blob の切り出し PNG を保存する(デバッグ用)",
    )
    ap.add_argument(
        "--dilate", type=int, default=0,
        help="連結成分検出前に N px の binary_dilation を適用して粒度を粗くする(default 0=無効)",
    )
    ap.add_argument(
        "--absorb-faint", action="store_true",
        help="faint pixel(diff 5-20、軽い水彩塗りの fill 等)を最近接 blob に距離変換で吸収する(セミナー型の cell fill leak 修正用、opt-in)",
    )
    args = ap.parse_args()

    blobs = split_diagram(
        raw_path=Path(args.raw),
        blobs_dir=Path(args.blobs_dir),
        save_crops=args.save_crops,
        dilate=args.dilate,
        absorb_faint=args.absorb_faint,
    )
    print(f"blob 数: {len(blobs)}")
    for b in blobs:
        print(f"  {b['id']}: ({b['x0']},{b['y0']})-({b['x1']},{b['y1']}) area={b['area']}")


if __name__ == "__main__":
    _main()
