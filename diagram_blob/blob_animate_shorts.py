"""blob_animate_shorts — Shorts 用 diagram 高速アニメ生成モジュール

blob_split_v2 が出力した RGBA PNG 群を stagger 0.2s 固定で順次出現させた
TARGET_W×TARGET_H 動画(mp4)を diagram ごとに生成する。

既存 blob_animate.py(order JSON 解決モジュール)とは役割が異なる:
  blob_animate.py       → order JSON の blob_filter 解決・書き出し(非映像)
  blob_animate_shorts.py → 実際の mp4 を生成する映像レンダラ

出力先:
  <shorts_dir>/diagrams_fast/<diag_name>.mp4

使い方(スクリプトから):
  from src.diagram_blob.blob_animate_shorts import render_diagram_shorts

使い方(CLI):
  python -m src.diagram_blob.blob_animate_shorts \\
    --diag-name diag_02_puzzle_building \\
    --blobs-dir channels/<channel>/videos/<topic>/work/blobs \\
    --orders-dir channels/<channel>/videos/<topic>/work/orders \\
    --out-dir channels/<channel>/videos/<topic>/shorts/short-A/diagrams_fast \\
    --duration 5.0

依存:
  Pillow, numpy, ffmpeg(h264_videotoolbox または libx264)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 定数(固定テンポ)
# ---------------------------------------------------------------------------

# 出力サイズは環境変数で変更できる(中立なサンプル既定値)
TARGET_W = int(os.environ.get("DIAGRAM_TARGET_W", "800"))
TARGET_H = int(os.environ.get("DIAGRAM_TARGET_H", "450"))
FPS = 30
STAGGER = 0.2       # 各 group の出現間隔(秒)
REVEAL_DUR = 0.4    # 各 group の描画時間(秒)

# Shorts は「セリフ同期 reveal」ではなく「最初の 2-3 秒で全要素出し切る高速描写」が原則。
# agent が誤って appear_t を 4s, 6s などセリフ同期に置いた場合、ここで強制的に
# FAST_REVEAL_WINDOW 内に等間隔再配置する(セーフネット)。
# 由来: 過去の事故 — agent がセリフ同期 appear_t を出し、Shorts 中盤で
# 要素がポップする違和感が発生。
FAST_REVEAL_WINDOW = 3.0   # 全 group をこの秒数以内に収める
FAST_REVEAL_MAX_APPEAR = 2.5  # 個別 appear_t の上限。これを超えてたらクランプ


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _to_letterbox(img: Any, target_w: int, target_h: int) -> Any:
    """RGBA 画像を letterbox fit でリサイズしてキャンバスに貼る。"""
    from PIL import Image
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    sw, sh = img.size
    src_ratio = sw / sh
    tgt_ratio = target_w / target_h
    if src_ratio < tgt_ratio:
        rh = target_h
        rw = int(rh * src_ratio)
    else:
        rw = target_w
        rh = int(rw / src_ratio)
    resized = img.resize((rw, rh), Image.LANCZOS)
    x0 = (target_w - rw) // 2
    y0 = (target_h - rh) // 2
    canvas.paste(resized, (x0, y0), resized)
    return canvas


def _merge_blobs(blob_names: list[str], blobs_dir: Path) -> tuple[Any | None, dict | None]:
    """blob PNG 群をアルファ合成して1枚の RGBA 配列と bbox info を返す。"""
    import numpy as np
    from PIL import Image

    merged = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    for bname in blob_names:
        p = blobs_dir / f"{bname}.png"
        if not p.exists():
            print(f"  [WARN] blob PNG なし: {p}", file=sys.stderr)
            continue
        img = _to_letterbox(Image.open(p).convert("RGBA"), TARGET_W, TARGET_H)
        a = np.array(img, dtype=np.uint8)
        src_a = a[..., 3:4].astype(np.float32) / 255.0
        dst_a = merged[..., 3:4].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1 - src_a)
        with np.errstate(divide="ignore", invalid="ignore"):
            out_rgb = np.where(
                out_a > 0,
                (
                    a[..., :3].astype(np.float32) * src_a
                    + merged[..., :3].astype(np.float32) * dst_a * (1 - src_a)
                ) / np.maximum(out_a, 1e-6),
                0,
            )
        merged[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        merged[..., 3:4] = (out_a * 255).astype(np.uint8)

    alpha = merged[..., 3] > 0
    ys, xs = np.where(alpha)
    if len(xs) == 0:
        return None, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    info = {
        "x0": x0, "x1": x1,
        "bw": x1 - x0 + 1,
        "bh": y1 - y0 + 1,
        "horizontal": (x1 - x0 + 1) / (y1 - y0 + 1) > 1.5,
    }
    return merged, info


def _resolve_blobs(
    grp: dict[str, Any],
    all_blob_info: list[dict[str, Any]],
    bands_cache: dict[int, list[str]],) -> list[str]:
    """group の blob_filter を解決して blob name リストを返す。"""
    bf = grp.get("blob_filter", "")

    def _norm(item: str) -> str:
        item = item.strip()
        return item if item.startswith("blob_") else f"blob_{int(item):02d}"

    if bf.startswith("y_band:"):
        return bands_cache.get(int(bf.split(":")[1]), [])
    elif bf.startswith("manual_list:"):
        return [_norm(n) for n in bf.split(":", 1)[1].split(",")]
    elif bf.startswith("manual:"):
        return [_norm(n) for n in bf.split(":", 1)[1].split(",")]
    # フォールバック: blobs フィールド直指定
    return grp.get("blobs", [])


def _cluster_y_bands(
    blob_info_list: list[dict[str, Any]], n: int) -> dict[int, list[str]]:
    """blob を y 座標でクラスタリングして n バンドに分割する。"""
    import numpy as np

    ys = np.array([b["center"][1] for b in blob_info_list])
    if len(ys) <= n:
        s = sorted(blob_info_list, key=lambda b: b["center"][1])
        return {i: [b["name"]] for i, b in enumerate(s[:n])}
    ys_s = np.sort(ys)
    diffs = np.diff(ys_s)
    top = np.argsort(diffs)[-(n - 1):]
    bnd = sorted([ys_s[i] + diffs[i] / 2 for i in top])

    def band_of(cy: float) -> int:
        for i, b in enumerate(bnd):
            if cy < b:
                return i
        return len(bnd)

    bands: dict[int, list[str]] = {i: [] for i in range(n)}
    for b in blob_info_list:
        bands[band_of(b["center"][1])].append(b["name"])
    return bands


def _apply_anim(
    arr: Any, info: dict, p: float, anim: str, x_grid: Any) -> Any:
    """アニメーション進捗 p(0.0-1.0)を適用してフレームを返す。"""
    import numpy as np
    from PIL import Image

    # ease-out cubic
    pe = 1 - (1 - p) ** 3
    if anim == "wipe_lr" or info["horizontal"]:
        x_thr = info["x0"] + info["bw"] * pe
        col_a = np.clip((x_thr - x_grid) / 4, 0.0, 1.0)
        a = arr.copy()
        a[..., 3] = (a[..., 3].astype(np.float32) * col_a[None, :]).astype(np.uint8)
        return Image.fromarray(a)
    # scale_pop / bounce_in / fade_in 系
    scale = 0.85 + 0.15 * pe
    af = pe
    a = arr.copy()
    a[..., 3] = (a[..., 3].astype(np.float32) * af).astype(np.uint8)
    img_l = Image.fromarray(a)
    bw_s = max(1, int(TARGET_W * scale))
    bh_s = max(1, int(TARGET_H * scale))
    rs = img_l.resize((bw_s, bh_s), Image.LANCZOS)
    canvas = Image.new("RGBA", (TARGET_W, TARGET_H), (255, 255, 255, 0))
    canvas.paste(rs, ((TARGET_W - bw_s) // 2, (TARGET_H - bh_s) // 2), rs)
    return canvas


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def render_diagram_shorts(
    diag_name: str,
    blobs_dir: Path,
    orders_dir: Path,
    out_dir: Path,
    duration: float,
    codec: str = "h264_videotoolbox",) -> Path | None:
    """diagram 1件の Shorts アニメ mp4 を生成する。

    Args:
        diag_name:  対象 diagram 名(例 diag_02_puzzle_building)
        blobs_dir:  blob PNG が格納された work/blobs/ ディレクトリ
        orders_dir: order JSON が格納された work/orders/ ディレクトリ
        out_dir:    mp4 出力先(shorts/short-A/diagrams_fast/ 等)
        duration:   この diagram の表示秒数
        codec:      映像コーデック(default: h264_videotoolbox / fallback: libx264)

    Returns:
        生成した mp4 のパス。必要ファイル不足でスキップした場合は None。
    """
    import numpy as np
    from PIL import Image

    blobs_dir = Path(blobs_dir)
    orders_dir = Path(orders_dir)
    out_dir = Path(out_dir)

    # Shorts 専用 order JSON を優先(本編 _order.json と分離)
    shorts_p = orders_dir / f"{diag_name}_shorts_order.json"
    main_p = orders_dir / f"{diag_name}_order.json"
    order_p = shorts_p if shorts_p.exists() else main_p
    blob_sub_dir = blobs_dir / diag_name

    if not order_p.exists():
        print(f"  [WARN] order JSON なし: {order_p} → スキップ", file=sys.stderr)
        return None
    if not blob_sub_dir.exists():
        print(f"  [WARN] blobs ディレクトリなし: {blob_sub_dir} → スキップ", file=sys.stderr)
        return None

    all_blobs_p = blobs_dir / "all_blobs.json"
    if not all_blobs_p.exists():
        print(f"  [WARN] all_blobs.json なし: {all_blobs_p} → スキップ", file=sys.stderr)
        return None

    order = json.loads(order_p.read_text())
    all_blobs: dict[str, Any] = json.loads(all_blobs_p.read_text())
    blob_info_list: list[dict[str, Any]] = all_blobs.get(diag_name, [])

    # Z 字読み順 sort(順序ずれ修正)
    # orderer agent が出力した groups[] 配列順は y/x で並び損ねるケースが多いので
    # blob_info_list の bbox を引いて Z 字(top→bottom→left→right)で再ソートする。
    from diagram_blob.blob_animate import sort_groups_z_order
    _bbox_by_name = {b["name"]: b.get("bbox", [0, 0, 0, 0]) for b in blob_info_list}

    def _get_yx(grp: dict[str, Any]) -> tuple[int, int]:
        bf = grp.get("blob_filter", "")
        if bf.startswith("y_band:"):
            return (10**9, 10**9)  # y_band は最後尾
        if bf.startswith("manual:") or bf.startswith("manual_list:"):
            ids = [x.strip() for x in bf.split(":", 1)[1].split(",") if x.strip()]
            names = [
                n if n.startswith("blob_") else f"blob_{int(n):02d}"
                for n in ids
            ]
            bboxes = [_bbox_by_name.get(n) for n in names]
            bboxes = [b for b in bboxes if b]
            if not bboxes:
                return (10**9, 10**9)
            y0 = min(int(b[1]) for b in bboxes)
            x0 = min(int(b[0]) for b in bboxes)
            return (y0, x0)
        return (10**9, 10**9)

    order = sort_groups_z_order(order, _get_yx)

    n_bands = sum(
        1 for g in order.get("groups", [])
        if g.get("blob_filter", "").startswith("y_band:")
    )
    bands_cache = _cluster_y_bands(blob_info_list, n_bands) if n_bands > 0 else {}

    items: list[dict[str, Any]] = []
    for idx, grp in enumerate(order.get("groups", [])):
        blob_names = _resolve_blobs(grp, blob_info_list, bands_cache)
        if not blob_names:
            continue
        # appear_t: order JSON に明示があれば優先、なければ stagger 計算
        appear_t = grp["appear_t"] if "appear_t" in grp else idx * STAGGER
        reveal_dur = grp.get("reveal_dur", REVEAL_DUR)
        merged, info = _merge_blobs(blob_names, blob_sub_dir)
        if merged is None:
            continue
        items.append({
            "label": grp.get("label", ""),
            "arr": merged,
            "info": info,
            "appear_t": appear_t,
            "reveal_dur": reveal_dur,
            "anim": grp.get("anim", "wipe_lr"),
        })

    # ---- Shorts 高速描写セーフネット(過去の事故由来)----
    # agent がセリフ同期 appear_t を出して Shorts 中盤で要素がポップする違和感を
    # 防ぐため、FAST_REVEAL_MAX_APPEAR を超える appear_t がある場合は全 group を
    # FAST_REVEAL_WINDOW 内に group 順序を維持したまま等間隔再配置する。
    if items:
        max_t = max(it["appear_t"] for it in items)
        if max_t > FAST_REVEAL_MAX_APPEAR:
            n = len(items)
            # 等間隔: 0, w/(n-1), 2w/(n-1), ..., w (n>=2 の場合)
            if n == 1:
                items[0]["appear_t"] = 0.0
            else:
                step = FAST_REVEAL_WINDOW / (n - 1)
                # group 順序(=現リスト順=order JSON の groups 順)で再配置
                items.sort(key=lambda it: it["appear_t"])  # 元の意図順序を保持
                for i, it in enumerate(items):
                    it["appear_t"] = round(i * step, 3)
            print(
                f"  [shorts-clamp] {diag_name}: appear_t max={max_t:.2f}s > "
                f"{FAST_REVEAL_MAX_APPEAR}s → 0..{FAST_REVEAL_WINDOW}s 窓に等間隔再配置 (n={n})",
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{diag_name}.mp4"
    bg = Image.new("RGBA", (TARGET_W, TARGET_H), (255, 255, 255, 255))
    x_grid = np.arange(TARGET_W)
    total_frames = int(duration * FPS)

    # コーデック選択: videotoolbox が使えなければ libx264 にフォールバック
    codec_args = ["-c:v", codec, "-b:v", "12M", "-allow_sw", "1"] if codec == "h264_videotoolbox" \
        else ["-c:v", "libx264", "-crf", "20"]

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgba", "-s", f"{TARGET_W}x{TARGET_H}", "-r", str(FPS),
            "-i", "-",
        ] + codec_args + ["-pix_fmt", "yuv420p", str(out_mp4)],
        stdin=subprocess.PIPE,
    )

    for f in range(total_frames):
        t = f / FPS
        frame = bg.copy()
        for it in items:
            if t < it["appear_t"]:
                continue
            p = min(1.0, (t - it["appear_t"]) / it["reveal_dur"])
            tinted = _apply_anim(it["arr"], it["info"], p, it["anim"], x_grid)
            frame = Image.alpha_composite(frame, tinted)
        proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()

    if proc.returncode != 0:
        print(f"  [ERROR] ffmpeg 失敗: {diag_name}", file=sys.stderr)
        return None

    last_appear = (len(items) - 1) * STAGGER + REVEAL_DUR if items else 0.0
    print(
        f"  ok {diag_name} -> {out_mp4.name}"
        f" ({duration:.2f}s, {len(items)} groups, all out by {last_appear:.1f}s)"
    )
    return out_mp4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Shorts 用 diagram アニメ mp4 を生成する")
    ap.add_argument("--diag-name", required=True, help="diagram 名")
    ap.add_argument("--blobs-dir", required=True, help="work/blobs/ ディレクトリ")
    ap.add_argument("--orders-dir", required=True, help="work/orders/ ディレクトリ")
    ap.add_argument("--out-dir", required=True, help="mp4 出力先")
    ap.add_argument("--duration", type=float, required=True, help="表示秒数")
    ap.add_argument(
        "--codec", default="h264_videotoolbox",
        help="映像コーデック (default: h264_videotoolbox)",
    )
    args = ap.parse_args()

    result = render_diagram_shorts(
        diag_name=args.diag_name,
        blobs_dir=Path(args.blobs_dir),
        orders_dir=Path(args.orders_dir),
        out_dir=Path(args.out_dir),
        duration=args.duration,
        codec=args.codec,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    _main()
