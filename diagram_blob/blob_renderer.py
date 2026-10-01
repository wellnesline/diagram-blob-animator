"""blob_renderer — blob 分離 + アニメ order JSON から reveal アニメ動画を生成。

解説動画 / セミナー型動画 共通の blob reveal アニメ実装。

入力:
  base_img:      完成形(全 blob 表示済)の raw PNG
  labels_npz:    blob_split が出力した labels.npz パス
                 (blobs_dir/<diag_name>/labels.npz、pixel value = blob_index+1, 0=背景)
  order:         resolve_blobs 済みの order JSON dict
                 (groups[].resolved_blobs に [[blob, ...], ...] が入っている)
  line_timings:  [{ "li": 0, "start": 0.0, "end": 4.84 }, ...]
                 line(=trigger_idx)ごとの start/end 秒
  duration:      全体の動画尺(slide 末尾無音込み)
  canvas_size:   出力解像度 (W, H)。None なら base_img のサイズと同じ

出力:
  out_mp4 に 1 slide / 1 diagram の reveal アニメ動画

中間ファイル(フレーム PNG)は tempfile.TemporaryDirectory に置き、
処理完了で自動削除。

依存: Pillow, numpy, ffmpeg(PATH)

実装方針:
- bbox ベースではなく **per-blob ピクセルマスク**(labels.npz)を使って描画
- 外枠 blob のように bbox が他要素を内包する場合でも、
  実際の描画ピクセルだけを reveal できる
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# easing helpers (0..1 → 0..1)
# ---------------------------------------------------------------------------

def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def _ease_out_back(t: float, overshoot: float = 1.7) -> float:
    """scale_pop 用。overshoot で1.0 を超えてから戻る。"""
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def _bounce(t: float) -> float:
    """bounce_in: 上下バウンドオフセット用(0..1 を返す)。"""
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def render_diagram_anim(
    base_img: Path,
    labels_npz: Path,
    order: dict[str, Any],
    duration: float,
    out_mp4: Path,
    line_timings: list[dict[str, Any]] | None = None,
    canvas_size: tuple[int, int] | None = None,
    fps: int = 30,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    auto_stagger: bool = True,
    stagger_gap: float = 0.7,
    initial_delay: float = 0.0,
    target_finish_time: float | None = None,
    max_reveal_cap: float = 0.5,
) -> Path:
    """blob 分離 + 順序計画から reveal アニメ動画を生成。

    Args:
        base_img:      完成形 PNG パス(全 blob 込みの元画像)
        labels_npz:    blob_split が出力した labels.npz パス
        order:         resolve_blobs 済みの order dict
        duration:      動画尺(秒)
        out_mp4:       出力 mp4 パス
        line_timings:  trigger_idx 連動を使う場合のみ必要
                       [{"li": 0, "start": 0.0, "end": 4.84}, ...]
                       auto_stagger=True なら無視される
        canvas_size:   出力解像度 (W, H)。None なら base_img のサイズと同じ
        fps:           フレームレート
        bg_color:      背景色 RGB
        auto_stagger:  True(デフォルト)= 配列順 + stagger_gap で自動配置
                       False = order の trigger_idx + delay_after_trigger を使う
        stagger_gap:   group 間の発火間隔(秒)。auto_stagger=True 時のみ
        initial_delay: 最初の group の発火タイミング(秒)。auto_stagger=True 時のみ
        target_finish_time: 全描写完了までの目標秒数(auto_stagger=True 時のみ)。
                       指定すると stagger_gap と reveal_dur を圧縮して target 内に収める。
                       None ならデフォルトの stagger_gap / reveal_dur を使う。
        max_reveal_cap: reveal_dur の上限(秒)。target_finish_time と組み合わせて
                       min(max_reveal_cap, target/2) で上限を決定する。default=0.5。

    Returns:
        out_mp4 のパス
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"blob_renderer には Pillow / numpy が必要: {e}") from e

    base_img = Path(base_img)
    labels_npz = Path(labels_npz)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    base = Image.open(base_img).convert("RGB")
    if canvas_size is None:
        canvas_size = base.size

    # base 解像度と canvas 解像度が違う場合は base を canvas にフィット
    # ただし labels も同じ解像度に合わせる必要がある(今回は base 解像度 = canvas 前提)
    if base.size != canvas_size:
        raise NotImplementedError(
            f"canvas_size {canvas_size} != base size {base.size}: "
            "labels の resize 未対応。同サイズで呼び出してください。")

    base_arr = np.array(base)  # H x W x 3
    H, W = base_arr.shape[:2]
    np_bg = np.array(bg_color, dtype=np.uint8)

    # labels: H x W の uint16、0=背景、N>=1 で blob_index = N-1
    labels = np.load(labels_npz)["labels"]
    if labels.shape != (H, W):
        raise ValueError(f"labels shape {labels.shape} != base shape {(H, W)}")

    # === per-group meta 構築 ===
    # auto_stagger=True: groups 配列順に stagger_gap 間隔で発火
    # auto_stagger=False: trigger_idx + delay_after_trigger を line_timings から解決
    line_start: dict[int, float] = {}
    if not auto_stagger:
        if line_timings is None:
            raise ValueError("auto_stagger=False の時は line_timings 必須")
        line_start = {l["li"]: float(l["start"]) for l in line_timings}

    # anim 種類別 default reveal_dur
    DEFAULT_REVEAL_DUR = {
        "wipe_lr":   0.6,
        "wipe_tb":   0.6,
        "scale_pop": 0.5,
        "bounce_in": 0.5,
        "fade_in":   0.5,
    }

    n_groups = len(order.get("groups", []))

    # target_finish_time が指定されたら stagger_gap と reveal_dur を target 内に収める
    effective_stagger = stagger_gap
    effective_reveal_dur_override: float | None = None
    if auto_stagger and target_finish_time is not None and n_groups > 0:
        # reveal_dur 上限 = min(max_reveal_cap, target/2)
        max_reveal = min(max_reveal_cap, target_finish_time / 2.0)
        # 全 group 完了 = (n-1) * stagger + reveal_dur ≦ target
        if n_groups > 1:
            effective_stagger = max(0.08, (target_finish_time - max_reveal) / (n_groups - 1))
        else:
            effective_stagger = stagger_gap
        # reveal_dur が effective_stagger * 1.5 を超えると group が重なりすぎるので圧縮
        effective_reveal_dur_override = min(max_reveal, effective_stagger * 1.5)

    per_group_meta: list[dict[str, Any]] = []
    for g_idx, g in enumerate(order.get("groups", [])):
        anim = g.get("anim", "fade_in")
        if effective_reveal_dur_override is not None:
            reveal_dur = effective_reveal_dur_override
        else:
            reveal_dur = float(g.get("reveal_dur", DEFAULT_REVEAL_DUR.get(anim, 0.5)))
        stagger = float(g.get("stagger", 0.0))

        if auto_stagger:
            # 配列順に effective_stagger 間隔で発火
            t_base = initial_delay + g_idx * effective_stagger
        else:
            trigger_idx = int(g.get("trigger_idx", 0))
            delay = float(g.get("delay_after_trigger", 0.0))
            t_base = line_start.get(trigger_idx, 0.0) + delay

        resolved_groups: list[list[dict]] = g.get("resolved_blobs", [])

        # 案 C: 最終防衛ガード (blob_finalize_guard spike)
        # blob_filter が非空(orderer が blob 指定済み)かつ resolved_blobs が欠落している場合、
        # reveal アニメが無音で素通りする。 run_all 経由では Step 1.85 が事前に防ぐが、
        # parallel_render を直接呼んだ / --skip-blob-check を使った経路の最終防衛として raise。
        # groups が空の slide(blob なし slide)は blob_filter も空なので誤 raise しない。
        _blob_filter = g.get("blob_filter", "")
        if _blob_filter and not resolved_groups:
            raise ValueError(
                f"[blob_renderer] resolved_blobs 欠落: group[{g_idx}] label={g.get('label')!r}, "
                f"blob_filter={_blob_filter!r}。\n"
                f"  blob_anim_finalize が未実行の可能性があります。\n"
                f"  以下を実行してから parallel_render を再起動してください:\n"
                f"    python -m src.post_video.blob_anim_finalize --topic <topic>\n"
                f"  run_all 経由なら Step 1.85 が自動対処します。"
            )

        per_blob: list[tuple[dict, float]] = []
        bi = 0
        for sub in resolved_groups:
            for b in sub:
                t_start = t_base + bi * stagger
                per_blob.append((b, t_start))
                bi += 1

        per_group_meta.append({
            "anim": anim,
            "reveal_dur": reveal_dur,
            "stagger": stagger,
            "blobs": per_blob,
        })

    # === revealed blob ID 集合(これらを order 描画対象とする)===
    revealed_blob_indices: set[int] = set()
    for g_meta in per_group_meta:
        for b, _ in g_meta["blobs"]:
            revealed_blob_indices.add(int(b["id"]))

    # === static_layer 構築 ===
    # static_layer = base から「revealed blob のピクセル」だけを背景色で消した画像
    # これにより:
    # - order に出てこない blob は最初から base そのまま見える
    # - revealed blob だけが anim 進捗で paste される
    static_layer = base_arr.copy()
    for bi in revealed_blob_indices:
        m = labels == (bi + 1)
        static_layer[m] = np_bg

    # === per-blob mask 事前計算 ===
    blob_masks: dict[int, "np.ndarray"] = {}
    for bi in revealed_blob_indices:
        blob_masks[bi] = (labels == (bi + 1))

    # === レンダリング ===
    total_frames = max(1, int(round(duration * fps)))

    with tempfile.TemporaryDirectory(prefix="blob_render_") as tmpd:
        tmp = Path(tmpd)

        # x/y grid を事前計算(wipe_lr / wipe_tb 用)
        xs_grid = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
        ys_grid = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)

        for fi in range(total_frames):
            t = fi / fps
            canvas = static_layer.copy()

            for g_meta in per_group_meta:
                anim = g_meta["anim"]
                rdur = g_meta["reveal_dur"]
                for b, t_start in g_meta["blobs"]:
                    bi = int(b["id"])
                    bmask = blob_masks[bi]
                    if t < t_start:
                        continue
                    if t >= t_start + rdur:
                        # 完了:mask 全体を base から paste
                        canvas[bmask] = base_arr[bmask]
                        continue
                    p = (t - t_start) / rdur
                    _apply_anim(
                        canvas, base_arr, b, bmask, anim, p, np_bg,
                        xs_grid, ys_grid,)

            Image.fromarray(canvas).save(tmp / f"f_{fi:05d}.png")

        # === ffmpeg で連番 → mp4 ===
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp / "f_%05d.png"),
            "-c:v", "h264_videotoolbox",
            "-b:v", "4M",
            "-pix_fmt", "yuv420p",
            "-an",
            str(out_mp4),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(tmp / "f_%05d.png"),
                "-c:v", "libx264",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-an",
                str(out_mp4),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {p.stderr[-2000:]}")

    return out_mp4


def _apply_anim(
    canvas,        # numpy ndarray HxWx3 uint8
    base,          # numpy ndarray HxWx3 uint8
    blob: dict,
    bmask,         # numpy ndarray HxW bool (この blob のピクセル)
    anim: str,
    p: float,      # 0..1 進捗
    bg,            # ndarray (3,) uint8
    xs_grid,       # H x W float32 (x座標)
    ys_grid,       # H x W float32 (y座標)
) -> None:
    """canvas を anim 種類 / 進捗 p で描画(blob mask ベース)。

    - canvas は static_layer のコピー(revealed blob は背景色で埋まっている前提)
    - blob のピクセルだけを mask で抽出して、anim 種類に応じて部分的に paste
    """
    import numpy as np

    x0, y0, x1, y1 = blob["x0"], blob["y0"], blob["x1"], blob["y1"]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return

    if anim == "fade_in":
        e = _ease_out_cubic(p)
        # mask 内のみ alpha blend
        cur = canvas.astype(np.float32)
        b = base.astype(np.float32)
        out = cur * (1 - e) + b * e
        canvas[bmask] = out[bmask].astype(np.uint8)
        return

    if anim == "wipe_lr":
        e = _ease_out_cubic(p)
        threshold_x = x0 + bw * e
        reveal = bmask & (xs_grid < threshold_x)
        canvas[reveal] = base[reveal]
        return

    if anim == "wipe_tb":
        e = _ease_out_cubic(p)
        threshold_y = y0 + bh * e
        reveal = bmask & (ys_grid < threshold_y)
        canvas[reveal] = base[reveal]
        return

    if anim == "scale_pop":
        # bbox 内の blob ピクセル群を、bbox 中心固定でスケール
        e = _ease_out_back(p, overshoot=1.7)
        scale = max(0.05, e)
        # blob bbox crop(base, bmask 両方)
        crop_base = base[y0:y1, x0:x1]
        crop_mask = bmask[y0:y1, x0:x1]
        # crop_base の非マスクピクセルは bg に置換しておく(scale 時の周辺漏れ防止)
        from PIL import Image as _Image
        rgb = crop_base.copy()
        rgb[~crop_mask] = bg
        # alpha=255 mask 部分のみ、255 以外は 0
        alpha = (crop_mask.astype(np.uint8)) * 255
        rgba = np.dstack([rgb, alpha])
        rgba_pil = _Image.fromarray(rgba, mode="RGBA")
        new_w = max(1, int(bw * scale))
        new_h = max(1, int(bh * scale))
        scaled = rgba_pil.resize((new_w, new_h), _Image.LANCZOS)
        scaled_arr = np.array(scaled)
        # 中心配置
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        nx0 = cx - new_w // 2
        ny0 = cy - new_h // 2
        nx1 = nx0 + new_w
        ny1 = ny0 + new_h
        ch_canvas, cw_canvas = canvas.shape[:2]
        sx0 = max(0, -nx0); sy0 = max(0, -ny0)
        nx0c = max(0, nx0); ny0c = max(0, ny0)
        nx1c = min(cw_canvas, nx1); ny1c = min(ch_canvas, ny1)
        sx1 = sx0 + (nx1c - nx0c); sy1 = sy0 + (ny1c - ny0c)
        if nx1c > nx0c and ny1c > ny0c:
            patch_rgba = scaled_arr[sy0:sy1, sx0:sx1]
            patch_rgb = patch_rgba[..., :3]
            patch_a = patch_rgba[..., 3:4].astype(np.float32) / 255.0
            cur = canvas[ny0c:ny1c, nx0c:nx1c].astype(np.float32)
            blended = cur * (1 - patch_a) + patch_rgb.astype(np.float32) * patch_a
            canvas[ny0c:ny1c, nx0c:nx1c] = blended.astype(np.uint8)
        return

    if anim == "bounce_in":
        # bounce easing(振動)で alpha blend(オフセット無し、shape mismatch 回避)
        e = _bounce(p)
        cur = canvas.astype(np.float32)
        b = base.astype(np.float32)
        out = cur * (1 - e) + b * e
        canvas[bmask] = out[bmask].astype(np.uint8)
        return

    # フォールバック
    _apply_anim(canvas, base, blob, bmask, "fade_in", p, bg, xs_grid, ys_grid)
