"""gen_bg_shorts — Shorts 用背景画像生成

本編 bg_raw_1536x1024.png を 90 度回転して縦型(1024×1536 = 9:14 相当)の
bg_shorts.png を生成する。

gpt-image-2 での新規生成は行わない。本編用背景からの転用がデフォルト。
assets/bg_shorts.png が既に存在する場合はスキップ(--force で上書き)。

出力先:
  <assets_dir>/bg_shorts.png   縦型背景(1024×1536)

使い方(スクリプトから):
  from src.diagram_blob.gen_bg_shorts import gen_bg_shorts

使い方(CLI):
  python -m src.diagram_blob.gen_bg_shorts \\
    --channel <channel> \\
    --force
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

_DEFAULT_CHANNEL = os.environ.get("DIAGRAM_CHANNEL", "demo")
_ROOT = Path(__file__).resolve().parent.parent  # リポジトリ直下(作業ルート)


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def gen_bg_shorts(
    assets_dir: Path,
    bg_raw_path: Path | None = None,
    force: bool = False,) -> Path:
    """bg_raw_1536x1024.png を 90 度回転して bg_shorts.png を生成する。

    Args:
        assets_dir:   出力先の assets ディレクトリ
        bg_raw_path:  本編 raw 背景 PNG(省略時は assets_dir/../assets/bg_raw_1536x1024.png)
        force:        既存ファイルがあっても再生成する

    Returns:
        生成した bg_shorts.png のパス

    Raises:
        FileNotFoundError: bg_raw_path が存在しない場合
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"gen_bg_shorts には Pillow が必要です: {e}") from e

    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "bg_shorts.png"

    if out_path.exists() and not force:
        print(f"[gen_bg_shorts] skip(既存): {out_path}")
        return out_path

    if bg_raw_path is None:
        bg_raw_path = assets_dir / "bg_raw_1536x1024.png"
    bg_raw_path = Path(bg_raw_path)

    if not bg_raw_path.exists():
        raise FileNotFoundError(
            f"本編背景 raw が見つかりません: {bg_raw_path}\n"
            "先に 別途用意した背景生成スクリプトで本編背景を生成してください。")

    img = Image.open(bg_raw_path).convert("RGB")
    # 90 度回転: ROTATE_90 = 反時計回り 90 度(縦長になる)
    rotated = img.rotate(90, expand=True)
    # 縦型 Shorts(1080×1920)に合わせて resize
    rotated_resized = rotated.resize((1080, 1920), Image.LANCZOS)
    rotated_resized.save(out_path)
    print(f"[gen_bg_shorts] ok: {out_path} ({rotated_resized.size})")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="本編背景 raw を 90 度回転して Shorts 用縦型背景を生成する")
    ap.add_argument("--channel", default=_DEFAULT_CHANNEL, help="チャンネル名")
    ap.add_argument(
        "--bg-raw",
        help="本編 raw 背景 PNG のパス(省略時は assets/bg_raw_1536x1024.png)",
    )
    ap.add_argument("--force", action="store_true", help="強制再生成")
    args = ap.parse_args()

    assets_dir = _ROOT / "channels" / args.channel / "assets"
    bg_raw_path = Path(args.bg_raw) if args.bg_raw else None
    result = gen_bg_shorts(
        assets_dir=assets_dir,
        bg_raw_path=bg_raw_path,
        force=args.force,)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    _main()
