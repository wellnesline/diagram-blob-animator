"""gen_short_title — Shorts 用タイトル画像生成(gpt-image-2。スタイルは DIAGRAM_STYLE_TEXT で指定)

短い説明文(title_text)と補足テキスト(sub_text)を受け取り、
タイトルカード画像を gpt-image-2 で生成する(スタイルは環境変数 DIAGRAM_STYLE_TEXT / DIAGRAM_STYLE_PALETTE で指定)。

出力先:
  <diagrams_dir>/_short_title_raw.png   生成 raw 画像(1536×1024)

使い方(スクリプトから):
  from diagram_blob.gen_short_title import gen_short_title

使い方(CLI):
  python -m diagram_blob.gen_short_title \\
    --topic <topic> \\
    --channel <channel> \\
    --title-text "<title text>" \\
    --sub-text "<sub text>" \\
    --force
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

_DEFAULT_CHANNEL = os.environ.get("DIAGRAM_CHANNEL", "demo")
# スタイルの既定値は中立なサンプル。配色は環境変数 DIAGRAM_STYLE_PALETTE で渡す。
_STYLE_PALETTE = os.environ.get("DIAGRAM_STYLE_PALETTE", "charcoal + teal + light grey")
# スタイル文の既定値は中立な短文。環境変数 DIAGRAM_STYLE_TEXT で差し替える。
_STYLE_TEXT = os.environ.get("DIAGRAM_STYLE_TEXT", "Simple flat card on a plain light background.")
_ROOT = Path(__file__).resolve().parent.parent  # リポジトリ直下(作業ルート)

# openai_image_config: model / quality の既定値を一箇所から取得(同梱の合成スタブ)
sys.path.insert(0, str(_ROOT))
from openai_image_config import (  # noqa: E402
    DEFAULT_IMAGE_MODEL as _DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_QUALITY as _DEFAULT_IMAGE_QUALITY,
    validate_image_defaults,)


# ---------------------------------------------------------------------------
# API キー取得
# ---------------------------------------------------------------------------

def _load_api_key(root: Path) -> str:
    """共通ローダー(_dotenv.py)経由で .env を読み、OPENAI_API_KEY を返す。
    .env が shell env を上書きする(キーローテ追従)。"""
    sys.path.insert(0, str(root))
    from _dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY が設定されていません。.env に記載してください。")
    return key


# ---------------------------------------------------------------------------
# プロンプト構築
# ---------------------------------------------------------------------------

def _build_prompt(title_text: str, sub_text: str) -> str:
    return f"""Title card for a Japanese vertical short video.

Layout (1536x1024 landscape, keep content in the center 80%):
- Large headline in the center: "{title_text}"
- Bottom-center small text: "{sub_text}" (omit if empty)

Style: {_STYLE_TEXT}
Colors: {_STYLE_PALETTE}.

CRITICAL safe area: keep upper 80px and lower 80px completely blank (white).
All elements must fit in central 1536x863 band.
"""


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def gen_short_title(
    title_text: str,
    diagrams_dir: Path,
    sub_text: str = "",
    force: bool = False,
    root: Path | None = None,) -> Path:
    """gpt-image-2 で Shorts タイトル画像を生成して _short_title_raw.png に保存する。

    Args:
        title_text:    タイトルのメインテキスト
        diagrams_dir:  出力先ディレクトリ(diagrams/ 等)
        sub_text:      補足テキスト(省略可)
        force:         既存ファイルがあっても再生成する
        root:          プロジェクトルート(省略時は自動検出)

    Returns:
        生成した raw PNG のパス
    """
    root = root or _ROOT
    credential = _load_api_key(root)
    diagrams_dir = Path(diagrams_dir)
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    raw_out = diagrams_dir / "_short_title_raw.png"

    if raw_out.exists() and not force:
        print(f"[gen_short_title] skip(既存): {raw_out}")
        return raw_out

    prompt = _build_prompt(title_text, sub_text)
    payload = json.dumps({
        "model": _DEFAULT_IMAGE_MODEL,
        "prompt": prompt,
        "size": "1536x1024",
        "n": 1,
        "quality": _DEFAULT_IMAGE_QUALITY,
    }).encode()
    validate_image_defaults(_DEFAULT_IMAGE_MODEL, _DEFAULT_IMAGE_QUALITY)
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=payload,
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        },
        method="POST",)
    print(f"[gen_short_title] 生成中: {title_text!r} ...", flush=True)
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    b = base64.b64decode(data["data"][0]["b64_json"])
    raw_out.write_bytes(b)
    print(f"[gen_short_title] raw: {raw_out}")
    return raw_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Shorts 用タイトル画像を gpt-image-2 で生成する")
    ap.add_argument("--topic", required=True, help="topic slug (例: <topic>)")
    ap.add_argument("--channel", default=_DEFAULT_CHANNEL, help="チャンネル名")
    ap.add_argument("--title-text", required=True, help="タイトルのメインテキスト")
    ap.add_argument("--sub-text", default="", help="補足テキスト(省略可)")
    ap.add_argument("--force", action="store_true", help="既存ファイルを上書きして再生成")
    args = ap.parse_args()

    diagrams_dir = _ROOT / "channels" / args.channel / "videos" / args.topic / "diagrams"
    result = gen_short_title(
        title_text=args.title_text,
        diagrams_dir=diagrams_dir,
        sub_text=args.sub_text,
        force=args.force,)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    _main()
