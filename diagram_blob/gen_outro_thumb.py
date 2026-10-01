"""gen_outro_thumb — Shorts 用アウトロサムネイル画像生成(gpt-image-2)

CTA 文言を含むアウトロカード画像を生成する。

出力先:
  <diagrams_dir>/_outro_thumb_raw.png   生成 raw 画像(1536×1024)

使い方(スクリプトから):
  from src.diagram_blob.gen_outro_thumb import gen_outro_thumb

使い方(CLI):
  python -m src.diagram_blob.gen_outro_thumb \\
    --topic <topic> \\
    --channel <channel> \\
    --main-title "<main title>" \\
    --cta-text "<cta text>" \\
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

def _build_prompt(main_title: str, cta_text: str) -> str:
    return f"""Outro card for a Japanese vertical short video.

Layout (1536x1024 landscape, keep content in the center 80%):
- Center: large text "{main_title}" inside a simple frame
- Below: an arrow pointing right, followed by the text "{cta_text}"

Style: {_STYLE_TEXT}
Colors: {_STYLE_PALETTE}.

CRITICAL safe area: keep upper 80px and lower 80px completely blank (white).
"""


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def gen_outro_thumb(
    main_title: str,
    diagrams_dir: Path,
    cta_text: str = "",
    force: bool = False,
    root: Path | None = None,) -> Path:
    """gpt-image-2 でアウトロサムネを生成して _outro_thumb_raw.png に保存する。

    Args:
        main_title:    動画のメインタイトル
        diagrams_dir:  出力先ディレクトリ(diagrams/ 等)
        cta_text:      CTA テキスト(省略時は空)
        force:         既存ファイルがあっても再生成する
        root:          プロジェクトルート(省略時は自動検出)

    Returns:
        生成した raw PNG のパス
    """
    root = root or _ROOT
    credential = _load_api_key(root)
    diagrams_dir = Path(diagrams_dir)
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    raw_out = diagrams_dir / "_outro_thumb_raw.png"

    if raw_out.exists() and not force:
        print(f"[gen_outro_thumb] skip(既存): {raw_out}")
        return raw_out

    prompt = _build_prompt(main_title, cta_text)
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
    print(f"[gen_outro_thumb] 生成中: {main_title!r} ...", flush=True)
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    b = base64.b64decode(data["data"][0]["b64_json"])
    raw_out.write_bytes(b)
    print(f"[gen_outro_thumb] raw: {raw_out}")
    return raw_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Shorts 用アウトロサムネを gpt-image-2 で生成する")
    ap.add_argument("--topic", required=True, help="topic slug")
    ap.add_argument("--channel", default=_DEFAULT_CHANNEL, help="チャンネル名")
    ap.add_argument("--main-title", required=True, help="動画メインタイトル")
    ap.add_argument("--cta-text", default="", help="CTA テキスト")
    ap.add_argument("--force", action="store_true", help="強制再生成")
    args = ap.parse_args()

    diagrams_dir = _ROOT / "channels" / args.channel / "videos" / args.topic / "diagrams"
    result = gen_outro_thumb(
        main_title=args.main_title,
        diagrams_dir=diagrams_dir,
        cta_text=args.cta_text,
        force=args.force,)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    _main()
