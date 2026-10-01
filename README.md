# diagram-blob-animator

Splits an explainer image into connected components ("blobs") and renders a reveal animation that shows the parts one by one, in the order of a JSON plan.

1 枚の図解画像を部品（連結成分）に分け、順番に出現させるアニメ動画（mp4）を作る Python モジュール群です。公開用に整理したサンプル実装です。

## 概要

図解の画像をそのまま見せるより、話の流れに合わせて部品を順に出す方が伝わります。しかし部品ごとに素材を作るのは手間です。このリポジトリは、完成した 1 枚絵を画像処理で部品に分解し、並び順を記述した JSON（order JSON）に従って段階的に表示する工程を提供します。

## 主な機能・設計のポイント

- **部品分解**: `blob_split.py`（v1）と `blob_split_v2.py`。v2 は、ラベリング用マスクは膨張させて連結性を確保し、出力 PNG の alpha は膨張なしの厳密なマスクで切り出して文字の縁をシャープに保ちます。出力は `labels.npz` / `blobs.json` / 部品ごとの RGBA PNG
- **order JSON の解決**: `blob_animate.py` が order JSON と部品情報から、描画に使う最終形式を作ります（映像は作らない前処理）
- **自動グルーピング**: `auto_grouping.py` は、部品の面積比などから機械的なルールで order JSON を組み立てる代替手段です。LLM による順序づけを使わない運用向けです。
- **レンダラ**: `blob_renderer.py` が reveal アニメ動画を出力（ffmpeg）。縦型向けの `blob_animate_shorts.py` は、部品を一定の間隔で順次出現させます
- **付属ユーティリティ**: Shorts 用の背景（本編背景の回転転用）、アウトロ画像、タイトル画像の生成（`gen_*.py`。画像モデルを使うのはアウトロ・タイトルのみ）。画像モデル名と品質の既定値は `openai_image_config.py` に一箇所に集約

## ディレクトリ構成（主要ファイル）

```
diagram-blob-animator/
├ diagram_blob/
│  ├ blob_split.py  blob_split_v2.py      部品分解
│  ├ blob_animate.py  auto_grouping.py    order JSON の解決・自動生成
│  ├ blob_renderer.py  blob_animate_shorts.py   動画レンダリング
│  ├ build_diagrams_anim.py               work ディレクトリ一括生成
│  └ gen_bg_shorts.py  gen_outro_thumb.py  gen_short_title.py
├ openai_image_config.py  _dotenv.py      ← 合成スタブ
```

## 動かし方

- 必要なもの: Python 3、ffmpeg、下記「同梱していない依存」の Python パッケージ
- 環境変数: `OPENAI_API_KEY`（`gen_outro_thumb.py` / `gen_short_title.py` のみ）、`DIAGRAM_CHANNEL`（`gen_*.py` が参照する設定名。既定は `demo`）、`DIAGRAM_STYLE_PALETTE` / `DIAGRAM_STYLE_TEXT`（`gen_outro_thumb.py` / `gen_short_title.py` の画像スタイル指定。既定は中立なサンプル値）
- 実行例: `python -m diagram_blob.build_diagrams_anim --topic-dir <作業ディレクトリ> ...`、`python -m diagram_blob.blob_split_v2 --raw <画像.png> ...`（引数は各ファイルの `--help` を参照）
- コードはホストプロジェクトの `src/` に置かれる前提のモジュール参照を含みます（スタブ `_dotenv.py` のコメント参照）

## 制限事項（閉包）

`ast` と import 文の調査で、リポジトリ内・標準ライブラリに無く、`requirements.txt` も無いため、次を同梱していない依存として挙げます。

- **同梱していない Python パッケージ（3 件）**: `PIL`（Pillow）, `numpy`, `scipy`
- **他の公開リポジトリへの依存**: なし（本リポジトリは `slide-to-video-pipeline` と `manga-panel-video` から呼び出されます）
- **合成スタブ**: `openai_image_config.py`（モデル名と品質の定数、検証関数のみ）, `_dotenv.py`（最小の .env ローダ。本来の実装ではありません）
- 外部サービス: OpenAI 画像生成（任意。入力画像・生成画像は利用規約と権利者の許諾の範囲で扱ってください）
- 処理対象の図解画像は同梱していません

## ライセンス

All rights reserved. 閲覧目的で公開しています。詳細は [LICENSE](LICENSE) を参照。
