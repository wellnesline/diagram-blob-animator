"""diagram_blob — 解説動画系統ユーティリティ

サブモジュール:
  blob_split          : 解説画像の連結成分(blob)分離(v1)
  blob_split_v2       : blob 分離 v2(alpha シャープ抽出 + RGBA PNG 出力)
  blob_animate        : blob_filter 解決 → order JSON 書き出し(非映像)
  blob_animate_shorts : Shorts 用 diagram アニメ mp4 レンダラ(stagger 0.2 固定)
  gen_short_title     : Shorts 用タイトル画像生成(gpt-image-2。スタイルは DIAGRAM_STYLE_TEXT で指定)
  gen_outro_thumb     : Shorts 用アウトロサムネ生成(gpt-image-2)
  gen_bg_shorts       : 本編背景 raw を 90 度回転して縦型 bg_shorts.png を生成

使い方は各モジュールの docstring を参照。
"""
