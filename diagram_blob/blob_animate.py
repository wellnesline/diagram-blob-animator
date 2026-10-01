"""blob_animate — blob 分析結果から reveal アニメ order JSON を生成するモジュール

orderer エージェントが書き出した order JSON と
blob_split が書き出した all_blobs.json を読み込み、
blob_animate 実行用の最終 JSON(orders 形式)を生成する。

出力先:
  <orders_dir>/<diag_name>_order.json   アニメ順序指示 JSON

orders_dir のデフォルトは topic ディレクトリ配下の work/orders/ を推奨。
/tmp 等の一時ディレクトリへのハードコードは禁止。

order JSON フォーマット(orderer 出力):
  {
    "diagram_id": "<diag_name>",
    "groups": [
      {
        "label": "title",
        "trigger_idx": 0,
        "delay_after_trigger": 0.3,
        "reveal_dur": 0.7,
        "anim": "wipe_lr",
        "blob_filter": "manual:01,02,07"
      },
      ...
    ]
  }

使い方(スクリプトから):
  from src.diagram_blob.blob_animate import load_order, resolve_blobs, write_order

使い方(CLI):
  python -m src.diagram_blob.blob_animate \\
    --order channels/<channel>/videos/<topic>/work/orders/diag_01_question_order.json \\
    --all-blobs channels/<channel>/videos/<topic>/work/blobs/all_blobs.json \\
    --out-dir channels/<channel>/videos/<topic>/work/orders
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# blob_filter 解析
# ---------------------------------------------------------------------------

def _norm_blob_id(s: str) -> str:
    """blob id をゼロパディング除去で正規化("00" → "0", "03" → "3", "10" → "10")。

    order JSON は orderer agent の出力癖で leading-zero 混在("manual:00,03,11" vs
    "manual:0,3,11")。 build_diagrams_anim の v2→v1 変換でキー側は除去済みなので、
    parse 側も同じ正規化を適用してマッチを揃える。 過去の事故由来。
    """
    s = s.strip()
    if not s:
        return s
    try:
        return str(int(s))
    except ValueError:
        return s


def _normalize_blob(b: dict[str, Any]) -> dict[str, Any]:
    """v1 schema ({id,x0,y0,x1,y1,area}) と v2 schema ({name,bbox,size,center}) の双方を
    v1 schema に正規化して返す。 元 dict は変更しない。"""
    if "id" in b:
        return b  # v1 そのまま
    if "name" in b and "bbox" in b:
        # v2 → v1
        name = b["name"]
        # "blob_03" → "03"
        if name.startswith("blob_"):
            bid = name[len("blob_"):]
        else:
            bid = name
        x0, y0, x1, y1 = b["bbox"]
        return {
            "id": bid,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "area": b.get("size", 0),
        }
    raise KeyError(f"blob に id も name/bbox も無い: {list(b.keys())}")


def _parse_blob_filter(
    blob_filter: str,
    blobs: list[dict[str, Any]],) -> list[list[dict[str, Any]]]:
    """blob_filter 文字列を解析して blob グループのリストを返す。

    Returns:
        list of groups. 各 group は blob のリスト。
        manual/y_band → [ [b1, b2, ...] ]  (1グループ)
        manual_list   → [[b1],[b2],...]  (各 blob が独立グループ=stagger)
    """
    # v1/v2 schema どちらでも受けられるよう正規化
    blobs = [_normalize_blob(b) for b in blobs]
    blob_by_id: dict[str, dict[str, Any]] = {_norm_blob_id(b["id"]): b for b in blobs}

    if blob_filter.startswith("manual:"):
        ids = [_norm_blob_id(x) for x in blob_filter[len("manual:"):].split(",") if x.strip()]
        selected = [blob_by_id[bid] for bid in ids if bid in blob_by_id]
        return [selected] if selected else []

    elif blob_filter.startswith("manual_list:"):
        ids = [_norm_blob_id(x) for x in blob_filter[len("manual_list:"):].split(",") if x.strip()]
        return [ [blob_by_id[bid]] for bid in ids if bid in blob_by_id ]

    elif blob_filter.startswith("y_band:"):
        band_idx = int(blob_filter[len("y_band:"):])
        if not blobs:
            return []
        y_vals = sorted({b["y0"] for b in blobs})
        # y 座標でクラスタリング(単純分位分割)
        n_bands = max(y_vals) // 100 + 1 if y_vals else 1
        band_h = (max(y_vals) - min(y_vals) + 1) / max(n_bands, 1)
        y_min = min(y_vals)
        band_blobs = [
            b for b in blobs
            if int((b["y0"] - y_min) / band_h) == band_idx
        ]
        return [band_blobs] if band_blobs else []

    else:
        # フォールバック: blob_filter をそのまま警告して空返却
        import sys
        print(f"[blob_animate] 未知の blob_filter: {blob_filter}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def load_order(order_path: Path) -> dict[str, Any]:
    """order JSON を読み込む。"""
    return json.loads(Path(order_path).read_text())


# ---------------------------------------------------------------------------
# Z 字読み順 sort(順序ずれ修正)
# ---------------------------------------------------------------------------

def sort_groups_z_order(
    order: dict[str, Any],
    get_yx,
    band_h: int = 80,) -> dict[str, Any]:
    """order の groups を Z 字読み順(上→下、 横同バンド内では左→右)で再ソート。

    `trigger_idx` あり: 同 trigger_idx の chunk 内のみソート。 chunk 間順序(trigger_idx 昇順)
    は不変。 これにより本編セリフ同期 mode を破壊しない。
    `appear_t` あり: chunk 内の元値の chronological 順を spatial-sorted な並びへ再配分。
    `appear_t` 無し: 配列順を入れ替えるだけ(下流で stagger 計算される前提)。

    Args:
        order:    groups[] を持つ order dict
        get_yx:   group → (y0, x0) を返す callable
        band_h:   y バンド幅(px)。 同一バンド内は x 昇順。 デフォルト 80

    Returns:
        新しい order dict(in-place 変更なし)
    """
    from collections import OrderedDict

    groups = order.get("groups", [])
    if not groups:
        return order

    def _sort_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunk_sorted = sorted(
            chunk,
            key=lambda g: (get_yx(g)[0] // band_h, get_yx(g)[1]),
        )
        # appear_t 再配分: chunk 内の元 chronological 順を spatial 順に対応付け
        original_ts = sorted(g["appear_t"] for g in chunk if "appear_t" in g)
        if original_ts:
            ti = iter(original_ts)
            for g in chunk_sorted:
                if "appear_t" in g:
                    g["appear_t"] = next(ti)
        return chunk_sorted

    has_trigger = any("trigger_idx" in g for g in groups)
    if has_trigger:
        by_trig: "OrderedDict[int, list]" = OrderedDict()
        for g in groups:
            ti = g.get("trigger_idx", 0)
            by_trig.setdefault(ti, []).append(g)
        result: list[dict[str, Any]] = []
        for ti in sorted(by_trig.keys()):
            result.extend(_sort_chunk(by_trig[ti]))
    else:
        result = _sort_chunk(groups)

    new_order = dict(order)
    new_order["groups"] = result
    return new_order


def _yx_from_resolved_blobs(group: dict[str, Any]) -> tuple[int, int]:
    """resolve_blobs 後の group から代表 (y0, x0) を返す(全 blob の min)。

    空 group / y_band 由来は (huge, huge) を返して末尾配置。
    """
    rb = group.get("resolved_blobs", [])
    flat: list[dict[str, Any]] = []
    for sub in rb:
        if isinstance(sub, list):
            flat.extend(sub)
        else:
            flat.append(sub)
    if not flat:
        return (10**9, 10**9)
    y0 = min(int(b.get("y0", 0)) for b in flat)
    x0 = min(int(b.get("x0", 0)) for b in flat)
    return (y0, x0)


def resolve_blobs(
    order: dict[str, Any],
    all_blobs: dict[str, list[dict[str, Any]]],
    sort_z_order: bool = True,) -> dict[str, Any]:
    """order の各 group に対して blob_filter を解決し、resolved_blobs を付与する。

    Args:
        order:        orderer が出力した order JSON
        all_blobs:    blob_split が書き出した all_blobs.json の内容
        sort_z_order: True なら resolved_blobs 確定後に Z 字読み順で groups を再ソート
                      (デフォルト True 順序ずれ修正)

    Returns:
        各 group に "resolved_blobs" フィールドが追加された order dict
    """
    diag_id = order.get("diagram_id", "")
    blobs = all_blobs.get(diag_id, [])

    resolved_groups = []
    for group in order.get("groups", []):
        blob_filter = group.get("blob_filter", "")
        groups_of_blobs = _parse_blob_filter(blob_filter, blobs)
        resolved_group = dict(group)
        resolved_group["resolved_blobs"] = groups_of_blobs
        resolved_groups.append(resolved_group)

    result = dict(order)
    result["groups"] = resolved_groups
    if sort_z_order:
        result = sort_groups_z_order(result, _yx_from_resolved_blobs)
    return result


def write_order(order: dict[str, Any], orders_dir: Path) -> Path:
    """resolve_blobs 済みの order を orders_dir/<diagram_id>_order.json に書き出す。

    Args:
        order:      resolve_blobs 済みの order dict
        orders_dir: 出力先ディレクトリ(work/orders/ 等)

    Returns:
        書き出したファイルのパス
    """
    orders_dir = Path(orders_dir)
    orders_dir.mkdir(parents=True, exist_ok=True)
    diag_id = order.get("diagram_id", "unknown")
    out_path = orders_dir / f"{diag_id}_order.json"
    out_path.write_text(
        json.dumps(order, ensure_ascii=False, indent=2) + "\n"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="blob_filter を解決して reveal アニメ order JSON を生成する")
    ap.add_argument(
        "--order", required=True,
        help="orderer が出力した order JSON のパス",)
    ap.add_argument(
        "--all-blobs", required=True,
        help="blob_split が書き出した all_blobs.json のパス",)
    ap.add_argument(
        "--out-dir", required=True,
        help="出力先ディレクトリ(例: channels/<channel>/videos/<topic>/work/orders)",
    )
    args = ap.parse_args()

    order = load_order(Path(args.order))
    all_blobs: dict[str, Any] = json.loads(Path(args.all_blobs).read_text())
    resolved = resolve_blobs(order, all_blobs)
    out_path = write_order(resolved, Path(args.out_dir))
    diag_id = order.get("diagram_id", "?")
    n_groups = len(resolved.get("groups", []))
    print(f"OK {diag_id}: groups={n_groups} → {out_path}")


if __name__ == "__main__":
    _main()
