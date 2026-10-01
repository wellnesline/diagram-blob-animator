"""本編 reveal mp4 生成モジュール。

work/orders/ にある order JSON + work/blobs/ の labels.npz から、
各 diagram について chain 全体尺の reveal アニメ mp4 を
work/diagrams_anim/<diag_name>.mp4 として書き出す。

使い方(Python から):
    from diagram_blob.build_diagrams_anim import build_diagrams_anim_main
    build_diagrams_anim_main(topic_dir, scenes_data, timings_data)

CLI:
    python -m diagram_blob.build_diagrams_anim \\
        --topic-dir channels/<channel>/videos/<slug>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _render_job_worker(job: dict) -> tuple[str, str]:
    """ProcessPool worker。トップレベル関数なので pickle 可能。

    job: build_diagrams_anim_main 内で構築した dict。
    Returns: (diag_name, out_mp4 path)
    """
    from diagram_blob.blob_renderer import render_diagram_anim
    render_diagram_anim(
        base_img=job["base_img"],
        labels_npz=job["labels_npz"],
        order=job["order"],
        duration=job["duration"],
        out_mp4=job["out_mp4"],
        **job["kwargs"],)
    return job["diag_name"], str(job["out_mp4"])


def build_diagrams_anim_main(
    topic_dir: Path,
    scenes_data: dict,
    timings_data: list[dict],
    initial_delay: float = 0.5,) -> dict[str, Path]:
    """各 diagram の reveal mp4 を生成し、{diag_name: mp4_path} を返す。

    skip 条件:
      - order JSON 不在 / labels.npz 不在 / base_img 不在
      - source_dur に diag 名なし(本編未使用)
      - diag_name が `_` で始まる(_outro_thumb / _short_title)
    """
    from diagram_blob.blob_renderer import render_diagram_anim
    from diagram_blob.blob_animate import resolve_blobs

    # all_blobs.json をロード(blob_filter → resolved_blobs 変換に必要)
    all_blobs_p = topic_dir / "work" / "blobs" / "all_blobs.json"
    if not all_blobs_p.exists():
        print(f"  [build_diagrams_anim] all_blobs.json 不在、スキップ: {all_blobs_p}")
        return {}
    all_blobs_data_raw = json.loads(all_blobs_p.read_text())

    # blob_split v2 schema {name, bbox, size, center} を
    # v1 schema {id, x0, y0, x1, y1, area} に変換(resolve_blobs/blob_renderer 互換)
    def _v2_to_v1(blob: dict) -> dict:
        if "id" in blob and "x0" in blob:
            return blob  # 既に v1 schema
        name = blob.get("name", "")
        # "blob_NN" → "NN"。ただし order JSON は "manual:0,8" のように leading zero
        # 無しで参照するため、int 変換して再フォーマット(leading zero 削除)。
        # 過去の事故由来: "08" vs "8" mismatch で resolved_blobs 空になる事故。
        raw_bid = name.split("_", 1)[1] if "_" in name else name
        try:
            bid = str(int(raw_bid))  # "08" → "8"
        except ValueError:
            bid = raw_bid
        bbox = blob.get("bbox", [0, 0, 0, 0])
        return {
            "id": bid,
            "x0": int(bbox[0]),
            "y0": int(bbox[1]),
            "x1": int(bbox[2]),
            "y1": int(bbox[3]),
            "area": int(blob.get("size", 0)),
        }

    all_blobs_data = {
        diag: [_v2_to_v1(b) for b in blobs]
        for diag, blobs in all_blobs_data_raw.items()
    }

    topic_dir = Path(topic_dir)
    orders_dir = topic_dir / "work" / "orders"
    blobs_dir = topic_dir / "work" / "blobs"
    diagrams_dir = topic_dir / "diagrams"
    out_dir = topic_dir / "work" / "diagrams_anim"

    if not orders_dir.exists():
        print("  [build_diagrams_anim] orders/ 不在、スキップ")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    # source_dur: 各 image_ai source を使う line の duration 合計
    # diag_chains: 各 diagram の chain 内 line_timings を構築
    #   {diag_name: [{"li": 0, "start": 0.0, "end": 5.82}, ...]}
    source_dur: dict[str, float] = {}
    diag_chains: dict[str, list[dict]] = {}
    chain_starts: dict[str, float] = {}
    first_diag: str | None = None  # 動画冒頭(L01)の diagram(追加)
    for line in timings_data:
        sid = line.get("scene_source")
        if not sid:
            continue
        if first_diag is None:
            first_diag = sid
        dur = float(line.get("duration", 0.0))
        if sid not in chain_starts:
            chain_starts[sid] = source_dur.get(sid, 0.0)  # 0 (新 chain は 0 start)
            diag_chains[sid] = []
        rel_start = source_dur.get(sid, 0.0)
        diag_chains[sid].append({
            "li": len(diag_chains[sid]),
            "start": round(rel_start, 4),
            "end": round(rel_start + dur, 4),
        })
        source_dur[sid] = source_dur.get(sid, 0.0) + dur

    results: dict[str, Path] = {}
    # 並列化(ProcessPool)対応。Phase 1=cache判定+mode決定、
    # Phase 2=並列 render(各 diagram は独立、out_mp4 競合なし)。
    pending_jobs: list[dict] = []  # 並列実行する job 群

    for order_path in sorted(orders_dir.glob("*_order.json")):
        # Shorts 専用 order(`_shorts_order.json`)は本編 blob_anim 対象外
        if order_path.name.endswith("_shorts_order.json"):
            continue
        diag_name = order_path.name.replace("_order.json", "")
        if diag_name.startswith("_"):
            continue  # _outro_thumb / _short_title は Shorts 専用

        if diag_name not in source_dur:
            print(f"  [build_diagrams_anim] {diag_name}: source_dur に不在(本編未使用)、スキップ")
            continue

        labels_npz = blobs_dir / diag_name / "labels.npz"
        if not labels_npz.exists():
            print(f"  [build_diagrams_anim] {diag_name}: labels.npz 不在、スキップ")
            continue

        # fix: blob_split は _raw.png(1536×1024)を入力にし labels.npz もその解像度。
        # diagrams/<diag>.png は letterbox 適用版で labels と shape が合わないため、
        # render_diagram_anim には _raw.png を渡す。
        base_img = diagrams_dir / f"{diag_name}_raw.png"
        if not base_img.exists():
            print(f"  [build_diagrams_anim] {diag_name}: base_img 不在、スキップ")
            continue

        out_mp4 = out_dir / f"{diag_name}.mp4"
        chain_dur = source_dur[diag_name]

        # cache: order_path / base_img / build_diagrams_anim.py / blob_renderer.py mtime
        # と既存 mp4 mtime 比較。ロジック変更時は cache invalidation。
        # 過去の事故: trigger_sync 対応追加時に旧 cache (auto_stagger)が
        # 流用されて本編が正しく動かない事故。
        _renderer_mtimes = []
        try:
            from pathlib import Path as _P
            _src = _P(__file__).resolve().parent
            for _f in [_src / "build_diagrams_anim.py", _src / "blob_renderer.py"]:
                if _f.exists():
                    _renderer_mtimes.append(_f.stat().st_mtime)
        except Exception:
            pass
        _max_dep_mtime = max(
            order_path.stat().st_mtime,
            base_img.stat().st_mtime,
            *_renderer_mtimes,
        ) if _renderer_mtimes else max(
            order_path.stat().st_mtime,
            base_img.stat().st_mtime,
        )
        if (
            out_mp4.exists()
            and out_mp4.stat().st_mtime > _max_dep_mtime
        ):
            print(f"  [build_diagrams_anim] {diag_name}: cache HIT")
            results[diag_name] = out_mp4
            continue

        order = json.loads(order_path.read_text())
        # blob_filter → resolved_blobs 変換が抜けていた事故対策。
        # render_diagram_anim は resolved_blobs を読むため、resolve_blobs 必須。
        order = resolve_blobs(order, all_blobs_data)
        n_groups = len(order.get("groups", []))

        # order の groups に trigger_idx があれば「セリフ同期 mode」として
        # auto_stagger=False で render(本編専用、同等品質を出すため)。
        # trigger_idx が無ければ後方互換で auto_stagger=True で render。
        groups = order.get("groups", [])
        has_trigger = any("trigger_idx" in g for g in groups)
        line_timings = diag_chains.get(diag_name, [])

        # 恒久ルール: 本編1スライド目(冒頭フック)はセリフ同期せず、
        # Shorts 同様 2-3 秒で一気に reveal させる(target_finish_time=3.0)。
        # 理由: 冒頭スライドは「セリフが導入問いかけ」で diagram 構造とズレやすく、
        # 順次出すと「セリフより先に何が出る/出ない」のチカチカが発生する。
        # 早期に絵を完成させて、以降セリフを聞かせる方が没入できる。
        is_first_slide = (diag_name == first_diag)

        # job 化(後で並列 render する)
        if is_first_slide:
            mode_label = f"FIRST-SLIDE burst (target=3.0s, n={n_groups})"
            kwargs = dict(
                auto_stagger=True,
                stagger_gap=0.4,
                initial_delay=0.1,
                target_finish_time=3.0,)
        elif has_trigger and line_timings:
            mode_label = f"trigger_sync (lines={len(line_timings)})"
            kwargs = dict(auto_stagger=False, line_timings=line_timings)
        else:
            usable = max(0.5, chain_dur * 0.7 - initial_delay)
            stagger_gap = usable / max(1, n_groups - 1) if n_groups > 1 else 0.0
            mode_label = f"auto_stagger={stagger_gap:.2f}s"
            kwargs = dict(
                auto_stagger=True,
                stagger_gap=stagger_gap,
                initial_delay=initial_delay,)

        pending_jobs.append({
            "diag_name": diag_name,
            "mode_label": mode_label,
            "base_img": base_img,
            "labels_npz": labels_npz,
            "order": order,
            "duration": chain_dur,
            "out_mp4": out_mp4,
            "n_groups": n_groups,
            "kwargs": kwargs,
        })

    # Phase 2: 並列 render
    # workers 上限 6(diag_07 等大型は 1個で 1.5GB メモリ食う)。
    # CPU bound + ffmpeg subprocess なので Process が無難。
    if pending_jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os as _os
        n_workers = max(1, min(6, (_os.cpu_count() or 4) // 2))
        print(
            f"  [build_diagrams_anim] parallel render workers={n_workers}, "
            f"jobs={len(pending_jobs)}",
            flush=True,
        )
        for job in pending_jobs:
            print(
                f"  [build_diagrams_anim] {job['diag_name']} dur={job['duration']:.2f}s "
                f"groups={job['n_groups']} {job['mode_label']} ...(queued)",
                flush=True,
            )

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            # _render_job_worker はトップレベル関数(pickle 可能)
            futures = {ex.submit(_render_job_worker, j): j["diag_name"] for j in pending_jobs}
            for fut in as_completed(futures):
                diag_name = futures[fut]
                try:
                    name, mp4 = fut.result()
                    results[name] = Path(mp4)
                    print(f"  [build_diagrams_anim] DONE {name}", flush=True)
                except Exception as e:
                    print(f"  [build_diagrams_anim] FAILED {diag_name}: {e}", flush=True)
                    raise

    return results


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-dir", type=Path, required=True)
    args = ap.parse_args()

    scenes = json.loads((args.topic_dir / "scenes.json").read_text())
    timings = json.loads((args.topic_dir / "work" / "timings.json").read_text())
    results = build_diagrams_anim_main(args.topic_dir, scenes, timings)
    print(f"\n[build_diagrams_anim] 生成完了: {len(results)} mp4")
    for diag, path in results.items():
        print(f"  {diag} → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
