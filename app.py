from __future__ import annotations

import json
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis_engine as engine
import risk_analysis
import storage
from pdf_report import build_pdf_report

APP_VERSION = "10.1"
APP_URL = "https://batting-form-analyzer.streamlit.app/"

st.set_page_config(
    page_title=f"打撃フォーム分析 ver.{APP_VERSION}",
    page_icon="⚾",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 1.5rem; padding-bottom: 4rem;}
    div[data-testid="stMetric"] {border: 1px solid #e7e9ee; padding: 12px; border-radius: 12px;}
    .small-note {font-size: .88rem; color: #667085;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"打撃フォーム分析 ver.{APP_VERSION}")
st.caption("比較選手と自分の打撃動画を比較し、動作の違い・身体負担に関連する動作・改善ポイントを表示します。")
st.info(
    "ver.10.1では100点評価を主表示にせず、比較選手との差を「方向・実測値・4段階の近さ」で示します。フォームの優劣や医学的な診断ではありません。"
)
st.link_button("🔗 打撃フォーム分析 ver.10.1 を開く・共有する", APP_URL, use_container_width=True)
st.caption(f"共有URL: {APP_URL}")

# ---------------------------- session ----------------------------
for key, default in {
    "analysis_bundle": None,
    "session_root": None,
    "selected_player_id": None,
    "cloud_players": [],
    "ref_roi_auto": None,
    "tar_roi_auto": None,
    "last_saved_analysis_key": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.session_root is None:
    st.session_state.session_root = tempfile.mkdtemp(prefix="bfa_v10_")

CLOUD = storage.is_cloud_runtime()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _cloud_list_players():
    players = deepcopy(st.session_state.cloud_players)
    players.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return players


def _list_players():
    return _cloud_list_players() if CLOUD else storage.list_players()


def _get_player(player_id):
    if not player_id:
        return None
    if not CLOUD:
        return storage.get_player(player_id)
    for p in st.session_state.cloud_players:
        if p.get("id") == player_id:
            return deepcopy(p)
    return None


def _create_player(name, batter_side, team="", note=""):
    if not CLOUD:
        return storage.create_player(name, batter_side, team, note)
    player = {
        "id": str(uuid.uuid4()), "name": name.strip(), "batter_side": batter_side,
        "team": team.strip(), "note": note.strip(), "created_at": _now(), "updated_at": _now(), "analyses": [],
    }
    st.session_state.cloud_players.append(player)
    return deepcopy(player)


def _add_analysis(player_id, record):
    if not CLOUD:
        return storage.add_analysis(player_id, record)
    record = deepcopy(record)
    record.setdefault("id", str(uuid.uuid4()))
    record.setdefault("created_at", _now())
    record.setdefault("app_version", APP_VERSION)
    for p in st.session_state.cloud_players:
        if p.get("id") == player_id:
            p.setdefault("analyses", []).append(record)
            p["updated_at"] = _now()
            return deepcopy(record)
    raise KeyError("選手ファイルが見つかりません。")


def _export_player(player_id):
    if not CLOUD:
        return storage.export_player_bundle(player_id)
    p = _get_player(player_id)
    payload = {
        "format": "BFA_PLAYER_BUNDLE", "schema_version": 1, "app_version": APP_VERSION,
        "exported_at": _now(), "player": p,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _import_player(raw):
    if not CLOUD:
        return storage.import_player_bundle(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != "BFA_PLAYER_BUNDLE":
        raise ValueError("対応する選手ファイルではありません。")
    p = deepcopy(payload.get("player", {}))
    if not p.get("name"):
        raise ValueError("選手データが壊れています。")
    ids = {x.get("id") for x in st.session_state.cloud_players}
    if p.get("id") in ids:
        p["id"] = str(uuid.uuid4())
        p["name"] = f"{p['name']}（読込）"
    p.setdefault("analyses", [])
    p["updated_at"] = _now()
    st.session_state.cloud_players.append(p)
    return deepcopy(p)


def _history_df(player):
    if not CLOUD:
        return storage.analysis_history_dataframe(player)
    rows = []
    for r in player.get("analyses", []):
        row = {
            "解析日時": r.get("created_at", ""),
            "比較対象": r.get("reference_name", "比較選手"),
            "比較結果": r.get("overall_comparison_label", ""),
            "平均差尺度": r.get("mean_normalized_difference"),
            "旧フォーム類似度": r.get("overall_score"),
            "解析信頼度": r.get("target_quality"),
            "身体負担判定": r.get("body_load_label", ""),
        }
        row.update(r.get("scores", {}) or {})
        rows.append(row)
    return pd.DataFrame(rows)


def _save_upload(uploaded, prefix):
    root = Path(st.session_state.session_root)
    suffix = Path(uploaded.name).suffix or ".mp4"
    path = root / f"{prefix}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return path


def _player_name_map(players):
    return {p["id"]: f"{p['name']}（{len(p.get('analyses', []))}回）" for p in players}


# ---------------------------- navigation ----------------------------
page = st.radio("画面", ["解析", "選手ファイル", "参考研究"], horizontal=True, label_visibility="collapsed")


# ============================ player files ============================
if page == "選手ファイル":
    st.header("選手ファイル")
    if CLOUD:
        st.warning(
            "公開Cloud版ではサーバー内保存を永続履歴として使いません。"
            "この画面を開いている間は複数選手を管理できますが、長期保存は各選手の .bfa ファイルを端末へ保存してください。"
        )
    else:
        st.success("ローカル版では、このPCに選手別の解析履歴を自動保存します。")

    with st.expander("＋ 新しい選手を作る", expanded=not bool(_list_players())):
        new_name = st.text_input("選手名・ニックネーム", key="new_player_name")
        new_side = st.radio("打席", ["右打者", "左打者"], horizontal=True, key="new_player_side")
        new_team = st.text_input("チーム・所属（任意）", key="new_player_team")
        new_note = st.text_area("メモ（任意）", key="new_player_note")
        if st.button("選手ファイルを作成", type="primary", use_container_width=True):
            try:
                p = _create_player(new_name, new_side, new_team, new_note)
                st.session_state.selected_player_id = p["id"]
                st.success(f"{p['name']} の選手ファイルを作成しました。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with st.expander("📥 .bfa選手ファイルを読み込む"):
        imported_file = st.file_uploader("選手ファイル", type=["bfa", "json"], key="player_import")
        if imported_file is not None and st.button("読み込む", use_container_width=True):
            try:
                p = _import_player(imported_file.getvalue())
                st.session_state.selected_player_id = p["id"]
                st.success(f"{p['name']} を読み込みました。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    players = _list_players()
    if not players:
        st.info("まだ選手ファイルがありません。上から作成してください。")
    else:
        name_map = _player_name_map(players)
        ids = [p["id"] for p in players]
        current = st.session_state.selected_player_id if st.session_state.selected_player_id in ids else ids[0]
        selected_id = st.selectbox("選手を選択", ids, index=ids.index(current), format_func=lambda x: name_map[x])
        st.session_state.selected_player_id = selected_id
        player = _get_player(selected_id)

        st.subheader(player["name"])
        st.write(f"**打席:** {player.get('batter_side', '-')}　 **解析回数:** {len(player.get('analyses', []))}回")
        if player.get("team"):
            st.write(f"**所属:** {player['team']}")
        if player.get("note"):
            st.caption(player["note"])

        st.download_button(
            "📤 この選手ファイルを端末へ保存（.bfa）",
            data=_export_player(selected_id),
            file_name=f"{player['name']}_batting_history.bfa",
            mime="application/json",
            use_container_width=True,
        )

        hist = _history_df(player)
        if hist.empty:
            st.info("まだ解析履歴がありません。『解析』画面でこの選手を選んで解析してください。")
        else:
            st.markdown("#### 解析履歴")
            st.dataframe(hist, use_container_width=True, hide_index=True)
            if "平均差尺度" in hist.columns:
                chart = hist[["解析日時", "平均差尺度"]].copy()
                chart["解析日時"] = pd.to_datetime(chart["解析日時"], errors="coerce")
                chart = chart.dropna(subset=["解析日時", "平均差尺度"]).set_index("解析日時")
                if not chart.empty:
                    st.markdown("#### 比較対象との差の推移")
                    st.caption("0に近いほど、その回の比較対象に近い動きです。比較対象が変わる場合は単純な成長曲線としては解釈しないでください。")
                    st.line_chart(chart)

            score_columns = [name for name in engine.FEATURES if name in hist.columns]
            if score_columns:
                selected_metric = st.selectbox("項目別の推移を見る", score_columns)
                item_chart = hist[["解析日時", selected_metric]].copy()
                item_chart["解析日時"] = pd.to_datetime(item_chart["解析日時"], errors="coerce")
                item_chart = item_chart.dropna(subset=["解析日時"]).set_index("解析日時")
                if not item_chart.empty:
                    st.line_chart(item_chart)


# ============================ sources ============================
elif page == "参考研究":
    st.header("参考研究")
    st.caption("改善アドバイスでは、動画で実際に測定できる運動学的proxyと、先行研究の論点を区別して使用します。")
    for source in risk_analysis.SOURCES.values():
        st.markdown(f"**{source['title']}**")
        st.caption(source["use"])
        st.link_button("論文・資料を開く", source["url"])
        st.divider()


# ============================ analysis ============================
else:
    st.header("1. 解析する選手")
    players = _list_players()
    name_map = _player_name_map(players)
    player_options = [None] + [p["id"] for p in players]
    current = st.session_state.selected_player_id if st.session_state.selected_player_id in player_options else None
    selected_player_id = st.selectbox(
        "選手ファイル（任意）",
        player_options,
        index=player_options.index(current),
        format_func=lambda x: "今回だけ解析（履歴に保存しない）" if x is None else name_map[x],
    )
    st.session_state.selected_player_id = selected_player_id
    selected_player = _get_player(selected_player_id) if selected_player_id else None

    with st.expander("解析条件", expanded=True):
        default_side = selected_player.get("batter_side", "右打者") if selected_player else "右打者"
        batter_side = st.radio("打席", ["右打者", "左打者"], index=0 if default_side == "右打者" else 1, horizontal=True)
        feature_names = list(engine.FEATURES.keys())
        selected_features = st.multiselect(
            "比較する5項目（8項目から選択）",
            feature_names,
            default=feature_names[:5],
            max_selections=5,
            format_func=lambda x: engine.FEATURES[x].get("display_name", x),
        )
        st.caption(f"選択中: {len(selected_features)} / 5")
        reference_name = st.text_input("比較選手名・比較対象（任意）", value="比較選手")

    st.header("2. 動画を選択")
    reference_file = st.file_uploader("比較選手の動画", type=["mp4", "mov", "avi", "mkv", "m4v"], key="reference_video")
    target_file = st.file_uploader("自分の動画", type=["mp4", "mov", "avi", "mkv", "m4v"], key="target_video")
    st.caption("打撃前後だけの2〜5秒程度の動画にすると、解析速度と人物追跡の安定性が上がります。")

    reference_roi = {"center": 0.5, "width": 0.60}
    target_roi = {"center": 0.5, "width": 0.60}

    if reference_file is not None and target_file is not None:
        st.header("3. 打者の解析範囲")
        st.info(
            "ver.10.1ではスマホで細いスライダーを合わせる必要を減らしました。"
            "まず『自動で打者位置を探す』を使い、合わない時だけ左・中央・右のプリセットか詳細調整を使ってください。"
        )

        ref_upload_path = _save_upload(reference_file, "preview_reference")
        tar_upload_path = _save_upload(target_file, "preview_target")

        if st.button("✨ 2本とも自動で打者位置を探す", type="primary", use_container_width=True):
            with st.spinner("打者候補を探しています…"):
                st.session_state.ref_roi_auto = engine.suggest_batter_roi(ref_upload_path)
                st.session_state.tar_roi_auto = engine.suggest_batter_roi(tar_upload_path)
            st.success("自動推定しました。下の赤枠を確認してください。")

        preview_percent = st.selectbox("確認する動画位置", [20, 35, 50, 65], index=1, format_func=lambda x: f"動画の{x}%付近")
        preview_ratio = preview_percent / 100.0

        def roi_editor(label, path, prefix, auto_value):
            st.markdown(f"#### {label}")
            default_center = 0.68 if batter_side == "右打者" else 0.32
            default = auto_value or {"center": default_center, "width": 0.55, "confidence": 0.0}
            mode = st.radio(
                f"{label}の打者位置",
                ["自動推定", "左寄り", "中央", "右寄り", "画面全体"],
                horizontal=True,
                key=f"{prefix}_roi_mode",
            )
            presets = {
                "自動推定": (default["center"], default["width"]),
                "左寄り": (0.28, 0.48),
                "中央": (0.50, 0.50),
                "右寄り": (0.72, 0.48),
                "画面全体": (0.50, 1.00),
            }
            center, width = presets[mode]
            with st.expander("細かく調整（必要な場合だけ）"):
                use_fine = st.checkbox(
                    "詳細調整を有効にする", value=False, key=f"{prefix}_use_fine"
                )
                if use_fine:
                    center_pct = st.number_input(
                        f"{label}: 横位置（中央）%", min_value=10, max_value=90,
                        value=int(round(center * 100)), step=2, key=f"{prefix}_center_num",
                    )
                    width_pct = st.number_input(
                        f"{label}: 範囲の幅%", min_value=25, max_value=100,
                        value=int(round(width * 100)), step=3, key=f"{prefix}_width_num",
                    )
                    center, width = center_pct / 100.0, width_pct / 100.0
                else:
                    st.caption("通常は上の自動推定・左寄り・中央・右寄りだけで十分です。")
            preview = engine.get_video_preview_frame(path, preview_ratio)
            overlay = engine.draw_roi_preview(preview, center, width)
            if overlay is not None:
                st.image(overlay, caption="赤枠内だけを高精度骨格解析します。", use_container_width=True)
            if mode == "自動推定" and auto_value and auto_value.get("confidence", 0) == 0:
                st.warning("自動推定に自信がないため初期値を表示しています。赤枠を目視確認してください。")
            return {"center": center, "width": width}

        reference_roi = roi_editor("比較選手", ref_upload_path, "ref", st.session_state.ref_roi_auto)
        st.divider()
        target_roi = roi_editor("自分", tar_upload_path, "tar", st.session_state.tar_roi_auto)

    st.header("4. 解析")
    can_run = reference_file is not None and target_file is not None and len(selected_features) == 5
    if len(selected_features) != 5:
        st.warning("比較する評価項目を5つ選択してください。")

    analyze_clicked = st.button("高精度解析を開始", type="primary", disabled=not can_run, use_container_width=True)

    if analyze_clicked:
        run_root = Path(tempfile.mkdtemp(prefix="bfa10_run_"))
        upload_dir = run_root / "uploads"
        output_dir = run_root / "results"
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_path = upload_dir / f"reference{Path(reference_file.name).suffix or '.mp4'}"
        tar_path = upload_dir / f"target{Path(target_file.name).suffix or '.mp4'}"
        ref_path.write_bytes(reference_file.getbuffer())
        tar_path.write_bytes(target_file.getbuffer())

        progress = st.progress(5, text="動画を準備しています…")
        try:
            progress.progress(15, text="打者を追跡し、骨格を高精度解析しています…")
            with st.spinner("MediaPipe Heavyで解析中です…"):
                result = engine.run_analysis(
                    reference_original_path=ref_path,
                    target_original_path=tar_path,
                    batter_side=batter_side,
                    selected_feature_names=selected_features,
                    output_dir=output_dir,
                    reference_batter_roi=reference_roi,
                    target_batter_roi=target_roi,
                )
            progress.progress(78, text="身体負担関連動作と先行研究アドバイスを計算しています…")
            result["body_load"] = risk_analysis.assess_body_load(result["reference_analysis"], result["target_analysis"], batter_side)
            result["research_advice"] = risk_analysis.research_based_advice(result["reference_analysis"], result["target_analysis"])
            progress.progress(90, text="PDFを作成しています…")
            pdf_bytes = build_pdf_report(
                result, output_dir, batter_side, selected_features,
                player_name=selected_player.get("name", "") if selected_player else "",
            )
            (output_dir / "打撃フォーム分析_ver10.pdf").write_bytes(pdf_bytes)
            st.session_state.analysis_bundle = {
                "result": result, "output_dir": output_dir, "pdf_bytes": pdf_bytes,
                "batter_side": batter_side, "selected_features": list(selected_features),
                "reference_name": reference_name, "player_id": selected_player_id,
                "analysis_key": str(uuid.uuid4()),
            }
            progress.progress(100, text="解析完了")
            st.success("解析が完了しました。インパクト候補を確認してください。")
        except Exception as exc:
            progress.empty()
            st.error("解析中にエラーが発生しました。")
            st.exception(exc)

    bundle = st.session_state.analysis_bundle
    if bundle is not None:
        result = bundle["result"]
        output_dir = Path(bundle["output_dir"])
        ref = result["reference_analysis"]
        tar = result["target_analysis"]

        st.divider()
        st.header("5. インパクト位置を確認")
        st.warning(
            "バットとボールを直接検出していないため、インパクトは身体動作からの推定です。"
            "ver.10.1では手首速度だけでなく肩・骨盤回転も使い、候補を最大3つ表示します。"
        )

        def impact_selector(label, analysis, prefix):
            candidates = list(dict.fromkeys([analysis.impact_frame] + list(analysis.impact_candidates)))[:3]
            options = candidates + ["手動"]
            choice = st.radio(
                f"{label}のインパクト候補",
                options,
                format_func=lambda x: "手動でフレーム番号を指定" if x == "手動" else f"{x / analysis.fps:.3f}秒（frame {x}）",
                key=f"{prefix}_impact_choice",
            )
            if choice == "手動":
                frame = int(st.number_input(
                    f"{label}: フレーム番号", min_value=0, max_value=max(0, analysis.frame_count-1),
                    value=int(analysis.impact_frame), step=1, key=f"{prefix}_manual_frame",
                ))
            else:
                frame = int(choice)
            preview = engine.get_pose_preview_frame(analysis, frame)
            if preview is not None:
                st.image(preview, caption=f"{label}: {frame / analysis.fps:.3f}秒", use_container_width=True)
            measured = engine.is_measured_frame(analysis, frame)
            if not measured:
                st.warning("このフレームは実測骨格ではありません。近くの候補を選んでください。")
            return frame, measured

        ref_frame, ref_ok = impact_selector("比較選手", ref, "ref")
        st.divider()
        tar_frame, tar_ok = impact_selector("自分", tar, "tar")

        if st.button("この2フレームをインパクトとして再計算", type="primary", use_container_width=True, disabled=not (ref_ok and tar_ok)):
            try:
                with st.spinner("インパクト位置を反映しています…"):
                    result = engine.recalculate_with_manual_impacts(
                        result, ref_frame, tar_frame, bundle["selected_features"], bundle["batter_side"]
                    )
                    result["body_load"] = risk_analysis.assess_body_load(result["reference_analysis"], result["target_analysis"], bundle["batter_side"])
                    result["research_advice"] = risk_analysis.research_based_advice(result["reference_analysis"], result["target_analysis"])
                    pdf_bytes = build_pdf_report(
                        result, output_dir, bundle["batter_side"], bundle["selected_features"],
                        player_name=_get_player(bundle["player_id"]).get("name", "") if bundle["player_id"] and _get_player(bundle["player_id"]) else "",
                    )
                    bundle["result"] = result
                    bundle["pdf_bytes"] = pdf_bytes
                    st.session_state.analysis_bundle = bundle
                st.success("得点・アドバイス・PDFを更新しました。")
            except Exception as exc:
                st.error("再計算中にエラーが発生しました。")
                st.exception(exc)

        result = st.session_state.analysis_bundle["result"]
        bundle = st.session_state.analysis_bundle
        ref = result["reference_analysis"]
        tar = result["target_analysis"]

        st.header("6. 解析結果")
        overall_label = result.get("overall_comparison_label", "評価保留")
        st.markdown(f"### 比較選手との動作傾向：**{overall_label}**")
        st.caption(
            "これはフォームの良し悪しではなく、今回選んだ5項目が比較選手とどの程度近いかをまとめた表示です。"
            "100点換算は内部互換用にのみ残し、主評価には使用しません。"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**最も近い動き**")
            st.subheader(result["best_item"])
        with c2:
            st.markdown("**最も違う動き**")
            st.subheader(result["lowest_item"])

        def quality_display(analysis):
            score = float(analysis.quality_score)
            if score >= 80:
                return "高", "この動画は安定して骨格解析できています。"
            if score >= 60:
                return "要確認", "一部のフレームで骨格検出が不安定です。結果は参考値として確認してください。"
            return "低", "骨格検出が不安定です。撮影条件や解析範囲を見直してください。"

        ref_quality_label, ref_quality_message = quality_display(ref)
        tar_quality_label, tar_quality_message = quality_display(tar)

        q1, q2 = st.columns(2)
        with q1:
            with st.container(border=True):
                st.markdown(f"**骨格検出品質：{ref_quality_label}**")
                st.caption(f"比較選手：{ref_quality_message}")
        with q2:
            with st.container(border=True):
                st.markdown(f"**骨格検出品質：{tar_quality_label}**")
                st.caption(f"自分：{tar_quality_message}")

        if min(float(ref.quality_score), float(tar.quality_score)) < 80:
            st.warning(
                "骨格検出品質が『要確認』の動画が含まれています。フォーム差だけでなく、"
                "画質・遮蔽・撮影角度・解析範囲の影響を受けている可能性があります。"
            )

        with st.expander("解析品質の詳細を見る", expanded=False):
            ref_q = ref.quality_details or {}
            tar_q = tar.quality_details or {}
            quality_detail = pd.DataFrame([
                {
                    "動画": "比較選手",
                    "内部品質スコア": round(float(ref.quality_score), 1),
                    "全体骨格検出率": f"{ref_q.get('overall_detection_rate', 0):.1%}",
                    "インパクト周辺検出率": f"{ref_q.get('impact_window_detection_rate', 0):.1%}",
                    "主要関節visibility": f"{ref_q.get('key_joint_visibility', 0):.2f}",
                },
                {
                    "動画": "自分",
                    "内部品質スコア": round(float(tar.quality_score), 1),
                    "全体骨格検出率": f"{tar_q.get('overall_detection_rate', 0):.1%}",
                    "インパクト周辺検出率": f"{tar_q.get('impact_window_detection_rate', 0):.1%}",
                    "主要関節visibility": f"{tar_q.get('key_joint_visibility', 0):.2f}",
                },
            ])
            st.dataframe(quality_detail, use_container_width=True, hide_index=True)
            st.caption(
                "内部品質スコアはフォームの良し悪しではありません。骨格推定がどの程度安定していたかを確認するための品質管理値です。"
            )

        st.markdown("#### 5項目の比較")
        for _, row in result["score_dataframe"].iterrows():
            level = row["比較判定"]
            with st.container(border=True):
                st.markdown(f"### {row['表示名']}　—　{level}")
                st.write(f"**比較選手** {row['比較選手表示']}　→　**自分** {row['自分表示']}")
                st.write(f"**差の方向:** {row['差の方向']}")
                st.caption(row["計測内容"])

        diff_chart = output_dir / "graphs" / "比較選手との差の方向.png"
        if diff_chart.exists():
            st.markdown("#### 差の方向をまとめて見る")
            st.image(str(diff_chart), use_container_width=True)
            st.caption("中央の0が比較選手です。左右は『良い／悪い』ではなく、比較選手より小さい／大きい方向を示します。")

        with st.expander("ver.10.1 追加指標を見る"):
            extra_keys = {
                "スイング開始→インパクト(s)": "swing_time_sec",
                "身体中心移動proxy": "body_center_shift_proxy",
                "身体中心速度proxy": "body_center_speed_proxy",
                "体幹側屈変化(deg)": "torso_side_bend_change_deg",
                "骨盤傾斜変化(deg)": "pelvis_tilt_change_deg",
                "骨盤→肩ピーク時間差(ms)": "hip_to_shoulder_peak_delay_ms",
                "左手首アライメントproxy(deg)": "left_wrist_alignment_proxy_deg",
                "右手首アライメントproxy(deg)": "right_wrist_alignment_proxy_deg",
            }
            extra_table = []
            for label, key in extra_keys.items():
                extra_table.append({
                    "指標": label,
                    "比較選手": round(float(ref.extra_metrics.get(key, float("nan"))), 3),
                    "自分": round(float(tar.extra_metrics.get(key, float("nan"))), 3),
                })
            st.dataframe(pd.DataFrame(extra_table), use_container_width=True, hide_index=True)
            st.caption("身体中心は画像上のproxyです。力学的な真の身体重心や地面反力を測定したものではありません。")

        st.header("7. 身体負担に関連する動作の確認")
        body = result["body_load"]
        st.markdown(f"### 今回の確認結果：**{body['overall_label']}**")
        st.caption("怪我の予測ではありません。先行研究で議論される動作と、今回の比較差を照合した参考表示です。")
        for _, row in body["regions"].iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['部位']}：{row['判定']}**")
                st.write(f"観測した動き: {row['主な計測proxy']}")
                st.caption(row['根拠と注意'])
        st.caption(body["limitations"])

        st.header("8. 今回優先して確認したいポイント")
        advice_df = result["advice_dataframe"]
        for i, row in advice_df.iterrows():
            with st.container(border=True):
                st.markdown(f"### {i+1}. {row['観点']}　—　{row['比較判定']}")
                st.markdown("**比較結果**")
                st.write(row["比較結果"])
                st.markdown("**動作の特徴**")
                st.write(row["動作の特徴"])
                st.markdown("**改善の方向性**")
                st.write(row["改善の方向性"])
                st.markdown("**練習例**")
                st.write(row["練習例"])
                st.caption(f"研究上の位置づけ: {row['研究上の位置づけ']}")

        research_df = result["research_advice"]
        if research_df is not None and not research_df.empty:
            with st.expander("先行研究から見る追加の観点", expanded=False):
                for _, row in research_df.iterrows():
                    st.markdown(f"**{row['観点']}**")
                    st.write(row["検出内容"])
                    st.write(f"改善の考え方: {row['改善アドバイス']}")
                    st.write(f"練習例: {row['練習例']}")
                    st.caption(f"参考研究: {row['参考研究']}")
                    st.link_button("参考研究を開く", row["リンク"])
                    st.divider()

        st.header("9. 保存")
        st.download_button(
            "📄 PDFレポートを保存",
            data=bundle["pdf_bytes"],
            file_name="打撃フォーム分析_ver10_1.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        if bundle.get("player_id"):
            player = _get_player(bundle["player_id"])
            if player:
                already_saved = st.session_state.last_saved_analysis_key == bundle["analysis_key"]
                if st.button(
                    "📁 この解析を選手ファイルに保存" if not already_saved else "✅ この解析は選手ファイルに保存済み",
                    type="primary", use_container_width=True, disabled=already_saved,
                ):
                    scores = dict(zip(result["score_dataframe"]["表示名"], result["score_dataframe"]["差の尺度"].abs()))
                    comparison_levels = dict(zip(result["score_dataframe"]["表示名"], result["score_dataframe"]["比較判定"]))
                    record = {
                        "created_at": _now(),
                        "reference_name": bundle.get("reference_name", "比較選手"),
                        "overall_score": round(float(result["overall_score"]), 1),  # 旧互換
                        "overall_comparison_label": result.get("overall_comparison_label", ""),
                        "mean_normalized_difference": round(float(result.get("mean_normalized_difference", float("nan"))), 3),
                        "target_quality": round(float(tar.quality_score), 1),
                        "body_load_label": body["overall_label"],
                        "scores": {k: float(v) for k, v in scores.items()},
                        "comparison_levels": comparison_levels,
                        "target_features": {k: float(v) for k, v in tar.features.items()},
                        "reference_features": {k: float(v) for k, v in ref.features.items()},
                        "target_extra_metrics": {k: float(v) for k, v in tar.extra_metrics.items()},
                        "reference_extra_metrics": {k: float(v) for k, v in ref.extra_metrics.items()},
                    }
                    _add_analysis(bundle["player_id"], record)
                    st.session_state.last_saved_analysis_key = bundle["analysis_key"]
                    st.success(f"{player['name']} の履歴に保存しました。")
                    if CLOUD:
                        st.info("公開Cloud版では、下の .bfa も端末に保存しておくと再起動後に復元できます。")
                    st.rerun()

                # Cloud版は保存後だけでなく常に最新選手ファイルを書き出せる。
                refreshed = _get_player(bundle["player_id"])
                if refreshed:
                    st.download_button(
                        "📤 最新の選手履歴を .bfa で保存",
                        data=_export_player(bundle["player_id"]),
                        file_name=f"{refreshed['name']}_batting_history.bfa",
                        mime="application/json",
                        use_container_width=True,
                    )
        else:
            st.caption("解析履歴を残したい場合は、次回から上部の『選手ファイル』を選んで解析してください。")
