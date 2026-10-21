from pathlib import Path
import tempfile
import shutil
import urllib.parse
import os

import cv2
import numpy as np
import streamlit as st

import analysis_engine as engine
from pdf_report import build_pdf_report
import risk_analysis


st.set_page_config(
    page_title="打撃フォーム分析",
    page_icon="⚾",
    layout="wide",
)

st.title("打撃フォーム分析")
st.caption("比較選手と自分の打撃動画を比較し、フォーム類似度と改善ポイントを表示します。")
st.info(
    "表示される点数は『比較選手とのフォーム類似度』です。"
    "フォームの優劣や医学的な傷害発生確率を示すものではありません。"
)

if "analysis_bundle" not in st.session_state:
    st.session_state.analysis_bundle = None


@st.cache_data(show_spinner=False)
def _video_preview(video_bytes, suffix, frame_ratio=0.35):
    """アップロード動画から指定割合付近のRGBプレビューを1枚取り出す。"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4")
    tmp_path = tmp.name
    try:
        tmp.write(video_bytes)
        tmp.close()
        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 1
        frame_index = int(np.clip(round((frame_count - 1) * frame_ratio), 0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _draw_roi_preview(image_rgb, center_percent, width_percent):
    """打者解析範囲を赤枠で可視化する。"""
    if image_rgb is None:
        return None
    image = image_rgb.copy()
    h, w = image.shape[:2]
    center = float(center_percent) / 100.0
    width_ratio = float(width_percent) / 100.0
    x0 = max(0.0, center - width_ratio / 2.0)
    x1 = min(1.0, center + width_ratio / 2.0)
    if x1 - x0 < width_ratio:
        if x0 <= 0:
            x1 = min(1.0, width_ratio)
        elif x1 >= 1:
            x0 = max(0.0, 1.0 - width_ratio)
    px0 = int(round(x0 * w))
    px1 = int(round(x1 * w))

    # 枠外を少し暗くし、解析対象が一目で分かるようにする。
    shaded = image.copy()
    if px0 > 0:
        shaded[:, :px0] = (shaded[:, :px0] * 0.35).astype(np.uint8)
    if px1 < w:
        shaded[:, px1:] = (shaded[:, px1:] * 0.35).astype(np.uint8)
    cv2.rectangle(shaded, (px0, 2), (max(px0 + 1, px1 - 1), h - 3), (255, 55, 55), 5)
    cv2.putText(
        shaded,
        "BATTER AREA",
        (max(10, px0 + 12), 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shaded


with st.sidebar:
    st.header("解析条件")

    batter_side = st.radio(
        "打席",
        ["右打者", "左打者"],
        horizontal=True,
    )

    st.markdown("#### 比較する評価項目（8項目から5項目）")
    st.caption("8項目をすべて表示しています。チェックを5つ選んでください。")

    feature_names = list(engine.FEATURES.keys())
    selected_features = []
    for i, feature_name in enumerate(feature_names):
        checked = st.checkbox(
            feature_name,
            value=(i < 5),
            key=f"feature_{i}",
        )
        if checked:
            selected_features.append(feature_name)

    st.write(f"選択中: **{len(selected_features)} / 5**")

    st.markdown("#### 解析方式")
    st.info("高精度モード（MediaPipe Pose Landmarker Heavy）で解析します。")
    st.caption("精度を優先するため、解析モードは高精度に固定しています。")

    st.markdown("#### 打者の人物指定")
    st.caption(
        "打者だけが入る解析範囲を指定します。"
        "キャッチャーや審判を赤枠の外へ出すことで、人物の取り違えを防ぎます。"
    )



st.subheader("1. 動画を選択")
col1, col2 = st.columns(2)
with col1:
    reference_file = st.file_uploader(
        "比較選手の動画",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        key="reference",
    )
with col2:
    target_file = st.file_uploader(
        "自分の動画",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        key="target",
    )

st.caption(
    "解析時間はファイル容量よりも『動画の長さ・解像度・フレーム数』の影響を強く受けます。"
    "打撃前後だけの短い動画（目安2〜5秒）にすると、さらに短縮できます。"
)
st.info(
    "動画全体から人物を選ぶのではなく、最初に『打者だけの解析範囲』を決めます。"
    "赤枠の外は骨格検出の対象から外れるため、捕手・審判の誤検出を大幅に減らせます。"
)

# --------------------------------------------------------
# 打者ROIの指定
# --------------------------------------------------------
reference_roi = {"center": 0.5, "width": 1.0}
target_roi = {"center": 0.5, "width": 1.0}

if reference_file is not None and target_file is not None:
    st.subheader("2. 打者だけを赤枠で囲む")
    st.caption(
        "打者のスイング全体が赤枠内に入り、キャッチャー・審判はできるだけ枠外になるように調整してください。"
        "上下方向は自動で全体を使うため、基本は『横位置』と『幅』だけ合わせればOKです。"
    )

    default_center = 68 if batter_side == "右打者" else 32
    preview_percent = st.slider(
        "赤枠確認用の動画位置（%）",
        5, 90, 35, 1,
        help="打者の全身が見やすい時点へ動かしてください。解析範囲そのものは動画全体で固定されます。",
    )
    preview_ratio = preview_percent / 100.0

    roi_col1, roi_col2 = st.columns(2)
    with roi_col1:
        st.markdown("##### 比較選手")
        ref_full = st.checkbox("画面全体を使う", value=False, key="ref_roi_full")
        if ref_full:
            ref_center, ref_width = 50, 100
        else:
            ref_center = st.slider("打者の横位置", 10, 90, default_center, 1, key="ref_roi_center")
            ref_width = st.slider("解析範囲の幅", 20, 100, 55, 1, key="ref_roi_width")
        ref_preview = _video_preview(
            reference_file.getvalue(),
            Path(reference_file.name).suffix,
            preview_ratio,
        )
        ref_overlay = _draw_roi_preview(ref_preview, ref_center, ref_width)
        if ref_overlay is not None:
            st.image(ref_overlay, caption="赤枠内だけを骨格解析", use_column_width=True)
        reference_roi = {"center": ref_center / 100.0, "width": ref_width / 100.0}

    with roi_col2:
        st.markdown("##### 自分")
        target_full = st.checkbox("画面全体を使う", value=False, key="target_roi_full")
        if target_full:
            target_center, target_width = 50, 100
        else:
            target_center = st.slider("打者の横位置", 10, 90, default_center, 1, key="target_roi_center")
            target_width = st.slider("解析範囲の幅", 20, 100, 55, 1, key="target_roi_width")
        target_preview = _video_preview(
            target_file.getvalue(),
            Path(target_file.name).suffix,
            preview_ratio,
        )
        target_overlay = _draw_roi_preview(target_preview, target_center, target_width)
        if target_overlay is not None:
            st.image(target_overlay, caption="赤枠内だけを骨格解析", use_column_width=True)
        target_roi = {"center": target_center / 100.0, "width": target_width / 100.0}

    st.success("赤枠に打者だけが入っていれば、そのまま解析へ進めます。")

st.subheader("3. 解析")
can_run = (
    reference_file is not None
    and target_file is not None
    and len(selected_features) == 5
)
if len(selected_features) != 5:
    st.warning(f"評価項目は5項目選択してください（現在 {len(selected_features)} 項目）。")

analyze_clicked = st.button(
    "解析開始",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
)

if analyze_clicked:
    session_root = Path(tempfile.mkdtemp(prefix="batting_form_"))
    upload_dir = session_root / "uploads"
    output_dir = session_root / "results"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_suffix = Path(reference_file.name).suffix or ".mp4"
    target_suffix = Path(target_file.name).suffix or ".mp4"
    reference_path = upload_dir / f"reference{reference_suffix}"
    target_path = upload_dir / f"target{target_suffix}"
    reference_path.write_bytes(reference_file.getbuffer())
    target_path.write_bytes(target_file.getbuffer())

    progress = st.progress(5, text="動画を準備しています…")
    try:
        progress.progress(15, text="骨格検出・フォーム解析を高精度モードで実行しています…")
        with st.spinner("解析中です。動画の長さ・解像度によって時間がかかります。"):
            result = engine.run_analysis(
                reference_original_path=reference_path,
                target_original_path=target_path,
                batter_side=batter_side,
                selected_feature_names=selected_features,
                output_dir=output_dir,
                reference_batter_position="自動",
                target_batter_position="自動",
                reference_batter_roi=reference_roi,
                target_batter_roi=target_roi,
            )

        # 身体負担・怪我リスク関連指標は、評価項目の選択に関係なく必ず計算する。
        result["body_load"] = risk_analysis.assess_body_load(
            result["reference_analysis"],
            result["target_analysis"],
            batter_side=batter_side,
        )
        result["body_load"]["regions"].to_csv(
            output_dir / "身体負担_部位別.csv", index=False, encoding="utf-8-sig"
        )
        result["body_load"]["components"].to_csv(
            output_dir / "身体負担_計測proxy.csv", index=False, encoding="utf-8-sig"
        )
        result["body_load"]["guidance"].to_csv(
            output_dir / "身体負担_改善候補.csv", index=False, encoding="utf-8-sig"
        )

        progress.progress(90, text="PDFレポートを作成しています…")
        pdf_bytes = build_pdf_report(
            result=result,
            output_dir=output_dir,
            batter_side=batter_side,
            selected_features=selected_features,
        )
        (output_dir / "打撃フォーム分析レポート.pdf").write_bytes(pdf_bytes)

        st.session_state.analysis_bundle = {
            "result": result,
            "session_root": session_root,
            "output_dir": output_dir,
            "pdf_bytes": pdf_bytes,
            "batter_side": batter_side,
            "selected_features": list(selected_features),
            "reference_batter_position": "自動",
            "target_batter_position": "自動",
            "reference_batter_roi": reference_roi,
            "target_batter_roi": target_roi,
        }
        st.session_state.ref_impact_frame = int(result["reference_analysis"].impact_frame)
        st.session_state.target_impact_frame = int(result["target_analysis"].impact_frame)

        progress.progress(100, text="解析が完了しました。")
        st.success("解析が完了しました。下の『インパクト確認・補正』も確認してください。")

    except Exception as error:
        progress.empty()
        st.error("解析中にエラーが発生しました。")
        st.exception(error)


bundle = st.session_state.analysis_bundle
if bundle is not None:
    result = bundle["result"]
    output_dir = Path(bundle["output_dir"])
    session_root = Path(bundle["session_root"])
    result_batter_side = bundle["batter_side"]
    result_selected_features = bundle["selected_features"]

    st.divider()
    st.subheader("4. インパクト位置を確認・補正")
    st.warning(
        "MediaPipe Poseはバットやボールそのものを検出していないため、インパクトは身体動作からの推定です。"
        "打者領域を限定してから自動推定しますが、画像を見て違う場合はここで正しいフレームへ修正してください。"
        "補正時は骨格検出をやり直さないため、再計算は短時間です。"
    )

    ref_analysis = result["reference_analysis"]
    target_analysis = result["target_analysis"]

    impact_col1, impact_col2 = st.columns(2)
    with impact_col1:
        st.markdown("##### 比較選手")
        ref_frame = st.slider(
            "比較選手のインパクトフレーム",
            min_value=0,
            max_value=max(0, ref_analysis.frame_count - 1),
            key="ref_impact_frame",
        )
        ref_preview = engine.get_pose_preview_frame(ref_analysis, ref_frame)
        ref_measured = engine.is_measured_frame(ref_analysis, ref_frame)
        if ref_preview is not None:
            st.image(
                ref_preview,
                caption=f"フレーム {ref_frame} / {ref_frame / ref_analysis.fps:.3f}秒",
                use_column_width=True,
            )
        if not ref_measured:
            st.warning("このフレームは補間値です。実測骨格のある近くのフレームを選んでください。")

    with impact_col2:
        st.markdown("##### 自分")
        target_frame = st.slider(
            "自分のインパクトフレーム",
            min_value=0,
            max_value=max(0, target_analysis.frame_count - 1),
            key="target_impact_frame",
        )
        target_preview = engine.get_pose_preview_frame(target_analysis, target_frame)
        target_measured = engine.is_measured_frame(target_analysis, target_frame)
        if target_preview is not None:
            st.image(
                target_preview,
                caption=f"フレーム {target_frame} / {target_frame / target_analysis.fps:.3f}秒",
                use_column_width=True,
            )
        if not target_measured:
            st.warning("このフレームは補間値です。実測骨格のある近くのフレームを選んでください。")

    can_recalculate_impact = bool(ref_measured and target_measured)
    if st.button(
        "この2フレームをインパクトとして再計算",
        type="primary",
        use_container_width=True,
        disabled=not can_recalculate_impact,
    ):
        try:
            with st.spinner("インパクト位置を反映して再計算しています…"):
                result = engine.recalculate_with_manual_impacts(
                    result=result,
                    reference_impact_frame=ref_frame,
                    target_impact_frame=target_frame,
                    selected_feature_names=result_selected_features,
                    batter_side=result_batter_side,
                )
                result["body_load"] = risk_analysis.assess_body_load(
                    result["reference_analysis"],
                    result["target_analysis"],
                    batter_side=result_batter_side,
                )
                result["body_load"]["regions"].to_csv(
                    output_dir / "身体負担_部位別.csv", index=False, encoding="utf-8-sig"
                )
                result["body_load"]["components"].to_csv(
                    output_dir / "身体負担_計測proxy.csv", index=False, encoding="utf-8-sig"
                )
                result["body_load"]["guidance"].to_csv(
                    output_dir / "身体負担_改善候補.csv", index=False, encoding="utf-8-sig"
                )
                pdf_bytes = build_pdf_report(
                    result=result,
                    output_dir=output_dir,
                    batter_side=result_batter_side,
                    selected_features=result_selected_features,
                )
                (output_dir / "打撃フォーム分析レポート.pdf").write_bytes(pdf_bytes)
                bundle["result"] = result
                bundle["pdf_bytes"] = pdf_bytes
                st.session_state.analysis_bundle = bundle
            st.success("インパクト位置を反映して、得点・改善提案・PDFを更新しました。")
        except Exception as error:
            st.error("再計算中にエラーが発生しました。")
            st.exception(error)

    # 再計算後の値を取得
    result = st.session_state.analysis_bundle["result"]
    bundle = st.session_state.analysis_bundle

    st.subheader("5. 解析結果")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("フォーム類似度", f"{result['overall_score']:.1f} / 100")
    metric2.metric("最も近い項目", result["best_item"], f"{result['best_score']:.1f}点")
    metric3.metric("差が大きい項目", result["lowest_item"], f"{result['lowest_score']:.1f}点")

    ref_analysis = result["reference_analysis"]
    target_analysis = result["target_analysis"]
    q1, q2 = st.columns(2)
    q1.metric(
        "解析信頼度：比較選手",
        f"{ref_analysis.quality_score:.1f} / 100",
        ref_analysis.quality_label,
    )
    q2.metric(
        "解析信頼度：自分",
        f"{target_analysis.quality_score:.1f} / 100",
        target_analysis.quality_label,
    )

    ref_q = ref_analysis.quality_details or {}
    tar_q = target_analysis.quality_details or {}
    st.caption(
        "解析信頼度はフォームの良し悪しではなく、骨格推定結果をどの程度信用してよいかの品質指標です。"
    )
    quality_table = {
        "項目": ["全体骨格検出率", "インパクト前後の検出率", "主要関節visibility"],
        "比較選手": [
            f"{ref_q.get('overall_detection_rate', 0):.1%}",
            f"{ref_q.get('impact_window_detection_rate', 0):.1%}",
            f"{ref_q.get('key_joint_visibility', 0):.2f}",
        ],
        "自分": [
            f"{tar_q.get('overall_detection_rate', 0):.1%}",
            f"{tar_q.get('impact_window_detection_rate', 0):.1%}",
            f"{tar_q.get('key_joint_visibility', 0):.2f}",
        ],
    }
    st.dataframe(quality_table, use_container_width=True, hide_index=True)
    if min(ref_analysis.quality_score, target_analysis.quality_score) < 80:
        st.warning(
            "解析信頼度が80点未満の動画が含まれています。結果は表示できますが、"
            "構え・インパクト画像と骨格の重なりを確認し、必要なら撮影条件や赤枠を調整してください。"
        )

    st.markdown("#### 項目別結果")
    st.dataframe(result["score_dataframe"], use_container_width=True, hide_index=True)

    radar_path = output_dir / "graphs" / "打撃フォーム比較レーダーチャート.png"
    bar_path = output_dir / "graphs" / "評価項目別類似度棒グラフ.png"
    graph1, graph2 = st.columns(2)
    with graph1:
        if radar_path.exists():
            st.image(str(radar_path), caption="フォーム比較レーダーチャート")
    with graph2:
        if bar_path.exists():
            st.image(str(bar_path), caption="項目別類似度")

    st.markdown("#### 構え・インパクト確認")
    image_names = [
        ("reference_stance_pose.jpg", "比較選手：構え"),
        ("reference_impact_pose.jpg", "比較選手：インパクト"),
        ("target_stance_pose.jpg", "自分：構え"),
        ("target_impact_pose.jpg", "自分：インパクト"),
    ]
    image_cols = st.columns(4)
    for col, (filename, caption) in zip(image_cols, image_names):
        image_path = output_dir / "impact_frames" / filename
        if image_path.exists():
            with col:
                st.image(str(image_path), caption=caption)

    st.subheader("6. 改善ポイント・練習")
    advice_df = result["advice_dataframe"]
    if advice_df.empty:
        st.success("現在の判定基準では、選択した5項目に大きな差は検出されませんでした。")
    else:
        st.dataframe(advice_df, use_container_width=True, hide_index=True)

    lowest_item = result["lowest_item"]
    youtube_query_map = {
        "身体を開くな": "野球 バッティング 開きを抑える 練習",
        "割れを作れ": "野球 バッティング 割れ 作り方 練習",
        "ボールを引きつけろ": "野球 バッティング ボールを引きつける 練習",
        "軸をぶらすな": "野球 バッティング 軸 ブレない 練習",
        "壁を作れ": "野球 バッティング 前足 壁 練習",
        "頭を残せ": "野球 バッティング 頭を残す 練習",
        "体重移動をしろ": "野球 バッティング 体重移動 練習",
        "腰を使え": "野球 バッティング 腰 回転 練習",
    }
    query = youtube_query_map.get(lowest_item, f"野球 バッティング {lowest_item}") + f" {result_batter_side}"
    youtube_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    st.markdown("#### 関連する練習動画を探す")
    st.write(f"今回最も差が大きかった「{lowest_item}」に合わせた検索候補です。")
    st.link_button("YouTubeで練習動画を検索", youtube_url, use_container_width=True)

    evidence_tip = risk_analysis.evidence_based_form_tip(lowest_item)
    if evidence_tip:
        st.markdown("#### 先行資料に基づく補足")
        st.info(evidence_tip["text"])
        st.markdown(f"[参考資料を開く]({evidence_tip['url']})")

    # --------------------------------------------------------
    # 身体負担関連動作（必須表示）
    # --------------------------------------------------------
    st.subheader("7. 身体負担関連動作（必須）")
    body_load = result.get("body_load")
    if body_load is not None:
        st.warning(
            "ここでは『怪我の確率』や医学的な安全性を点数化しません。"
            "先行研究で論点となった動作を、比較選手との差に基づいて段階表示します。"
        )
        st.metric("総合判定", body_load["overall_label"])
        st.caption(body_load["limitations"])

        st.markdown("#### 部位別")
        st.dataframe(body_load["regions"], use_container_width=True, hide_index=True)

        with st.expander("どの数値から判定したかを見る"):
            st.dataframe(body_load["components"], use_container_width=True, hide_index=True)
            st.caption(
                "『低い』は『安全』を意味しません。比較選手よりproxy値が小さいという意味だけです。"
            )

        st.markdown("#### 負担関連動作を抑えるための改善候補")
        st.dataframe(body_load["guidance"], use_container_width=True, hide_index=True)

        st.markdown("#### 根拠として使用した先行研究・資料")
        sources = body_load["sources"]
        source_markdown = (
            f"- **腹斜筋・フォロースルー**: [{sources['oblique']['title']}]({sources['oblique']['url']})  \n"
            f"  {sources['oblique']['note']}\n"
            f"- **腰部・脊柱**: [{sources['low_back']['title']}]({sources['low_back']['url']})  \n"
            f"  {sources['low_back']['note']}\n"
            f"- **肩・肘の投球障害研究（判定には不使用）**: [{sources['throwing']['title']}]({sources['throwing']['url']})  \n"
            f"  {sources['throwing']['note']}\n"
            f"- **身体の開きの改善参考**: [{sources['opening']['title']}]({sources['opening']['url']})  \n"
            f"  {sources['opening']['note']}"
        )
        st.markdown(source_markdown)
        st.info(
            "宮崎大学の論文は投球障害についての研究です。今回の打撃動画だけでは同じ可動域項目を測れないため、"
            "肩・肘の傷害判定には使用していません。"
        )

    st.subheader("8. 結果を保存")
    st.download_button(
        "PDFレポートをダウンロード",
        data=bundle["pdf_bytes"],
        file_name="打撃フォーム分析レポート.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    with st.expander("CSV・画像などの詳細データも保存する"):
        archive_base = session_root / "batting_analysis_results"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))
        st.download_button(
            "詳細データをZIPでダウンロード",
            data=archive_path.read_bytes(),
            file_name="batting_analysis_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.subheader("9. スマートフォン・他の人と共有")
    st.info(
        "http://localhost:... は『このPC自身』だけを指すURLなので、スマートフォンへ送っても開けません。"
    )
    share_col1, share_col2 = st.columns(2)
    with share_col1:
        st.markdown("##### 同じWi-Fi内で使う")
        st.write(
            "付属の `start_web_app_network.bat` で起動し、黒い画面に表示される "
            "**Network URL** をスマートフォンで開いてください。PCとスマホは同じWi-Fiに接続します。"
        )
    with share_col2:
        st.markdown("##### 離れた人にも共有する")
        st.write(
            "Streamlit Community Cloudへ公開すると、"
            "`https://○○○.streamlit.app` のような共有用URLを取得できます。"
        )
        st.link_button(
            "Streamlit Community Cloudを開く",
            "https://share.streamlit.io/",
            use_container_width=True,
        )
        st.caption("このフォルダはCommunity Cloudへアップロードできる構成にしてあります。")

