from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_NAME = "HeiseiKakuGo-W5"
pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def _p(text, style):
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def build_pdf_report(result, output_dir, batter_side, selected_features, player_name=""):
    output_dir = Path(output_dir)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=13*mm, leftMargin=13*mm, topMargin=13*mm, bottomMargin=13*mm,
        title="打撃フォーム分析 ver.10.0 レポート",
        author="Batting Form Analyzer ver.10.0",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("JPTitle", parent=styles["Title"], fontName=FONT_NAME, fontSize=19, leading=24, alignment=TA_CENTER, spaceAfter=5*mm)
    h1 = ParagraphStyle("JPH1", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=13, leading=17, spaceBefore=4*mm, spaceAfter=2.5*mm)
    body = ParagraphStyle("JPBody", parent=styles["BodyText"], fontName=FONT_NAME, fontSize=9.2, leading=13.5)
    small = ParagraphStyle("JPSmall", parent=body, fontSize=7.8, leading=10.5)
    score_style = ParagraphStyle("JPScore", parent=body, fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=3*mm)

    story = [_p("打撃フォーム分析 ver.10.0", title)]
    if player_name:
        story.append(_p(f"選手: {player_name}", body))
    story.append(_p(f"解析日時: {datetime.now().strftime('%Y/%m/%d %H:%M')} / 打席: {batter_side}", small))
    story.append(_p("解析方式: MediaPipe Pose Landmarker Heavy / 画像座標＋World座標の用途別解析", small))
    story.append(Spacer(1, 3*mm))

    story.append(_p("比較選手との総合類似度", h1))
    story.append(_p(f"{result['overall_score']:.1f} / 100", score_style))
    story.append(_p(
        "この点数は比較選手との動作特徴の近さを表す類似度です。フォームの優劣や医学的な傷害発生確率ではありません。",
        small,
    ))

    ref = result["reference_analysis"]
    tar = result["target_analysis"]
    story.append(_p("解析信頼度", h1))
    q_data = [[_p("動画", small), _p("信頼度", small), _p("判定", small), _p("全体検出率", small), _p("インパクト周辺", small), _p("visibility", small)]]
    for label, analysis in [("比較選手", ref), ("自分", tar)]:
        q = analysis.quality_details or {}
        q_data.append([
            _p(label, small), _p(f"{analysis.quality_score:.1f}/100", small), _p(analysis.quality_label, small),
            _p(f"{q.get('overall_detection_rate', 0):.1%}", small), _p(f"{q.get('impact_window_detection_rate', 0):.1%}", small),
            _p(f"{q.get('key_joint_visibility', 0):.2f}", small),
        ])
    qt = Table(q_data, colWidths=[24*mm, 25*mm, 22*mm, 30*mm, 33*mm, 24*mm], repeatRows=1)
    qt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT_NAME), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EEF6EE")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(qt)

    story.append(_p("項目別結果", h1))
    df = result["score_dataframe"]
    data = [[_p("評価項目", small), _p("比較選手", small), _p("自分", small), _p("差", small), _p("類似度", small)]]
    for _, row in df.iterrows():
        data.append([
            _p(row["評価項目"], small), _p(row["比較選手"], small), _p(row["自分"], small),
            _p(row["差の絶対値"], small), _p(f"{float(row['得点']):.1f}", small),
        ])
    t = Table(data, colWidths=[46*mm, 31*mm, 31*mm, 27*mm, 24*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT_NAME), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9EEF5")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(t)

    story.append(_p("ver.10.0 追加指標", h1))
    extra_rows = [[_p("指標", small), _p("比較選手", small), _p("自分", small)]]
    labels = [
        ("スイング開始→インパクト(s)", "swing_time_sec"),
        ("身体中心移動proxy", "body_center_shift_proxy"),
        ("身体中心速度proxy", "body_center_speed_proxy"),
        ("体幹側屈変化(deg)", "torso_side_bend_change_deg"),
        ("骨盤傾斜変化(deg)", "pelvis_tilt_change_deg"),
        ("骨盤→肩ピーク時間差(ms)", "hip_to_shoulder_peak_delay_ms"),
        ("左手首アライメントproxy(deg)", "left_wrist_alignment_proxy_deg"),
        ("右手首アライメントproxy(deg)", "right_wrist_alignment_proxy_deg"),
    ]
    for label, key in labels:
        extra_rows.append([
            _p(label, small),
            _p(f"{ref.extra_metrics.get(key, float('nan')):.3f}", small),
            _p(f"{tar.extra_metrics.get(key, float('nan')):.3f}", small),
        ])
    et = Table(extra_rows, colWidths=[80*mm, 40*mm, 40*mm], repeatRows=1)
    et.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT_NAME), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F4F0FA")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")), ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))
    story.append(et)
    story.append(_p("身体中心は画像上のproxyであり、力学的な真の身体重心ではありません。", small))

    graph_dir = output_dir / "graphs"
    radar = graph_dir / "打撃フォーム比較レーダーチャート.png"
    bar = graph_dir / "評価項目別類似度棒グラフ.png"
    if radar.exists() or bar.exists():
        story.append(_p("グラフ", h1))
        cells = []
        for path in [radar, bar]:
            if path.exists():
                cells.append(Image(str(path), width=80*mm, height=58*mm, kind="proportional"))
        if len(cells) == 1:
            cells.append("")
        story.append(Table([cells], colWidths=[86*mm, 86*mm]))

    body_load = result.get("body_load")
    if body_load:
        story.append(_p("身体負担関連動作", h1))
        story.append(_p(f"総合判定: {body_load['overall_label']}", body))
        risk_data = [[_p("部位", small), _p("判定", small), _p("主な計測proxy", small)]]
        for _, row in body_load["regions"].iterrows():
            risk_data.append([_p(row["部位"], small), _p(row["判定"], small), _p(row["主な計測proxy"], small)])
        rt = Table(risk_data, colWidths=[37*mm, 33*mm, 95*mm], repeatRows=1)
        rt.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), FONT_NAME), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FFF0E8")),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(rt)
        story.append(_p(body_load["limitations"], small))

    story.append(PageBreak())
    story.append(_p("改善ポイント・推奨練習", h1))
    advice_df = result.get("advice_dataframe")
    if advice_df is not None and not advice_df.empty:
        for i, row in advice_df.iterrows():
            story.append(KeepTogether([
                _p(f"{i+1}. {row.get('分類', '')}（優先度: {row.get('優先度', '')}）", h1),
                _p(f"検出内容: {row.get('検出内容', '')}", body),
                _p(f"改善の方向性: {row.get('改善の方向性', '')}", body),
                _p(f"推奨練習: {row.get('推奨練習', '')}", body),
                _p(f"参考: {row.get('参考', '')}", small),
                Spacer(1, 2*mm),
            ]))

    research_df = result.get("research_advice")
    if research_df is not None and not research_df.empty:
        story.append(_p("先行研究を参考にした追加アドバイス", h1))
        for _, row in research_df.iterrows():
            story.append(_p(f"・{row.get('観点', '')}: {row.get('改善アドバイス', '')}", body))
            story.append(_p(f"練習例: {row.get('練習例', '')}", small))
            story.append(_p(f"参考研究: {row.get('参考研究', '')}", small))
            story.append(_p(row.get("リンク", ""), small))
            story.append(Spacer(1, 1.5*mm))

    story.append(_p("構え・インパクト確認", h1))
    specs = [
        ("reference_stance_pose.jpg", "比較選手: 構え"),
        ("reference_impact_pose.jpg", "比較選手: インパクト"),
        ("target_stance_pose.jpg", "自分: 構え"),
        ("target_impact_pose.jpg", "自分: インパクト"),
    ]
    blocks = []
    for filename, caption in specs:
        path = output_dir / "impact_frames" / filename
        if path.exists():
            blocks.append([Image(str(path), width=76*mm, height=54*mm, kind="proportional"), _p(caption, small)])
    for i in range(0, len(blocks), 2):
        pair = blocks[i:i+2]
        if len(pair) == 1:
            pair.append("")
        story.append(Table([pair], colWidths=[86*mm, 86*mm]))

    if body_load:
        story.append(_p("参考研究", h1))
        for source in body_load["sources"].values():
            story.append(_p(f"・{source['title']}", small))
            story.append(_p(source["url"], small))
            story.append(_p(source["use"], small))
            story.append(Spacer(1, 1*mm))

    story.append(_p("注意事項", h1))
    story.append(_p(
        "本レポートは動画から推定した骨格座標に基づく比較情報です。撮影角度、画質、遮蔽、フレームレートで結果は変動します。"
        "身体負担関連動作は医学的診断や怪我発生確率ではありません。痛みやしびれ等がある場合は医療専門職へ相談してください。",
        small,
    ))

    doc.build(story)
    data = buffer.getvalue()
    buffer.close()
    return data
