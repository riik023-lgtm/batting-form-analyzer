from io import BytesIO
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)

FONT_NAME = "HeiseiKakuGo-W5"
pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def _p(text, style):
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def build_pdf_report(result, output_dir, batter_side, selected_features):
    """解析結果を1つの日本語PDFレポートにまとめて返す。"""
    output_dir = Path(output_dir)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="打撃フォーム分析レポート",
        author="Batting Form Analyzer",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "JPTitle", parent=styles["Title"], fontName=FONT_NAME,
        fontSize=20, leading=25, alignment=TA_CENTER, spaceAfter=8 * mm
    )
    h1 = ParagraphStyle(
        "JPH1", parent=styles["Heading1"], fontName=FONT_NAME,
        fontSize=14, leading=18, spaceBefore=4 * mm, spaceAfter=3 * mm
    )
    body = ParagraphStyle(
        "JPBody", parent=styles["BodyText"], fontName=FONT_NAME,
        fontSize=9.5, leading=14
    )
    small = ParagraphStyle(
        "JPSmall", parent=body, fontSize=8, leading=11
    )
    score_style = ParagraphStyle(
        "JPScore", parent=body, fontSize=26, leading=30,
        alignment=TA_CENTER, spaceAfter=4 * mm
    )

    story = []
    story.append(_p("打撃フォーム分析レポート", title))
    story.append(_p(f"解析日時: {datetime.now().strftime('%Y/%m/%d %H:%M')}", small))
    story.append(_p(f"打席: {batter_side} / 解析モード: {result.get('analysis_mode', '高精度')}", small))
    story.append(Spacer(1, 4 * mm))
    story.append(_p("比較選手との総合類似度", h1))
    story.append(_p(f"{result['overall_score']:.1f} / 100", score_style))
    story.append(_p(
        "この点数は比較選手との動作特徴の近さを表す類似度であり、フォームの優劣や医学的な傷害発生確率を示すものではありません。",
        small
    ))

    # v1.9: フォーム得点とは別に解析信頼度を表示
    ref_analysis = result["reference_analysis"]
    tar_analysis = result["target_analysis"]
    story.append(_p("解析信頼度", h1))
    story.append(_p(
        "解析信頼度はフォームの良し悪しではなく、骨格推定結果をどの程度信用してよいかを表す品質指標です。",
        small
    ))
    quality_data = [[
        _p("動画", small), _p("信頼度", small), _p("判定", small),
        _p("全体検出率", small), _p("インパクト周辺検出率", small), _p("主要関節visibility", small)
    ]]
    for label, analysis in [("比較選手", ref_analysis), ("自分", tar_analysis)]:
        q = analysis.quality_details or {}
        quality_data.append([
            _p(label, small),
            _p(f"{analysis.quality_score:.1f}/100", small),
            _p(analysis.quality_label, small),
            _p(f"{q.get('overall_detection_rate', 0):.1%}", small),
            _p(f"{q.get('impact_window_detection_rate', 0):.1%}", small),
            _p(f"{q.get('key_joint_visibility', 0):.2f}", small),
        ])
    qt = Table(quality_data, colWidths=[24*mm, 25*mm, 24*mm, 28*mm, 35*mm, 34*mm], repeatRows=1)
    qt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT_NAME),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EEF6EE")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(qt)

    story.append(_p("項目別結果", h1))
    df = result["score_dataframe"]
    table_data = [[
        _p("評価項目", small), _p("比較選手", small), _p("自分", small),
        _p("差", small), _p("類似度", small)
    ]]
    for _, row in df.iterrows():
        table_data.append([
            _p(row["評価項目"], small),
            _p(row["比較選手"], small),
            _p(row["自分"], small),
            _p(row["差の絶対値"], small),
            _p(f"{float(row['得点']):.1f}", small),
        ])
    t = Table(table_data, colWidths=[45*mm, 30*mm, 30*mm, 25*mm, 25*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT_NAME),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9EEF5")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)

    graph_dir = output_dir / "graphs"
    radar = graph_dir / "打撃フォーム比較レーダーチャート.png"
    bar = graph_dir / "評価項目別類似度棒グラフ.png"
    images = []
    for path, caption in [(radar, "フォーム比較レーダーチャート"), (bar, "評価項目別類似度")]:
        if path.exists():
            img = Image(str(path), width=82*mm, height=68*mm, kind="proportional")
            images.append([img, _p(caption, small)])
    if images:
        story.append(_p("グラフ", h1))
        row = []
        for img, cap in images:
            row.append([img, cap])
        story.append(Table([row], colWidths=[88*mm] * len(row)))

    body_load = result.get("body_load")
    if body_load is not None:
        story.append(_p("身体負担関連動作", h1))
        story.append(_p(
            f"総合判定: {body_load['overall_label']}",
            body
        ))
        story.append(_p(
            "怪我の確率や医学的な安全性を点数化せず、先行研究で論点となった動作を比較選手との差から段階表示しています。",
            small
        ))

        risk_table = [[
            _p("部位", small), _p("判定", small), _p("主な計測proxy", small)
        ]]
        for _, row in body_load["regions"].iterrows():
            risk_table.append([
                _p(row["部位"], small),
                _p(row["判定"], small),
                _p(row["主な計測proxy"], small),
            ])
        rt = Table(risk_table, colWidths=[38*mm, 34*mm, 94*mm], repeatRows=1)
        rt.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), FONT_NAME),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FFF0E8")),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#AAB2BD")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(rt)
        story.append(Spacer(1, 2*mm))
        story.append(_p(body_load["limitations"], small))

    story.append(PageBreak())
    story.append(_p("改善ポイント・推奨練習", h1))
    if body_load is not None:
        story.append(_p("身体負担関連動作からの改善候補", h1))
        for _, row in body_load["guidance"].iterrows():
            story.append(_p(f"{row.get('部位', '')}: {row.get('改善の方向性', '')}", body))
            story.append(_p(f"練習例: {row.get('練習例', '')}", small))
            story.append(Spacer(1, 2*mm))

    advice_df = result["advice_dataframe"]
    if advice_df.empty:
        story.append(_p("現在の判定基準では、選択した項目に大きな差は検出されませんでした。", body))
    else:
        for idx, row in advice_df.iterrows():
            block = [
                _p(f"{idx+1}. {row.get('分類', '改善ポイント')}（優先度: {row.get('優先度', '-')}）", h1),
                _p(f"検出内容: {row.get('検出内容', '')}", body),
                _p(f"改善の方向性: {row.get('改善の方向性', '')}", body),
                _p(f"推奨練習: {row.get('推奨練習', '')}", body),
                Spacer(1, 3*mm),
            ]
            story.append(KeepTogether(block))

    story.append(_p("構え・インパクト確認", h1))
    impact_dir = output_dir / "impact_frames"
    image_specs = [
        ("reference_stance_pose.jpg", "比較選手: 構え"),
        ("reference_impact_pose.jpg", "比較選手: インパクト"),
        ("target_stance_pose.jpg", "自分: 構え"),
        ("target_impact_pose.jpg", "自分: インパクト"),
    ]
    cells = []
    for filename, caption in image_specs:
        path = impact_dir / filename
        if path.exists():
            cells.append([
                Image(str(path), width=78*mm, height=55*mm, kind="proportional"),
                _p(caption, small)
            ])
    for i in range(0, len(cells), 2):
        pair = cells[i:i+2]
        if len(pair) == 1:
            pair.append("")
        story.append(Table([pair], colWidths=[88*mm, 88*mm]))
        story.append(Spacer(1, 2*mm))

    if body_load is not None:
        story.append(_p("身体負担関連動作の参考文献", h1))
        sources = body_load["sources"]
        for key in ["oblique", "low_back", "throwing", "opening"]:
            source = sources[key]
            story.append(_p(f"・{source['title']}", small))
            story.append(_p(source['url'], small))
            story.append(_p(source['note'], small))
            story.append(Spacer(1, 1.5*mm))

    story.append(_p("注意事項", h1))
    story.append(_p(
        "本レポートは動画から推定した骨格座標に基づくフォーム比較の参考情報です。撮影角度、画質、遮蔽、フレームレート等により結果は変動します。身体負担関連動作の判定は医学的な診断や実際の傷害発生確率ではありません。痛みやしびれ等の症状がある場合は本結果だけで判断せず、医療専門職へ相談してください。",
        small
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
