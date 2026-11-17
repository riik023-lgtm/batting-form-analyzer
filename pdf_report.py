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
        title="打撃フォーム分析 ver.10.1 レポート", author="Batting Form Analyzer ver.10.1",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("JPTitle", parent=styles["Title"], fontName=FONT_NAME, fontSize=19, leading=24, alignment=TA_CENTER, spaceAfter=5*mm)
    h1 = ParagraphStyle("JPH1", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=13, leading=17, spaceBefore=4*mm, spaceAfter=2.5*mm)
    body = ParagraphStyle("JPBody", parent=styles["BodyText"], fontName=FONT_NAME, fontSize=9.2, leading=13.5)
    small = ParagraphStyle("JPSmall", parent=body, fontSize=7.8, leading=10.5)

    story = [_p("打撃フォーム分析 ver.10.1", title)]
    if player_name:
        story.append(_p(f"選手: {player_name}", body))
    story.append(_p(f"解析日時: {datetime.now().strftime('%Y/%m/%d %H:%M')} / 打席: {batter_side}", small))
    story.append(_p("解析方式: MediaPipe Pose Landmarker Heavy / 画像座標＋World座標の用途別解析", small))
    story.append(Spacer(1, 3*mm))

    story.append(_p("比較選手との動作傾向", h1))
    story.append(_p(result.get("overall_comparison_label", "評価保留"), body))
    story.append(_p(
        "この表示は比較選手との近さを4段階で表したもので、フォームの優劣を示す点数ではありません。内部互換用の100点換算値は本レポートでは表示しません。",
        small,
    ))

    ref = result["reference_analysis"]; tar = result["target_analysis"]
    story.append(_p("解析信頼度", h1))
    q_data = [[_p("動画", small), _p("信頼度", small), _p("判定", small), _p("全体検出率", small), _p("インパクト周辺", small), _p("visibility", small)]]
    for label, analysis in [("比較選手", ref), ("自分", tar)]:
        q = analysis.quality_details or {}
        q_data.append([_p(label, small), _p(f"{analysis.quality_score:.1f}/100", small), _p(analysis.quality_label, small), _p(f"{q.get('overall_detection_rate',0):.1%}", small), _p(f"{q.get('impact_window_detection_rate',0):.1%}", small), _p(f"{q.get('key_joint_visibility',0):.2f}", small)])
    qt = Table(q_data, colWidths=[24*mm,25*mm,22*mm,30*mm,33*mm,24*mm], repeatRows=1)
    qt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),FONT_NAME),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EEF6EE")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#AAB2BD")),("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("ALIGN",(1,1),(-1,-1),"CENTER")]))
    story.append(qt)
    story.append(_p("解析信頼度は骨格推定品質の指標であり、フォーム評価ではありません。", small))

    story.append(_p("5項目の比較", h1))
    df = result["score_dataframe"]
    data = [[_p("項目", small), _p("比較選手", small), _p("自分", small), _p("差の方向", small), _p("比較判定", small)]]
    for _, row in df.iterrows():
        data.append([_p(row["表示名"], small), _p(row["比較選手表示"], small), _p(row["自分表示"], small), _p(row["差の方向"], small), _p(row["比較判定"], small)])
    t=Table(data,colWidths=[37*mm,30*mm,30*mm,45*mm,28*mm],repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),FONT_NAME),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E9EEF5")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#AAB2BD")),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(t)

    diff_chart = output_dir / "graphs" / "比較選手との差の方向.png"
    if diff_chart.exists():
        story.append(_p("差の方向", h1))
        story.append(Image(str(diff_chart), width=165*mm, height=92*mm, kind="proportional"))
        story.append(_p("中央の0が比較選手です。左右は良否ではなく、比較選手より小さい／大きい方向を表します。", small))

    body_load = result.get("body_load")
    if body_load:
        story.append(_p("身体負担に関連する動作の確認", h1))
        story.append(_p(f"今回の確認結果: {body_load['overall_label']}", body))
        risk_data=[[_p("部位",small),_p("確認結果",small),_p("観測した動き",small)]]
        for _, row in body_load["regions"].iterrows():
            risk_data.append([_p(row["部位"],small),_p(row["判定"],small),_p(row["主な計測proxy"],small)])
        rt=Table(risk_data,colWidths=[38*mm,34*mm,94*mm],repeatRows=1)
        rt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),FONT_NAME),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#FFF0E8")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#AAB2BD")),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(rt)
        story.append(_p(body_load["limitations"], small))

    story.append(PageBreak())
    story.append(_p("今回優先して確認したいポイント", h1))
    advice_df=result.get("advice_dataframe")
    if advice_df is not None and not advice_df.empty:
        for i,row in advice_df.iterrows():
            story.append(KeepTogether([
                _p(f"{i+1}. {row.get('観点','')} — {row.get('比較判定','')}", h1),
                _p(f"比較結果: {row.get('比較結果','')}", body),
                _p(f"動作の特徴: {row.get('動作の特徴','')}", body),
                _p(f"改善の方向性: {row.get('改善の方向性','')}", body),
                _p(f"練習例: {row.get('練習例','')}", body),
                _p(f"研究上の位置づけ: {row.get('研究上の位置づけ','')}", small),
                Spacer(1,2*mm),
            ]))

    research_df=result.get("research_advice")
    if research_df is not None and not research_df.empty:
        story.append(_p("先行研究から見る追加の観点", h1))
        for _, row in research_df.iterrows():
            story.append(_p(f"・{row.get('観点','')}: {row.get('検出内容','')}", body))
            story.append(_p(f"改善の考え方: {row.get('改善アドバイス','')}", small))
            story.append(_p(f"参考研究: {row.get('参考研究','')}", small))
            story.append(Spacer(1,1.5*mm))

    story.append(_p("注意事項", h1))
    story.append(_p(
        "本レポートは動画から推定した骨格座標に基づく比較情報です。撮影角度、画質、遮蔽、フレームレートで結果は変動します。"
        "身体負担に関連する動作の表示は医学的診断や怪我発生確率ではありません。痛みやしびれ等がある場合は医療専門職へ相談してください。",
        small,
    ))
    doc.build(story)
    data=buffer.getvalue(); buffer.close(); return data
