"""ver.10.0: 先行研究を参照した身体負担・改善アドバイス。

重要:
- 医学的な怪我発生確率は算出しない。
- 動画で直接測れない筋力、筋厚、関節トルク、地面反力、真の身体重心は断定しない。
- MediaPipeから得られる運動学的proxyと比較選手との差を段階表示する。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SOURCES = {
    "swing_characteristics": {
        "title": "スイング計測装置を用いた大学野球選手の特性分析に関する研究",
        "url": "https://doi.org/10.1299/transjsme.20-00240",
        "use": "スイング時間・スイング特性を考える際の参考。センサーバットの値を動画proxyと同一視しない。",
    },
    "upper_limb_energy": {
        "title": "野球の打撃における上肢のエネルギーフロー：バット・ヘッドスピードの上位群と下位群のスイング局面の比較",
        "url": "https://doi.org/10.5432/jjpehss.17038",
        "use": "上肢へのエネルギー伝達の研究。動画ではエネルギーを直接測らず、骨盤→肩の回転タイミングの参考にする。",
    },
    "swing_angle": {
        "title": "スイング角度およびローリング角速度が打球特性に及ぼす影響（野球のインパクトシミュレーション）",
        "url": "https://doi.org/10.32226/jjbse.2012_007",
        "use": "バット軌道の解釈に利用。Poseだけではバット角度を直接測れないため、現版では採点しない。",
    },
    "impact_reproducibility": {
        "title": "インパクト位置の再現性に関わる野球打撃動作の検討－大学野球打者によるティー打撃を対象として－",
        "url": "https://www.ritsumei.ac.jp/~isaka/ronbun/misaki.pdf",
        "use": "骨盤傾斜・体幹・上肢などのばらつきとインパクト再現性を考える参考。再現性判定には複数スイングが必要。",
    },
    "trunk_muscle": {
        "title": "野球競技におけるバットスイングパフォーマンスと体幹筋形態の関係に関する研究",
        "url": "https://www.ritsumei.ac.jp/~isaka/ronbun/tutikane.pdf",
        "use": "体幹筋とバットスイングの関係の参考。動画から筋厚・筋力は推定せず、回旋動作だけを評価する。",
    },
    "batting_running": {
        "title": "打撃に走塁条件が加わる複合運動が打撃スイング速度および打撃動作へ及ぼす影響",
        "url": "https://doi.org/10.5432/jjpehss.039-24118",
        "use": "肩・腰角速度、身体重心位置・速度などの観点を参考。動画では身体中心速度proxyとして扱う。",
    },
    "follow_through": {
        "title": "Torso dynamics during follow through in baseball batting",
        "url": "https://doi.org/10.1080/14763141.2022.2071328",
        "use": "フォロースルー時の体幹回旋・腹斜筋負担を考える参考。",
    },
    "low_back_case": {
        "title": "バッティング動作により腰痛を発症した高校野球選手の1症例",
        "url": "https://doi.org/10.14902/kinkipt.2008.0.98.0",
        "use": "体幹側屈・回旋と腰部負担を考える参考。1症例であり一般化には注意。",
    },
    "opening": {
        "title": "『身体が開く』とは？青木宣親から学ぶ、身体の開きを抑える方法",
        "url": "https://victorysportsnews.com/articles/7155",
        "use": "傷害根拠ではなくフォーム改善の実践参考。",
    },
}


def _relative_change(target, reference, floor):
    if not np.isfinite(target) or not np.isfinite(reference):
        return np.nan
    scale = max(abs(float(reference)), float(floor))
    return (float(target) - float(reference)) / scale


def _label_from_change(change, moderate=0.20, high=0.40):
    if not np.isfinite(change):
        return "評価困難"
    magnitude = abs(float(change))
    if magnitude >= high:
        return "要確認"
    if magnitude >= moderate:
        return "少し差あり"
    return "大きな差は確認されず"


def _overall(labels: List[str]):
    if "要確認" in labels:
        return "要確認"
    if "少し差あり" in labels:
        return "少し差あり"
    if all(v == "評価困難" for v in labels):
        return "評価困難"
    return "大きな差は確認されず"


def assess_body_load(reference_analysis, target_analysis, batter_side="右打者") -> Dict:
    """比較選手との差から身体負担関連動作を段階表示する。"""
    re = reference_analysis.extra_metrics
    ta = target_analysis.extra_metrics

    side_change = _relative_change(
        ta.get("torso_side_bend_change_deg", np.nan),
        re.get("torso_side_bend_change_deg", np.nan),
        5.0,
    )
    sep_change = _relative_change(
        target_analysis.features.get("hip_shoulder_separation", np.nan),
        reference_analysis.features.get("hip_shoulder_separation", np.nan),
        12.0,
    )
    hip_speed_change = _relative_change(
        ta.get("hip_rotation_speed_deg_s", np.nan),
        re.get("hip_rotation_speed_deg_s", np.nan),
        150.0,
    )
    shoulder_speed_change = _relative_change(
        ta.get("shoulder_rotation_speed_deg_s", np.nan),
        re.get("shoulder_rotation_speed_deg_s", np.nan),
        150.0,
    )
    pelvis_tilt_change = _relative_change(
        ta.get("pelvis_tilt_change_deg", np.nan),
        re.get("pelvis_tilt_change_deg", np.nan),
        5.0,
    )

    low_back_magnitude = np.nanmax(np.abs([side_change, sep_change, hip_speed_change]))
    low_back_label = _label_from_change(low_back_magnitude, moderate=0.25, high=0.50)
    oblique_magnitude = np.nanmax(np.abs([hip_speed_change, shoulder_speed_change, sep_change]))
    oblique_label = _label_from_change(oblique_magnitude, moderate=0.25, high=0.50)
    pelvis_label = _label_from_change(pelvis_tilt_change, moderate=0.35, high=0.70)

    regions = pd.DataFrame([
        {
            "部位": "腰部・体幹",
            "判定": low_back_label,
            "主な計測proxy": "体幹側屈proxy、肩-骨盤捻転差、骨盤回転速度",
            "根拠と注意": "腰痛症例・体幹回旋研究を参考。単眼動画から腰椎負荷そのものは測定していません。",
        },
        {
            "部位": "脇腹・腹斜筋周辺",
            "判定": oblique_label,
            "主な計測proxy": "骨盤/肩回転速度、肩-骨盤捻転差",
            "根拠と注意": "フォロースルー・体幹筋研究を参考。筋活動・筋厚は動画から測定していません。",
        },
        {
            "部位": "骨盤・下肢連動",
            "判定": pelvis_label,
            "主な計測proxy": "骨盤左右傾斜変化、身体中心移動proxy",
            "根拠と注意": "インパクト再現性研究を参考。地面反力や真の重心位置は測定していません。",
        },
    ])

    components = pd.DataFrame([
        {"指標": "体幹側屈変化", "比較選手": re.get("torso_side_bend_change_deg"), "自分": ta.get("torso_side_bend_change_deg"), "単位": "deg"},
        {"指標": "骨盤傾斜変化", "比較選手": re.get("pelvis_tilt_change_deg"), "自分": ta.get("pelvis_tilt_change_deg"), "単位": "deg"},
        {"指標": "身体中心移動proxy", "比較選手": re.get("body_center_shift_proxy"), "自分": ta.get("body_center_shift_proxy"), "単位": "shoulder-width ratio"},
        {"指標": "身体中心速度proxy", "比較選手": re.get("body_center_speed_proxy"), "自分": ta.get("body_center_speed_proxy"), "単位": "shoulder-width/sec"},
        {"指標": "骨盤→肩ピーク時間差", "比較選手": re.get("hip_to_shoulder_peak_delay_ms"), "自分": ta.get("hip_to_shoulder_peak_delay_ms"), "単位": "ms"},
        {"指標": "スイング開始→インパクト", "比較選手": re.get("swing_time_sec"), "自分": ta.get("swing_time_sec"), "単位": "sec"},
    ])

    guidance = []
    if low_back_label in {"少し差あり", "要確認"}:
        guidance.append({
            "部位": "腰部・体幹",
            "改善の方向性": "比較選手との差が大きい体幹側屈・捻転・骨盤回転を個別に確認し、強い捻りや急な制動だけを増やさないでください。",
            "練習例": "50〜70%強度のティー打撃、フォロースルーを途中で止めない連続素振り、スロー動画で体幹傾きを確認",
            "参考": SOURCES["low_back_case"]["title"],
        })
    if oblique_label in {"少し差あり", "要確認"}:
        guidance.append({
            "部位": "脇腹・腹斜筋周辺",
            "改善の方向性": "筋力不足とは断定せず、骨盤と肩の回転タイミングおよびフォロースルーまでの回転の滑らかさを確認してください。",
            "練習例": "軽いバット→通常バットの段階的スイング、メディシンボール回旋、分解素振り",
            "参考": SOURCES["follow_through"]["title"],
        })
    if pelvis_label in {"少し差あり", "要確認"}:
        guidance.append({
            "部位": "骨盤・姿勢",
            "改善の方向性": "骨盤傾斜の変化が大きい場合、単発の良否ではなく同じ撮影条件で複数スイングのばらつきを確認してください。",
            "練習例": "同条件で3〜5スイング撮影し、インパクト付近の骨盤傾斜と体幹傾きを比較",
            "参考": SOURCES["impact_reproducibility"]["title"],
        })
    if not guidance:
        guidance.append({
            "部位": "全体",
            "改善の方向性": "今回測定できたproxyでは、比較選手より明確に大きい身体負担関連動作は検出されませんでした。これは安全を保証する意味ではありません。",
            "練習例": "同じ撮影条件で複数回測定し、フォームのばらつきを確認",
            "参考": "複数の先行研究",
        })

    return {
        "overall_label": _overall(regions["判定"].tolist()),
        "regions": regions,
        "components": components,
        "guidance": pd.DataFrame(guidance),
        "sources": SOURCES,
        "limitations": (
            "この表示は医学的診断や怪我発生確率ではありません。単眼動画とMediaPipeから得られる運動学的proxyを、"
            "比較選手との差および先行研究で論点となった動作と照合した参考情報です。筋力、筋厚、関節トルク、"
            "地面反力、真の身体重心、バットヘッド速度は直接測定していません。"
        ),
    }


def research_based_advice(reference_analysis, target_analysis) -> pd.DataFrame:
    """新規6研究を中心に、追加指標から改善候補を出す。"""
    re = reference_analysis.extra_metrics
    ta = target_analysis.extra_metrics
    rows = []

    def add(topic, finding, advice, training, key):
        src = SOURCES[key]
        rows.append({
            "観点": topic,
            "検出内容": finding,
            "改善アドバイス": advice,
            "練習例": training,
            "参考研究": src["title"],
            "リンク": src["url"],
        })

    ref_time, tar_time = re.get("swing_time_sec", np.nan), ta.get("swing_time_sec", np.nan)
    if np.isfinite(ref_time) and np.isfinite(tar_time) and tar_time > ref_time * 1.15:
        add(
            "スイング時間",
            f"スイング開始からインパクトまでが比較選手より{tar_time-ref_time:.3f}秒長く推定されました。",
            "速く振ることだけを目的にせず、動作開始からインパクトまでに余分な停滞がないか確認してください。",
            "50〜70%強度の連続ティー→通常スイング、同じテンポでの反復",
            "swing_characteristics",
        )

    ref_delay, tar_delay = re.get("hip_to_shoulder_peak_delay_ms", np.nan), ta.get("hip_to_shoulder_peak_delay_ms", np.nan)
    if np.isfinite(ref_delay) and np.isfinite(tar_delay) and abs(tar_delay - ref_delay) >= 60:
        add(
            "骨盤→肩の運動連鎖",
            f"骨盤回転速度ピークから肩回転速度ピークまでの時間差が比較選手と約{abs(tar_delay-ref_delay):.0f}ms異なります。",
            "エネルギー伝達そのものは測定していませんが、骨盤と肩の回転順序をスロー動画で確認してください。",
            "ステップ→骨盤→肩の順を意識した分解素振り、メディシンボール回旋",
            "trunk_muscle",
        )

    # 上肢エネルギーフロー研究では手関節を含む上肢末端でのエネルギー伝達が論点となる。
    # 本アプリは力学的エネルギーを測れないため、手首アライメントproxyの差だけを補助表示する。
    wrist_pairs = [
        ("left_wrist_alignment_proxy_deg", "左手首"),
        ("right_wrist_alignment_proxy_deg", "右手首"),
    ]
    wrist_diffs = []
    for key, label in wrist_pairs:
        rv, tv = re.get(key, np.nan), ta.get(key, np.nan)
        if np.isfinite(rv) and np.isfinite(tv):
            wrist_diffs.append((abs(tv-rv), label, rv, tv))
    if wrist_diffs:
        diff, label, rv, tv = max(wrist_diffs)
        if diff >= 15:
            add(
                "手首アライメントproxy",
                f"{label}の前腕―手掌方向の2D proxyが比較選手と約{diff:.1f}°異なります。",
                "力学的エネルギーや手関節トルクを測定した値ではありません。インパクト付近で手首が大きく折れていないか、バット軌道と合わせて確認してください。",
                "短いバットでの低強度ティー、インパクト前後を正面・側方のスロー動画で確認",
                "upper_limb_energy",
            )

    ref_com, tar_com = re.get("body_center_speed_proxy", np.nan), ta.get("body_center_speed_proxy", np.nan)
    if np.isfinite(ref_com) and np.isfinite(tar_com) and abs(tar_com-ref_com) / max(abs(ref_com), 0.2) >= 0.25:
        add(
            "身体中心速度proxy",
            "スイング中盤までの身体中心速度proxyが比較選手と大きく異なります。",
            "真の身体重心速度ではありません。身体全体の前方移動と回転がどのタイミングで起きているかを確認してください。",
            "歩きティー、ステップ打撃、打撃後に一塁へ1〜2歩出る実戦ドリル",
            "batting_running",
        )

    ref_tilt, tar_tilt = re.get("pelvis_tilt_change_deg", np.nan), ta.get("pelvis_tilt_change_deg", np.nan)
    ref_side, tar_side = re.get("torso_side_bend_change_deg", np.nan), ta.get("torso_side_bend_change_deg", np.nan)
    if (
        np.isfinite(ref_tilt) and np.isfinite(tar_tilt) and abs(tar_tilt-ref_tilt) >= 8
    ) or (
        np.isfinite(ref_side) and np.isfinite(tar_side) and abs(tar_side-ref_side) >= 8
    ):
        add(
            "インパクト再現性に関わる姿勢proxy",
            "骨盤傾斜または体幹側屈の変化が比較選手と大きく異なります。",
            "1スイングだけで再現性は判定できないため、同じ条件で3〜5回撮影してばらつきを確認してください。",
            "ティー位置を固定した3〜5球の連続測定、各回の骨盤傾斜・体幹傾きの比較",
            "impact_reproducibility",
        )

    if not rows:
        add(
            "追加指標",
            "今回の追加proxyでは大きな差は検出されませんでした。",
            "単発の結果だけでなく、複数スイングの再現性を見ると改善点を見つけやすくなります。",
            "同条件で3〜5スイングを保存し、履歴グラフを比較",
            "impact_reproducibility",
        )

    return pd.DataFrame(rows)
