"""先行研究に基づく、打撃動画からの身体負担『参考判定』。

v1.9 方針
- 0〜100の「怪我リスク風スコア」は廃止する。
- 比較選手との相対差を、根拠となるproxyごとに段階表示する。
- 「低い＝安全」とは扱わない。
- 単眼MediaPipe Poseで直接測れないバットヘッド速度、関節トルク、筋活動、
  真の腰椎側屈は、直接値として扱わない。
- 速度系proxyは、連続する実測骨格フレーム間だけで計算する。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

SOURCES = {
    "oblique": {
        "title": "Torso dynamics during follow through in baseball batting",
        "url": "https://doi.org/10.1080/14763141.2022.2071328",
        "note": (
            "大学野球選手29名のモーションキャプチャ研究。右打者のフォロースルーで"
            "左外腹斜筋の伸張性収縮が示唆され、インパクト直後のバットヘッド速度が高いほど"
            "体幹回旋トルク・パワーとの関連が大きかった。"
        ),
    },
    "low_back": {
        "title": "バッティング動作により腰痛を発症した高校野球選手の1症例",
        "url": "https://doi.org/10.14902/kinkipt.2008.0.98.0",
        "note": (
            "腰痛を発症した高校野球選手1例の三次元動作解析。外角打ちでは体幹側屈と"
            "腰椎回旋が大きく、腰椎ストレス増大の可能性が考察された。空振り時の急激な"
            "脊柱運動についても傷害要因の可能性が述べられている。症例報告のため一般化には注意が必要。"
        ),
    },
    "throwing": {
        "title": "Risk factors associated with throwing injuries in young baseball players",
        "url": "https://doi.org/10.1016/j.jse.2023.01.034",
        "note": (
            "小学生野球選手2466名の投球障害研究。肩内旋可動域、非利き側股関節内旋可動域、"
            "柔軟性などが投球障害と関連した。ただし『投球』研究であり、打撃動画だけから"
            "肩・肘の傷害リスクを採点する根拠には直接用いない。"
        ),
    },
    "opening": {
        "title": "『身体が開く』とは？青木宣親から学ぶ、身体の開きを抑える方法",
        "url": "https://victorysportsnews.com/articles/7155",
        "note": (
            "実践解説記事。身体の開きを『腰の回転開始が早い状態』として説明し、"
            "後ろ足・股関節で開きを制御する練習としてアウトエッジランジ等を紹介している。"
            "傷害研究ではなくフォーム改善の参考として扱う。"
        ),
    },
}


def _horizontal_line_angle(left_point, right_point):
    dx = right_point[0] - left_point[0]
    dz = right_point[2] - left_point[2]
    return float(np.degrees(np.arctan2(dz, dx)))


def _signed_angle_difference(a, b):
    return np.degrees(
        np.arctan2(
            np.sin(np.radians(a - b)),
            np.cos(np.radians(a - b)),
        )
    )


def _nan_rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return (
        pd.Series(values)
        .rolling(window=max(1, int(window)), center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _safe_max(values, default=np.nan):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float(default)


def _masked_pair_speed(points, fps, valid_mask):
    points = np.asarray(points, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(points) < 2:
        return np.full(len(points), np.nan)
    speed = np.full(len(points), np.nan, dtype=float)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
    pair_valid = valid_mask[:-1] & valid_mask[1:]
    speed[1:][pair_valid] = diffs[pair_valid]
    return speed


def _masked_angle_velocity(angles, fps, valid_mask):
    angles = np.asarray(angles, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    unwrapped = np.degrees(np.unwrap(np.radians(angles)))
    velocity = np.full(len(unwrapped), np.nan, dtype=float)
    if len(unwrapped) >= 2:
        diffs = np.diff(unwrapped) * fps
        pair_valid = valid_mask[:-1] & valid_mask[1:]
        velocity[1:][pair_valid] = diffs[pair_valid]
    return velocity


def _motion_metrics(analysis) -> Dict[str, float]:
    world = np.asarray(analysis.world_landmarks, dtype=float)
    valid = np.asarray(analysis.valid_mask, dtype=bool)
    fps = max(float(analysis.fps), 1e-6)
    impact = int(np.clip(analysis.impact_frame, 0, len(world) - 1))

    shoulder_angles = np.asarray([
        _horizontal_line_angle(frame[LEFT_SHOULDER], frame[RIGHT_SHOULDER])
        for frame in world
    ], dtype=float)
    hip_angles = np.asarray([
        _horizontal_line_angle(frame[LEFT_HIP], frame[RIGHT_HIP])
        for frame in world
    ], dtype=float)

    signed_sep = _signed_angle_difference(shoulder_angles, hip_angles)
    sep_abs = np.abs(signed_sep).astype(float)
    sep_abs[~valid] = np.nan

    torso_ang_vel = _masked_angle_velocity(signed_sep, fps, valid)
    torso_speed = np.abs(torso_ang_vel)
    smooth_window = max(3, int(round(fps * 0.06)))
    torso_speed_smooth = _nan_rolling_mean(torso_speed, smooth_window)

    # 急減速も、速度値が連続して実測由来の箇所だけを使う。
    torso_deceleration = np.full(len(torso_speed_smooth), np.nan, dtype=float)
    if len(torso_speed_smooth) >= 2:
        delta = np.diff(torso_speed_smooth) * fps
        finite_pair = np.isfinite(torso_speed_smooth[:-1]) & np.isfinite(torso_speed_smooth[1:])
        decel = np.maximum(0.0, -delta)
        torso_deceleration[1:][finite_pair] = decel[finite_pair]

    left_wrist = world[:, LEFT_WRIST, :3]
    right_wrist = world[:, RIGHT_WRIST, :3]
    left_speed = _masked_pair_speed(left_wrist, fps, valid)
    right_speed = _masked_pair_speed(right_wrist, fps, valid)
    wrist_speed = 0.5 * (left_speed + right_speed)
    wrist_speed = _nan_rolling_mean(wrist_speed, smooth_window)

    pre = max(0, impact - int(round(0.25 * fps)))
    post = min(len(world), impact + int(round(0.35 * fps)) + 1)
    wrist_post_end = min(len(wrist_speed), impact + int(round(0.15 * fps)) + 1)

    return {
        "max_torso_separation_deg": _safe_max(sep_abs[pre:post]),
        "post_torso_rotation_speed_deg_s": _safe_max(torso_speed[impact:post]),
        "post_torso_rotation_deceleration_deg_s2": _safe_max(torso_deceleration[impact:post]),
        "post_wrist_speed_proxy_m_s": _safe_max(wrist_speed[max(0, impact - 1):wrist_post_end]),
    }


def _relative_change(target, reference, floor):
    if not np.isfinite(target) or not np.isfinite(reference):
        return np.nan
    scale = max(abs(float(reference)), float(floor))
    return (float(target) - float(reference)) / scale


def _comparison_label(change):
    if not np.isfinite(change):
        return "評価困難"
    if change >= 0.40:
        return "大きく高い"
    if change >= 0.15:
        return "やや高い"
    if change <= -0.40:
        return "大きく低い"
    if change <= -0.15:
        return "やや低い"
    return "同程度"


def _component_row(name, target, reference, unit, rationale, floor):
    change = _relative_change(target, reference, floor=floor)
    return {
        "指標": name,
        "自分": round(float(target), 2) if np.isfinite(target) else np.nan,
        "比較選手": round(float(reference), 2) if np.isfinite(reference) else np.nan,
        "差率": round(float(change * 100.0), 1) if np.isfinite(change) else np.nan,
        "比較結果": _comparison_label(change),
        "単位": unit,
        "根拠": rationale,
    }


def _region_judgement(component_labels):
    labels = list(component_labels)
    if any(label == "評価困難" for label in labels):
        return "評価困難"
    high = sum(label == "大きく高い" for label in labels)
    mild = sum(label == "やや高い" for label in labels)
    if high >= 1 or mild >= 2:
        return "注意要素あり"
    if mild == 1:
        return "やや注意"
    return "明確な増大なし"


def _overall_judgement(region_labels):
    labels = list(region_labels)
    if any(label == "評価困難" for label in labels):
        return "一部評価困難"
    if any(label == "注意要素あり" for label in labels):
        return "注意要素あり"
    if any(label == "やや注意" for label in labels):
        return "やや注意"
    return "明確な増大なし"


def assess_body_load(reference_analysis, target_analysis, batter_side="右打者"):
    """部位別の身体負担関連動作を、点数ではなく段階評価で返す。"""
    ref = _motion_metrics(reference_analysis)
    tar = _motion_metrics(target_analysis)

    component_rows = [
        _component_row(
            "最大肩-骨盤相対回旋",
            tar["max_torso_separation_deg"], ref["max_torso_separation_deg"],
            "deg", "腰部回旋負荷のproxy", 8.0,
        ),
        _component_row(
            "インパクト後の体幹回旋速度",
            tar["post_torso_rotation_speed_deg_s"], ref["post_torso_rotation_speed_deg_s"],
            "deg/s", "フォロースルーの回旋動態proxy", 80.0,
        ),
        _component_row(
            "インパクト後の体幹回旋急減速",
            tar["post_torso_rotation_deceleration_deg_s2"], ref["post_torso_rotation_deceleration_deg_s2"],
            "deg/s²", "急激な回旋制動のproxy", 800.0,
        ),
        _component_row(
            "インパクト直後の手首速度",
            tar["post_wrist_speed_proxy_m_s"], ref["post_wrist_speed_proxy_m_s"],
            "m/s (Pose proxy)", "バットヘッド速度の代替proxy（同一ではない）", 0.6,
        ),
    ]
    components = pd.DataFrame(component_rows)

    lookup = {row["指標"]: row["比較結果"] for row in component_rows}
    low_back_label = _region_judgement([
        lookup["最大肩-骨盤相対回旋"],
        lookup["インパクト後の体幹回旋速度"],
        lookup["インパクト後の体幹回旋急減速"],
    ])
    oblique_label = _region_judgement([
        lookup["インパクト直後の手首速度"],
        lookup["インパクト後の体幹回旋速度"],
        lookup["インパクト後の体幹回旋急減速"],
    ])

    side_note = (
        "右打者の研究結果を直接参照。投手側は左側腹斜筋を想定。"
        if batter_side == "右打者"
        else "右打者研究を左右反転して参考表示。左打者での直接検証ではありません。"
    )

    regions = pd.DataFrame([
        {
            "部位": "腰部・脊柱",
            "判定": low_back_label,
            "主な計測proxy": "肩-骨盤の相対回旋量、インパクト後の回旋速度・急減速",
            "根拠": "腰痛症例報告で体幹側屈・腰椎回旋増大、急激な脊柱運動がストレス要因として考察",
            "エビデンス上の注意": "1症例。側屈は単眼動画では信頼性が低いため本判定から除外",
        },
        {
            "部位": "投手側の脇腹・腹斜筋",
            "判定": oblique_label,
            "主な計測proxy": "インパクト直後の手首速度、体幹回旋速度、フォロースルーの急減速",
            "根拠": "フォロースルーで外腹斜筋の伸張性収縮が示唆され、バットヘッド速度と体幹回旋トルク/パワーが関連",
            "エビデンス上の注意": "バット速度・トルクを直接測定していないためproxy。" + side_note,
        },
    ])

    overall_label = _overall_judgement(regions["判定"].tolist())

    guidance: List[Dict[str, str]] = []
    if low_back_label in {"注意要素あり", "やや注意"}:
        guidance.append({
            "部位": "腰部・脊柱",
            "改善の方向性": (
                "比較選手より体幹の相対回旋またはフォロースルーの急な回旋変化が大きい項目があります。"
                "低強度のティー打撃で、胸郭と骨盤が過度にねじれたまま急停止しないか確認してください。"
            ),
            "練習例": "50〜70%強度のコース別ティー、スロー動画でインパクト後0.3秒の体幹回旋を確認、段階的に強度を上げる",
        })
    if oblique_label in {"注意要素あり", "やや注意"}:
        guidance.append({
            "部位": "脇腹・腹斜筋",
            "改善の方向性": (
                "フォロースルーでの回旋速度・制動proxyの一部が比較選手より高い傾向です。"
                "強振だけを繰り返さず、フォロースルーまで滑らかに回転を逃がす動作を確認してください。"
            ),
            "練習例": "軽いバット/低強度スイング→通常スイングへ段階的に移行、フォロースルーを途中で止めない連続素振り",
        })

    if not guidance:
        guidance.append({
            "部位": "全体",
            "改善の方向性": (
                "今回計測できたproxyでは、比較選手より明確に増大した身体負担関連動作は検出されませんでした。"
                "これは『安全』や『怪我しない』ことを意味しません。"
            ),
            "練習例": "現在のフォームを維持しつつ、疲労や痛みがある日は強度を下げてください。",
        })

    return {
        "overall_label": overall_label,
        "regions": regions,
        "components": components,
        "guidance": pd.DataFrame(guidance),
        "sources": SOURCES,
        "limitations": (
            "本判定は医学的な怪我発生確率ではありません。比較選手との差と先行研究で論点となった動作を"
            "段階表示する参考情報です。撮影角度、MediaPipeの単眼3D推定、インパクト推定誤差の影響を受けます。"
            "痛み・既往歴・疲労・筋力・可動域などは評価していません。"
        ),
    }


def evidence_based_form_tip(lowest_item):
    """提供された実践記事は傷害根拠ではなく、フォーム改善の補足だけに使う。"""
    if lowest_item == "身体を開くな":
        return {
            "title": "身体の開きへの補足",
            "text": (
                "VICTORYの実践解説では、身体の開きを腰の回転開始が早い状態として説明し、"
                "後ろ足側の股関節操作と『アウトエッジランジ』を練習例として紹介しています。"
                "これは傷害リスク研究ではなく、フォーム改善の参考情報です。"
            ),
            "url": SOURCES["opening"]["url"],
        }
    return None
