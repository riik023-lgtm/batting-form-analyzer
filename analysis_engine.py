from __future__ import annotations

import math
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import font_manager

APP_VERSION = "10.0"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "batting_analysis_output"
CONVERTED_DIR = OUTPUT_DIR / "converted_videos"
IMPACT_DIR = OUTPUT_DIR / "impact_frames"
GRAPH_DIR = OUTPUT_DIR / "graphs"
MAX_INPUT_WIDTH = 1280

POSE_MODEL = {
    "filename": "pose_landmarker_heavy.task",
    "url": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
}
HAND_MODEL = {
    "filename": "hand_landmarker.task",
    "url": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

FEATURES = {
    "身体を開くな": {
        "key": "shoulder_opening",
        "unit": "degree",
        "description": "構えからインパクトまでの肩の回転量",
        "tolerance": 18.0,
    },
    "割れを作れ": {
        "key": "hip_shoulder_separation",
        "unit": "degree",
        "description": "インパクト前の肩と骨盤の最大捻転差",
        "tolerance": 15.0,
    },
    "ボールを引きつけろ": {
        "key": "hand_torso_distance",
        "unit": "shoulder-width ratio",
        "description": "インパクト時の手と体幹中心の距離",
        "tolerance": 0.30,
    },
    "軸をぶらすな": {
        "key": "axis_sway",
        "unit": "shoulder-width ratio",
        "description": "画像座標で見た構え〜インパクトの体幹中心の横移動幅",
        "tolerance": 0.35,
    },
    "壁を作れ": {
        "key": "front_knee_angle",
        "unit": "degree",
        "description": "インパクト時の前脚膝角度",
        "tolerance": 20.0,
    },
    "頭を残せ": {
        "key": "head_movement",
        "unit": "shoulder-width ratio",
        "description": "画像座標で見た構え〜インパクトの頭部移動量",
        "tolerance": 0.30,
    },
    "体重移動をしろ": {
        "key": "weight_transfer",
        "unit": "shoulder-width ratio",
        "description": "画像座標で見た骨盤中心の構え〜インパクト水平移動量",
        "tolerance": 0.35,
    },
    "腰を使え": {
        "key": "hip_rotation_speed",
        "unit": "degree/sec",
        "description": "World座標で見たインパクト前後の最大骨盤回転速度",
        "tolerance": 180.0,
    },
}

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

KEY_JOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
]


@dataclass
class VideoAnalysis:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    world_landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    raw_normalized_landmarks: np.ndarray
    valid_mask: np.ndarray
    impact_frame: int
    stance_frame: int
    swing_onset_frame: int
    impact_candidates: List[int]
    features: Dict[str, float]
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    roi_x0: float = 0.0
    roi_x1: float = 1.0
    quality_score: float = 0.0
    quality_label: str = "未評価"
    quality_details: Dict[str, float] = field(default_factory=dict)


def configure_output_directory(base_dir=None):
    global OUTPUT_DIR, CONVERTED_DIR, IMPACT_DIR, GRAPH_DIR
    OUTPUT_DIR = Path(base_dir) if base_dir is not None else BASE_DIR / "batting_analysis_output"
    CONVERTED_DIR = OUTPUT_DIR / "converted_videos"
    IMPACT_DIR = OUTPUT_DIR / "impact_frames"
    GRAPH_DIR = OUTPUT_DIR / "graphs"
    for d in [OUTPUT_DIR, CONVERTED_DIR, IMPACT_DIR, GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


configure_output_directory()


def setup_japanese_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    return None


JAPANESE_FONT_NAME = setup_japanese_font()


def ensure_pose_model() -> Path:
    path = BASE_DIR / POSE_MODEL["filename"]
    if not path.exists():
        urllib.request.urlretrieve(POSE_MODEL["url"], str(path))
    return path


def ensure_hand_model() -> Path:
    path = BASE_DIR / HAND_MODEL["filename"]
    if not path.exists():
        urllib.request.urlretrieve(HAND_MODEL["url"], str(path))
    return path


def can_opencv_read_video(video_path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    ok = cap.isOpened()
    success, frame = cap.read() if ok else (False, None)
    cap.release()
    return bool(ok and success and frame is not None)


def convert_video_to_mp4(input_path, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(output_path),
    ]
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace"
    )
    if p.returncode != 0 or not Path(output_path).exists():
        raise RuntimeError("動画のMP4変換に失敗しました。")
    return Path(output_path)


def prepare_video_for_analysis(original_path, prefix):
    original_path = Path(original_path)
    if original_path.suffix.lower() == ".mp4" and can_opencv_read_video(original_path):
        return original_path
    converted = CONVERTED_DIR / f"{prefix}_converted.mp4"
    convert_video_to_mp4(original_path, converted)
    if not can_opencv_read_video(converted):
        raise RuntimeError("MP4変換後も動画を読み込めませんでした。")
    return converted


def midpoint(a, b):
    return (np.asarray(a) + np.asarray(b)) / 2.0


def joint_angle(a, b, c):
    ba = np.asarray(a[:3], dtype=float) - np.asarray(b[:3], dtype=float)
    bc = np.asarray(c[:3], dtype=float) - np.asarray(b[:3], dtype=float)
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-8:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def horizontal_line_angle(left_point, right_point):
    dx = right_point[0] - left_point[0]
    dz = right_point[2] - left_point[2]
    return float(np.degrees(np.arctan2(dz, dx)))


def image_line_angle(left_point, right_point):
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def circular_angle_difference(a, b):
    return np.abs(np.degrees(np.arctan2(
        np.sin(np.radians(np.asarray(a) - np.asarray(b))),
        np.cos(np.radians(np.asarray(a) - np.asarray(b))),
    )))


def signed_angle_difference(a, b):
    return np.degrees(np.arctan2(
        np.sin(np.radians(np.asarray(a) - np.asarray(b))),
        np.cos(np.radians(np.asarray(a) - np.asarray(b))),
    ))


def moving_average(values, window=5):
    return pd.Series(np.asarray(values, dtype=float)).rolling(
        window=max(1, int(window)), center=True, min_periods=1
    ).mean().to_numpy()


def _safe_nanmax(values, default=np.nan):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float(default)


def _safe_nanmedian(values, default=np.nan):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float(default)


def interpolate_landmarks(landmark_array, valid_mask):
    result = np.asarray(landmark_array, dtype=float).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    frames = np.arange(len(result))
    if len(result) == 0:
        return result
    for landmark in range(result.shape[1]):
        for dim in range(3):
            values = result[:, landmark, dim]
            good = valid_mask & np.isfinite(values)
            if good.sum() >= 2:
                result[:, landmark, dim] = np.interp(frames, frames[good], values[good])
    return result


def _pose_to_arrays(world_pose, normalized_pose):
    world = np.array([
        [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in world_pose
    ], dtype=float)
    norm = np.array([
        [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in normalized_pose
    ], dtype=float)
    return world, norm


def _angle_2d(a, b, c):
    ba = np.asarray(a[:2]) - np.asarray(b[:2])
    bc = np.asarray(c[:2]) - np.asarray(b[:2])
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))


def _batter_pose_score(norm):
    required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
                LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    points = norm[required]
    if not np.all(np.isfinite(points[:, :2])):
        return -1e9
    visibility = float(np.nanmean(points[:, 3]))
    ls, rs = norm[LEFT_SHOULDER], norm[RIGHT_SHOULDER]
    lw, rw = norm[LEFT_WRIST], norm[RIGHT_WRIST]
    lh, rh = norm[LEFT_HIP], norm[RIGHT_HIP]
    sh = midpoint(ls[:2], rs[:2])
    hip = midpoint(lh[:2], rh[:2])
    wrist = midpoint(lw[:2], rw[:2])
    shoulder_width = max(float(np.linalg.norm(ls[:2] - rs[:2])), 1e-4)
    torso_h = max(float(abs(hip[1] - sh[1])), 1e-4)
    hand_close = float(np.clip(1.4 - np.linalg.norm(lw[:2] - rw[:2]) / shoulder_width, 0, 1))
    hand_height = float(np.clip((hip[1] - wrist[1]) / torso_h, -0.2, 1.4))
    hand_height_score = max(0.0, hand_height / 1.4)
    left_knee = _angle_2d(norm[LEFT_HIP], norm[LEFT_KNEE], norm[LEFT_ANKLE])
    right_knee = _angle_2d(norm[RIGHT_HIP], norm[RIGHT_KNEE], norm[RIGHT_ANKLE])
    knees = [v for v in [left_knee, right_knee] if np.isfinite(v)]
    knee_extension = float(np.clip((np.mean(knees) - 65) / 95, 0, 1)) if knees else 0
    ankle = midpoint(norm[LEFT_ANKLE, :2], norm[RIGHT_ANKLE, :2])
    standing = float(np.clip(abs(ankle[1] - hip[1]) / max(torso_h * 1.5, 1e-4), 0, 1))
    # 捕手は深い屈曲＋手が低いことが多いため、その両方を弱く減点。
    crouch_penalty = 0.8 if knee_extension < 0.25 and hand_height_score < 0.20 else 0.0
    return 2.2 * hand_close + 1.8 * hand_height_score + 1.8 * knee_extension + 0.8 * standing + 0.7 * visibility - crouch_penalty


def _select_batter_pose(result, previous_center=None):
    if not result.pose_world_landmarks or not result.pose_landmarks:
        return None, None, None
    candidates = []
    count = min(len(result.pose_world_landmarks), len(result.pose_landmarks))
    for i in range(count):
        world, norm = _pose_to_arrays(result.pose_world_landmarks[i], result.pose_landmarks[i])
        hip = midpoint(norm[LEFT_HIP, :2], norm[RIGHT_HIP, :2])
        if not np.all(np.isfinite(hip)):
            continue
        score = _batter_pose_score(norm)
        if previous_center is not None and np.all(np.isfinite(previous_center)):
            dist = float(np.linalg.norm(hip - previous_center))
            score += 4.8 * math.exp(-0.5 * (dist / 0.11) ** 2)
        candidates.append((score, world, norm, hip))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, world, norm, hip = candidates[0]
    return world, norm, hip


def _normalize_batter_roi(roi):
    if not roi:
        return 0.0, 1.0
    center = float(np.clip(roi.get("center", 0.5), 0, 1))
    width = float(np.clip(roi.get("width", 0.6), 0.20, 1.0))
    x0 = max(0.0, center - width / 2)
    x1 = min(1.0, center + width / 2)
    if x1 - x0 < width:
        if x0 <= 0:
            x1 = min(1.0, width)
        else:
            x0 = max(0.0, 1.0 - width)
    return float(x0), float(x1)


def _map_crop_to_full(norm, x0, x1):
    mapped = norm.copy()
    mapped[:, 0] = x0 + mapped[:, 0] * max(x1 - x0, 1e-6)
    return mapped


def _is_temporally_consistent(current, previous):
    if previous is None:
        return True
    stable = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    if not np.all(np.isfinite(current[stable, :2])):
        return False
    if not np.all(np.isfinite(previous[stable, :2])):
        return True
    cur_torso = np.mean(current[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], :2], axis=0)
    prev_torso = np.mean(previous[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], :2], axis=0)
    if float(np.linalg.norm(cur_torso - prev_torso)) > 0.20:
        return False
    return True


def _longest_true_run(mask):
    best = current = 0
    for v in np.asarray(mask, dtype=bool):
        if v:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _nearest_valid_frame(frame, valid_mask, max_distance):
    valid = np.asarray(valid_mask, dtype=bool)
    if len(valid) == 0:
        return None
    frame = int(np.clip(frame, 0, len(valid) - 1))
    if valid[frame]:
        return frame
    for d in range(1, max(0, int(max_distance)) + 1):
        for idx in (frame - d, frame + d):
            if 0 <= idx < len(valid) and valid[idx]:
                return idx
    return None


def _masked_speed(points, fps, valid_mask):
    points = np.asarray(points, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    speed = np.full(len(points), np.nan, dtype=float)
    if len(points) >= 2:
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
        good = valid[:-1] & valid[1:]
        speed[1:][good] = diffs[good]
    return speed


def _masked_angle_velocity(angles, fps, valid_mask):
    angles = np.asarray(angles, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    unwrapped = np.degrees(np.unwrap(np.radians(angles)))
    velocity = np.full(len(angles), np.nan, dtype=float)
    if len(angles) >= 2:
        diff = np.diff(unwrapped) * fps
        good = valid[:-1] & valid[1:]
        velocity[1:][good] = diff[good]
    return velocity


def _robust_scale(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [20, 95])
    den = max(float(hi - lo), 1e-9)
    out = (arr - lo) / den
    return np.clip(out, 0.0, 1.0)


def _motion_signals(world, norm, valid_mask, fps):
    left_wrist = world[:, LEFT_WRIST, :3]
    right_wrist = world[:, RIGHT_WRIST, :3]
    wrist_center = midpoint(left_wrist, right_wrist)
    wrist_speed = 0.35 * _masked_speed(left_wrist, fps, valid_mask) + 0.35 * _masked_speed(right_wrist, fps, valid_mask) + 0.30 * _masked_speed(wrist_center, fps, valid_mask)

    sh_angles = np.array([horizontal_line_angle(f[LEFT_SHOULDER], f[RIGHT_SHOULDER]) for f in world])
    hip_angles = np.array([horizontal_line_angle(f[LEFT_HIP], f[RIGHT_HIP]) for f in world])
    sh_speed = np.abs(_masked_angle_velocity(sh_angles, fps, valid_mask))
    hip_speed = np.abs(_masked_angle_velocity(hip_angles, fps, valid_mask))

    # 画像上の身体中心速度も加え、World原点依存のない移動情報を利用。
    torso2d = midpoint(
        midpoint(norm[:, LEFT_SHOULDER, :2], norm[:, RIGHT_SHOULDER, :2]),
        midpoint(norm[:, LEFT_HIP, :2], norm[:, RIGHT_HIP, :2]),
    )
    torso_speed2d = _masked_speed(torso2d, fps, valid_mask)

    win = max(3, int(round(fps * 0.06)))
    if win % 2 == 0:
        win += 1
    signals = {
        "wrist_speed": moving_average(wrist_speed, win),
        "shoulder_speed": moving_average(sh_speed, win),
        "hip_speed": moving_average(hip_speed, win),
        "torso_speed2d": moving_average(torso_speed2d, win),
        "shoulder_angles": sh_angles,
        "hip_angles": hip_angles,
    }
    return signals


def estimate_impact_candidates(world, norm, fps, valid_mask, top_k=3):
    """手首速度だけでなく、肩・骨盤回転と身体移動を合わせた候補推定。"""
    signals = _motion_signals(world, norm, valid_mask, fps)
    n = len(world)
    if n <= 2:
        return [0], 0

    wrist_n = _robust_scale(signals["wrist_speed"])
    shoulder_n = _robust_scale(signals["shoulder_speed"])
    hip_n = _robust_scale(signals["hip_speed"])
    torso_n = _robust_scale(signals["torso_speed2d"])

    combined = 0.50 * wrist_n + 0.22 * shoulder_n + 0.20 * hip_n + 0.08 * torso_n
    valid = np.asarray(valid_mask, dtype=bool)
    combined[~valid] = np.nan

    search_start = max(1, int(round(0.05 * n)))
    search_end = min(n - 2, int(round(0.95 * n)))
    finite = combined[search_start:search_end + 1]
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        idx = int(np.flatnonzero(valid)[0]) if valid.any() else 0
        return [idx], idx

    # スイング開始: 序盤基準より明確に上がった最初の連続区間。
    baseline_end = max(search_start + 2, int(round(n * 0.35)))
    baseline = _safe_nanmedian(combined[search_start:baseline_end], 0.0)
    threshold = max(float(baseline) + 0.18, float(np.nanmax(finite)) * 0.25)
    onset = search_start
    run = 0
    needed = max(2, int(round(fps * 0.04)))
    for i in range(search_start, search_end + 1):
        if np.isfinite(combined[i]) and combined[i] >= threshold:
            run += 1
            if run >= needed:
                onset = max(search_start, i - run + 1)
                break
        else:
            run = 0

    # 手首速度が高く、肩・腰回転も立ち上がっている局所ピークを候補にする。
    peaks = []
    peak_floor = max(0.42, float(np.nanmax(finite)) * 0.62)
    candidate_start = max(
        onset + max(2, int(round(0.10 * fps))),
        search_start + 1,
    )
    for i in range(candidate_start, search_end):
        if not np.isfinite(combined[i]):
            continue
        if combined[i] < peak_floor:
            continue
        if combined[i] >= combined[i - 1] and combined[i] >= combined[i + 1]:
            # 後半フォロースルーを少しだけ不利にする。
            progress = (i - onset) / max(1, search_end - onset)
            score = float(combined[i] - 0.10 * max(progress - 0.45, 0.0))
            peaks.append((score, i))

    if not peaks:
        best = int(np.nanargmax(combined[search_start:search_end + 1])) + search_start
        peaks = [(float(combined[best]), best)]

    peaks.sort(key=lambda x: x[0], reverse=True)
    candidates = []
    min_gap = max(2, int(round(fps * 0.06)))
    for _, idx in peaks:
        if all(abs(idx - j) >= min_gap for j in candidates):
            snapped = _nearest_valid_frame(idx, valid, max(2, int(round(0.08 * fps))))
            if snapped is not None:
                candidates.append(int(snapped))
        if len(candidates) >= top_k:
            break

    if not candidates:
        candidates = [int(_nearest_valid_frame(onset, valid, max(2, int(round(0.1 * fps)))) or onset)]

    # 表示は時系列順でなく、最有力→次点の順。自動採用は最有力。
    return candidates, int(onset)


def estimate_stance_frame(world, norm, valid_mask, impact_frame, onset_frame, fps):
    """固定0.8秒前ではなく、スイング開始前の低動作フレームから構えを選ぶ。"""
    valid = np.asarray(valid_mask, dtype=bool)
    signals = _motion_signals(world, norm, valid, fps)
    wrist = signals["wrist_speed"]
    hip = signals["hip_speed"]
    sh = signals["shoulder_speed"]
    motion = _robust_scale(wrist) + 0.4 * _robust_scale(hip) + 0.4 * _robust_scale(sh)

    start = max(0, int(impact_frame - round(1.20 * fps)))
    end = min(max(start + 1, onset_frame), int(impact_frame - round(0.12 * fps)))
    if end <= start:
        fallback = impact_frame - int(round(0.8 * fps))
        return int(_nearest_valid_frame(fallback, valid, max(3, int(round(0.15 * fps)))) or max(0, fallback))

    indices = [i for i in range(start, end + 1) if valid[i] and np.isfinite(motion[i])]
    if not indices:
        fallback = impact_frame - int(round(0.8 * fps))
        return int(_nearest_valid_frame(fallback, valid, max(3, int(round(0.15 * fps)))) or max(0, fallback))
    # 低動作かつなるべくスイング開始に近いフレームを優先。
    scores = [(float(motion[i]) + 0.12 * (end - i) / max(1, end - start), i) for i in indices]
    scores.sort(key=lambda x: x[0])
    return int(scores[0][1])


def _analysis_quality(valid_mask, raw_norm, impact_frame, fps):
    valid = np.asarray(valid_mask, dtype=bool)
    n = len(valid)
    if n == 0:
        return {"score": 0.0, "label": "解析不可", "overall_detection_rate": 0.0,
                "impact_window_detection_rate": 0.0, "impact_window_longest_run": 0,
                "impact_window_transition_rate": 0.0, "key_joint_visibility": 0.0}
    pre = max(0, impact_frame - int(round(0.70 * fps)))
    post = min(n, impact_frame + int(round(0.30 * fps)) + 1)
    wm = valid[pre:post]
    overall = float(valid.mean())
    impact_rate = float(wm.mean()) if len(wm) else 0.0
    longest = _longest_true_run(wm)
    transition = float((wm[:-1] & wm[1:]).mean()) if len(wm) >= 2 else 0.0
    vis = 0.0
    try:
        sec = np.asarray(raw_norm)[pre:post][:, KEY_JOINTS, 3]
        finite = sec[np.isfinite(sec)]
        if finite.size:
            vis = float(np.clip(np.nanmean(finite), 0, 1))
    except Exception:
        pass
    score = 100 * (
        0.10 * min(overall / 0.75, 1.0)
        + 0.35 * min(impact_rate / 0.85, 1.0)
        + 0.20 * min(longest / max(1.0, 0.25 * fps), 1.0)
        + 0.15 * min(transition / 0.75, 1.0)
        + 0.20 * np.clip((vis - 0.35) / 0.50, 0, 1)
    )
    label = "高" if score >= 80 else "要注意" if score >= 60 else "解析不可"
    return {
        "score": round(float(score), 1), "label": label,
        "overall_detection_rate": overall,
        "impact_window_detection_rate": impact_rate,
        "impact_window_longest_run": int(longest),
        "impact_window_transition_rate": transition,
        "key_joint_visibility": vis,
        "impact_window_start": pre,
        "impact_window_end": max(pre, post - 1),
    }


def _assert_analysis_quality(q, fps):
    problems = []
    if q["impact_window_detection_rate"] < 0.55:
        problems.append("インパクト前後の骨格検出率が55%未満")
    if q["impact_window_longest_run"] < max(3, int(round(0.12 * fps))):
        problems.append("インパクト前後の連続検出区間が短い")
    if q["impact_window_transition_rate"] < 0.45:
        problems.append("実測フレーム間の連続性が不足")
    if q["key_joint_visibility"] < 0.45:
        problems.append("主要関節visibilityが低い")
    if q["score"] < 60:
        problems.append("解析信頼度が60点未満")
    if problems:
        raise ValueError(
            "この動画は解析品質が不足しているため、数値化を停止しました。\n"
            f"解析信頼度: {q['score']:.1f}/100（{q['label']}）\n"
            + " / ".join(problems)
            + "\n打者が大きく映る2〜5秒程度の動画にし、必要なら解析範囲を調整してください。"
        )


def _shoulder_width_2d(norm, frame):
    return max(float(np.linalg.norm(norm[frame, LEFT_SHOULDER, :2] - norm[frame, RIGHT_SHOULDER, :2])), 1e-5)


def _body_center_proxy(xy):
    """画像座標の複数部位から作る2D身体中心proxy。真の身体重心ではない。"""
    shoulder = midpoint(xy[:, LEFT_SHOULDER, :2], xy[:, RIGHT_SHOULDER, :2])
    hip = midpoint(xy[:, LEFT_HIP, :2], xy[:, RIGHT_HIP, :2])
    knees = midpoint(xy[:, LEFT_KNEE, :2], xy[:, RIGHT_KNEE, :2])
    # 体幹を重めにした単純proxy。生体力学的な真のCOMとは呼ばない。
    return 0.30 * shoulder + 0.50 * hip + 0.20 * knees


def calculate_features(world, norm, fps, impact_frame, stance_frame, onset_frame, batter_side, valid_mask, image_aspect_ratio=1.0):
    valid = np.asarray(valid_mask, dtype=bool)
    shoulder_width_world = max(float(np.linalg.norm(world[stance_frame, LEFT_SHOULDER, :3] - world[stance_frame, RIGHT_SHOULDER, :3])), 1e-6)
    # normalized x/y は単位が異なるため、xを幅/高さ比で補正してから2D距離・角度を計算する。
    xy = np.asarray(norm[:, :, :2], dtype=float).copy()
    xy[:, :, 0] *= max(float(image_aspect_ratio), 1e-6)
    shoulder_width_2d = max(float(np.linalg.norm(xy[stance_frame, LEFT_SHOULDER] - xy[stance_frame, RIGHT_SHOULDER])), 1e-5)

    lead_hip, lead_knee, lead_ankle = (
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE) if batter_side == "右打者"
        else (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    )

    sh_angles = np.array([horizontal_line_angle(f[LEFT_SHOULDER], f[RIGHT_SHOULDER]) for f in world])
    hip_angles = np.array([horizontal_line_angle(f[LEFT_HIP], f[RIGHT_HIP]) for f in world])
    sh_unwrap = np.degrees(np.unwrap(np.radians(sh_angles)))
    hip_unwrap = np.degrees(np.unwrap(np.radians(hip_angles)))

    shoulder_opening = abs(float(sh_unwrap[impact_frame] - sh_unwrap[stance_frame]))

    pre_start = max(stance_frame, impact_frame - int(round(0.7 * fps)))
    separation = circular_angle_difference(sh_angles, hip_angles).astype(float)
    sec = separation[pre_start:impact_frame + 1]
    sec_valid = valid[pre_start:impact_frame + 1]
    sec[~sec_valid] = np.nan
    hip_shoulder_separation = _safe_nanmax(sec)

    shoulder_centers_world = midpoint(world[:, LEFT_SHOULDER, :3], world[:, RIGHT_SHOULDER, :3])
    hip_centers_world = midpoint(world[:, LEFT_HIP, :3], world[:, RIGHT_HIP, :3])
    hand_centers_world = midpoint(world[:, LEFT_WRIST, :3], world[:, RIGHT_WRIST, :3])
    torso_world = midpoint(shoulder_centers_world, hip_centers_world)
    hand_torso_distance = float(np.linalg.norm(hand_centers_world[impact_frame] - torso_world[impact_frame]) / shoulder_width_world)

    shoulder_centers_2d = midpoint(xy[:, LEFT_SHOULDER, :2], xy[:, RIGHT_SHOULDER, :2])
    hip_centers_2d = midpoint(xy[:, LEFT_HIP, :2], xy[:, RIGHT_HIP, :2])
    torso_2d = midpoint(shoulder_centers_2d, hip_centers_2d)

    torso_x = torso_2d[stance_frame:impact_frame + 1, 0].astype(float)
    torso_x[~valid[stance_frame:impact_frame + 1]] = np.nan
    axis_sway = float((_safe_nanmax(torso_x) - np.nanmin(torso_x[np.isfinite(torso_x)])) / shoulder_width_2d) if np.isfinite(torso_x).any() else np.nan

    front_knee_angle = joint_angle(world[impact_frame, lead_hip], world[impact_frame, lead_knee], world[impact_frame, lead_ankle])

    head2d = xy[:, NOSE, :2]
    head_sec = head2d[stance_frame:impact_frame + 1]
    head_dist = np.linalg.norm(head_sec - head2d[stance_frame], axis=1).astype(float)
    head_dist[~valid[stance_frame:impact_frame + 1]] = np.nan
    head_movement = _safe_nanmax(head_dist) / shoulder_width_2d

    # v10.0の重要修正: 体重移動はWorld座標でなく画像上の骨盤中心の絶対移動を使う。
    weight_transfer_signed = float((hip_centers_2d[impact_frame, 0] - hip_centers_2d[stance_frame, 0]) / shoulder_width_2d)
    weight_transfer = abs(weight_transfer_signed)

    hip_velocity = _masked_angle_velocity(hip_angles, fps, valid)
    vel_start = max(stance_frame, impact_frame - int(round(0.5 * fps)))
    vel_end = min(len(hip_velocity), impact_frame + int(round(0.15 * fps)) + 1)
    hip_rotation_speed = _safe_nanmax(np.abs(hip_velocity[vel_start:vel_end]))

    # ---- 新しい補助指標 ----
    shoulder_velocity = _masked_angle_velocity(sh_angles, fps, valid)
    shoulder_rotation_speed = _safe_nanmax(np.abs(shoulder_velocity[vel_start:vel_end]))

    body_center = _body_center_proxy(xy)
    body_shift = float(np.linalg.norm(body_center[impact_frame] - body_center[stance_frame]) / shoulder_width_2d)
    body_speed = _masked_speed(body_center, fps, valid) / shoulder_width_2d
    body_speed_peak = _safe_nanmax(body_speed[max(stance_frame, onset_frame):impact_frame + 1])

    # 体幹側屈proxy: 画像上の肩中心→骨盤中心ベクトルの鉛直からの傾き。
    torso_vec = shoulder_centers_2d - hip_centers_2d
    torso_side_angles = np.degrees(np.arctan2(torso_vec[:, 0], -torso_vec[:, 1]))
    torso_side_bend_impact = float(torso_side_angles[impact_frame])
    torso_side_bend_change = abs(float(torso_side_angles[impact_frame] - torso_side_angles[stance_frame]))

    pelvis_tilt_angles = np.degrees(np.arctan2(
        xy[:, RIGHT_HIP, 1] - xy[:, LEFT_HIP, 1],
        xy[:, RIGHT_HIP, 0] - xy[:, LEFT_HIP, 0],
    ))
    pelvis_tilt_impact = float(pelvis_tilt_angles[impact_frame])
    pelvis_tilt_change = abs(float(pelvis_tilt_angles[impact_frame] - pelvis_tilt_angles[stance_frame]))

    # 運動連鎖: インパクト前0.6秒〜直後0.1秒の骨盤/肩速度ピーク時刻差。
    chain_start = max(0, impact_frame - int(round(0.60 * fps)))
    chain_end = min(len(world), impact_frame + int(round(0.10 * fps)) + 1)
    hip_abs = np.abs(hip_velocity[chain_start:chain_end])
    sh_abs = np.abs(shoulder_velocity[chain_start:chain_end])
    hip_peak_frame = chain_start + int(np.nanargmax(hip_abs)) if np.isfinite(hip_abs).any() else impact_frame
    shoulder_peak_frame = chain_start + int(np.nanargmax(sh_abs)) if np.isfinite(sh_abs).any() else impact_frame
    sequence_delay_ms = float((shoulder_peak_frame - hip_peak_frame) / fps * 1000.0)

    swing_time_sec = max(0.0, float((impact_frame - onset_frame) / fps))

    return {
        "shoulder_opening": float(shoulder_opening),
        "hip_shoulder_separation": float(hip_shoulder_separation),
        "hand_torso_distance": float(hand_torso_distance),
        "axis_sway": float(axis_sway),
        "front_knee_angle": float(front_knee_angle),
        "head_movement": float(head_movement),
        "weight_transfer": float(weight_transfer),
        "hip_rotation_speed": float(hip_rotation_speed),
    }, {
        "weight_transfer_signed": weight_transfer_signed,
        "body_center_shift_proxy": body_shift,
        "body_center_speed_proxy": body_speed_peak,
        "torso_side_bend_impact_deg": torso_side_bend_impact,
        "torso_side_bend_change_deg": torso_side_bend_change,
        "pelvis_tilt_impact_deg": pelvis_tilt_impact,
        "pelvis_tilt_change_deg": pelvis_tilt_change,
        "shoulder_rotation_speed_deg_s": shoulder_rotation_speed,
        "hip_rotation_speed_deg_s": hip_rotation_speed,
        "hip_peak_frame": float(hip_peak_frame),
        "shoulder_peak_frame": float(shoulder_peak_frame),
        "hip_to_shoulder_peak_delay_ms": sequence_delay_ms,
        "swing_time_sec": swing_time_sec,
    }


def _draw_pose_on_frame(frame, norm):
    h, w = frame.shape[:2]
    pairs = [
        (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP),
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    pts = {}
    for idx in set(sum(([a, b] for a, b in pairs), [])) | {NOSE}:
        if idx < len(norm) and np.all(np.isfinite(norm[idx, :2])):
            pts[idx] = (int(norm[idx, 0] * w), int(norm[idx, 1] * h))
    for a, b in pairs:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 80), 3, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(frame, p, 4, (0, 60, 255), -1, cv2.LINE_AA)
    return frame


def save_pose_frame_image(video_path, frame_number, norm, output_path, label):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return
    frame = _draw_pose_on_frame(frame, norm)
    cv2.putText(frame, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), frame)


def get_pose_preview_frame(analysis, frame_number):
    frame_number = int(np.clip(int(frame_number), 0, max(0, analysis.frame_count - 1)))
    cap = cv2.VideoCapture(str(analysis.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    norm = analysis.normalized_landmarks[frame_number]
    if np.isfinite(norm[:, :3]).any():
        frame = _draw_pose_on_frame(frame, norm)
    cv2.putText(frame, f"FRAME {frame_number}  {frame_number / analysis.fps:.3f}s", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_video_preview_frame(video_path, frame_ratio=0.35):
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        return None
    idx = int(np.clip(round((count - 1) * float(frame_ratio)), 0, count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_roi_preview(image_rgb, center, width):
    if image_rgb is None:
        return None
    image = image_rgb.copy()
    h, w = image.shape[:2]
    x0, x1 = _normalize_batter_roi({"center": center, "width": width})
    px0, px1 = int(x0 * w), int(x1 * w)
    if px0 > 0:
        image[:, :px0] = (image[:, :px0] * 0.35).astype(np.uint8)
    if px1 < w:
        image[:, px1:] = (image[:, px1:] * 0.35).astype(np.uint8)
    cv2.rectangle(image, (px0, 2), (max(px0 + 1, px1 - 1), h - 3), (255, 55, 55), 5)
    return image


def suggest_batter_roi(video_path, frame_ratio=0.35):
    """1枚のプレビューから打者候補を推定し、横ROIを返す。失敗時は中央60%。"""
    frame_rgb = get_video_preview_frame(video_path, frame_ratio)
    if frame_rgb is None:
        return {"center": 0.5, "width": 0.60, "confidence": 0.0}
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    if w > MAX_INPUT_WIDTH:
        scale = MAX_INPUT_WIDTH / w
        frame_bgr = cv2.resize(frame_bgr, (MAX_INPUT_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Options = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        output_segmentation_masks=False,
    )
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
        world, norm, _ = _select_batter_pose(result)
        if world is None or norm is None:
            raise ValueError("no pose")
        visible = norm[(norm[:, 3] >= 0.35) & np.isfinite(norm[:, 0])]
        if len(visible) < 6:
            visible = norm[np.isfinite(norm[:, 0])]
        x_min, x_max = float(np.min(visible[:, 0])), float(np.max(visible[:, 0]))
        center = float(np.clip((x_min + x_max) / 2, 0.1, 0.9))
        width = float(np.clip((x_max - x_min) * 1.55, 0.34, 0.72))
        return {"center": center, "width": width, "confidence": 1.0}
    except Exception:
        return {"center": 0.5, "width": 0.60, "confidence": 0.0}




def _angle_between_2d(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / den, -1.0, 1.0))))


def _hand_wrist_alignment_metrics(video_path, frame_number, pose_norm, roi_x0, roi_x1):
    """
    Hand Landmarkerをインパクト付近の1フレームだけに使う。
    前腕（肘→手首）と手掌（手首→中指MCP）の2D方向差をproxyとして返す。
    真の橈屈/尺屈角度ではないため、補助指標としてのみ使用する。
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {}
    h, w = frame.shape[:2]
    px0 = int(np.clip(round(roi_x0 * w), 0, max(0, w - 2)))
    px1 = int(np.clip(round(roi_x1 * w), px0 + 2, w))
    crop = frame[:, px0:px1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    if not result.hand_landmarks:
        return {}

    crop_width = max(float(roi_x1 - roi_x0), 1e-6)
    hands = []
    for hand in result.hand_landmarks:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=float)
        arr[:, 0] = roi_x0 + arr[:, 0] * crop_width
        hands.append(arr)

    metrics = {}
    used = set()
    for side_name, wrist_idx, elbow_idx in [
        ("left", LEFT_WRIST, LEFT_ELBOW),
        ("right", RIGHT_WRIST, RIGHT_ELBOW),
    ]:
        aspect = w / max(h, 1)
        pose_wrist = pose_norm[wrist_idx, :2].copy()
        pose_elbow = pose_norm[elbow_idx, :2].copy()
        pose_wrist[0] *= aspect
        pose_elbow[0] *= aspect
        best_i, best_dist = None, float("inf")
        for i, hand in enumerate(hands):
            if i in used:
                continue
            hand_wrist_for_match = hand[0, :2].copy()
            hand_wrist_for_match[0] *= aspect
            dist = float(np.linalg.norm(hand_wrist_for_match - pose_wrist))
            if dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is None or best_dist > 0.18:
            continue
        used.add(best_i)
        hand = hands[best_i]
        forearm = pose_wrist - pose_elbow
        palm_start = hand[0, :2].copy()
        palm_end = hand[9, :2].copy()
        palm_start[0] *= aspect
        palm_end[0] *= aspect
        palm = palm_end - palm_start  # wrist -> middle MCP
        angle = _angle_between_2d(forearm, palm)
        if np.isfinite(angle):
            metrics[f"{side_name}_wrist_alignment_proxy_deg"] = float(angle)
            metrics[f"{side_name}_hand_match_distance"] = best_dist
    return metrics


def analyze_video(video_path, batter_side, prefix, batter_roi=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Options = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=RunningMode.VIDEO,
        num_poses=2 if batter_roi else 4,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.40,
        output_segmentation_masks=False,
    )

    x0, x1 = _normalize_batter_roi(batter_roi)
    world_list, norm_list, valid_list = [], [], []
    previous_center = None
    previous_norm = None
    reject_run = 0
    frame_no = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            px0 = int(np.clip(round(x0 * w), 0, max(0, w - 2)))
            px1 = int(np.clip(round(x1 * w), px0 + 2, w))
            crop = frame[:, px0:px1]
            if crop.shape[1] > MAX_INPUT_WIDTH:
                scale = MAX_INPUT_WIDTH / crop.shape[1]
                crop = cv2.resize(crop, (MAX_INPUT_WIDTH, max(1, int(round(crop.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(round(frame_no * 1000.0 / fps))
            result = landmarker.detect_for_video(image, timestamp)
            world, norm_crop, center = _select_batter_pose(result, previous_center)
            raw_found = world is not None and norm_crop is not None
            temporal_ok = raw_found and _is_temporally_consistent(norm_crop, previous_norm)
            pose_ok = bool(temporal_ok)
            if raw_found and not temporal_ok:
                reject_run += 1
                if reject_run >= 2 and _batter_pose_score(norm_crop) >= 2.0:
                    pose_ok = True
            elif raw_found:
                reject_run = 0
            else:
                reject_run += 1
                if reject_run >= 3:
                    previous_center = None
                    previous_norm = None

            if pose_ok:
                mapped = _map_crop_to_full(norm_crop, x0, x1)
                world_list.append(world)
                norm_list.append(mapped)
                valid_list.append(True)
                previous_center = center
                previous_norm = norm_crop.copy()
                reject_run = 0
            else:
                world_list.append(np.full((33, 4), np.nan))
                norm_list.append(np.full((33, 4), np.nan))
                valid_list.append(False)
            frame_no += 1
    cap.release()

    if frame_no < 5:
        raise ValueError("動画のフレーム数が少なすぎます。")
    valid = np.asarray(valid_list, dtype=bool)
    valid_count = int(valid.sum())
    longest = _longest_true_run(valid)
    if valid_count < max(10, int(round(fps * 0.60))) or longest < max(5, int(round(fps * 0.18))):
        raise ValueError(
            "打撃解析に必要な骨格フレームを十分に確保できませんでした。\n"
            f"骨格検出率: {valid.mean():.1%}\n"
            "打者が大きく映るよう解析範囲を少し狭めるか、2〜5秒程度の動画にしてください。"
        )

    raw_norm = np.asarray(norm_list, dtype=float)
    world = interpolate_landmarks(np.asarray(world_list), valid)
    norm = interpolate_landmarks(raw_norm, valid)

    candidates, onset = estimate_impact_candidates(world, norm, fps, valid, top_k=3)
    impact = int(candidates[0])
    stance = estimate_stance_frame(world, norm, valid, impact, onset, fps)
    q = _analysis_quality(valid, raw_norm, impact, fps)
    _assert_analysis_quality(q, fps)
    features, extra = calculate_features(
        world, norm, fps, impact, stance, onset, batter_side, valid,
        image_aspect_ratio=(width / max(height, 1)),
    )
    # 手関節はPoseだけでは詳細が不足するため、インパクト1フレームだけHand Landmarkerを追加利用。
    try:
        extra.update(_hand_wrist_alignment_metrics(video_path, impact, norm[impact], x0, x1))
    except Exception:
        # 手が隠れる・モデル取得不可などでは全体解析を止めない。
        pass

    analysis = VideoAnalysis(
        path=str(video_path), fps=fps, frame_count=frame_no, width=width, height=height,
        world_landmarks=world, normalized_landmarks=norm, raw_normalized_landmarks=raw_norm,
        valid_mask=valid, impact_frame=impact, stance_frame=stance, swing_onset_frame=onset,
        impact_candidates=[int(v) for v in candidates], features=features, extra_metrics=extra,
        roi_x0=x0, roi_x1=x1, quality_score=float(q["score"]), quality_label=str(q["label"]), quality_details=q,
    )
    save_pose_frame_image(video_path, stance, norm[stance], IMPACT_DIR / f"{prefix}_stance_pose.jpg", "STANCE")
    save_pose_frame_image(video_path, impact, norm[impact], IMPACT_DIR / f"{prefix}_impact_pose.jpg", "IMPACT")
    return analysis


def similarity_score(reference_value, target_value, tolerance):
    if not np.isfinite(reference_value) or not np.isfinite(target_value):
        return np.nan
    diff = abs(float(target_value) - float(reference_value))
    # v10: 測定ノイズで極端な0点に落ちにくいよう、指数をやや緩やかに。
    z = diff / max(float(tolerance), 1e-9)
    score = 100.0 * math.exp(-0.5 * (z ** 1.6))
    return float(np.clip(score, 0, 100))


def build_score_table(reference_analysis, target_analysis, selected_names):
    rows = []
    for name in selected_names:
        cfg = FEATURES[name]
        ref = reference_analysis.features[cfg["key"]]
        tar = target_analysis.features[cfg["key"]]
        score = similarity_score(ref, tar, cfg["tolerance"])
        rows.append({
            "評価項目": name,
            "計測内容": cfg["description"],
            "比較選手": round(float(ref), 3),
            "自分": round(float(tar), 3),
            "単位": cfg["unit"],
            "差の絶対値": round(abs(float(tar) - float(ref)), 3),
            "得点": round(float(score), 1),
        })
    return pd.DataFrame(rows)


def save_radar_chart(df):
    labels = df["評価項目"].tolist()
    scores = df["得点"].astype(float).tolist()
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + angles[:1]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_c, [100] * (n + 1), linestyle="--", marker="o", label="比較選手の基準")
    ax.plot(angles_c, scores + scores[:1], linewidth=2.5, marker="o", label="自分の類似度")
    ax.fill(angles_c, scores + scores[:1], alpha=0.20)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(f"打撃フォーム比較\n総合類似度: {df['得点'].mean():.1f}点", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "打撃フォーム比較レーダーチャート.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_bar_chart(df):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    labels = df["評価項目"].tolist()
    scores = df["得点"].astype(float).tolist()
    bars = ax.bar(labels, scores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("比較選手との類似度（点）")
    ax.set_title("評価項目別のフォーム類似度")
    ax.tick_params(axis="x", rotation=20)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2, f"{score:.1f}", ha="center")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "評価項目別類似度棒グラフ.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_improvement_advice(score_df, ref, tar):
    rows = []
    scores = dict(zip(score_df["評価項目"], score_df["得点"]))

    def add(category, priority, finding, direction, training, evidence=""):
        rows.append({
            "分類": category, "優先度": priority, "検出内容": finding,
            "改善の方向性": direction, "推奨練習": training, "参考": evidence,
        })

    for name in score_df["評価項目"]:
        score = float(scores[name])
        if score >= 70:
            continue
        key = FEATURES[name]["key"]
        rv, tv = ref.features[key], tar.features[key]
        if name == "身体を開くな":
            add("上半身", "高" if score < 55 else "中",
                f"肩の回転量が比較選手と{abs(tv-rv):.1f}度異なります。",
                "骨盤の回転開始と肩の回転開始の順序を確認してください。",
                "前肩を残したティー打撃、下半身主導の分解素振り",
                "VICTORY『身体が開く』解説／運動連鎖研究")
        elif name == "割れを作れ":
            add("捻転差", "高" if score < 55 else "中",
                f"肩と骨盤の最大捻転差が比較選手と{abs(tv-rv):.1f}度異なります。",
                "捻りを大きくすること自体ではなく、骨盤→肩への回転の時間差を整えてください。",
                "ステップドリル、メディシンボール回旋、ゆっくりした連続素振り",
                "体幹・骨盤回旋に関する先行研究")
        elif name == "ボールを引きつけろ":
            add("手部軌道", "高" if score < 55 else "中",
                "インパクト時の手と体幹中心の距離が比較選手と異なります。",
                "手が過度に体から離れる／近づきすぎる動きを確認してください。",
                "コース別ティー、短いバットでのインサイドアウト練習",
                "スイング半径・時間に関する研究")
        elif name == "軸をぶらすな":
            add("姿勢安定", "高" if score < 55 else "中",
                "画像上の体幹中心の横移動幅が比較選手と異なります。",
                "身体全体が横へ流れるのか、股関節を中心に回転できているのかを確認してください。",
                "ノーステップ打撃、軸足固定ティー、スプリットスクワット")
        elif name == "壁を作れ":
            add("前脚", "高" if score < 55 else "中",
                f"インパクト時の前脚膝角度が比較選手と{abs(tv-rv):.1f}度異なります。",
                "前脚が早く伸び切る／深く曲がり続けるタイミングを確認してください。",
                "踏み込み後静止ドリル、スプリットスクワット")
        elif name == "頭を残せ":
            add("頭部安定", "中",
                "画像上の頭部移動量が比較選手と異なります。",
                "視線位置と頭部の前突・横流れをスロー動画で確認してください。",
                "ノーステップ打撃、頭の横に目印を置く素振り")
        elif name == "体重移動をしろ":
            add("体重移動proxy", "中",
                "画像上の骨盤中心の水平移動量が比較選手と異なります。",
                "移動量だけでなく、インパクトまでに移動が収束しているかを確認してください。",
                "歩きティー、ステップ打撃、踏み込み後静止ドリル",
                "身体重心速度・走塁意識条件に関する研究")
        elif name == "腰を使え":
            add("骨盤回旋", "高" if score < 55 else "中",
                "骨盤回転速度が比較選手と異なります。",
                "筋力不足とは断定せず、骨盤回転の立ち上がりと肩回転との順序を確認してください。",
                "メディシンボール回旋、ステップ打撃、分解素振り",
                "体幹筋・骨盤回旋に関する研究")

    delay = tar.extra_metrics.get("hip_to_shoulder_peak_delay_ms", np.nan)
    ref_delay = ref.extra_metrics.get("hip_to_shoulder_peak_delay_ms", np.nan)
    if np.isfinite(delay) and np.isfinite(ref_delay) and abs(delay - ref_delay) >= 60:
        add("運動連鎖", "中",
            f"骨盤回転速度ピーク→肩回転速度ピークの時間差が比較選手と約{abs(delay-ref_delay):.0f}ms異なります。",
            "回転の順番とつながりを確認してください。大きい・小さいだけで良否は断定しません。",
            "50〜70%強度の分解素振り、ステップ→骨盤→肩の順を確認",
            "上肢へのエネルギー伝達・体幹回旋研究")

    if not rows:
        add("総合評価", "低", "選択項目では大きな差は検出されませんでした。",
            "同じ撮影条件で複数スイングを比較し、再現性を確認してください。",
            "同条件で3〜5スイング撮影")
    order = {"高": 0, "中": 1, "低": 2}
    return pd.DataFrame(rows).sort_values("優先度", key=lambda s: s.map(order)).reset_index(drop=True)


def _refresh_result(result, selected_feature_names, batter_side):
    ref = result["reference_analysis"]
    tar = result["target_analysis"]
    score_df = build_score_table(ref, tar, selected_feature_names)
    score_df.to_csv(OUTPUT_DIR / "batting_form_comparison_result.csv", index=False, encoding="utf-8-sig")
    save_radar_chart(score_df)
    save_comparison_bar_chart(score_df)
    advice = generate_improvement_advice(score_df, ref, tar)
    advice.to_csv(OUTPUT_DIR / "打撃フォーム改善提案.csv", index=False, encoding="utf-8-sig")
    overall = float(score_df["得点"].mean())
    best = score_df.loc[score_df["得点"].idxmax()]
    low = score_df.loc[score_df["得点"].idxmin()]
    result.update({
        "overall_score": overall,
        "score_dataframe": score_df,
        "advice_dataframe": advice,
        "best_item": str(best["評価項目"]),
        "best_score": float(best["得点"]),
        "lowest_item": str(low["評価項目"]),
        "lowest_score": float(low["得点"]),
        "selected_feature_names": list(selected_feature_names),
        "batter_side": batter_side,
    })
    return result


def run_analysis(reference_original_path, target_original_path, batter_side, selected_feature_names,
                 output_dir=None, reference_batter_roi=None, target_batter_roi=None, **_):
    if batter_side not in {"右打者", "左打者"}:
        raise ValueError("打席を選択してください。")
    selected_feature_names = list(selected_feature_names)
    if len(selected_feature_names) != 5:
        raise ValueError("評価項目は5項目選択してください。")
    if output_dir is not None:
        configure_output_directory(output_dir)
    ref_path = prepare_video_for_analysis(reference_original_path, "reference")
    tar_path = prepare_video_for_analysis(target_original_path, "target")
    ref = analyze_video(ref_path, batter_side, "reference", reference_batter_roi)
    tar = analyze_video(tar_path, batter_side, "target", target_batter_roi)
    result = {
        "reference_analysis": ref,
        "target_analysis": tar,
        "output_dir": Path(OUTPUT_DIR),
        "analysis_mode": "高精度 ver.10.0",
        "reference_batter_roi": reference_batter_roi,
        "target_batter_roi": target_batter_roi,
    }
    return _refresh_result(result, selected_feature_names, batter_side)


def is_measured_frame(analysis, frame_number):
    idx = int(np.clip(int(frame_number), 0, max(0, analysis.frame_count - 1)))
    return bool(analysis.valid_mask[idx])


def _apply_manual_impact(analysis, impact_frame, batter_side, prefix):
    impact = int(np.clip(int(impact_frame), 0, analysis.frame_count - 1))
    if not analysis.valid_mask[impact]:
        nearest = _nearest_valid_frame(impact, analysis.valid_mask, max(3, int(round(0.12 * analysis.fps))))
        if nearest is None:
            raise ValueError("選択フレーム付近に実測骨格がありません。")
        impact = int(nearest)
    stance = estimate_stance_frame(
        analysis.world_landmarks, analysis.normalized_landmarks, analysis.valid_mask,
        impact, analysis.swing_onset_frame, analysis.fps,
    )
    q = _analysis_quality(analysis.valid_mask, analysis.raw_normalized_landmarks, impact, analysis.fps)
    _assert_analysis_quality(q, analysis.fps)
    features, extra = calculate_features(
        analysis.world_landmarks, analysis.normalized_landmarks, analysis.fps,
        impact, stance, analysis.swing_onset_frame, batter_side, analysis.valid_mask,
        image_aspect_ratio=(analysis.width / max(analysis.height, 1)),
    )
    try:
        extra.update(_hand_wrist_alignment_metrics(
            analysis.path, impact, analysis.normalized_landmarks[impact], analysis.roi_x0, analysis.roi_x1
        ))
    except Exception:
        pass
    analysis.impact_frame = impact
    analysis.stance_frame = stance
    analysis.features = features
    analysis.extra_metrics = extra
    analysis.quality_score = float(q["score"])
    analysis.quality_label = str(q["label"])
    analysis.quality_details = q
    save_pose_frame_image(analysis.path, stance, analysis.normalized_landmarks[stance], IMPACT_DIR / f"{prefix}_stance_pose.jpg", "STANCE")
    save_pose_frame_image(analysis.path, impact, analysis.normalized_landmarks[impact], IMPACT_DIR / f"{prefix}_impact_pose.jpg", "IMPACT")


def recalculate_with_manual_impacts(result, reference_impact_frame, target_impact_frame, selected_feature_names, batter_side):
    configure_output_directory(result["output_dir"])
    _apply_manual_impact(result["reference_analysis"], reference_impact_frame, batter_side, "reference")
    _apply_manual_impact(result["target_analysis"], target_impact_frame, batter_side, "target")
    return _refresh_result(result, selected_feature_names, batter_side)
