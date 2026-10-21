# ============================================================
# 野球打撃フォーム比較・改善提案システム
# Jupyter Notebook / Windows用 完全版
#
# 主な機能
# 1. 比較選手と自分の動画を選択
# 2. MOV・AVI・MKVなどをMP4へ自動変換
# 3. MediaPipeで骨格を検出
# 4. 8項目から評価する5項目を選択
# 5. 比較選手とのフォーム類似度を採点
# 6. 肩と腰の角度差を0～180度で正しく計算
# 7. 日本語レーダーチャートを作成
# 8. 改善点と推奨練習を自動提示
# 9. CSV・TXT・PNG・確認画像を保存
# ============================================================


# ============================================================
# 2. ライブラリの読み込み
# ============================================================

import os
import math
import subprocess
import urllib.request
import traceback

from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import imageio_ffmpeg

from matplotlib import font_manager

# ============================================================
# 3. 保存フォルダの作成
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def configure_output_directory(base_dir=None):
    """
    解析結果の保存先を設定する。
    Webアプリから解析ごとに別フォルダを指定できる。
    """
    global OUTPUT_DIR, CONVERTED_DIR, IMPACT_DIR, GRAPH_DIR

    if base_dir is None:
        OUTPUT_DIR = BASE_DIR / "batting_analysis_output"
    else:
        OUTPUT_DIR = Path(base_dir)

    CONVERTED_DIR = OUTPUT_DIR / "converted_videos"
    IMPACT_DIR = OUTPUT_DIR / "impact_frames"
    GRAPH_DIR = OUTPUT_DIR / "graphs"

    for directory in [OUTPUT_DIR, CONVERTED_DIR, IMPACT_DIR, GRAPH_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return OUTPUT_DIR

configure_output_directory()


# ============================================================
# 4. 日本語フォントの設定
# ============================================================

def setup_japanese_font():
    """
    Windows内にある日本語フォントを探し、
    Matplotlibのフォントとして設定する。
    """

    candidate_paths = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\msgothic.ttc"
    ]

    for font_path in candidate_paths:

        if os.path.exists(font_path):

            font_manager.fontManager.addfont(
                font_path
            )

            font_name = font_manager.FontProperties(
                fname=font_path
            ).get_name()

            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False

            print()
            print("日本語フォントを設定しました:")
            print(font_name)

            return font_name

    plt.rcParams["axes.unicode_minus"] = False

    print()
    print(
        "日本語フォントが見つからなかったため、"
        "既定フォントを使用します。"
    )

    return None


JAPANESE_FONT_NAME = setup_japanese_font()


# ============================================================
# 5. MediaPipeモデルの準備
# ============================================================

MODEL_SPEC = {
    "filename": "pose_landmarker_heavy.task",
    "url": (
        "https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
}

# 高精度モードに固定。入力画像は最大1280px幅で解析する。
MAX_INPUT_WIDTH = 1280


def ensure_pose_model():
    """高精度のMediaPipe Heavyモデルを初回にダウンロードする。"""
    model_path = BASE_DIR / MODEL_SPEC["filename"]
    if not model_path.exists():
        print()
        print("MediaPipe Heavyモデルをダウンロードしています。")
        urllib.request.urlretrieve(MODEL_SPEC["url"], str(model_path))
        print("モデルのダウンロードが完了しました。")
    return model_path


# ============================================================
# 6. 評価項目
# ============================================================

FEATURES = {

    "身体を開くな": {
        "key": "shoulder_opening",
        "unit": "degree",
        "description":
            "構えからインパクトまでの肩の回転量",
        "tolerance": 18.0
    },

    "割れを作れ": {
        "key": "hip_shoulder_separation",
        "unit": "degree",
        "description":
            "肩と腰の最大捻転差",
        "tolerance": 15.0
    },

    "ボールを引きつけろ": {
        "key": "hand_torso_distance",
        "unit": "shoulder-width ratio",
        "description":
            "インパクト時の手と体幹中心の距離",
        "tolerance": 0.30
    },

    "軸をぶらすな": {
        "key": "axis_sway",
        "unit": "shoulder-width ratio",
        "description":
            "構えからインパクトまでの体幹中心の横移動",
        "tolerance": 0.35
    },

    "壁を作れ": {
        "key": "front_knee_angle",
        "unit": "degree",
        "description":
            "インパクト時の前脚膝角度",
        "tolerance": 20.0
    },

    "頭を残せ": {
        "key": "head_movement",
        "unit": "shoulder-width ratio",
        "description":
            "構えからインパクトまでの頭部移動量",
        "tolerance": 0.30
    },

    "体重移動をしろ": {
        "key": "weight_transfer",
        "unit": "shoulder-width ratio",
        "description":
            "構えからインパクトまでの骨盤中心の移動量",
        "tolerance": 0.35
    },

    "腰を使え": {
        "key": "hip_rotation_speed",
        "unit": "degree/sec",
        "description":
            "インパクト前後の最大腰回転速度",
        "tolerance": 180.0
    }
}


# ============================================================
# 7. MediaPipeのランドマーク番号
# ============================================================

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# 8. 解析結果を保存するクラス
# ============================================================

@dataclass
class VideoAnalysis:

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    world_landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    valid_mask: np.ndarray

    impact_frame: int
    stance_frame: int

    features: Dict[str, float]

    # v1.6: 打者だけを解析する固定横ROI（元動画の正規化座標）
    roi_x0: float = 0.0
    roi_x1: float = 1.0

    # v1.9: 解析品質。フォーム得点とは別に表示する。
    quality_score: float = 0.0
    quality_label: str = "未評価"
    quality_details: Dict[str, float] = None


# ============================================================
# 9. 動画ファイルを選択する関数
# ============================================================

def select_video_file(title):
    """
    ファイル選択画面を表示する。
    開けない場合は、パスを直接入力する。
    """

    try:

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes(
            "-topmost",
            True
        )

        selected_path = filedialog.askopenfilename(

            title=title,

            filetypes=[

                (
                    "Video files",
                    "*.mp4 *.mov *.MOV "
                    "*.avi *.AVI "
                    "*.mkv *.MKV "
                    "*.m4v *.M4V"
                ),

                (
                    "All files",
                    "*.*"
                )
            ]
        )

        root.destroy()

        if selected_path:

            return Path(
                selected_path
            )

    except Exception as error:

        print()
        print(
            "ファイル選択画面を開けませんでした。"
        )

        print(
            "動画パスの入力方式に切り替えます。"
        )

        print(
            "詳細:",
            error
        )

    print()
    print(title)

    selected_path = input(
        "動画ファイルのパスを入力してください: "
    ).strip()

    selected_path = (
        selected_path
        .strip('"')
        .strip("'")
    )

    selected_path = Path(
        selected_path
    )

    if not selected_path.exists():

        raise FileNotFoundError(
            f"指定した動画が見つかりません: "
            f"{selected_path}"
        )

    return selected_path


# ============================================================
# 10. OpenCVで動画を読めるか確認
# ============================================================

def can_opencv_read_video(video_path):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        capture.release()

        return False

    success, frame = capture.read()

    capture.release()

    return (
        success
        and frame is not None
    )


# ============================================================
# 11. MOVなどをMP4に変換
# ============================================================

def convert_video_to_mp4(
    input_path,
    output_path
):
    """
    ffmpegを使用して動画をMP4に変換する。
    """

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    ffmpeg_exe = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )

    print()
    print("=" * 60)
    print("動画をMP4へ変換しています。")
    print("入力:", input_path)
    print("出力:", output_path)
    print("=" * 60)

    command = [

        ffmpeg_exe,

        "-y",

        "-i",
        str(input_path),

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "20",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-movflags",
        "+faststart",

        str(output_path)
    ]

    process = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        encoding="utf-8",

        errors="replace"
    )

    if process.returncode != 0:

        print()
        print(
            process.stderr[-3000:]
        )

        raise RuntimeError(
            "動画のMP4変換に失敗しました。"
        )

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
    ):

        raise RuntimeError(
            "MP4ファイルが正常に作成されませんでした。"
        )

    print()
    print(
        "MP4変換が完了しました。"
    )

    print(
        output_path
    )

    return output_path


# ============================================================
# 12. 解析用動画の準備
# ============================================================

def prepare_video_for_analysis(
    original_path,
    prefix
):
    """
    MP4で読み込める動画はそのまま使用する。
    その他の動画、または読み込めないMP4は再変換する。
    """

    original_path = Path(
        original_path
    )

    extension = (
        original_path
        .suffix
        .lower()
    )

    converted_path = (
        CONVERTED_DIR
        / f"{prefix}_converted.mp4"
    )

    if extension == ".mp4":

        print()
        print(
            f"{prefix}動画はMP4形式です。"
        )

        if can_opencv_read_video(
            original_path
        ):

            print(
                "OpenCVで読み込み可能です。"
            )

            return original_path

        print(
            "MP4ですが読み込めないため、"
            "再変換します。"
        )

    converted_path = convert_video_to_mp4(
        original_path,
        converted_path
    )

    if not can_opencv_read_video(
        converted_path
    ):

        raise RuntimeError(
            "MP4変換後もOpenCVで"
            "読み込めませんでした。"
        )

    return converted_path


# ============================================================
# 13. 数学計算用関数
# ============================================================

def midpoint(
    point_a,
    point_b
):

    return (
        point_a
        + point_b
    ) / 2.0


def distance_3d(
    point_a,
    point_b
):

    return float(
        np.linalg.norm(
            point_a[:3]
            - point_b[:3]
        )
    )


def joint_angle(
    point_a,
    point_b,
    point_c
):
    """
    A-B-CのBを中心とする関節角度を計算する。
    """

    vector_ba = (
        point_a[:3]
        - point_b[:3]
    )

    vector_bc = (
        point_c[:3]
        - point_b[:3]
    )

    denominator = (
        np.linalg.norm(
            vector_ba
        )
        * np.linalg.norm(
            vector_bc
        )
    )

    if denominator < 1e-8:

        return np.nan

    cosine_value = (
        np.dot(
            vector_ba,
            vector_bc
        )
        / denominator
    )

    cosine_value = np.clip(
        cosine_value,
        -1.0,
        1.0
    )

    return float(

        np.degrees(

            np.arccos(
                cosine_value
            )
        )
    )


def horizontal_line_angle(
    left_point,
    right_point
):
    """
    左右の関節点を結ぶ線の
    水平面上の角度を計算する。
    """

    difference_x = (
        right_point[0]
        - left_point[0]
    )

    difference_z = (
        right_point[2]
        - left_point[2]
    )

    return float(

        np.degrees(

            np.arctan2(
                difference_z,
                difference_x
            )
        )
    )


def circular_angle_difference(
    angle_a,
    angle_b
):
    """
    2つの角度の最短差を0～180度で求める。

    例:
    170度と-170度の差は
    340度ではなく20度になる。
    """

    difference = np.degrees(

        np.arctan2(

            np.sin(
                np.radians(
                    angle_a
                    - angle_b
                )
            ),

            np.cos(
                np.radians(
                    angle_a
                    - angle_b
                )
            )
        )
    )

    return np.abs(
        difference
    )


def moving_average(
    values,
    window=5
):
    """
    移動平均で値を平滑化する。
    """

    return (

        pd.Series(
            values
        )

        .rolling(
            window=window,
            center=True,
            min_periods=1
        )

        .mean()

        .to_numpy()
    )


# ============================================================
# 14. 骨格未検出フレームの補間
# ============================================================

def interpolate_landmarks(
    landmark_array,
    valid_mask
):
    """
    骨格を検出できなかったフレームを
    前後の値から線形補間する。
    """

    result = landmark_array.copy()

    frame_numbers = np.arange(
        len(result)
    )

    for landmark_number in range(
        result.shape[1]
    ):

        for dimension in range(3):

            values = result[
                :,
                landmark_number,
                dimension
            ]

            valid_values = (
                valid_mask
                & np.isfinite(
                    values
                )
            )

            if valid_values.sum() < 2:

                continue

            result[
                :,
                landmark_number,
                dimension
            ] = np.interp(

                frame_numbers,

                frame_numbers[
                    valid_values
                ],

                values[
                    valid_values
                ]
            )

    return result


# ============================================================
# 15. インパクトフレームの推定
# ============================================================

def estimate_impact_frame(
    world_landmarks,
    fps,
    valid_mask=None
):
    """
    インパクトの自動推定（改良版）。

    旧方式は「両手首の中心速度が動画中で最大のフレーム」を
    そのまま採用していたため、フォロースルー側の大きな動きを
    インパクトと誤判定することがあった。

    改良版では、
    1. 左右の手首それぞれの速度
    2. 両手首中心の速度
    を組み合わせてスイング速度を作り、
    「最大値そのもの」ではなく、最大値に近い最初の主要ピークを
    インパクト候補として採用する。

    それでもバット・ボール自体はMediaPipe Poseでは検出していないため、
    最終的にはWeb画面上でフレームを目視補正できるようにしている。
    """

    left_wrist_positions = world_landmarks[:, LEFT_WRIST, :3]
    right_wrist_positions = world_landmarks[:, RIGHT_WRIST, :3]
    hand_center_positions = midpoint(
        left_wrist_positions,
        right_wrist_positions
    )

    left_speed = np.linalg.norm(
        np.diff(left_wrist_positions, axis=0),
        axis=1
    ) * fps
    right_speed = np.linalg.norm(
        np.diff(right_wrist_positions, axis=0),
        axis=1
    ) * fps
    center_speed = np.linalg.norm(
        np.diff(hand_center_positions, axis=0),
        axis=1
    ) * fps

    # 両手首中心だけだと、左右の手の動きが相殺されることがある。
    # そのため個々の手首速度を主成分として使用する。
    swing_speed = (
        0.40 * left_speed
        + 0.40 * right_speed
        + 0.20 * center_speed
    )

    # v1.9: 補間フレーム間の見かけの速度を、インパクト推定に使わない。
    # 連続する2フレームがどちらも実測された場合だけ速度を有効にする。
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(valid_mask) == len(world_landmarks):
            transition_valid = valid_mask[:-1] & valid_mask[1:]
            swing_speed = swing_speed.astype(float)
            swing_speed[~transition_valid] = np.nan

    # FPSに応じて約0.08秒分を平滑化する。
    smooth_window = max(3, int(round(fps * 0.08)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    swing_speed = moving_average(
        swing_speed,
        window=smooth_window
    )

    n = len(swing_speed)
    if n == 0:
        return 0

    search_start = max(1, int(n * 0.08))
    search_end = min(n - 1, int(n * 0.92))
    if search_end <= search_start:
        return min(len(world_landmarks) - 1, int(np.nanargmax(swing_speed)) + 1)

    section = swing_speed[search_start:search_end + 1]
    finite_section = section[np.isfinite(section)]
    if finite_section.size == 0:
        return max(0, min(len(world_landmarks) - 1, search_start + 1))

    peak_value = float(np.nanmax(finite_section))

    # 動画序盤の比較的静かな部分からスイング開始の目安を作る。
    baseline_end = max(search_start + 2, int(n * 0.35))
    baseline_values = swing_speed[search_start:min(baseline_end, n)]
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    baseline = float(np.nanmedian(baseline_values)) if baseline_values.size else 0.0
    onset_threshold = max(baseline * 2.2, peak_value * 0.20)

    onset = search_start
    consecutive_needed = max(2, int(round(fps * 0.04)))
    run = 0
    for i in range(search_start, search_end + 1):
        if np.isfinite(swing_speed[i]) and swing_speed[i] >= onset_threshold:
            run += 1
            if run >= consecutive_needed:
                onset = max(search_start, i - run + 1)
                break
        else:
            run = 0

    # フォロースルーの後半ピークを拾いにくくするため、
    # 最大ピークの82%以上に達する「最初の局所ピーク」を採用する。
    major_peak_threshold = peak_value * 0.82
    min_candidate = min(
        search_end,
        onset + max(1, int(round(fps * 0.05)))
    )

    candidates = []
    for i in range(max(search_start + 1, min_candidate), search_end):
        current = swing_speed[i]
        if not np.isfinite(current):
            continue
        if (
            current >= major_peak_threshold
            and current >= swing_speed[i - 1]
            and current >= swing_speed[i + 1]
        ):
            candidates.append(i)

    if candidates:
        speed_index = candidates[0]
    else:
        speed_index = (
            int(np.nanargmax(swing_speed[search_start:search_end + 1]))
            + search_start
        )

    # 手首速度のピークは実際の接触よりわずかに後ろに出る場合があるため、
    # 約0.03秒だけ前へ補正する。
    lead_frames = max(0, int(round(fps * 0.03)))
    impact_frame = speed_index + 1 - lead_frames

    return int(np.clip(
        impact_frame,
        0,
        len(world_landmarks) - 1
    ))


def estimate_stance_frame(
    impact_frame,
    fps
):
    """
    インパクトの約0.8秒前を
    構え・準備動作の基準フレームとする。
    """

    stance_frame = (
        impact_frame
        - int(
            round(
                0.8 * fps
            )
        )
    )

    return max(
        0,
        stance_frame
    )


# ============================================================
# 16. 特徴量の計算
# ============================================================

def calculate_features(
    world,
    fps,
    impact_frame,
    stance_frame,
    batter_side,
    valid_mask=None
):
    """
    打撃フォームの8つの特徴量を計算する。

    v1.9では、速度・最大値などの計算で補間フレームを極力除外する。
    """

    if valid_mask is None:
        valid_mask = np.ones(len(world), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(valid_mask) != len(world):
            valid_mask = np.ones(len(world), dtype=bool)

    shoulder_width = distance_3d(

        world[
            stance_frame,
            LEFT_SHOULDER
        ],

        world[
            stance_frame,
            RIGHT_SHOULDER
        ]
    )

    shoulder_width = max(
        shoulder_width,
        1e-6
    )

    if batter_side == "右打者":

        lead_hip = LEFT_HIP
        lead_knee = LEFT_KNEE
        lead_ankle = LEFT_ANKLE

    else:

        lead_hip = RIGHT_HIP
        lead_knee = RIGHT_KNEE
        lead_ankle = RIGHT_ANKLE

    shoulder_centers = midpoint(

        world[
            :,
            LEFT_SHOULDER,
            :3
        ],

        world[
            :,
            RIGHT_SHOULDER,
            :3
        ]
    )

    hip_centers = midpoint(

        world[
            :,
            LEFT_HIP,
            :3
        ],

        world[
            :,
            RIGHT_HIP,
            :3
        ]
    )

    hand_centers = midpoint(

        world[
            :,
            LEFT_WRIST,
            :3
        ],

        world[
            :,
            RIGHT_WRIST,
            :3
        ]
    )

    torso_centers = midpoint(
        shoulder_centers,
        hip_centers
    )

    head_positions = world[
        :,
        NOSE,
        :3
    ]

    shoulder_angles = []

    hip_angles = []

    for frame_number in range(
        len(world)
    ):

        shoulder_angles.append(

            horizontal_line_angle(

                world[
                    frame_number,
                    LEFT_SHOULDER
                ],

                world[
                    frame_number,
                    RIGHT_SHOULDER
                ]
            )
        )

        hip_angles.append(

            horizontal_line_angle(

                world[
                    frame_number,
                    LEFT_HIP
                ],

                world[
                    frame_number,
                    RIGHT_HIP
                ]
            )
        )

    shoulder_angles = np.asarray(
        shoulder_angles,
        dtype=float
    )

    hip_angles = np.asarray(
        hip_angles,
        dtype=float
    )

    shoulder_angles_unwrapped = np.degrees(

        np.unwrap(

            np.radians(
                shoulder_angles
            )
        )
    )

    hip_angles_unwrapped = np.degrees(

        np.unwrap(

            np.radians(
                hip_angles
            )
        )
    )

    # --------------------------------------------------------
    # 1. 身体を開くな
    # --------------------------------------------------------

    shoulder_opening = abs(

        shoulder_angles_unwrapped[
            impact_frame
        ]

        - shoulder_angles_unwrapped[
            stance_frame
        ]
    )

    # --------------------------------------------------------
    # 2. 割れを作れ
    # --------------------------------------------------------

    pre_impact_start = max(

        stance_frame,

        impact_frame
        - int(
            round(
                0.7 * fps
            )
        )
    )

    separation = (
        circular_angle_difference(
            shoulder_angles,
            hip_angles
        )
    )

    separation_section = separation[
        pre_impact_start:
        impact_frame + 1
    ].astype(float)
    separation_valid = valid_mask[pre_impact_start:impact_frame + 1]
    separation_section[~separation_valid] = np.nan

    hip_shoulder_separation = float(

        np.nanmax(
            separation_section
        )
    )

    # --------------------------------------------------------
    # 3. ボールを引きつけろ
    # --------------------------------------------------------

    hand_torso_distance = (

        np.linalg.norm(

            hand_centers[
                impact_frame
            ]

            - torso_centers[
                impact_frame
            ]
        )

        / shoulder_width
    )

    # --------------------------------------------------------
    # 4. 軸をぶらすな
    # --------------------------------------------------------

    torso_x_section = torso_centers[
        stance_frame:
        impact_frame + 1,
        0
    ].astype(float)
    torso_valid = valid_mask[stance_frame:impact_frame + 1]
    torso_x_section[~torso_valid] = np.nan

    axis_sway = (

        float(

            np.nanmax(
                torso_x_section
            )

            - np.nanmin(
                torso_x_section
            )
        )

        / shoulder_width
    )

    # --------------------------------------------------------
    # 5. 壁を作れ
    # --------------------------------------------------------

    front_knee_angle = joint_angle(

        world[
            impact_frame,
            lead_hip
        ],

        world[
            impact_frame,
            lead_knee
        ],

        world[
            impact_frame,
            lead_ankle
        ]
    )

    # --------------------------------------------------------
    # 6. 頭を残せ
    # --------------------------------------------------------

    head_section = head_positions[
        stance_frame:
        impact_frame + 1
    ]

    head_distances = np.linalg.norm(

        head_section

        - head_positions[
            stance_frame
        ],

        axis=1
    ).astype(float)
    head_valid = valid_mask[stance_frame:impact_frame + 1]
    head_distances[~head_valid] = np.nan

    head_movement = (

        float(
            np.nanmax(
                head_distances
            )
        )

        / shoulder_width
    )

    # --------------------------------------------------------
    # 7. 体重移動をしろ
    # --------------------------------------------------------

    weight_transfer = (

        abs(

            hip_centers[
                impact_frame,
                0
            ]

            - hip_centers[
                stance_frame,
                0
            ]
        )

        / shoulder_width
    )

    # --------------------------------------------------------
    # 8. 腰を使え
    # --------------------------------------------------------

    # v1.9: 連続する実測フレーム間だけで角速度を計算する。
    hip_angular_velocity = np.full(len(hip_angles_unwrapped), np.nan, dtype=float)
    if len(hip_angles_unwrapped) >= 2:
        delta = np.diff(hip_angles_unwrapped) * fps
        transition_valid = valid_mask[:-1] & valid_mask[1:]
        hip_angular_velocity[1:][transition_valid] = delta[transition_valid]

    velocity_start = max(

        stance_frame,

        impact_frame
        - int(
            round(
                0.5 * fps
            )
        )
    )

    velocity_end = min(

        len(
            hip_angular_velocity
        ),

        impact_frame
        + int(
            round(
                0.15 * fps
            )
        )
        + 1
    )

    hip_rotation_speed = float(

        np.nanmax(

            np.abs(

                hip_angular_velocity[
                    velocity_start:
                    velocity_end
                ]
            )
        )
    )

    return {

        "shoulder_opening":
            float(
                shoulder_opening
            ),

        "hip_shoulder_separation":
            float(
                hip_shoulder_separation
            ),

        "hand_torso_distance":
            float(
                hand_torso_distance
            ),

        "axis_sway":
            float(
                axis_sway
            ),

        "front_knee_angle":
            float(
                front_knee_angle
            ),

        "head_movement":
            float(
                head_movement
            ),

        "weight_transfer":
            float(
                weight_transfer
            ),

        "hip_rotation_speed":
            float(
                hip_rotation_speed
            )
    }


# ============================================================
# 17. 構え・インパクト画像の保存
# ============================================================

def save_frame_image(
    video_path,
    frame_number,
    output_path
):
    """
    指定フレームを画像として保存する。
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number)
    )

    success, frame = capture.read()

    capture.release()

    if success and frame is not None:

        cv2.imwrite(
            str(output_path),
            frame
        )


# ============================================================
# 18. 骨格付き確認画像の保存
# ============================================================

def draw_pose_on_frame(
    frame,
    normalized_landmarks
):
    """
    骨格ランドマークを画像上に描画する。
    """

    if normalized_landmarks is None:

        return frame

    output_frame = frame.copy()

    connections = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE)
    ]

    height, width = frame.shape[:2]

    points = {}

    for landmark_number in [
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE
    ]:

        x = int(
            normalized_landmarks[
                landmark_number,
                0
            ]
            * width
        )

        y = int(
            normalized_landmarks[
                landmark_number,
                1
            ]
            * height
        )

        points[
            landmark_number
        ] = (
            x,
            y
        )

    for start, end in connections:

        if (
            start in points
            and end in points
        ):

            cv2.line(

                output_frame,

                points[start],

                points[end],

                (
                    0,
                    255,
                    0
                ),

                3
            )

    for point in points.values():

        cv2.circle(

            output_frame,

            point,

            5,

            (
                0,
                0,
                255
            ),

            -1
        )

    return output_frame


def save_pose_frame_image(
    video_path,
    frame_number,
    normalized_landmarks,
    output_path,
    label_text
):
    """
    骨格を重ねた確認画像を保存する。
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number)
    )

    success, frame = capture.read()

    capture.release()

    if not success or frame is None:

        return

    frame = draw_pose_on_frame(
        frame,
        normalized_landmarks
    )

    cv2.putText(

        frame,

        label_text,

        (
            30,
            50
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.2,

        (
            255,
            255,
            255
        ),

        3,

        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        frame
    )


# ============================================================
# 19. 複数人物から「打者」を選ぶ補助処理
# ============================================================

def _pose_to_arrays(world_pose, normalized_pose):
    """MediaPipeの1人物分のランドマークをNumPy配列へ変換する。"""
    world_array = np.array([
        [
            landmark.x,
            landmark.y,
            landmark.z,
            getattr(landmark, "visibility", 1.0),
        ]
        for landmark in world_pose
    ], dtype=float)

    normalized_array = np.array([
        [
            landmark.x,
            landmark.y,
            landmark.z,
            getattr(landmark, "visibility", 1.0),
        ]
        for landmark in normalized_pose
    ], dtype=float)

    return world_array, normalized_array


def _angle_2d(point_a, point_b, point_c):
    """正規化画像座標上でA-B-CのBを中心とする角度を返す。"""
    ba = np.asarray(point_a[:2], dtype=float) - np.asarray(point_b[:2], dtype=float)
    bc = np.asarray(point_c[:2], dtype=float) - np.asarray(point_b[:2], dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return np.nan
    cosine_value = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_value)))


def _position_matches(hip_x, preference):
    """打者のおおまかな画面位置指定に合うか判定する。"""
    if preference in (None, "自動"):
        return True
    if preference == "左":
        return hip_x <= 0.42
    if preference == "中央":
        return 0.28 <= hip_x <= 0.72
    if preference == "右":
        return hip_x >= 0.58
    return True


def _batter_pose_score(normalized_array):
    """
    1人物が「打者らしい」度合いを簡易評価する。

    キャッチャーは一般に深くしゃがみ、両手が低い位置に来やすい。
    打者は立位に近く、両手が比較的近く、肩〜胸付近にあることが多い。
    ここでは医学的・競技的評価ではなく、人物選択のためだけに用いる。
    """
    required = [
        NOSE,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    ]

    points = normalized_array[required]
    if not np.all(np.isfinite(points[:, :2])):
        return -1e9

    visibility = float(np.nanmean(points[:, 3]))

    left_shoulder = normalized_array[LEFT_SHOULDER]
    right_shoulder = normalized_array[RIGHT_SHOULDER]
    left_wrist = normalized_array[LEFT_WRIST]
    right_wrist = normalized_array[RIGHT_WRIST]
    left_hip = normalized_array[LEFT_HIP]
    right_hip = normalized_array[RIGHT_HIP]

    shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
    hip_center = (left_hip[:2] + right_hip[:2]) / 2.0
    wrist_center = (left_wrist[:2] + right_wrist[:2]) / 2.0

    shoulder_width = max(float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])), 1e-4)
    torso_height = max(float(abs(hip_center[1] - shoulder_center[1])), 1e-4)

    # 両手が近いほど、打撃時のグリップ姿勢らしさを加点。
    wrist_distance_ratio = float(np.linalg.norm(left_wrist[:2] - right_wrist[:2]) / shoulder_width)
    hand_closeness = float(np.clip(1.4 - wrist_distance_ratio, 0.0, 1.0))

    # 手が腰より上、できれば肩付近にあるほど加点。
    hand_height_ratio = float((hip_center[1] - wrist_center[1]) / torso_height)
    hand_height_score = float(np.clip(hand_height_ratio, 0.0, 1.4) / 1.4)

    # キャッチャーの深いしゃがみを避けるため膝伸展角度を利用。
    left_knee_angle = _angle_2d(
        normalized_array[LEFT_HIP],
        normalized_array[LEFT_KNEE],
        normalized_array[LEFT_ANKLE],
    )
    right_knee_angle = _angle_2d(
        normalized_array[RIGHT_HIP],
        normalized_array[RIGHT_KNEE],
        normalized_array[RIGHT_ANKLE],
    )
    knee_angles = [v for v in [left_knee_angle, right_knee_angle] if np.isfinite(v)]
    if knee_angles:
        knee_extension = float(np.clip((np.mean(knee_angles) - 70.0) / 90.0, 0.0, 1.0))
    else:
        knee_extension = 0.0

    # 立位らしい縦方向の脚の長さ。
    ankle_center = (
        normalized_array[LEFT_ANKLE, :2] + normalized_array[RIGHT_ANKLE, :2]
    ) / 2.0
    leg_vertical = abs(float(ankle_center[1] - hip_center[1]))
    standing_score = float(np.clip(leg_vertical / max(torso_height * 1.5, 1e-4), 0.0, 1.0))

    return (
        2.2 * hand_closeness
        + 1.8 * hand_height_score
        + 1.6 * knee_extension
        + 0.8 * standing_score
        + 0.6 * visibility
    )


def _normalize_batter_roi(batter_roi):
    """
    v1.6: 打者だけを解析する横方向ROIを0〜1で正規化する。
    batter_roi は {"center": 0〜1, "width": 0〜1} を想定。
    None の場合は画面全体。
    """
    if not batter_roi:
        return 0.0, 1.0

    center = float(np.clip(batter_roi.get("center", 0.5), 0.0, 1.0))
    width = float(np.clip(batter_roi.get("width", 1.0), 0.18, 1.0))
    half = width / 2.0
    x0 = max(0.0, center - half)
    x1 = min(1.0, center + half)

    # 端に寄ったときも、できるだけ指定幅を確保する。
    if x1 - x0 < width:
        if x0 <= 0.0:
            x1 = min(1.0, width)
        elif x1 >= 1.0:
            x0 = max(0.0, 1.0 - width)

    if x1 - x0 < 0.10:
        return 0.0, 1.0
    return float(x0), float(x1)


def _map_crop_landmarks_to_full_frame(normalized_array, roi_x0, roi_x1):
    """切り出し画像上の正規化座標を、元動画全体の座標へ戻す。"""
    mapped = normalized_array.copy()
    crop_width = max(float(roi_x1 - roi_x0), 1e-6)
    mapped[:, 0] = roi_x0 + mapped[:, 0] * crop_width
    return mapped


def _segment_length(points, a, b):
    if not np.all(np.isfinite(points[[a, b], :2])):
        return np.nan
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def _is_temporally_consistent_pose(current_normalized, previous_normalized):
    """
    v1.6: 1フレームだけ捕手と打者の関節が混ざったような骨格を弾く。
    打撃中に大きく動く手首は判定に使わず、肩・骨盤・脚など
    比較的安定する骨格長と胴体位置の連続性だけを確認する。
    """
    if previous_normalized is None:
        return True

    stable_indices = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    ]
    cur = current_normalized
    prev = previous_normalized

    if not np.all(np.isfinite(cur[stable_indices, :2])):
        return False
    if not np.all(np.isfinite(prev[stable_indices, :2])):
        return True

    cur_torso = (
        cur[LEFT_SHOULDER, :2] + cur[RIGHT_SHOULDER, :2]
        + cur[LEFT_HIP, :2] + cur[RIGHT_HIP, :2]
    ) / 4.0
    prev_torso = (
        prev[LEFT_SHOULDER, :2] + prev[RIGHT_SHOULDER, :2]
        + prev[LEFT_HIP, :2] + prev[RIGHT_HIP, :2]
    ) / 4.0
    if float(np.linalg.norm(cur_torso - prev_torso)) > 0.18:
        return False

    segments = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    ratios = []
    for a, b in segments:
        cur_len = _segment_length(cur, a, b)
        prev_len = _segment_length(prev, a, b)
        if np.isfinite(cur_len) and np.isfinite(prev_len) and prev_len > 1e-5:
            ratios.append(cur_len / prev_len)

    if ratios:
        ratios = np.asarray(ratios, dtype=float)
        # 1フレームで骨格長が半分以下/約2倍以上になるのは人物混線の疑いが強い。
        if np.nanmin(ratios) < 0.48 or np.nanmax(ratios) > 2.10:
            return False

    return True


def _select_batter_pose(result, previous_hip_center=None, position_preference="自動"):
    """
    MediaPipeが検出した複数人物から打者を1人選ぶ。

    1) 指定された画面位置（任意）
    2) 打者らしい立位・手位置
    3) 前フレームからの位置連続性
    を組み合わせる。
    """
    if not result.pose_world_landmarks or not result.pose_landmarks:
        return None, None, None

    candidates = []
    count = min(len(result.pose_world_landmarks), len(result.pose_landmarks))

    for index in range(count):
        world_array, normalized_array = _pose_to_arrays(
            result.pose_world_landmarks[index],
            result.pose_landmarks[index],
        )

        hip_center = (
            normalized_array[LEFT_HIP, :2] + normalized_array[RIGHT_HIP, :2]
        ) / 2.0

        if not np.all(np.isfinite(hip_center)):
            continue

        base_score = _batter_pose_score(normalized_array)
        matches = _position_matches(float(hip_center[0]), position_preference)

        # 手動位置指定時は、該当領域の人物を強く優先する。
        position_bonus = 5.0 if (position_preference != "自動" and matches) else 0.0
        position_penalty = -5.0 if (position_preference != "自動" and not matches) else 0.0

        continuity_bonus = 0.0
        if previous_hip_center is not None and np.all(np.isfinite(previous_hip_center)):
            distance = float(np.linalg.norm(hip_center - previous_hip_center))
            continuity_bonus = 4.5 * math.exp(-0.5 * (distance / 0.12) ** 2)

        total_score = base_score + position_bonus + position_penalty + continuity_bonus
        candidates.append((total_score, world_array, normalized_array, hip_center))

    if not candidates:
        return None, None, None

    # 位置指定した場合、該当人物が1人でもいれば該当候補だけで競わせる。
    if position_preference != "自動":
        matching_candidates = [
            item for item in candidates
            if _position_matches(float(item[3][0]), position_preference)
        ]
        if matching_candidates:
            candidates = matching_candidates

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, world_array, normalized_array, hip_center = candidates[0]
    return world_array, normalized_array, hip_center


def _longest_true_run(mask):
    """Trueが連続する最長フレーム数を返す。"""
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _nearest_valid_frame(frame_number, valid_mask, max_distance):
    """指定フレーム近傍で最も近い実測フレームを返す。見つからなければNone。"""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(valid_mask) == 0:
        return None
    frame_number = int(np.clip(frame_number, 0, len(valid_mask) - 1))
    if valid_mask[frame_number]:
        return frame_number
    max_distance = max(0, int(max_distance))
    for distance in range(1, max_distance + 1):
        left = frame_number - distance
        right = frame_number + distance
        if left >= 0 and valid_mask[left]:
            return left
        if right < len(valid_mask) and valid_mask[right]:
            return right
    return None


def _analysis_quality(valid_mask, normalized_raw, impact_frame, fps):
    """
    解析品質を0〜100で評価する。
    類似度とは別指標で、主にインパクト周辺の実測骨格の充足度を見る。
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(valid_mask)
    fps = max(float(fps), 1e-6)
    if n == 0:
        return {
            "score": 0.0, "label": "解析不可",
            "overall_detection_rate": 0.0,
            "impact_window_detection_rate": 0.0,
            "impact_window_longest_run": 0,
            "impact_window_transition_rate": 0.0,
            "key_joint_visibility": 0.0,
        }

    pre = max(0, int(impact_frame - round(0.70 * fps)))
    post = min(n, int(impact_frame + round(0.30 * fps)) + 1)
    window_mask = valid_mask[pre:post]
    overall_rate = float(valid_mask.mean())
    impact_rate = float(window_mask.mean()) if len(window_mask) else 0.0
    longest_run = _longest_true_run(window_mask)

    if len(window_mask) >= 2:
        transitions = window_mask[:-1] & window_mask[1:]
        transition_rate = float(transitions.mean())
    else:
        transition_rate = 0.0

    # 主要関節のvisibility。実測フレームだけを対象にする。
    key_joints = [
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
    ]
    key_visibility = 0.0
    try:
        normalized_raw = np.asarray(normalized_raw, dtype=float)
        section = normalized_raw[pre:post][:, key_joints, 3]
        finite = section[np.isfinite(section)]
        if finite.size:
            key_visibility = float(np.clip(np.nanmean(finite), 0.0, 1.0))
    except Exception:
        key_visibility = 0.0

    # 100点満点。全体率より、インパクト前後の品質を重視する。
    overall_component = min(overall_rate / 0.75, 1.0)
    impact_component = min(impact_rate / 0.85, 1.0)
    continuity_target = max(1.0, 0.25 * fps)
    continuity_component = min(longest_run / continuity_target, 1.0)
    transition_component = min(transition_rate / 0.75, 1.0)
    visibility_component = float(np.clip((key_visibility - 0.35) / 0.50, 0.0, 1.0))

    score = 100.0 * (
        0.10 * overall_component
        + 0.35 * impact_component
        + 0.20 * continuity_component
        + 0.15 * transition_component
        + 0.20 * visibility_component
    )

    if score >= 80:
        label = "高"
    elif score >= 60:
        label = "要注意"
    else:
        label = "解析不可"

    return {
        "score": round(float(score), 1),
        "label": label,
        "overall_detection_rate": overall_rate,
        "impact_window_detection_rate": impact_rate,
        "impact_window_longest_run": int(longest_run),
        "impact_window_transition_rate": transition_rate,
        "key_joint_visibility": key_visibility,
        "impact_window_start": int(pre),
        "impact_window_end": int(max(pre, post - 1)),
    }


def _assert_analysis_quality(quality, fps):
    """低品質動画は、もっともらしい数値を出さずに停止する。"""
    minimum_run = max(3, int(round(0.12 * max(float(fps), 1.0))))
    problems = []
    if quality["impact_window_detection_rate"] < 0.55:
        problems.append("インパクト前後の骨格検出率が55%未満")
    if quality["impact_window_longest_run"] < minimum_run:
        problems.append("インパクト前後の連続検出区間が短い")
    if quality["impact_window_transition_rate"] < 0.45:
        problems.append("実測フレーム間の連続性が不足")
    if quality["key_joint_visibility"] < 0.45:
        problems.append("主要関節のvisibilityが低い")
    if quality["score"] < 60:
        problems.append("解析信頼度が60点未満")

    if problems:
        raise ValueError(
            "この動画は解析品質が不足しているため、結果の数値化を停止しました。\n"
            f"解析信頼度: {quality['score']:.1f}/100（{quality['label']}）\n"
            f"全体骨格検出率: {quality['overall_detection_rate']:.1%}\n"
            f"インパクト前後の検出率: {quality['impact_window_detection_rate']:.1%}\n"
            f"主要関節visibility: {quality['key_joint_visibility']:.2f}\n"
            + " / ".join(problems)
            + "\n赤枠を打者全身に合わせ、2〜5秒程度の動画で、遮蔽の少ない映像を使用してください。"
        )


# ============================================================
# 20. 動画解析
# ============================================================

def analyze_video(
    video_path,
    batter_side,
    prefix,
    batter_position="自動",
    batter_roi=None
):
    """
    動画全体をMediaPipeで解析する。
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise ValueError(
            f"動画を開けませんでした: "
            f"{video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:

        fps = 30.0

    width = int(

        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(

        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # 解析方式は高精度（Heavy）に固定
    model_path = ensure_pose_model()
    max_input_width = MAX_INPUT_WIDTH

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    PoseLandmarker = (
        mp.tasks.vision.PoseLandmarker
    )

    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = PoseLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                model_path
            )
        ),

        running_mode=RunningMode.VIDEO,

        # v1.6: ROI内だけを解析するため、通常は2人までで十分。
        # 画面全体解析時のみ4人まで検出する。
        num_poses=(2 if batter_roi else 4),

        min_pose_detection_confidence=0.35,

        min_pose_presence_confidence=0.35,

        min_tracking_confidence=0.40,

        output_segmentation_masks=False
    )

    world_list = []
    normalized_list = []
    valid_list = []

    frame_number = 0
    previous_batter_hip_center = None
    previous_batter_normalized = None
    consecutive_rejections = 0
    roi_x0, roi_x1 = _normalize_batter_roi(batter_roi)

    print()
    print(
        "骨格検出を開始します。"
    )

    with PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = capture.read()

            if not success:

                break

            # v1.6: 先に「打者だけ」の横方向ROIへ切り出す。
            # これによりキャッチャー・審判・投手をMediaPipeへ極力入力しない。
            original_height, original_width = frame.shape[:2]
            crop_x0 = int(round(roi_x0 * original_width))
            crop_x1 = int(round(roi_x1 * original_width))
            crop_x0 = int(np.clip(crop_x0, 0, max(0, original_width - 2)))
            crop_x1 = int(np.clip(crop_x1, crop_x0 + 2, original_width))
            frame_for_pose = frame[:, crop_x0:crop_x1]

            if max_input_width and frame_for_pose.shape[1] > max_input_width:
                scale = max_input_width / frame_for_pose.shape[1]
                resized_height = max(1, int(round(frame_for_pose.shape[0] * scale)))
                frame_for_pose = cv2.resize(
                    frame_for_pose,
                    (max_input_width, resized_height),
                    interpolation=cv2.INTER_AREA
                )

            frame_rgb = cv2.cvtColor(
                frame_for_pose,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=frame_rgb
            )

            timestamp_ms = int(

                round(

                    frame_number
                    * 1000.0
                    / fps
                )
            )

            result = (
                landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            world_array, normalized_array_crop, selected_hip_center = _select_batter_pose(
                result,
                previous_hip_center=previous_batter_hip_center,
                # ROI内では左右指定を使わず、打者らしさ+追跡で選ぶ。
                position_preference="自動" if batter_roi else batter_position,
            )

            raw_pose_found = (
                world_array is not None
                and normalized_array_crop is not None
            )

            temporal_ok = (
                raw_pose_found
                and _is_temporally_consistent_pose(
                    normalized_array_crop,
                    previous_batter_normalized,
                )
            )

            # v1.8: 一度追跡が外れた後に、以前の骨格との差だけで
            # 以後のフレームを連続して棄却し続ける「追跡ロック」を防ぐ。
            # ROI内で打者候補が再検出され、2フレーム以上連続して不一致なら
            # 新しい打者位置として再取得する。
            pose_ok = bool(temporal_ok)
            if raw_pose_found and not temporal_ok:
                consecutive_rejections += 1
                candidate_score = _batter_pose_score(normalized_array_crop)
                if consecutive_rejections >= 2 and candidate_score >= 2.0:
                    pose_ok = True
            elif raw_pose_found:
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
                if consecutive_rejections >= 3:
                    previous_batter_hip_center = None
                    previous_batter_normalized = None

            if pose_ok:
                # 描画用座標だけ元動画全体へ戻す。
                normalized_array = _map_crop_landmarks_to_full_frame(
                    normalized_array_crop,
                    roi_x0,
                    roi_x1,
                )

                world_list.append(world_array)
                normalized_list.append(normalized_array)
                valid_list.append(True)

                previous_batter_hip_center = selected_hip_center
                previous_batter_normalized = normalized_array_crop.copy()
                consecutive_rejections = 0

            else:

                world_list.append(

                    np.full(
                        (
                            33,
                            4
                        ),
                        np.nan
                    )
                )

                normalized_list.append(

                    np.full(
                        (
                            33,
                            4
                        ),
                        np.nan
                    )
                )

                valid_list.append(
                    False
                )

            frame_number += 1

            if frame_number % 100 == 0:

                print(
                    f"{frame_number}"
                    "フレーム処理済み"
                )

    capture.release()

    if frame_number < 5:

        raise ValueError(
            "動画のフレーム数が少なすぎます。"
        )

    valid_mask = np.asarray(
        valid_list,
        dtype=bool
    )

    detection_rate = float(
        valid_mask.mean()
    )

    # v1.8: 動画全体の50%以上を必須にする判定を廃止。
    # 打撃の前後だけ打者が明瞭に映る実戦動画では、全体率が低くても
    # 解析に必要な有効フレームが確保できるため、
    # 「総有効フレーム数」と「連続して追跡できた区間」で品質判定する。
    valid_frame_count = int(valid_mask.sum())
    longest_valid_run = _longest_true_run(valid_mask)
    minimum_valid_frames = max(10, int(round(fps * 0.60)))
    minimum_contiguous_frames = max(5, int(round(fps * 0.18)))

    if (
        valid_frame_count < minimum_valid_frames
        or longest_valid_run < minimum_contiguous_frames
    ):
        raise ValueError(
            "打撃解析に必要な骨格フレームを十分に確保できませんでした。\n"
            f"骨格検出率: {detection_rate:.1%}\n"
            f"有効フレーム: {valid_frame_count}/{frame_number}\n"
            f"最長連続検出: {longest_valid_run}フレーム\n"
            "赤枠を少し広げ、構えからフォロースルーまで打者の全身が"
            "枠内に入るよう調整して再解析してください。"
        )

    world_landmarks = interpolate_landmarks(

        np.asarray(
            world_list
        ),

        valid_mask
    )

    normalized_landmarks = interpolate_landmarks(

        np.asarray(
            normalized_list
        ),

        valid_mask
    )

    impact_frame = estimate_impact_frame(
        world_landmarks,
        fps,
        valid_mask=valid_mask
    )

    # 自動推定位置が補間フレームなら、近傍の実測フレームへ寄せる。
    snapped_impact = _nearest_valid_frame(
        impact_frame, valid_mask, max_distance=max(2, int(round(0.10 * fps)))
    )
    if snapped_impact is None:
        raise ValueError(
            "インパクト候補付近に実測骨格がありません。赤枠や動画を見直してください。"
        )
    impact_frame = int(snapped_impact)

    stance_frame = estimate_stance_frame(
        impact_frame,
        fps
    )
    snapped_stance = _nearest_valid_frame(
        stance_frame, valid_mask, max_distance=max(3, int(round(0.15 * fps)))
    )
    if snapped_stance is not None:
        stance_frame = int(snapped_stance)

    raw_normalized = np.asarray(normalized_list, dtype=float)
    quality = _analysis_quality(
        valid_mask, raw_normalized, impact_frame, fps
    )
    _assert_analysis_quality(quality, fps)

    features = calculate_features(

        world_landmarks,

        fps,

        impact_frame,

        stance_frame,

        batter_side,

        valid_mask=valid_mask
    )

    # 通常画像の保存

    save_frame_image(

        video_path,

        stance_frame,

        IMPACT_DIR
        / f"{prefix}_stance.jpg"
    )

    save_frame_image(

        video_path,

        impact_frame,

        IMPACT_DIR
        / f"{prefix}_impact.jpg"
    )

    # 骨格付き画像の保存

    save_pose_frame_image(

        video_path,

        stance_frame,

        normalized_landmarks[
            stance_frame
        ],

        IMPACT_DIR
        / f"{prefix}_stance_pose.jpg",

        "STANCE"
    )

    save_pose_frame_image(

        video_path,

        impact_frame,

        normalized_landmarks[
            impact_frame
        ],

        IMPACT_DIR
        / f"{prefix}_impact_pose.jpg",

        "IMPACT"
    )

    print()
    print("-" * 60)
    print(
        "動画:",
        Path(video_path).name
    )
    print(
        "フレーム数:",
        frame_number
    )
    print(
        f"FPS: {fps:.2f}"
    )
    print(
        f"骨格検出率: "
        f"{detection_rate:.1%}"
    )
    print(
        f"構え推定時刻: "
        f"{stance_frame / fps:.2f}秒"
    )
    print(
        f"インパクト推定時刻: "
        f"{impact_frame / fps:.2f}秒"
    )
    print("-" * 60)

    return VideoAnalysis(

        path=str(
            video_path
        ),

        fps=fps,

        frame_count=frame_number,

        width=width,

        height=height,

        world_landmarks=world_landmarks,

        normalized_landmarks=normalized_landmarks,

        valid_mask=valid_mask,

        impact_frame=impact_frame,

        stance_frame=stance_frame,

        features=features,
        roi_x0=roi_x0,
        roi_x1=roi_x1,
        quality_score=float(quality["score"]),
        quality_label=str(quality["label"]),
        quality_details=quality
    )


# ============================================================
# 20. 類似度得点の計算
# ============================================================

def similarity_score(
    reference_value,
    target_value,
    tolerance
):
    """
    差が0なら100点。
    差が大きくなるほど得点を下げる。
    """

    if (
        not np.isfinite(
            reference_value
        )
        or not np.isfinite(
            target_value
        )
    ):

        return np.nan

    difference = abs(

        target_value
        - reference_value
    )

    score = (

        100.0

        * math.exp(

            -0.5

            * (
                difference
                / tolerance
            ) ** 2
        )
    )

    return float(

        np.clip(
            score,
            0.0,
            100.0
        )
    )


# ============================================================
# 21. 採点表の作成
# ============================================================

def build_score_table(
    reference_analysis,
    target_analysis,
    selected_names
):
    """
    選択した5項目の採点表を作成する。
    """

    rows = []

    for name in selected_names:

        setting = FEATURES[
            name
        ]

        key = setting[
            "key"
        ]

        reference_value = (
            reference_analysis
            .features[key]
        )

        target_value = (
            target_analysis
            .features[key]
        )

        score = similarity_score(

            reference_value,

            target_value,

            setting[
                "tolerance"
            ]
        )

        rows.append({

            "評価項目":
                name,

            "計測内容":
                setting[
                    "description"
                ],

            "比較選手":
                round(
                    reference_value,
                    3
                ),

            "自分":
                round(
                    target_value,
                    3
                ),

            "単位":
                setting[
                    "unit"
                ],

            "差の絶対値":
                round(
                    abs(
                        target_value
                        - reference_value
                    ),
                    3
                ),

            "得点":
                round(
                    score,
                    1
                )
        })

    score_dataframe = pd.DataFrame(
        rows
    )

    overall_score = float(

        score_dataframe[
            "得点"
        ].mean()
    )

    print()
    print("=" * 60)
    print(
        f"総合点: "
        f"{overall_score:.1f} / 100点"
    )
    print("=" * 60)

    print(score_dataframe.to_string(index=False))

    return score_dataframe


# ============================================================
# 22. 日本語レーダーチャート
# ============================================================

def save_radar_chart(
    score_dataframe
):
    """
    日本語のレーダーチャートを作成する。

    比較選手を100点の基準として、
    自分の類似度を表示する。
    """

    labels = (
        score_dataframe[
            "評価項目"
        ]
        .tolist()
    )

    target_scores = (

        score_dataframe[
            "得点"
        ]

        .astype(
            float
        )

        .tolist()
    )

    reference_scores = [

        100.0

        for _ in labels
    ]

    number_of_items = len(
        labels
    )

    angles = np.linspace(

        0,

        2 * np.pi,

        number_of_items,

        endpoint=False

    ).tolist()

    angles_closed = (
        angles
        + angles[:1]
    )

    target_scores_closed = (
        target_scores
        + target_scores[:1]
    )

    reference_scores_closed = (
        reference_scores
        + reference_scores[:1]
    )

    figure = plt.figure(
        figsize=(
            11,
            10
        )
    )

    axis = figure.add_subplot(
        111,
        polar=True
    )

    axis.set_theta_offset(
        np.pi / 2
    )

    axis.set_theta_direction(
        -1
    )

    axis.plot(

        angles_closed,

        reference_scores_closed,

        linewidth=2,

        linestyle="--",

        marker="o",

        label="比較選手の基準"
    )

    axis.fill(

        angles_closed,

        reference_scores_closed,

        alpha=0.05
    )

    axis.plot(

        angles_closed,

        target_scores_closed,

        linewidth=2.5,

        marker="o",

        label="自分の類似度"
    )

    axis.fill(

        angles_closed,

        target_scores_closed,

        alpha=0.20
    )

    axis.set_xticks(
        angles
    )

    axis.set_xticklabels(
        labels,
        fontsize=13
    )

    axis.tick_params(
        axis="x",
        pad=18
    )

    axis.set_ylim(
        0,
        100
    )

    axis.set_yticks([
        20,
        40,
        60,
        80,
        100
    ])

    axis.set_yticklabels([
        "20",
        "40",
        "60",
        "80",
        "100"
    ])

    axis.grid(
        True
    )

    overall_score = float(

        score_dataframe[
            "得点"
        ].mean()
    )

    axis.set_title(

        "打撃フォーム比較レーダーチャート\n"

        f"総合類似度："
        f"{overall_score:.1f}点",

        fontsize=17,

        pad=40
    )

    axis.legend(

        loc="upper right",

        bbox_to_anchor=(
            1.28,
            1.15
        ),

        fontsize=11
    )

    for angle, score in zip(
        angles,
        target_scores
    ):

        display_radius = min(
            score + 8,
            105
        )

        axis.text(

            angle,

            display_radius,

            f"{score:.1f}",

            horizontalalignment="center",

            verticalalignment="center",

            fontsize=11,

            fontweight="bold"
        )

    plt.tight_layout()

    output_path = (
        GRAPH_DIR
        / "打撃フォーム比較レーダーチャート.png"
    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print()
    print(
        "日本語版レーダーチャートを保存しました。"
    )

    print(
        output_path
    )


# ============================================================
# 23. 比較用棒グラフ
# ============================================================

def save_comparison_bar_chart(
    score_dataframe
):
    """
    各項目の類似度を棒グラフで表示する。
    """

    figure = plt.figure(
        figsize=(
            11,
            7
        )
    )

    axis = figure.add_subplot(
        111
    )

    labels = (
        score_dataframe[
            "評価項目"
        ].tolist()
    )

    scores = (

        score_dataframe[
            "得点"
        ]

        .astype(
            float
        )

        .tolist()
    )

    bars = axis.bar(
        labels,
        scores
    )

    axis.set_ylim(
        0,
        100
    )

    axis.set_ylabel(
        "比較選手との類似度（点）",
        fontsize=12
    )

    axis.set_xlabel(
        "評価項目",
        fontsize=12
    )

    axis.set_title(
        "評価項目別のフォーム類似度",
        fontsize=16
    )

    axis.tick_params(
        axis="x",
        rotation=20
    )

    axis.grid(
        axis="y",
        alpha=0.3
    )

    for bar, score in zip(
        bars,
        scores
    ):

        axis.text(

            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 2,

            f"{score:.1f}",

            ha="center",

            va="bottom",

            fontsize=11
        )

    plt.tight_layout()

    output_path = (
        GRAPH_DIR
        / "評価項目別類似度棒グラフ.png"
    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print()
    print(
        "評価項目別棒グラフを保存しました。"
    )

    print(
        output_path
    )


# ============================================================
# 24. 全8項目の特徴量保存
# ============================================================

def save_all_features(
    reference_analysis,
    target_analysis
):
    """
    選択の有無にかかわらず、
    全8項目の特徴量を保存する。
    """

    rows = []

    for name, setting in FEATURES.items():

        key = setting[
            "key"
        ]

        reference_value = (
            reference_analysis
            .features[key]
        )

        target_value = (
            target_analysis
            .features[key]
        )

        rows.append({

            "評価項目":
                name,

            "計測内容":
                setting[
                    "description"
                ],

            "比較選手":
                round(
                    reference_value,
                    4
                ),

            "自分":
                round(
                    target_value,
                    4
                ),

            "単位":
                setting[
                    "unit"
                ],

            "差の絶対値":
                round(
                    abs(
                        target_value
                        - reference_value
                    ),
                    4
                )
        })

    all_features_dataframe = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIR
        / "all_extracted_features.csv"
    )

    all_features_dataframe.to_csv(

        output_path,

        index=False,

        encoding="utf-8-sig"
    )

    print()
    print(
        "全8項目の特徴量を保存しました。"
    )

    print(
        output_path
    )

    return all_features_dataframe


# ============================================================
# 25. 改善提案の作成
# ============================================================

def generate_improvement_advice(
    score_dataframe,
    reference_analysis,
    target_analysis
):
    """
    得点と全8特徴量から、
    改善点と推奨練習を自動生成する。

    注意:
    筋力を直接測定しているわけではないため、
    「筋力不足」と断定せず、
    動作改善の可能性として提示する。
    """

    advice_rows = []

    score_dictionary = dict(

        zip(

            score_dataframe[
                "評価項目"
            ],

            score_dataframe[
                "得点"
            ]
        )
    )

    reference_features = (
        reference_analysis
        .features
    )

    target_features = (
        target_analysis
        .features
    )

    def add_advice(
        category,
        priority,
        finding,
        advice,
        training
    ):

        advice_rows.append({

            "分類":
                category,

            "優先度":
                priority,

            "検出内容":
                finding,

            "改善の方向性":
                advice,

            "推奨練習":
                training
        })

    # --------------------------------------------------------
    # 身体の開き
    # --------------------------------------------------------

    if (
        "身体を開くな"
        in score_dictionary
    ):

        score = score_dictionary[
            "身体を開くな"
        ]

        reference_value = (
            reference_features[
                "shoulder_opening"
            ]
        )

        target_value = (
            target_features[
                "shoulder_opening"
            ]
        )

        if score < 60:

            if target_value > reference_value:

                add_advice(

                    "上半身",

                    "高",

                    (
                        "構えからインパクトまでの"
                        "肩の回転量が比較選手より"
                        f"{target_value - reference_value:.1f}度"
                        "大きくなっています。"
                    ),

                    (
                        "上半身が先に開いている可能性があります。"
                        "腰の回転を先行させ、"
                        "その後に肩と腕を回す順序を"
                        "意識してください。"
                    ),

                    (
                        "前肩を残したティー打撃、"
                        "ステップ打撃、"
                        "下半身主導のシャドースイング"
                    )
                )

            else:

                add_advice(

                    "上半身",

                    "中",

                    (
                        "肩の回転量が比較選手より"
                        f"{reference_value - target_value:.1f}度"
                        "小さくなっています。"
                    ),

                    (
                        "肩の開きを抑えすぎて、"
                        "回転が止まっていないか"
                        "確認してください。"
                    ),

                    (
                        "ゆっくりした連続素振り、"
                        "腰から肩へ回転を伝える練習"
                    )
                )

    # --------------------------------------------------------
    # 割れ
    # --------------------------------------------------------

    if (
        "割れを作れ"
        in score_dictionary
    ):

        score = score_dictionary[
            "割れを作れ"
        ]

        reference_value = (
            reference_features[
                "hip_shoulder_separation"
            ]
        )

        target_value = (
            target_features[
                "hip_shoulder_separation"
            ]
        )

        if score < 60:

            if target_value < reference_value:

                add_advice(

                    "捻転差",

                    "高",

                    (
                        "肩と腰の最大捻転差が"
                        "比較選手より"
                        f"{reference_value - target_value:.1f}度"
                        "小さくなっています。"
                    ),

                    (
                        "踏み込み時に胸を残しながら、"
                        "骨盤を先に回転させる動作を"
                        "意識してください。"
                    ),

                    (
                        "割れを作るステップドリル、"
                        "メディシンボール回旋投げ、"
                        "チューブを使った体幹回旋"
                    )
                )

            else:

                add_advice(

                    "捻転差",

                    "中",

                    (
                        "肩と腰の最大捻転差が"
                        "比較選手より"
                        f"{target_value - reference_value:.1f}度"
                        "大きくなっています。"
                    ),

                    (
                        "捻りを大きくしすぎて、"
                        "動作が遅れたり姿勢が"
                        "崩れていないか確認してください。"
                    ),

                    (
                        "小さなステップから始める連続打撃、"
                        "ゆっくりした分解素振り"
                    )
                )

    # --------------------------------------------------------
    # ボールを引きつける動作
    # --------------------------------------------------------

    if (
        "ボールを引きつけろ"
        in score_dictionary
    ):

        score = score_dictionary[
            "ボールを引きつけろ"
        ]

        reference_value = (
            reference_features[
                "hand_torso_distance"
            ]
        )

        target_value = (
            target_features[
                "hand_torso_distance"
            ]
        )

        if score < 70:

            if target_value > reference_value:

                add_advice(

                    "バット軌道",

                    "高",

                    (
                        "インパクト推定時の"
                        "手と体幹中心の距離が、"
                        "比較選手より大きくなっています。"
                    ),

                    (
                        "手が体から離れ、"
                        "バットが外回りしている可能性があります。"
                        "グリップを体の近くから"
                        "出す意識が有効です。"
                    ),

                    (
                        "内角ティー、"
                        "脇にタオルを挟んだ素振り、"
                        "短いバットを使った"
                        "インサイドアウト練習"
                    )
                )

            else:

                add_advice(

                    "バット軌道",

                    "中",

                    (
                        "手と体幹中心の距離が"
                        "比較選手より小さくなっています。"
                    ),

                    (
                        "手を体に近づけすぎて、"
                        "窮屈になっていないか"
                        "確認してください。"
                    ),

                    (
                        "コース別ティー打撃、"
                        "インパクト位置の確認"
                    )
                )

    # --------------------------------------------------------
    # 軸の安定性
    # --------------------------------------------------------

    if (
        "軸をぶらすな"
        in score_dictionary
    ):

        score = score_dictionary[
            "軸をぶらすな"
        ]

        reference_value = (
            reference_features[
                "axis_sway"
            ]
        )

        target_value = (
            target_features[
                "axis_sway"
            ]
        )

        if score < 70:

            if target_value > reference_value:

                finding = (
                    "体幹中心の横移動量が、"
                    "比較選手より大きくなっています。"
                )

            else:

                finding = (
                    "体幹中心の横移動量が、"
                    "比較選手より小さくなっています。"
                )

            add_advice(

                "姿勢安定",

                "高",

                finding,

                (
                    "頭と骨盤が同時に大きく流れないように、"
                    "股関節を中心とした回転を"
                    "意識してください。"
                ),

                (
                    "片脚立ち、"
                    "スプリットスクワット、"
                    "軸足を固定したティー打撃"
                )
            )

    # --------------------------------------------------------
    # 前脚の壁
    # --------------------------------------------------------

    if (
        "壁を作れ"
        in score_dictionary
    ):

        score = score_dictionary[
            "壁を作れ"
        ]

        reference_value = (
            reference_features[
                "front_knee_angle"
            ]
        )

        target_value = (
            target_features[
                "front_knee_angle"
            ]
        )

        if score < 60:

            if target_value > reference_value:

                add_advice(

                    "前脚",

                    "高",

                    (
                        "インパクト推定時の前脚膝角度が"
                        "比較選手より"
                        f"{target_value - reference_value:.1f}度"
                        "大きく、前脚がより伸びています。"
                    ),

                    (
                        "前脚が早く伸び切っていないか、"
                        "踏み込みから回転を受け止める"
                        "タイミングを確認してください。"
                    ),

                    (
                        "前脚で止まるステップ打撃、"
                        "スプリットスクワット、"
                        "片脚スクワット"
                    )
                )

            else:

                add_advice(

                    "前脚",

                    "高",

                    (
                        "インパクト推定時の前脚膝角度が"
                        "比較選手より"
                        f"{reference_value - target_value:.1f}度"
                        "小さく、膝が深く曲がっています。"
                    ),

                    (
                        "前脚で回転を受け止めきれず、"
                        "体が前へ流れていないか"
                        "確認してください。"
                    ),

                    (
                        "踏み込み後に静止するドリル、"
                        "ランジ、"
                        "前脚の等尺性トレーニング"
                    )
                )

    # --------------------------------------------------------
    # 頭部の安定性
    # --------------------------------------------------------

    head_reference = (
        reference_features[
            "head_movement"
        ]
    )

    head_target = (
        target_features[
            "head_movement"
        ]
    )

    if (
        head_target
        > head_reference + 0.20
    ):

        add_advice(

            "頭部安定",

            "中",

            (
                "比較選手より頭部の移動量が"
                "大きくなっています。"
            ),

            (
                "ボールを見る位置を安定させ、"
                "頭を前へ突っ込ませずに"
                "回転してください。"
            ),

            (
                "頭の横に目印を置いた素振り、"
                "ノーステップ打撃、"
                "体幹の抗回旋トレーニング"
            )
        )

    # --------------------------------------------------------
    # 体重移動
    # --------------------------------------------------------

    transfer_reference = (
        reference_features[
            "weight_transfer"
        ]
    )

    transfer_target = (
        target_features[
            "weight_transfer"
        ]
    )

    transfer_difference = abs(

        transfer_target
        - transfer_reference
    )

    if transfer_difference > 0.25:

        if transfer_target < transfer_reference:

            add_advice(

                "体重移動",

                "中",

                (
                    "骨盤中心の移動量が"
                    "比較選手より小さくなっています。"
                ),

                (
                    "軸足から前脚へ力を移す動作が"
                    "十分でない可能性があります。"
                ),

                (
                    "歩きながらのティー打撃、"
                    "ステップ打撃、"
                    "サイドランジ"
                )
            )

        else:

            add_advice(

                "体重移動",

                "中",

                (
                    "骨盤中心の移動量が"
                    "比較選手より大きくなっています。"
                ),

                (
                    "重心が前へ流れすぎていないか確認し、"
                    "踏み込んだ前脚で止まる動作を"
                    "意識してください。"
                ),

                (
                    "ノーステップ打撃、"
                    "前脚で静止するドリル、"
                    "スプリットスクワット"
                )
            )

    # --------------------------------------------------------
    # 腰の回転速度
    # --------------------------------------------------------

    hip_speed_reference = (
        reference_features[
            "hip_rotation_speed"
        ]
    )

    hip_speed_target = (
        target_features[
            "hip_rotation_speed"
        ]
    )

    if hip_speed_target < (
        hip_speed_reference
        * 0.75
    ):

        add_advice(

            "下半身",

            "高",

            (
                "腰の最大回転速度が、"
                "比較選手の75％未満です。"
            ),

            (
                "下半身から生み出した力を"
                "骨盤回転へつなげる能力を"
                "高めることが有効と考えられます。"
            ),

            (
                "スクワット、"
                "ブルガリアンスクワット、"
                "サイドランジ、"
                "メディシンボール回旋投げ"
            )
        )

    # --------------------------------------------------------
    # 複合判定：下半身の強化
    # --------------------------------------------------------

    lower_body_problem_count = 0

    if hip_speed_target < (
        hip_speed_reference
        * 0.80
    ):

        lower_body_problem_count += 1

    if transfer_difference > 0.25:

        lower_body_problem_count += 1

    if (
        "壁を作れ"
        in score_dictionary
        and score_dictionary[
            "壁を作れ"
        ] < 60
    ):

        lower_body_problem_count += 1

    if lower_body_problem_count >= 2:

        add_advice(

            "総合提案",

            "高",

            (
                "前脚、体重移動、腰回転のうち、"
                "複数項目で比較選手との差が"
                "検出されました。"
            ),

            (
                "下半身の筋力だけでなく、"
                "踏み込み、前脚での受け止め、"
                "骨盤回転を連動させる練習が"
                "有効と考えられます。"
            ),

            (
                "スクワット、"
                "ブルガリアンスクワット、"
                "片脚立ち、"
                "メディシンボール回旋投げ、"
                "ステップ打撃"
            )
        )

    # --------------------------------------------------------
    # 改善点が検出されなかった場合
    # --------------------------------------------------------

    if not advice_rows:

        add_advice(

            "総合評価",

            "低",

            (
                "選択した評価項目では、"
                "比較選手との差が比較的小さい結果でした。"
            ),

            (
                "現在の動きを維持しながら、"
                "インパクト位置とフォームの再現性を"
                "確認してください。"
            ),

            (
                "同じ撮影条件で複数回スイングし、"
                "平均値とばらつきを確認する"
            )
        )

    advice_dataframe = pd.DataFrame(
        advice_rows
    )

    priority_order = {
        "高": 0,
        "中": 1,
        "低": 2
    }

    advice_dataframe[
        "優先順位"
    ] = (

        advice_dataframe[
            "優先度"
        ]

        .map(
            priority_order
        )
    )

    advice_dataframe = (

        advice_dataframe

        .sort_values(
            "優先順位"
        )

        .drop(
            columns="優先順位"
        )

        .reset_index(
            drop=True
        )
    )

    return advice_dataframe


# ============================================================
# 26. 改善提案の保存
# ============================================================

def save_improvement_advice(
    advice_dataframe
):
    """
    改善提案をCSVとテキストで保存する。
    """

    csv_path = (
        OUTPUT_DIR
        / "打撃フォーム改善提案.csv"
    )

    text_path = (
        OUTPUT_DIR
        / "打撃フォーム改善提案.txt"
    )

    advice_dataframe.to_csv(

        csv_path,

        index=False,

        encoding="utf-8-sig"
    )

    with open(

        text_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "打撃フォーム改善提案\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(

            "注意事項：\n"

            "この結果は、動画から取得した骨格座標と"
            "比較選手との差に基づく推定です。\n"

            "筋力不足や技術的な問題を"
            "直接診断するものではありません。\n\n"
        )

        for index, row in (
            advice_dataframe
            .iterrows()
        ):

            file.write(
                f"【提案{index + 1}】\n"
            )

            file.write(
                f"分類："
                f"{row['分類']}\n"
            )

            file.write(
                f"優先度："
                f"{row['優先度']}\n"
            )

            file.write(
                f"検出内容："
                f"{row['検出内容']}\n"
            )

            file.write(
                f"改善の方向性："
                f"{row['改善の方向性']}\n"
            )

            file.write(
                f"推奨練習："
                f"{row['推奨練習']}\n\n"
            )

    print()
    print(
        "改善提案を保存しました。"
    )

    print(
        csv_path
    )

    print(
        text_path
    )


# ============================================================
# 27. 解析情報の保存
# ============================================================

def save_analysis_information(
    reference_analysis,
    target_analysis,
    batter_side
):
    """
    動画のFPS、フレーム数、推定時刻などを保存する。
    """

    information_dataframe = pd.DataFrame([

        {
            "区分":
                "比較選手",

            "動画パス":
                reference_analysis.path,

            "打席":
                batter_side,

            "FPS":
                reference_analysis.fps,

            "フレーム数":
                reference_analysis.frame_count,

            "骨格検出率":
                reference_analysis
                .valid_mask
                .mean(),

            "解析信頼度":
                reference_analysis.quality_score,

            "解析信頼度判定":
                reference_analysis.quality_label,

            "インパクト周辺検出率":
                (reference_analysis.quality_details or {}).get("impact_window_detection_rate", np.nan),

            "構え推定フレーム":
                reference_analysis
                .stance_frame,

            "構え推定時刻":
                reference_analysis
                .stance_frame
                / reference_analysis
                .fps,

            "インパクト推定フレーム":
                reference_analysis
                .impact_frame,

            "インパクト推定時刻":
                reference_analysis
                .impact_frame
                / reference_analysis
                .fps
        },

        {
            "区分":
                "自分",

            "動画パス":
                target_analysis.path,

            "打席":
                batter_side,

            "FPS":
                target_analysis.fps,

            "フレーム数":
                target_analysis.frame_count,

            "骨格検出率":
                target_analysis
                .valid_mask
                .mean(),

            "解析信頼度":
                target_analysis.quality_score,

            "解析信頼度判定":
                target_analysis.quality_label,

            "インパクト周辺検出率":
                (target_analysis.quality_details or {}).get("impact_window_detection_rate", np.nan),

            "構え推定フレーム":
                target_analysis
                .stance_frame,

            "構え推定時刻":
                target_analysis
                .stance_frame
                / target_analysis
                .fps,

            "インパクト推定フレーム":
                target_analysis
                .impact_frame,

            "インパクト推定時刻":
                target_analysis
                .impact_frame
                / target_analysis
                .fps
        }
    ])

    output_path = (
        OUTPUT_DIR
        / "video_analysis_information.csv"
    )

    information_dataframe.to_csv(

        output_path,

        index=False,

        encoding="utf-8-sig"
    )

    print()
    print(
        "動画解析情報を保存しました。"
    )

    print(
        output_path
    )



# ============================================================
# 28. Web/GUIから呼び出す統合解析関数
# ============================================================

def run_analysis(
    reference_original_path,
    target_original_path,
    batter_side,
    selected_feature_names,
    output_dir=None,
    reference_batter_position="自動",
    target_batter_position="自動",
    reference_batter_roi=None,
    target_batter_roi=None
):
    """
    現在の解析アルゴリズムを、画面UIから1回の関数呼び出しで実行する。
    """
    if batter_side not in ("右打者", "左打者"):
        raise ValueError("batter_side は「右打者」または「左打者」を指定してください。")

    selected_feature_names = list(selected_feature_names)

    if len(selected_feature_names) != 5:
        raise ValueError("評価項目は5項目選択してください。")

    invalid = [name for name in selected_feature_names if name not in FEATURES]
    if invalid:
        raise ValueError(f"存在しない評価項目があります: {invalid}")

    valid_positions = {"自動", "左", "中央", "右"}
    if reference_batter_position not in valid_positions:
        raise ValueError(f"比較選手の打者位置指定が不正です: {reference_batter_position}")
    if target_batter_position not in valid_positions:
        raise ValueError(f"自分の打者位置指定が不正です: {target_batter_position}")

    if output_dir is not None:
        configure_output_directory(output_dir)

    reference_video_path = prepare_video_for_analysis(
        Path(reference_original_path),
        "reference"
    )
    target_video_path = prepare_video_for_analysis(
        Path(target_original_path),
        "target"
    )

    reference_analysis = analyze_video(
        reference_video_path,
        batter_side,
        "reference",
        batter_position=reference_batter_position,
        batter_roi=reference_batter_roi
    )
    target_analysis = analyze_video(
        target_video_path,
        batter_side,
        "target",
        batter_position=target_batter_position,
        batter_roi=target_batter_roi
    )

    score_dataframe = build_score_table(
        reference_analysis,
        target_analysis,
        selected_feature_names
    )

    score_csv_path = OUTPUT_DIR / "batting_form_comparison_result.csv"
    score_dataframe.to_csv(
        score_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    save_radar_chart(score_dataframe)
    save_comparison_bar_chart(score_dataframe)

    all_features_dataframe = save_all_features(
        reference_analysis,
        target_analysis
    )

    advice_dataframe = generate_improvement_advice(
        score_dataframe,
        reference_analysis,
        target_analysis
    )
    save_improvement_advice(advice_dataframe)

    save_analysis_information(
        reference_analysis,
        target_analysis,
        batter_side
    )

    overall_score = float(score_dataframe["得点"].mean())
    best_row = score_dataframe.loc[score_dataframe["得点"].idxmax()]
    lowest_row = score_dataframe.loc[score_dataframe["得点"].idxmin()]

    return {
        "overall_score": overall_score,
        "score_dataframe": score_dataframe,
        "all_features_dataframe": all_features_dataframe,
        "advice_dataframe": advice_dataframe,
        "reference_analysis": reference_analysis,
        "target_analysis": target_analysis,
        "best_item": str(best_row["評価項目"]),
        "best_score": float(best_row["得点"]),
        "lowest_item": str(lowest_row["評価項目"]),
        "lowest_score": float(lowest_row["得点"]),
        "output_dir": Path(OUTPUT_DIR),
        "analysis_mode": "高精度",
        "reference_batter_position": reference_batter_position,
        "target_batter_position": target_batter_position,
        "reference_batter_roi": reference_batter_roi,
        "target_batter_roi": target_batter_roi,
    }


# ============================================================
# 29. インパクト目視確認・手動補正用
# ============================================================

def get_pose_preview_frame(analysis, frame_number):
    """指定フレームを骨格付きRGB画像として返す。Streamlitプレビュー用。"""
    frame_number = int(np.clip(
        int(frame_number),
        0,
        max(0, analysis.frame_count - 1)
    ))

    capture = cv2.VideoCapture(str(analysis.path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        return None

    if 0 <= frame_number < len(analysis.normalized_landmarks):
        landmarks = analysis.normalized_landmarks[frame_number]
        if np.isfinite(landmarks[:, :3]).any():
            frame = draw_pose_on_frame(frame, landmarks)

    cv2.putText(
        frame,
        f"FRAME {frame_number}  {frame_number / analysis.fps:.3f}s",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def is_measured_frame(analysis, frame_number):
    """指定フレームが補間ではなく実測骨格かを返す。"""
    frame_number = int(np.clip(int(frame_number), 0, max(0, analysis.frame_count - 1)))
    return bool(analysis.valid_mask[frame_number])


def _apply_manual_impact(analysis, impact_frame, batter_side, prefix):
    """指定したインパクトフレームで特徴量と確認画像を作り直す。"""
    impact_frame = int(np.clip(
        int(impact_frame),
        0,
        max(0, analysis.frame_count - 1)
    ))
    stance_frame = estimate_stance_frame(
        impact_frame,
        analysis.fps
    )

    analysis.impact_frame = impact_frame
    analysis.stance_frame = stance_frame
    if not bool(analysis.valid_mask[impact_frame]):
        raise ValueError(
            "選択したインパクトフレームは補間フレームです。骨格が実測されている近くのフレームを選択してください。"
        )

    analysis.features = calculate_features(
        analysis.world_landmarks,
        analysis.fps,
        impact_frame,
        stance_frame,
        batter_side,
        valid_mask=analysis.valid_mask
    )

    quality = _analysis_quality(
        analysis.valid_mask,
        analysis.normalized_landmarks,
        impact_frame,
        analysis.fps
    )
    _assert_analysis_quality(quality, analysis.fps)
    analysis.quality_score = float(quality["score"])
    analysis.quality_label = str(quality["label"])
    analysis.quality_details = quality

    save_frame_image(
        analysis.path,
        stance_frame,
        IMPACT_DIR / f"{prefix}_stance.jpg"
    )
    save_frame_image(
        analysis.path,
        impact_frame,
        IMPACT_DIR / f"{prefix}_impact.jpg"
    )
    save_pose_frame_image(
        analysis.path,
        stance_frame,
        analysis.normalized_landmarks[stance_frame],
        IMPACT_DIR / f"{prefix}_stance_pose.jpg",
        "STANCE"
    )
    save_pose_frame_image(
        analysis.path,
        impact_frame,
        analysis.normalized_landmarks[impact_frame],
        IMPACT_DIR / f"{prefix}_impact_pose.jpg",
        "IMPACT"
    )


def recalculate_with_manual_impacts(
    result,
    reference_impact_frame,
    target_impact_frame,
    selected_feature_names,
    batter_side,
):
    """
    すでにMediaPipe解析済みの骨格座標を再利用して、
    インパクトフレームだけを変更して高速に再採点する。
    MediaPipeの再実行は行わない。
    """
    output_dir = Path(result["output_dir"])
    configure_output_directory(output_dir)

    reference_analysis = result["reference_analysis"]
    target_analysis = result["target_analysis"]

    _apply_manual_impact(
        reference_analysis,
        reference_impact_frame,
        batter_side,
        "reference"
    )
    _apply_manual_impact(
        target_analysis,
        target_impact_frame,
        batter_side,
        "target"
    )

    score_dataframe = build_score_table(
        reference_analysis,
        target_analysis,
        selected_feature_names
    )
    score_dataframe.to_csv(
        OUTPUT_DIR / "batting_form_comparison_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_radar_chart(score_dataframe)
    save_comparison_bar_chart(score_dataframe)

    all_features_dataframe = save_all_features(
        reference_analysis,
        target_analysis
    )
    advice_dataframe = generate_improvement_advice(
        score_dataframe,
        reference_analysis,
        target_analysis
    )
    save_improvement_advice(advice_dataframe)
    save_analysis_information(
        reference_analysis,
        target_analysis,
        batter_side
    )

    overall_score = float(score_dataframe["得点"].mean())
    best_row = score_dataframe.loc[score_dataframe["得点"].idxmax()]
    lowest_row = score_dataframe.loc[score_dataframe["得点"].idxmin()]

    return {
        "overall_score": overall_score,
        "score_dataframe": score_dataframe,
        "all_features_dataframe": all_features_dataframe,
        "advice_dataframe": advice_dataframe,
        "reference_analysis": reference_analysis,
        "target_analysis": target_analysis,
        "best_item": str(best_row["評価項目"]),
        "best_score": float(best_row["得点"]),
        "lowest_item": str(lowest_row["評価項目"]),
        "lowest_score": float(lowest_row["得点"]),
        "output_dir": Path(OUTPUT_DIR),
        "analysis_mode": "高精度",
    }
