from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

APP_VERSION = "10.1"
DATA_ROOT = Path(
    os.environ.get(
        "BFA_DATA_DIR",
        str(Path.home() / ".batting_form_analyzer_v10"),
    )
)
DB_PATH = DATA_ROOT / "players.json"


def is_cloud_runtime() -> bool:
    """Streamlit Community Cloud等の一時ファイル環境をおおまかに判定。"""
    markers = [
        "/mount/src",
        "/home/adminuser",
    ]
    cwd = str(Path.cwd())
    home = str(Path.home())
    return any(marker in cwd or marker in home for marker in markers)


def _empty_db() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "app_version": APP_VERSION,
        "players": [],
    }


def _ensure_dir() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_db() -> Dict[str, Any]:
    _ensure_dir()
    if not DB_PATH.exists():
        return _empty_db()
    try:
        payload = json.loads(DB_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _empty_db()
        payload.setdefault("schema_version", 1)
        payload.setdefault("app_version", APP_VERSION)
        payload.setdefault("players", [])
        return payload
    except Exception:
        return _empty_db()


def save_db(db: Dict[str, Any]) -> None:
    _ensure_dir()
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(DB_PATH)


def list_players() -> List[Dict[str, Any]]:
    db = load_db()
    players = deepcopy(db.get("players", []))
    players.sort(
        key=lambda item: item.get("updated_at", item.get("created_at", "")),
        reverse=True,
    )
    return players


def get_player(player_id: str) -> Optional[Dict[str, Any]]:
    if not player_id:
        return None
    for player in load_db().get("players", []):
        if player.get("id") == player_id:
            return deepcopy(player)
    return None


def create_player(
    name: str,
    batter_side: str,
    team: str = "",
    note: str = "",
) -> Dict[str, Any]:
    name = str(name).strip()
    if not name:
        raise ValueError("選手名またはニックネームを入力してください。")
    if batter_side not in {"右打者", "左打者"}:
        raise ValueError("打席は右打者または左打者を指定してください。")

    now = datetime.now().isoformat(timespec="seconds")
    player = {
        "id": str(uuid.uuid4()),
        "name": name,
        "batter_side": batter_side,
        "team": str(team).strip(),
        "note": str(note).strip(),
        "created_at": now,
        "updated_at": now,
        "analyses": [],
    }
    db = load_db()
    db["players"].append(player)
    save_db(db)
    return deepcopy(player)


def update_player(
    player_id: str,
    name: Optional[str] = None,
    batter_side: Optional[str] = None,
    team: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    db = load_db()
    for player in db.get("players", []):
        if player.get("id") != player_id:
            continue
        if name is not None and str(name).strip():
            player["name"] = str(name).strip()
        if batter_side is not None:
            if batter_side not in {"右打者", "左打者"}:
                raise ValueError("打席は右打者または左打者を指定してください。")
            player["batter_side"] = batter_side
        if team is not None:
            player["team"] = str(team).strip()
        if note is not None:
            player["note"] = str(note).strip()
        player["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_db(db)
        return deepcopy(player)
    raise KeyError("選手ファイルが見つかりません。")


def delete_player(player_id: str) -> None:
    db = load_db()
    before = len(db.get("players", []))
    db["players"] = [p for p in db.get("players", []) if p.get("id") != player_id]
    if len(db["players"]) == before:
        raise KeyError("選手ファイルが見つかりません。")
    save_db(db)


def add_analysis(player_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    db = load_db()
    now = datetime.now().isoformat(timespec="seconds")
    saved = deepcopy(record)
    saved.setdefault("id", str(uuid.uuid4()))
    saved.setdefault("created_at", now)
    saved.setdefault("app_version", APP_VERSION)

    for player in db.get("players", []):
        if player.get("id") != player_id:
            continue
        player.setdefault("analyses", []).append(saved)
        player["updated_at"] = now
        save_db(db)
        return deepcopy(saved)
    raise KeyError("選手ファイルが見つかりません。")


def export_player_bundle(player_id: str) -> bytes:
    player = get_player(player_id)
    if player is None:
        raise KeyError("選手ファイルが見つかりません。")
    bundle = {
        "format": "BFA_PLAYER_BUNDLE",
        "schema_version": 1,
        "app_version": APP_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "player": player,
    }
    return json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")


def import_player_bundle(raw: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("選手ファイルを読み込めませんでした。") from exc

    if payload.get("format") != "BFA_PLAYER_BUNDLE":
        raise ValueError("このファイルは対応する選手ファイルではありません。")

    incoming = payload.get("player")
    if not isinstance(incoming, dict) or not incoming.get("name"):
        raise ValueError("選手データが壊れています。")

    db = load_db()
    imported = deepcopy(incoming)
    existing_ids = {p.get("id") for p in db.get("players", [])}
    if imported.get("id") in existing_ids:
        imported["id"] = str(uuid.uuid4())
        imported["name"] = f"{imported.get('name', '選手')}（読込）"
    imported.setdefault("analyses", [])
    imported.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    imported["updated_at"] = datetime.now().isoformat(timespec="seconds")
    db["players"].append(imported)
    save_db(db)
    return deepcopy(imported)


def analysis_history_dataframe(player: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for record in player.get("analyses", []):
        row = {
            "解析日時": record.get("created_at", ""),
            "比較対象": record.get("reference_name", "比較選手"),
            "比較結果": record.get("overall_comparison_label", ""),
            "平均差尺度": record.get("mean_normalized_difference"),
            "旧フォーム類似度": record.get("overall_score"),
            "解析信頼度": record.get("target_quality"),
            "身体負担判定": record.get("body_load_label", ""),
        }
        scores = record.get("scores", {}) or {}
        for name, value in scores.items():
            row[name] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "解析日時" in df.columns:
        df = df.sort_values("解析日時")
    return df.reset_index(drop=True)
