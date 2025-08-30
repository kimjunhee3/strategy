# Str_cache.py
# 안정적 시드+증분 캐시 모듈 (드롭다운/초기 UI 안전화)
# - 시드 경로: ./static/seed/*.csv
# - 캐시 경로: ./cache/*.csv
# - 메타: ./cache/manifest.json

from __future__ import annotations
import os, json, tempfile
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# =========================
# 경로/상수
# =========================
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
SEED_DIR  = BASE_DIR / "static" / "seed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 파일 스키마 버전 (컬럼/형식 바뀌면 반드시 갱신)
SCHEMA_VERSION = "2025-08-30-v1"

# 드롭다운 최소 보장용
TEAM_LIST_FALLBACK = ["LG","KT","KIA","SSG","두산","롯데","삼성","NC","키움","한화"]

# 캐시 대상(파일명 -> 필수 컬럼)
REQUIRED: Dict[str, List[str]] = {
    "df_hit.csv":    ["team","date","H","HR","AB"],
    "df_run.csv":    ["team","date","SB","CS"],
    "df_pitch1.csv": ["team","date","ERA","IP","WHIP"],
    "df_pitch2.csv": ["team","date","QS","SV","HLD"],
    "df_def.csv":    ["team","date","E","FPCT","A"],
}

MANIFEST = CACHE_DIR / "manifest.json"


# =========================
# 유틸
# =========================
def _atomic_write_csv(path: Path, df: pd.DataFrame):
    """임시 파일에 저장 후 교체로 부분 손상 방지"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=str(CACHE_DIR)) as tmp:
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        tmp.flush()
        os.replace(tmp.name, path)

def _normalize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "team" in df.columns:
        # 팀 문자열 자잘한 공백/널 방지
        df["team"] = df["team"].astype(str).str.strip()
    return df

def _load_or_seed(filename: str, columns: List[str]) -> pd.DataFrame:
    """캐시→시드→빈헤더 순으로 로드하고 캐시에 보장 저장"""
    cache_path = CACHE_DIR / filename
    # 1) 캐시 존재 시
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            df = _normalize_columns(df, columns)
            return df
        except Exception:
            pass  # 손상 시 아래로 폴백

    # 2) 시드 존재 시
    seed_path = SEED_DIR / filename
    if seed_path.exists():
        try:
            df = pd.read_csv(seed_path)
            df = _normalize_columns(df, columns)
            _atomic_write_csv(cache_path, df)
            return df
        except Exception:
            pass

    # 3) 빈 헤더
    df = pd.DataFrame(columns=columns)
    _atomic_write_csv(cache_path, df)
    return df

def _ensure_manifest() -> dict:
    if MANIFEST.exists():
        try:
            meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("schema changed")
            # 필수 키 보정
            if "high_watermark" not in meta:
                meta["high_watermark"] = {k: None for k in REQUIRED}
            for k in REQUIRED:
                meta["high_watermark"].setdefault(k, None)
            return meta
        except Exception:
            pass
    meta = {
        "schema_version": SCHEMA_VERSION,
        "last_updated": None,
        "high_watermark": {k: None for k in REQUIRED},  # 파일별 마지막 날짜(YYYY-MM-DD)
    }
    MANIFEST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def _save_manifest(meta: dict):
    MANIFEST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _drop_dupes_by_team_date(df: pd.DataFrame) -> pd.DataFrame:
    subset = [c for c in ["team","date"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    if "date" in df.columns:
        df = df.sort_values("date")
    return df


# =========================
# 공개 API (부트/로드/옵션)
# =========================
_cache_frames: Dict[str, pd.DataFrame] = {}
_manifest: Optional[dict] = None

def bootstrap_cache() -> None:
    """앱 시작 시 1회 호출: 캐시 프레임/매니페스트 보장"""
    global _cache_frames, _manifest
    _cache_frames = {name: _load_or_seed(name, cols) for name, cols in REQUIRED.items()}
    _manifest = _ensure_manifest()

def load_cache_frames() -> Dict[str, pd.DataFrame]:
    """현재 메모리 상의 캐시 프레임 반환 (필요 시 bootstrap_cache 먼저 호출)"""
    if not _cache_frames:
        bootstrap_cache()
    return _cache_frames

def get_team_options() -> List[str]:
    """드롭다운 팀 목록(항상 값 보장)"""
    frames = load_cache_frames()
    for key in ["df_hit.csv", "df_pitch1.csv", "df_def.csv", "df_pitch2.csv", "df_run.csv"]:
        df = frames.get(key)
        if df is not None and "team" in df and df["team"].notna().any():
            uniq = sorted([t for t in df["team"].dropna().unique().tolist() if str(t).strip()])
            if uniq:
                return uniq
    return TEAM_LIST_FALLBACK


# =========================
# 증분 갱신
# =========================
class CacheScraper:
    """
    프로젝트 내 기존 크롤러를 얇게 감싼 인터페이스.
    반드시 아래 메서드를 구현하여 주입하세요.

    def fetch_between(self, start: date, end: date) -> Dict[str, pd.DataFrame]:
        # 반환: 파일명 -> 해당 구간 신규 데이터프레임
        # 예: {
        #   "df_hit.csv": pd.DataFrame([...]),
        #   "df_pitch1.csv": pd.DataFrame([...]),
        #   ...
        # }
        ...
    """
    def fetch_between(self, start: date, end: date) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


def refresh_cache_incremental(days_back: int = 3, scraper: Optional[CacheScraper] = None) -> dict:
    """
    증분 갱신 실행:
    - 파일별 HWM(마지막 날짜)+1 ~ 오늘 사이만 수집/병합
    - 중복 제거, 정렬, 아토믹 저장
    - manifest 갱신
    반환: {"ok": bool, "last_updated": str, "hwm": dict}
    """
    global _cache_frames, _manifest
    if not _cache_frames or _manifest is None:
        bootstrap_cache()

    if scraper is None:
        # 사용자가 주입 안 했으면 no-op 갱신
        return {
            "ok": False,
            "last_updated": _manifest.get("last_updated"),
            "hwm": _manifest.get("high_watermark"),
            "msg": "scraper가 주입되지 않아 갱신을 수행하지 않았습니다.",
        }

    end = date.today()
    default_start = end - timedelta(days=days_back)

    # 파일별 갱신 필요 구간 계산
    need_update = {}
    for fname in REQUIRED:
        hwm = _manifest["high_watermark"].get(fname)
        start = default_start if hwm is None else max(default_start, pd.to_datetime(hwm).date() + timedelta(days=1))
        if start <= end:
            need_update[fname] = (start, end)

    if not need_update:
        return {
            "ok": True,
            "last_updated": _manifest.get("last_updated"),
            "hwm": _manifest.get("high_watermark"),
            "msg": "이미 최신 상태입니다.",
        }

    # 원천 수집 (최소 구간으로 한 번 호출; 내부에서 파일별 라우팅 권장)
    min_start = min(s for s, _ in need_update.values())
    fetched = scraper.fetch_between(min_start, end) or {}

    # 파일별 병합/저장/HWM 갱신
    for fname, columns in REQUIRED.items():
        # 신규 조각
        newdf = fetched.get(fname, pd.DataFrame(columns=columns))
        newdf = _normalize_columns(newdf, columns)

        # 현재 캐시
        cur = _cache_frames.get(fname, pd.DataFrame(columns=columns))
        cur = _normalize_columns(cur, columns)

        merged = pd.concat([cur, newdf], ignore_index=True)
        merged = _drop_dupes_by_team_date(merged)
        _atomic_write_csv(CACHE_DIR / fname, merged)
        _cache_frames[fname] = merged

        # HWM 갱신
        if "date" in merged.columns and len(merged):
            _manifest["high_watermark"][fname] = str(max(merged["date"]))

    _manifest["last_updated"] = datetime.now().isoformat(timespec="seconds")
    _save_manifest(_manifest)

    return {
        "ok": True,
        "last_updated": _manifest["last_updated"],
        "hwm": _manifest["high_watermark"],
    }


# =========================
# 통합형 헬퍼 (기존 코드와의 연결)
# =========================
def get_cached_df(name: str) -> pd.DataFrame:
    """캐시된 특정 DF 바로 가져오기"""
    frames = load_cache_frames()
    return frames.get(name, pd.DataFrame(columns=REQUIRED.get(name, [])))

def get_df_hit() -> pd.DataFrame:
    return get_cached_df("df_hit.csv")

def get_df_pitch1() -> pd.DataFrame:
    return get_cached_df("df_pitch1.csv")

def get_df_pitch2() -> pd.DataFrame:
    return get_cached_df("df_pitch2.csv")

def get_df_def() -> pd.DataFrame:
    return get_cached_df("df_def.csv")

def get_df_run() -> pd.DataFrame:
    return get_cached_df("df_run.csv")


# =========================
# 예시: 기존 크롤러 연결 가이드
# =========================
# 사용 중인 크롤링 함수가 아래처럼 있다면:
#   - fetch_hit_stats(start, end) -> pd.DataFrame(columns=["team","date","H","HR","AB", ...])
#   - fetch_pitch_stats1(start, end) -> pd.DataFrame(columns=["team","date","ERA","IP","WHIP", ...])
#   - fetch_pitch_stats2(start, end) -> pd.DataFrame(columns=["team","date","QS","SV","HLD", ...])
#   - fetch_def_stats(start, end) -> pd.DataFrame(columns=["team","date","E","FPCT","A", ...])
#   - fetch_run_stats(start, end) -> pd.DataFrame(columns=["team","date","SB","CS", ...])
#
# 아래 클래스를 프로젝트에 맞춰 구현하고, refresh_cache_incremental(days_back, scraper=YourScraper())로 호출하세요.

class YourProjectScraper(CacheScraper):
    def __init__(self):
        # 필요 시 세션/헤더/쿠키 준비
        pass

    def fetch_between(self, start: date, end: date) -> Dict[str, pd.DataFrame]:
        # TODO: 아래를 실제 함수로 교체
        # from your_module import fetch_hit_stats, fetch_pitch_stats1, fetch_pitch_stats2, fetch_def_stats, fetch_run_stats
        def _empty(cols): return pd.DataFrame(columns=cols)

        result: Dict[str, pd.DataFrame] = {}
        try:
            # 예시: 존재하는 함수로 바꿔 연결
            # result["df_hit.csv"]    = fetch_hit_stats(start, end)
            # result["df_pitch1.csv"] = fetch_pitch_stats1(start, end)
            # result["df_pitch2.csv"] = fetch_pitch_stats2(start, end)
            # result["df_def.csv"]    = fetch_def_stats(start, end)
            # result["df_run.csv"]    = fetch_run_stats(start, end)
            # 임시(빈 반환) — 연결 전까지 안전 동작
            for fname, cols in REQUIRED.items():
                result[fname] = _empty(cols)
        except Exception:
            # 실패 시라도 구조는 유지
            for fname, cols in REQUIRED.items():
                result.setdefault(fname, _empty(cols))

        # 스키마 정규화
        for fname, cols in REQUIRED.items():
            result[fname] = _normalize_columns(result[fname], cols)
        return result


# =========================
# 스크립트 실행 테스트 (선택)
# =========================
if __name__ == "__main__":
    # 1) 부트
    bootstrap_cache()
    print("팀 옵션(샘플):", get_team_options())

    # 2) 증분 갱신 테스트 (연결 전에는 빈 병합만 수행)
    resp = refresh_cache_incremental(days_back=3, scraper=YourProjectScraper())
    print("갱신 결과:", resp)

    # 3) 개별 DF 접근
    print("df_hit rows:", len(get_df_hit()))
