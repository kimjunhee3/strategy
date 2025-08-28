# Str_cache.py
import os, io, time, json
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# 경로/캐시 기본설정
# -----------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUT_DIR    = os.path.join(STATIC_DIR, "output")
CACHE_DIR  = os.path.join(STATIC_DIR, "cache")
CACHE_TTL_HOURS = 6  # 캐시 유효시간


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) < (CACHE_TTL_HOURS * 3600)

# -----------------------------
# 피처맵 / 역지표 / 설명
# -----------------------------
batting_features = {
    "공격력 전반": ["OPS", "OBP", "SLG"],
    "찬스 대응": ["RISP"],
    "작전 적합도": ["SO_per_G", "GDP_per_G", "SAC_per_G", "SF_per_G"],
}


pitching_features = {
    "선발 운용": ["ERA", "QS_per_G", "IP_per_G", "WHIP", "SO_p_per_G", "BB_p_per_G", "P/IP"],
    "불펜 전략": ["HLD_per_G", "SV_per_G", "BSV_per_G", "WPCT"],
    "교체 운용": ["NP_per_G", "WP_per_G", "BK_per_G"],
}


# ✅ 수비/주루에서 같은 이름의 지표가 ‘의미’가 달라서 분리
defense_features = {
    "수비 안정성": ["FPCT", "E_per_G"],
    "수비 관여/범위": ["RF_per_G", "PO_per_G", "A_per_G"],
    "연계 플레이": ["DP_per_G"],
    "도루 억제": ["CS_def_per_G", "CS%", "PB_per_G", "PKO_def_per_G", "SB_per_G"],
}


running_features = {
    "도루 전략 판단": ["SB", "SB%"],
    "위험 주루 억제": ["CS_run_per_G", "PKO_run_per_G"],
    "주루 감각 평가": ["OOB_per_G"],
}


# ✅ 역지표(높을수록 나쁨)
inverse_metrics = {
    # 투수
    "ERA", "WHIP", "BSV_per_G", "BB_p_per_G", "P/IP", "WP_per_G", "BK_per_G",
    # 타격
    "SO_per_G", "GDP_per_G",
    # 수비
    "E_per_G", "PB_per_G", "SB_per_G",   # (CS_def_per_G / PKO_def_per_G 는 정상지표 → 제외)
    # 주루(주루팀 관점에서 ‘잡힘/견제사’는 나쁨)
    "OOB_per_G", "CS_run_per_G", "PKO_run_per_G",
}


# 설명(없으면 기본 문구)
metric_info = {
    # 타격
    "OPS": {"desc": "종합적인 공격 생산력이 부족한 것으로 판단됩니다."},
    "SLG": {"desc": "장타력이 부족하여 큰 점수를 내기 어렵습니다."},
    "RISP": {"desc": "득점권 해결 능력이 떨어집니다."},
    "SO_per_G": {"desc": "삼진이 많아 공격 흐름이 자주 끊깁니다."},
    "GDP_per_G": {"desc": "병살타가 많아 공격 흐름이 단절됩니다."},
    "SAC_per_G": {"desc": "희생번트 활용도가 낮습니다."},
    "SF_per_G": {"desc": "희생플라이로 주자 진루가 부족합니다."},
    "OBP": {"desc": "출루율이 낮아 타선 연결이 매끄럽지 않습니다."},


    # 투수
    "WHIP": {"desc": "이닝당 출루 허용이 많아 위기 노출이 잦습니다."},
    "HLD_per_G": {"desc": "불펜이 약해 리드를 지키지 못합니다."},
    "ERA": {"desc": "실점 억제가 잘 되지 않습니다."},
    "QS_per_G": {"desc": "선발 이닝 소화가 부족합니다."},
    "P/IP": {"desc": "투구 효율이 낮아 교체가 잦습니다."},
    "IP_per_G": {"desc": "선발 이닝 소화가 부족해 불펜 부담이 커집니다."},
    "SO_p_per_G": {"desc": "삼진 생산이 부족해 위기에서 탈출하기 어렵습니다."},
    "BB_p_per_G": {"desc": "볼넷 허용이 많아 불필요한 주자 관리가 발생합니다."},
    "NP_per_G": {"desc": "경기당 투구 수가 과도해 효율적이지 못합니다."},
    "WP_per_G": {"desc": "폭투가 잦아 주자 관리가 불안정합니다."},
    "BK_per_G": {"desc": "보크 빈도가 높아 투구 동작 안정성이 떨어집니다."},


    # 수비(강화)
    "FPCT": {"desc": "수비율이 낮아 기본적인 안정성이 부족합니다."},
    "E_per_G": {"desc": "실책이 많아 수비에서 실점 위험이 큽니다."},
    "PO_per_G": {"desc": "자살(포구 관여)이 적어 타구 처리 개입이 부족합니다."},
    "A_per_G":  {"desc": "보조(Assists)가 적어 송구/중계·연계 개입이 드뭅니다."},
    "RF_per_G": {"desc": "PO+A 기준 수비 관여가 적어 전체 수비 범위가 좁습니다."},
    "DP_per_G": {"desc": "병살 전환 빈도가 낮아 수비 연계력이 제한적입니다."},
    "CS_def_per_G": {"desc": "도루 저지 수가 적어 상대 주루 억제가 부족합니다."},
    "PKO_def_per_G": {"desc": "견제 아웃이 적어 1루 리드 폭을 줄이지 못합니다."},
    "PB_per_G": {"desc": "포일/패스트볼이 잦아 공 수비 안정감이 떨어집니다."},
    "SB_per_G": {"desc": "상대 도루 허용이 많아 배터리 억제력이 부족합니다."},
    "CS%": {"desc": "도루 저지율이 낮아 2루 견제가 효과적이지 못합니다."},


    # 주루(강화)
    "SB": {"desc": "도루 시도가 많지만 효율성 점검이 필요합니다."},
    "SB%": {"desc": "도루 성공률이 낮아 작전 효율이 떨어집니다."},
    "OOB_per_G": {"desc": "주루사/무리한 플레이로 아웃이 많습니다."},
    "CS_run_per_G": {"desc": "주루 과정에서 잡히는 빈도가 높아 리스크 관리가 부족합니다."},
    "PKO_run_per_G": {"desc": "견제사 빈도가 높아 리드·스타트 타이밍 조정이 필요합니다."},
}


# -----------------------------
# 유틸
# -----------------------------
def _fetch_html_tables(url: str) -> pd.DataFrame:
    """requests + read_html"""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36"),
        "Referer": "https://www.koreabaseball.com/",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    buf = io.StringIO(r.text)
    tables = pd.read_html(buf, flavor="lxml")
    if not tables:
        raise ValueError("No table found in page")
    return tables[0]
def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('%','', regex=False),
                                  errors="coerce").fillna(0)


# -----------------------------
# 메인 파이프라인
# -----------------------------
def clean_and_extract(df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
    df = df.copy()
    for cols in feature_map.values():
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('%','', regex=False),
                                        errors="coerce").fillna(0)
            else:
                df[col] = 0.0
    return df


def score_by_area(df: pd.DataFrame, feature_map: dict) -> pd.DataFrame:
    scored = pd.DataFrame({"팀": df["팀"]})
    scaler = MinMaxScaler()
    for area, cols in feature_map.items():
        valid = [c for c in cols if c in df.columns]
        if not valid:
            continue
        area_scores = pd.DataFrame(index=df.index)
        for col in valid:
            series = df[col].copy()
            if col in inverse_metrics:   # 역지표 반전
                series = series.max() - series
            scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
            area_scores[col] = scaled
        scored[area] = area_scores.mean(axis=1)
    return scored


def get_all_scores(force: bool=False):
    """
    반환: (score_hit, score_pitch, score_def, score_run,
           df_hit, df_pitch, df_def, df_run,
           clean_hit, clean_pitch, clean_def, clean_run)
    """
    ensure_dirs()


    # 캐시 경로
    p_hit  = cache_path("df_hit.csv")
    p_run  = cache_path("df_run.csv")
    p_pit1 = cache_path("df_pitch1.csv")
    p_pit2 = cache_path("df_pitch2.csv")
    p_def  = cache_path("df_def.csv")


    fresh = all(cache_fresh(p) for p in [p_hit, p_run, p_pit1, p_pit2, p_def])
    if fresh and not force:
        df_hit  = pd.read_csv(p_hit)
        df_run  = pd.read_csv(p_run)
        df_p1   = pd.read_csv(p_pit1)
        df_p2   = pd.read_csv(p_pit2)
        df_def  = pd.read_csv(p_def)
    else:
        # 크롤링
        df_hit1 = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Hitter/Basic1.aspx")
        df_hit2 = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Hitter/Basic2.aspx")
        df_hit  = pd.merge(df_hit1, df_hit2, on="팀명", how="outer").rename(columns={"팀명":"팀"})


        df_run  = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Runner/Basic.aspx").rename(columns={"팀명":"팀"})


        df_p1   = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Pitcher/Basic1.aspx")
        df_p2   = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Pitcher/Basic2.aspx")
        dup_cols = [c for c in df_p2.columns if c in df_p1.columns and c!="팀명"]
        df_pitch = pd.merge(df_p1, df_p2.drop(columns=dup_cols), on="팀명", how="outer").rename(columns={"팀명":"팀"})
        df_p1, df_p2 = df_pitch.copy(), df_pitch.copy()


        df_def  = _fetch_html_tables("https://www.koreabaseball.com/Record/Team/Defense/Basic.aspx").rename(columns={"팀명":"팀"})


        # 캐시 저장
        df_hit.to_csv(p_hit, index=False, encoding="utf-8-sig")
        df_run.to_csv(p_run, index=False, encoding="utf-8-sig")
        df_p1.to_csv(p_pit1, index=False, encoding="utf-8-sig")
        df_p2.to_csv(p_pit2, index=False, encoding="utf-8-sig")
        df_def.to_csv(p_def, index=False, encoding="utf-8-sig")


    # -----------------------------
    # 파생지표/경기당 환산
    # -----------------------------
    # 타자
    if "G" in df_hit.columns:
        _to_numeric(df_hit, ["G","SO","GDP","SAC","SF"])
        g = df_hit["G"].replace(0,1)
        for col in ["SO","GDP","SAC","SF"]:
            if col in df_hit.columns:
                df_hit[f"{col}_per_G"] = pd.to_numeric(df_hit[col], errors="coerce").fillna(0) / g


    # 투수 (IP decimal + per_G + P/IP)
    df_pitch = df_p1
    if "IP" in df_pitch.columns:
        def _ip_to_decimal(v):
            try:
                s=str(v).strip()
                if not s: return 0.0
                if ' ' in s:
                    whole, frac = s.split()
                    num,den = frac.split('/')
                    return float(whole) + float(num)/float(den)
                return float(s)
            except: return 0.0
        df_pitch["IP"] = df_pitch["IP"].apply(_ip_to_decimal)


    _to_numeric(df_pitch, ["G","QS","SO","BB","HLD","SV","BSV","NP","WP","BK","WPCT","ERA","WHIP"])
    g_p = df_pitch["G"].replace(0,1) if "G" in df_pitch.columns else 1
    for col, new in [("QS","QS_per_G"), ("SO","SO_p_per_G"), ("BB","BB_p_per_G"),
                     ("HLD","HLD_per_G"), ("SV","SV_per_G"), ("BSV","BSV_per_G"),
                     ("NP","NP_per_G"), ("WP","WP_per_G"), ("BK","BK_per_G")]:
        if col in df_pitch.columns:
            df_pitch[new] = pd.to_numeric(df_pitch[col], errors="coerce").fillna(0) / g_p
    if "NP" in df_pitch.columns and "IP" in df_pitch.columns:
        df_pitch["P/IP"] = (pd.to_numeric(df_pitch["NP"], errors="coerce").fillna(0) /
                            df_pitch["IP"].replace(0, pd.NA)).fillna(0)


    # 수비
    _to_numeric(df_def, ["G","E","PKO","PO","A","DP","PB","SB","CS","FPCT","CS%"])
    g_d = df_def["G"].replace(0,1) if "G" in df_def.columns else 1
    for col in ["E","PKO","PO","A","DP","PB","SB","CS"]:
        if col in df_def.columns:
            df_def[f"{col}_per_G"] = pd.to_numeric(df_def[col], errors="coerce").fillna(0) / g_d
    if {"PO","A"}.issubset(df_def.columns):
        df_def["RF_per_G"] = (pd.to_numeric(df_def["PO"], errors="coerce").fillna(0) +
                              pd.to_numeric(df_def["A"],  errors="coerce").fillna(0)) / g_d
    else:
        df_def["RF_per_G"] = 0.0
    # 수비/주루 관점 분리
    df_def["CS_def_per_G"]  = df_def.get("CS_per_G",  0)
    df_def["PKO_def_per_G"] = df_def.get("PKO_per_G", 0)


    # 주루
    _to_numeric(df_run, ["G","SB","CS","OOB","PKO","SB%"])
    g_r = df_run["G"].replace(0,1) if "G" in df_run.columns else 1
    for col in ["SB","CS","OOB","PKO"]:
        if col in df_run.columns:
            df_run[f"{col}_per_G"] = pd.to_numeric(df_run[col], errors="coerce").fillna(0) / g_r
    df_run["CS_run_per_G"]  = df_run.get("CS_per_G",  0)
    df_run["PKO_run_per_G"] = df_run.get("PKO_per_G", 0)


    # -----------------------------
    # 정제/스코어링
    # -----------------------------
    clean_hit   = clean_and_extract(df_hit.rename(columns={"팀":"팀"}),   batting_features)
    clean_pitch = clean_and_extract(df_pitch.rename(columns={"팀":"팀"}), pitching_features)
    clean_def   = clean_and_extract(df_def.rename(columns={"팀":"팀"}),   defense_features)
    clean_run   = clean_and_extract(df_run.rename(columns={"팀":"팀"}),   running_features)


    score_hit   = score_by_area(clean_hit,   batting_features)
    score_pitch = score_by_area(clean_pitch, pitching_features)
    score_def   = score_by_area(clean_def,   defense_features)
    score_run   = score_by_area(clean_run,   running_features)


    return (score_hit, score_pitch, score_def, score_run,
            df_hit, df_pitch, df_def, df_run,
            clean_hit, clean_pitch, clean_def, clean_run)


if __name__ == "__main__":
    print("Running pipeline to refresh cache...")
    ensure_dirs()
    _ = get_all_scores(force=True)
    with open(cache_path("last_update.json"), "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, f, ensure_ascii=False)
    print("Done.")





