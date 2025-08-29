from flask import Flask, render_template, request
import os, time, threading, random
import numpy as np
import pandas as pd

# --------- Matplotlib 폰트/백엔드 세팅 ---------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

CHART_VER = "v5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(BASE_DIR, 'static', 'fonts', 'NanumGothic.ttf'),
    os.path.join(BASE_DIR, 'static', 'NanumGothic.ttf'),
]
FONT_PATH = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

KFONT = None
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    try:
        font_manager._rebuild()  # 내부 API: 폰트 캐시 재생성
    except Exception:
        pass
    KFONT = font_manager.FontProperties(fname=FONT_PATH)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [KFONT.get_name()]
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] = 22
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 12
else:
    import logging
    logging.warning("NanumGothic.ttf not found. looked at: %s", candidate_paths)

# ---------- 애플리케이션 캐시 ----------
DATA_CACHE = {"ts": 0, "payload": None}   # payload는 get_all_scores() 반환 12-튜플
DATA_TTL   = 60*60*6  # 6시간

# ---------- 파이프라인 의존 ----------
from Str_cache import (
    ensure_dirs, get_all_scores,
    batting_features, pitching_features, defense_features, running_features,
    metric_info, inverse_metrics as INV_METRICS
)

# ---------- 유틸 ----------
def empty_stub_payload():
    """템플릿이 기대하는 12-튜플(스코어4, raw4, clean4)을 빈 값으로 반환."""
    empty_scores = [pd.DataFrame(columns=["팀"]) for _ in range(4)]  # score_hit, score_pitch, score_def, score_run
    empty_raws   = [pd.DataFrame(columns=["팀"]) for _ in range(4)]  # df_hit,   df_pitch,   df_def,   df_run
    empty_cleans = [pd.DataFrame(columns=["팀"]) for _ in range(4)]  # clean_hit,clean_pitch,clean_def,clean_run
    return (*empty_scores, *empty_raws, *empty_cleans)
    
def read_payload_from_cache():
    """애플리케이션 메모리 캐시 사용(파일 캐시 X)."""
    if DATA_CACHE["payload"] is None:
        return None
    if (time.time() - DATA_CACHE["ts"]) > DATA_TTL:
        return None
    return DATA_CACHE["payload"]

def save_payload_to_cache(payload):
    DATA_CACHE["payload"] = payload
    DATA_CACHE["ts"] = time.time()

def warmup_matplotlib():
    matplotlib.get_cachedir()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)

# ---------- 차트 ----------
def draw_radar_chart(
    df_score: pd.DataFrame,
    team_name: str,
    category_name: str,
    compare_team_name: str = "상위 3팀 평균",
) -> str:
    if df_score is None or df_score.empty or "팀" not in df_score.columns:
        # 빈 차트 생성(자리에만 그림)
        output_dir = os.path.join("static", "output")
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{team_name}_{category_name}_radar_{CHART_VER}.png"
        save_path = os.path.join(output_dir, file_name)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_title("데이터 없음")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor='white', edgecolor='none')
        plt.close()
        return f"output/{file_name}"

    # 1) 축 라벨/각도
    labels = df_score.columns[1:]  # 첫 컬럼은 '팀'
    if team_name not in df_score["팀"].values or len(labels) == 0:
        # 팀 미존재/라벨 없음 보호
        output_dir = os.path.join("static", "output")
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{team_name}_{category_name}_radar_{CHART_VER}.png"
        save_path = os.path.join(output_dir, file_name)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_title("데이터 없음")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=180, facecolor='white', edgecolor='none')
        plt.close()
        return f"output/{file_name}"

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

    # 2) 비교 데이터
    team_row = df_score[df_score["팀"] == team_name].iloc[0]
    score_col = str(df_score.columns[1])
    df_sorted = df_score.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    top3 = df_sorted.head(3)
    avg_row = top3[labels].mean()
    avg_row["팀"] = compare_team_name

    compare_df = pd.concat([team_row.to_frame().T, avg_row.to_frame().T], ignore_index=True)

    # 3) 기본 설정
    fig, ax = plt.subplots(figsize=(11.0, 11.0), subplot_kw=dict(polar=True))
    r_max = 1.0
    ax.set_ylim(0, r_max)

    plot_angles = angles + angles[:1]
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    # 4) 스타일
    team_line_color = "#007bff"
    avg_line_color  = "#dc3545"
    team_fill_rgba = (0/255, 123/255, 255/255, 0.25)
    avg_fill_rgba  = (220/255, 53/255, 69/255, 0.12)

    # 5) 폴리곤
    for idx, row in compare_df.iterrows():
        values = row[labels].values.tolist()
        values += values[:1]
        if idx == 0:
            line_color, fill_rgba, lw, marker, ls = team_line_color, team_fill_rgba, 3, 'o', '-'
        else:
            line_color, fill_rgba, lw, marker, ls = avg_line_color,  avg_fill_rgba,  2, 's', '--'
        ax.plot(plot_angles, values, linewidth=lw, marker=marker, linestyle=ls, color=line_color, zorder=3)
        ax.fill(plot_angles, values, color=fill_rgba, zorder=2)

    # 6) 라벨 수동 배치
    label_radius = r_max * 1.15
    def _ha_for_angle(rad):
        deg = np.degrees(rad) % 360
        if 5 < deg < 175:
            return 'left'
        if 185 < deg < 355:
            return 'right'
        return 'center'

    for ang, lab in zip(angles, labels):
        ha = _ha_for_angle(ang)
        if KFONT is not None:
            ax.text(ang, label_radius, lab, ha=ha, va='center', fontproperties=KFONT, fontsize=18)
        else:
            ax.text(ang, label_radius, lab, ha=ha, va='center', fontsize=18)

    # 7) 반지름 눈금
    if KFONT is not None:
        for t in ax.get_yticklabels():
            t.set_fontproperties(KFONT)
            t.set_fontsize(13)
    else:
        ax.tick_params(labelsize=13)

    # 8) 범례
    if KFONT is not None:
        ax.legend(["해당팀", compare_team_name], loc='upper right', bbox_to_anchor=(1.20, 1.02), prop=KFONT, fontsize=15)
    else:
        ax.legend(["해당팀", compare_team_name], loc='upper right', bbox_to_anchor=(1.20, 1.02), fontsize=15)

    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1)
    ax.set_facecolor('white')

    # 9) 저장
    output_dir = os.path.join("static", "output")
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{team_name}_{category_name}_radar_{CHART_VER}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=220, facecolor='white', edgecolor='none')
    plt.close()

    return f"output/{file_name}"

def draw_radar_chart_if_needed(df_score, team, category, compare_label, data_ts):
    output_dir = os.path.join("static", "output")
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{team}_{category}_radar_{CHART_VER}.png"
    save_path = os.path.join(output_dir, file_name)
    if os.path.exists(save_path) and (os.path.getmtime(save_path) >= data_ts - 1):
        return f"output/{file_name}"
    return draw_radar_chart(df_score, team, category, compare_team_name=compare_label)

# ---------- Flask 앱 ----------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.before_request
def _fast_head_probe():
    # Render가 보내는 루트 HEAD를 가볍게 통과시킴
    if request.method == "HEAD":
        return ("", 200)

# 앱 시작 시 준비
ensure_dirs()
try:
    warmup_matplotlib()
except Exception as e:
    import logging
    logging.exception("Warmup failed: %s", e)

# ---------- 백그라운드 워밍업 ----------
def _fetch_and_update_cache(force=False, attempts=3, base_sleep=2):
    """네트워크 수집은 여기서만 수행."""
    for attempt in range(attempts):
        try:
            payload = get_all_scores(force=force)
            save_payload_to_cache(payload)
            return True
        except Exception as e:
            time.sleep(base_sleep * (attempt + 1) + random.random())
    return False

def warm_up_async(force=False):
    threading.Thread(target=_fetch_and_update_cache, kwargs={"force": force}, daemon=True).start()

# ---------- 렌더링 ----------
def render_index(payload):
    # 8-튜플(과거 스텁 등)이 들어와도 12-튜플로 보정
    if isinstance(payload, (list, tuple)) and len(payload) == 8:
        (score_hit, score_pitch, score_def, score_run,
         df_hit, df_pitch, df_def, df_run) = payload
        clean_hit, clean_pitch, clean_def, clean_run = df_hit, df_pitch, df_def, df_run
        payload = (score_hit, score_pitch, score_def, score_run,
                   df_hit, df_pitch, df_def, df_run,
                   clean_hit, clean_pitch, clean_def, clean_run)

    (score_hit, score_pitch, score_def, score_run,
     df_hit, df_pitch, df_def, df_run,
     clean_hit, clean_pitch, clean_def, clean_run) = payload


    def _team_list_from_disk_cache():
        try:
            p = os.path.join("static", "cache", "df_hit.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                if "팀" in df.columns:
                   return df["팀"].dropna().unique().tolist()
        except Exception:
            pass
        return []

def render_index(payload):
    ...
    # 캐시/데이터가 아직 없으면 빈 화면
    if score_hit is None or (isinstance(score_hit, pd.DataFrame) and score_hit.empty):
        team_list_fallback = _team_list_from_disk_cache()
        return render_template(
            "Bgraph.html",
            team_list=team_list_fallback,   # ← 최소한 팀 선택은 가능
            charts={}, analysis={}, warnings={},
            last_update=None
        )


    
    # 캐시/데이터가 아직 없으면 빈 화면
    if score_hit is None or isinstance(score_hit, pd.DataFrame) and score_hit.empty:
        return render_template("Bgraph.html",
                               team_list=[],
                               charts={}, analysis={}, warnings={},
                               last_update=None)

    team_list = score_hit["팀"].tolist()
    charts, analysis_results = {}, {}
    team = (request.args.get("team") or request.form.get("team_name") or "").strip()

    if team and team in team_list:
        analysis_results = {
            "team": team,
            "categories": {
                "타자": {"main": [], "detail": []},
                "투수": {"main": [], "detail": []},
                "수비": {"main": [], "detail": []},
                "주루": {"main": [], "detail": []},
            }
        }

        # 레이더: 팀 vs 상위3 평균(혹은 전체 평균)
        for cat, df in [("타자", score_hit), ("투수", score_pitch), ("수비", score_def), ("주루", score_run)]:
            if df is None or df.empty or "팀" not in df.columns or len(df.columns) < 2:
                charts[cat] = draw_radar_chart_if_needed(pd.DataFrame({"팀": []}), team, cat, "상위 3팀 평균", DATA_CACHE["ts"])
                continue

            score_col = str(df.columns[1])
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            if team in df_sorted.head(3)["팀"].values:
                compare_label = "전체 평균"
            else:
                compare_label = "상위 3팀 평균"

            chart_path = draw_radar_chart_if_needed(
                df, team, cat, compare_label=compare_label, data_ts=DATA_CACHE["ts"]
            )
            charts[cat] = chart_path

        # ------- 전략 요약 -------
        def get_zone(v):
            if v >= 0.75: return "상"
            if v >= 0.5:  return "중상"
            if v >= 0.25: return "중하"
            return "하"

        def add_strategy(cat_name, df_score, label, msgs):
            if df_score is None or df_score.empty or label not in df_score.columns: 
                return
            v = float(df_score.loc[df_score["팀"]==team, label].values[0])
            z = get_zone(v)
            analysis_results["categories"][cat_name]["main"].append(f"· {label} {msgs[z]}")

        # 투수 - 불펜 전략
        add_strategy("투수", score_pitch, "불펜 전략", {
            "상":   "안정적입니다. 리드 상황은 잘 지키고 있으니 필승조(핵심 불펜) 휴식일만 관리하세요.",
            "중상": "대체로 좋습니다. 좌우 투수 활용을 더 세밀하게 하고, 연속 등판만 줄이면 됩니다.",
            "중하": "기복이 있습니다. 롱릴리프(긴 이닝 소화)와 브릿지(중간 계투) 역할을 명확히 나누세요.",
            "하":   "불안합니다. 셋업·마무리 구성을 다시 짜고 경기 후반 운영 방식을 재정비하세요."
        })

        # 타자 - 찬스 대응
        add_strategy("타자", score_hit, "찬스 대응", {
            "상":   "매우 좋습니다. 클러치 상황과 2스트라이크 대응을 그대로 유지하세요.",
            "중상": "양호합니다. 빠른 공/변화구 대응 루틴을 조금만 다듬으세요.",
            "중하": "아쉽습니다. 중심타선에서 진루타 전략(번트·히트앤런)을 늘리세요.",
            "하":   "취약합니다. 타순 재배열과 초구 공략, 존 통제로 선구를 강화하세요."
        })

        # 수비 - 관여/범위
        add_strategy("수비", score_def, "수비 관여/범위", {
            "상":   "범위가 넓습니다. 현재 포지셔닝을 유지하세요.",
            "중상": "무난합니다. 중계·백업 동선을 반복 훈련해 안정성을 높이세요.",
            "중하": "다소 좁습니다. 외야 전진 수비와 내야 범위 훈련을 늘리세요.",
            "하":   "제한적입니다. 타구 분포에 맞춘 수비 위치와 팀 간 소통을 강화하세요."
        })

        # 수비 - 도루 억제
        add_strategy("수비", score_def, "도루 억제", {
            "상":   "잘 억제합니다. 지금 수준을 유지하세요.",
            "중상": "대체로 양호합니다. 투수 견제 동작을 조금 더 다양화하세요.",
            "중하": "허용이 늘었습니다. 포수 송구 정확도와 1루 견제를 강화하세요.",
            "하":   "취약합니다. 투수 동작과 포수 송구 모두 전면 재정비가 필요합니다."
        })

        # 수비 - 안정성
        add_strategy("수비", score_def, "수비 안정성", {
            "상":   "안정적입니다. 강한 타구 대응과 송구 속도를 유지하세요.",
            "중상": "대체로 좋습니다. 실책 유형을 분석해 맞춤 훈련하세요.",
            "중하": "흔들립니다. 송구 정확도와 기본 동작을 보완하세요.",
            "하":   "불안정합니다. 기본기 루틴을 재학습해 쉬운 플레이 성공률부터 회복하세요."
        })

        # 주루 - 도루 전략 판단
        add_strategy("주루", score_run, "도루 전략 판단", {
            "상":   "효율적입니다. 도루 시도를 적극적으로 이어가세요.",
            "중상": "무난합니다. 좌투수 상대에서 도루를 조금 더 활용하세요.",
            "중하": "아쉽습니다. 도루 성공률이 낮아 리드 폭을 줄이고 히트앤런 같은 대체 작전을 고려하세요.",
            "하":   "취약합니다. 도루가 잘 통하지 않아 비중을 줄이고 대주자 카드를 상황 한정으로 활용하세요."
        })

        # ------- 세부 진단 -------
        def detailed(team, raw_df, scaled_df, features, category_name):
            out=[]
            if raw_df is None or raw_df.empty or scaled_df is None or scaled_df.empty:
                return out

            def zone_from_quantile(val, qs, inverse=False):
                q1,q2,q3 = qs
                if not inverse:
                    if val>=q3: return "최상위권"
                    if val>=q2: return "평균 이상"
                    if val>=q1: return "평균 이하"
                    return "최하위권"
                else:
                    if val<=q1: return "최상위권"
                    if val<=q2: return "평균 이상"
                    if val<=q3: return "평균 이하"
                    return "최하위권"

            t_raw = raw_df[raw_df["팀"]==team].iloc[0]
            t_scl = scaled_df[scaled_df["팀"]==team].iloc[0]
            for area, metrics in features.items():
                if area not in scaled_df.columns:
                    continue
                area_score = float(t_scl[area])
                if area_score < 0.4:
                    for m in metrics:
                        if m in raw_df.columns:
                            val = float(t_raw[m])
                            qs = raw_df[m].quantile([0.25,0.5,0.75]).values
                            inverse = (m in INV_METRICS)
                            z = zone_from_quantile(val, qs, inverse)
                            if z not in ("평균 이하", "최하위권"):
                                continue
                            desc = metric_info.get(m, {}).get("desc", f"{m} 개선 필요")
                            out.append({
                                "category": category_name, "metric": m, "value": f"{val:.3f}",
                                "zone": z, "area_name": area, "desc": desc
                            })
            return out

        detailed_all=[]
        detailed_all += detailed(team, clean_hit,  score_hit,  batting_features, "타자")
        detailed_all += detailed(team, clean_pitch,score_pitch, pitching_features,"투수")
        detailed_all += detailed(team, clean_def,  score_def,  defense_features,"수비")
        detailed_all += detailed(team, clean_run,  score_run,  running_features,"주루")

        for d in detailed_all:
            analysis_results["categories"][d["category"]]["detail"].append(d)

        # ------- 경고 -------
        warnings = {}
        bundle = {
            "타자": (score_hit,  clean_hit,  batting_features),
            "투수": (score_pitch,clean_pitch,pitching_features),
            "수비": (score_def,  clean_def,  defense_features),
            "주루": (score_run,  clean_run,  running_features),
        }
        for cat, (df_s, df_r, fmap) in bundle.items():
            if df_s is None or df_s.empty or df_r is None or df_r.empty:
                continue
            labels = df_s.columns[1:]
            row_s = df_s[df_s["팀"]==team].iloc[0]
            row_r = df_r[df_r["팀"]==team].iloc[0]
            weak_msgs=[]
            for lab in labels:
                val = float(row_s[lab])
                if val < 0.3:
                    metrics = fmap.get(lab, [])
                    raw_hint=""
                    if len(metrics)==1 and metrics[0] in row_r.index:
                        raw_hint = f", 원시: {metrics[0]}={float(row_r[metrics[0]]):.3f}"
                    weak_msgs.append(f"📉 {lab}: {val:.3f} (즉시 개선 필요{raw_hint})")
            if weak_msgs:
                warnings[cat] = [f"⚠️ 주의: {len(labels)}개 지표 중 {len(weak_msgs)}개가 위험 수준입니다."] + weak_msgs

        last_update = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(DATA_CACHE["ts"] or time.time()))
        return render_template("Bgraph.html",
                               team_list=team_list, charts=charts,
                               analysis=analysis_results, warnings=warnings,
                               last_update=last_update)

    # 팀 미선택: 팀 리스트만 출력
    return render_template("Bgraph.html",
                           team_list=team_list, charts={}, analysis={}, warnings={},
                           last_update=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(DATA_CACHE["ts"] or time.time())))

# ---------- 라우트 ----------
@app.route("/", methods=["GET", "POST"])
def index():
    payload = read_payload_from_cache()
    if payload is None:
        # ➊ 동기 한 번 시도 (빠른 실패)
        ok = _fetch_and_update_cache(force=False, attempts=1, base_sleep=0.5)
        if ok:
            payload = read_payload_from_cache()
        else:
            # ➋ 실패하면 비동기 워밍 + 스텁 반환
            warm_up_async(force=False)
            payload = empty_stub_payload()
    return render_index(payload)

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/refresh")
def refresh():
    # 강제 최신화(네트워크). 호출은 관리자/수동으로만.
    ok = _fetch_and_update_cache(force=True)
    if not ok:
        return ("FETCH FAILED", 503)
    return ("OK", 200)

# ---------- 실행 ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port, debug=False)
