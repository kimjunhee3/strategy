from flask import Flask, render_template, request
import os, time
import numpy as np
import pandas as pd


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
        # 내부 API지만 캐시 문제 해결에 도움됨
        font_manager._rebuild()
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

# ---------- 캐시 ----------
DATA_CACHE = {"ts": 0, "payload": None}
DATA_TTL = 60*60*6  # 6시간

# ---------- 파이프라인/설정 ----------
from Str_cache import (
    ensure_dirs, get_all_scores,
    batting_features, pitching_features, defense_features, running_features,
    metric_info, inverse_metrics as INV_METRICS
)

def get_cached_scores():
    now = time.time()
    if DATA_CACHE["payload"] and now - DATA_CACHE["ts"] < DATA_TTL:
        return DATA_CACHE["payload"]
    payload = get_all_scores()  # force=False
    DATA_CACHE["payload"] = payload
    DATA_CACHE["ts"] = now
    return payload

# ---------- 차트 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_radar_chart(
    df_score: pd.DataFrame,
    team_name: str,
    category_name: str,
    compare_team_name: str = "상위 3팀 평균",
) -> str:
    # 1) 축 라벨/각도 준비
    labels = df_score.columns[1:]  # 첫 컬럼은 '팀'
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

    # 2) 비교용 데이터 (해당 팀 vs 상위3 평균)
    team_row = df_score[df_score["팀"] == team_name].iloc[0]
    score_col = df_score.columns[1]
    df_sorted = df_score.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    top3 = df_sorted.head(3)
    avg_row = top3[labels].mean()
    avg_row["팀"] = compare_team_name

    compare_df = pd.concat([team_row.to_frame().T, avg_row.to_frame().T], ignore_index=True)

    # 3) 차트 기본 설정 (라벨을 밖으로 빼기 위해 여백/반경 여유)
    fig, ax = plt.subplots(figsize=(11.0, 11.0), subplot_kw=dict(polar=True))
    r_max = 1.0
    ax.set_ylim(0, r_max)

    plot_angles = angles + angles[:1]
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # 기본 각도 라벨 숨김(바깥쪽에 직접 배치)

    # 4) 스타일 (색/투명도)
    team_line_color = "#007bff"
    avg_line_color  = "#dc3545"
    team_fill_rgba = (0/255, 123/255, 255/255, 0.25)
    avg_fill_rgba  = (220/255, 53/255, 69/255, 0.12)

    # 5) 폴리곤 그리기 (팀 / 비교대상)
    for idx, row in compare_df.iterrows():
        values = row[labels].values.tolist()
        values += values[:1]

        if idx == 0:
            line_color = team_line_color
            fill_rgba  = team_fill_rgba
            lw, marker, ls = 3, 'o', '-'
        else:
            line_color = avg_line_color
            fill_rgba  = avg_fill_rgba
            lw, marker, ls = 2, 's', '--'

        ax.plot(plot_angles, values, linewidth=lw, marker=marker, linestyle=ls,
                color=line_color, zorder=3)
        ax.fill(plot_angles, values, color=fill_rgba, zorder=2)

    # 6) 바깥쪽 라벨 수동 배치 (원 밖으로, 글씨 크게)
    label_radius = r_max * 1.15  # 바깥쪽으로 10% 더
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
            ax.text(ang, label_radius, lab, ha=ha, va='center',
                    fontproperties=KFONT, fontsize=18)
        else:
            ax.text(ang, label_radius, lab, ha=ha, va='center', fontsize=18)

    # 7) 반지름 눈금 폰트
    if KFONT is not None:
        for t in ax.get_yticklabels():
            t.set_fontproperties(KFONT)
            t.set_fontsize(13)
    else:
        ax.tick_params(labelsize=13)

    # 8) 제목/범례
    if KFONT is not None:
        ax.legend(["해당팀", compare_team_name],
                  loc='upper right', bbox_to_anchor=(1.20, 1.02),
                  prop=KFONT, fontsize=15)
    else:
        #ax.set_title(f"{team_name}", fontsize=24, pad=32, fontweight='bold')
        ax.legend(["해당팀", compare_team_name],
                  loc='upper right', bbox_to_anchor=(1.20, 1.02),
                  fontsize=15)

    # 9) 격자/배경
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1)
    ax.set_facecolor('white')

    # 10) 저장 (전역 CHART_VER 사용)
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

    # 캐시된 이미지가 있고 데이터보다 최신이면 재사용
    if os.path.exists(save_path) and (os.path.getmtime(save_path) >= data_ts - 1):
        return f"output/{file_name}"

    # 없으면 새로 그림 (저장도 같은 파일명으로)
    return draw_radar_chart(df_score, team, category, compare_team_name=compare_label)

# ---------- 워밍업 ----------
def warmup():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.get_cachedir()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)

# ---------- Flask 앱 ----------
app = Flask(__name__, template_folder="templates", static_folder="static")

# 앱 시작 시 1회 준비
ensure_dirs()
try:
    warmup()
except Exception as e:
    import logging
    logging.exception("Warmup failed: %s", e)

# ---------- 라우트 ----------
@app.route("/", methods=["GET", "POST"])
def index():
    (score_hit, score_pitch, score_def, score_run,
     df_hit, df_pitch, df_def, df_run,
     clean_hit, clean_pitch, clean_def, clean_run) = get_cached_scores()

    if score_hit is None or score_hit.empty:
        return render_template("Bgraph.html", team_list=[], charts={}, analysis={}, warnings={}, last_update=None)

    team_list = score_hit["팀"].tolist()
    charts, analysis_results = {}, {}
    team = request.args.get("team") or request.form.get("team_name")

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
            score_col = str(df.columns[1])
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            if team in df_sorted.head(3)["팀"].values:
                avg_row = df_sorted.iloc[:, 1:].mean(numeric_only=True); avg_row["팀"] = "전체 평균"
            else:
                top3 = df_sorted.head(3)
                avg_row = top3.iloc[:, 1:].mean(numeric_only=True); avg_row["팀"] = "상위 3팀 평균"
            chart_path = draw_radar_chart_if_needed(
                df, team, cat,
                compare_label=avg_row["팀"],
                data_ts=DATA_CACHE["ts"]
            )
            charts[cat] = chart_path

        # 전략 요약
        def get_zone(v):
            if v >= 0.75: return "상"
            if v >= 0.5:  return "중상"
            if v >= 0.25: return "중하"
            return "하"

        def add_strategy(cat_name, df_score, label, msgs):
            if label in df_score.columns:
                v = float(df_score.loc[df_score["팀"]==team, label].values[0])
                z = get_zone(v)
                analysis_results["categories"][cat_name]["main"].append(f"· {label} {msgs[z]}")

        # 투수
        add_strategy("투수", score_pitch, "불펜 전략", {
            "상":"우수 → 리드 시 중·후반 매니지먼트 유지(핵심 셋업 휴식일 관리)",
            "중상":"양호 → 좌우 매치업/백투백 제한으로 효율 개선",
            "중하":"다소 약함 → 롱릴리프 롤 명확화·역할 단순화",
            "하":"취약 → 셋업/클로저 재정의·하이레버리지 투입 규칙 재설계"
        })
        # 타자
        add_strategy("타자", score_hit, "찬스 대응", {
            "상":"매우 우수 → 클러치 라인업 유지·2스트라이크 접근 유지",
            "중상":"양호 → 고구속/변화구 분업 타석전략 미세조정",
            "중하":"부족 → 중심타선 진루타 설계(번트/히트앤런) 가동",
            "하":"취약 → 타순 재배열·선구 강화(초구 스윙률/존 통제)"
        })
        # 수비 - 관여/범위
        add_strategy("수비", score_def, "수비 관여/범위", {
            "상":"우수 → 타자성향 시프트 미세튜닝, 외야 시작 위치 유지",
            "중상":"양호 → 중계 라인/백업 동선 반복으로 커버 범위 확대",
            "중하":"다소 부족 → 코너 외야 2~3m 전진, 내야 수비 범위 훈련",
            "하":"취약 → 타구분포 기반 시프트 규칙화·커뮤니케이션 루틴 재정립"
        })
        # 수비 - 도루 억제
        add_strategy("수비", score_def, "도루 억제", {
            "상":"우수 → 시도 자체 억제, 상황별 콜만 정교화",
            "중상":"양호 → 투수 홀드/퀵모션 다양화(간헐적 2단 홀드)",
            "중하":"다소 약함 → 포수 팝타임·송구 정확도 드릴, 1루 견제 상향",
            "하":"취약 → 퀵모션 1.35s↓·팝타임 2.0s대 목표로 전면 재정비"
        })
        # 수비 - 안정성
        add_strategy("수비", score_def, "수비 안정성", {
            "상":"우수 → 강한 타구 대응·송구 속도 유지",
            "중상":"양호 → 실책 유형(포구/송구) 분해 후 맞춤 훈련",
            "중하":"다소 부족 → 송구 발끝 정렬·원바운드 정확도 강화",
            "하":"취약 → 기본기 루틴 재학습, 쉬운 플레이 성공률부터 회복"
        })
        # 주루
        add_strategy("주루", score_run, "도루 전략 판단", {
            "상":"적극적 → 하이라스크·하이리턴 상황 선별 확대",
            "중상":"양호 → 좌투수 상대로만 선택적 확대",
            "중하":"다소 약함 → 스타트/리드 폭 보수화, 히트앤런로 대체",
            "하":"소극적 → 도루 비중 축소·대주자 카드 상황 한정 운용"
        })

        # 세부 진단
        def detailed(team, raw_df, scaled_df, features, category_name):
            out=[]
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

        # 경고
        warnings = {}
        bundle = {
            "타자": (score_hit,  clean_hit,  batting_features),
            "투수": (score_pitch,clean_pitch,pitching_features),
            "수비": (score_def,  clean_def,  defense_features),
            "주루": (score_run,  clean_run,  running_features),
        }
        for cat, (df_s, df_r, fmap) in bundle.items():
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

        last_update = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return render_template("Bgraph.html",
                               team_list=team_list, charts=charts,
                               analysis=analysis_results, warnings=warnings,
                               last_update=last_update)

    # 팀 선택 안 했을 때
    return render_template("Bgraph.html", team_list=team_list, charts={}, analysis={}, warnings={}, last_update=None)

@app.route("/refresh")
def refresh():
    payload = get_all_scores(force=True)
    DATA_CACHE["payload"] = payload
    DATA_CACHE["ts"] = time.time()
    return ("OK", 200)

# ---------- 실행 ----------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port, debug=False)
