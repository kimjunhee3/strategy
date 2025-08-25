# Str_flask.py
from flask import Flask, render_template, request
import os, time
import numpy as np
import pandas as pd


# 파이프라인/설정
from Str_cache import (
    ensure_dirs, get_all_scores,
    batting_features, pitching_features, defense_features, running_features,
    metric_info, inverse_metrics as INV_METRICS
)


# ---- 레이더 차트 (내장) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_radar_chart(df_score: pd.DataFrame, team_name: str, category_name: str, compare_team_name="상위 3팀 평균") -> str:
    labels = df_score.columns[1:]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()


    team_row = df_score[df_score["팀"] == team_name].iloc[0]
    score_col = df_score.columns[1]
    df_sorted = df_score.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    top3 = df_sorted.head(3)
    avg_row = top3[labels].mean()
    avg_row["팀"] = compare_team_name


    compare_df = pd.concat([team_row.to_frame().T, avg_row.to_frame().T], ignore_index=True)


    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 1.0)
    plot_angles = angles + angles[:1]


    # 색상 지정: 팀=파랑, 평균=빨강
    team_line_color = "#007bff"
    team_fill_color = "rgba(0,123,255,0.25)"  # 참고용 문자열, matplotlib에는 아래 fill에서 직접 RGBA 사용
    avg_line_color  = "#dc3545"
    # matplotlib RGBA는 0~1 튜플 사용
    team_fill_rgba = (0/255, 123/255, 255/255, 0.25)
    avg_fill_rgba  = (220/255, 53/255, 69/255, 0.12)


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


        ax.plot(
            plot_angles, values,
            linewidth=lw,
            marker=marker,
            linestyle=ls,
            color=line_color,         # ★ 선 색 명시
            zorder=3
        )
        ax.fill(plot_angles, values, color=fill_rgba, zorder=2)  # ★ 채우기 색도 통일


    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(f"{team_name}", fontsize=18, pad=30, fontweight='bold')
    ax.legend(["해당팀", compare_team_name], loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1)


    # 배경은 흰색 권장 (기존 파란색이면 대비가 떨어짐)
    ax.set_facecolor('white')


    output_dir = os.path.join("static", "output")
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{team_name}_{category_name}_radar.png"
    save_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white', edgecolor='none')
    plt.close()
    return f"output/{file_name}"


# ---- Flask ----
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_dirs()
    (score_hit, score_pitch, score_def, score_run,
     df_hit, df_pitch, df_def, df_run,
     clean_hit, clean_pitch, clean_def, clean_run) = get_all_scores()


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


        # --- 레이더: 팀 vs 상위3 평균(혹은 전체 평균) ---
        for cat, df in [("타자", score_hit), ("투수", score_pitch), ("수비", score_def), ("주루", score_run)]:
            score_col = str(df.columns[1])
            df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
            if team in df_sorted.head(3)["팀"].values:
                avg_row = df_sorted.iloc[:, 1:].mean(numeric_only=True); avg_row["팀"] = "전체 평균"
            else:
                top3 = df_sorted.head(3)
                avg_row = top3.iloc[:, 1:].mean(numeric_only=True); avg_row["팀"] = "상위 3팀 평균"
            chart_path = draw_radar_chart(df, team, cat, compare_team_name=avg_row["팀"])
            charts[cat] = chart_path


        # --- 전략 요약 (액션형 문구) ---
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


        # --- 세부 진단 (정규화 낮은 영역 + 원시 ‘평균 이하/최하위권’만) ---
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


        # --- 경고 (정규화 약함 + 단일지표는 원시값 힌트) ---
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
    _ = get_all_scores(force=True)
    return ("OK", 200)


if __name__ == "__main__":
    ensure_dirs()
    import os
    port = int(os.environ.get("PORT", 5055))  # 배포환경 PORT 사용, 없으면 로컬 5055
    app.run(host="0.0.0.0", port=port, debug=True)
