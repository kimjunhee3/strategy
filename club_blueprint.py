# club_blueprint.py
import os
from flask import Blueprint, render_template, request, jsonify
from crawl_club import get_youtube_videos

club_bp = Blueprint("club", __name__, template_folder="templates")

@club_bp.route("/", methods=["GET"])
def club_index():
    # 템플릿 내부 JS가 ?team=LG 등을 처리
    return render_template("Club.html")

@club_bp.route("/search", methods=["POST"])
def club_search():
    data = request.get_json(silent=True) or {}
    clubs = data.get("clubs")
    options = data.get("options", {})

    # 하위호환: 이전 형식 {"club": "..."}
    if not clubs and "club" in data:
        club_name = str(data.get("club"))
        clubs = [club_name] if club_name else []

    if not clubs:
        return jsonify({"error": "클럽(팀)명이 비어있습니다.", "long": [], "short": []}), 400

    clubs = [c for c in clubs if isinstance(c, str) and c.strip()]
    if not clubs:
        return jsonify({"error": "유효한 클럽(팀)명이 없습니다.", "long": [], "short": []}), 400

    try:
        long_videos, short_videos = get_youtube_videos(clubs, options=options)
        return jsonify({"long": long_videos, "short": short_videos})
    except Exception as e:
        return jsonify({"error": str(e), "long": [], "short": []}), 500
