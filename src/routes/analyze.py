import os
import uuid
from flask import Blueprint, request, jsonify

from src.utils.extract_text import extract_text_from_file
from src.utils.engine import detect_ai_content

bp = Blueprint("analyze", __name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@bp.route("/analyze-ai", methods=["POST"])
def analyze_ai():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400

    # save temp file
    ext = os.path.splitext(file.filename)[1]

    temp_name = f"{uuid.uuid4()}{ext}"

    temp_path = os.path.join(
        UPLOAD_FOLDER,
        temp_name
    )

    file.save(temp_path)

    try:

        # extract text
        text = extract_text_from_file(temp_path)

        if not text or len(text.strip()) < 50:
            return jsonify({
                "success": False,
                "error": "Unable to extract enough text"
            })

        # AI detection
        result = detect_ai_content(text)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "result": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        try:
            os.remove(temp_path)
        except:
            pass