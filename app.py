import io
import json
import os
import re
from typing import Any, Dict, List

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


APP_TITLE = "Resume ATS Analyzer"
DEFAULT_MODEL = "gemini-3.8-flash"
MAX_RESUME_CHARS = 50000


def get_api_key() -> str:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    sections = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            sections.append(text)

    # Also capture text from tables because resumes often use tables
    # for skills, experience, and education.
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                sections.append(row_text)

    return "\n".join(sections)


def extract_resume_text(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)

    if filename.endswith(".docx"):
        return extract_docx_text(file_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")


def clean_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON even if the model accidentally wraps it in a markdown fence."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Gemini returned an invalid JSON response.")
        return json.loads(match.group(0))


def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the UI safe even if Gemini returns slightly different values."""
    try:
        score = int(float(result.get("ats_score", 0)))
    except (TypeError, ValueError):
        score = 0

    result["ats_score"] = max(0, min(100, score))

    for key in [
        "summary",
        "formatting_analysis",
        "keyword_analysis",
        "job_match_analysis",
        "rewritten_summary",
        "overall_recommendation",
    ]:
        value = result.get(key, "")
        result[key] = str(value) if value is not None else ""

    for key in [
        "strengths",
        "weaknesses",
        "missing_keywords",
        "recommended_keywords",
        "improvements",
    ]:
        value = result.get(key, [])
        if not isinstance(value, list):
            value = [str(value)]
        result[key] = [str(item) for item in value if str(item).strip()]

    return result


def analyze_resume(
    resume_text: str,
    job_description: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml or your environment variables."
        )

    client = genai.Client(api_key=api_key)

    job_section = (
        job_description.strip()
        if job_description.strip()
        else "No job description was provided. Evaluate general ATS compatibility."
    )

    prompt = f"""
You are an expert Applicant Tracking System (ATS) resume reviewer and professional career coach.

Analyze the resume below. Your ATS score must be an evidence-based estimate, not a claim that it is an exact score produced by a specific commercial ATS.

Evaluate:
- ATS readability and parsing friendliness
- Section headings and structure
- Keyword relevance
- Skills and experience relevance
- Quantifiable achievements
- Clarity and conciseness
- Formatting risks such as tables, columns, graphics, unusual symbols, headers/footers, or missing standard sections when they can be inferred from the extracted text
- Alignment with the supplied job description, if present

Important:
- Do not invent qualifications, experience, education, certifications, employers, or skills.
- Missing keywords should be based on the job description when one is provided.
- Distinguish between a keyword being truly absent and simply not visible in extracted text.
- Give practical, specific improvements.
- The rewritten summary must only use facts supported by the resume.
- Return ONLY valid JSON. No markdown and no commentary outside the JSON.

JOB DESCRIPTION:
{job_section}

RESUME:
{resume_text[:MAX_RESUME_CHARS]}

Return this exact JSON structure:
{{
  "ats_score": 0,
  "summary": "2-4 sentence overall assessment.",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "formatting_analysis": "Specific ATS formatting assessment.",
  "keyword_analysis": "Keyword and skills assessment.",
  "job_match_analysis": "Job-description alignment assessment.",
  "missing_keywords": ["...", "..."],
  "recommended_keywords": ["...", "..."],
  "improvements": ["Specific action 1", "Specific action 2"],
  "rewritten_summary": "An improved professional summary using only resume facts. If there is not enough information, say so.",
  "overall_recommendation": "A concise prioritized recommendation."
}}
"""

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=6000,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return normalize_result(clean_json_response(response.text))


def score_label(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Needs improvement"
    return "Weak"


def display_bullets(items: List[str]) -> None:
    if not items:
        st.write("No items returned.")
        return

    for item in items:
        st.markdown(f"- {item}")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.caption(
    "Upload a PDF or DOCX resume to get an AI-based ATS estimate, keyword analysis, "
    "and actionable improvement suggestions."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_MODEL,
        help="Use a model available to your Gemini API account.",
    )

    st.info(
        "Your Gemini API key should be stored in Streamlit Secrets or an environment "
        "variable—not in this source code."
    )

    with st.expander("How the score works"):
        st.write(
            "The score is an AI-based estimate of ATS compatibility and job relevance. "
            "It is not an official score from LinkedIn, Workday, Greenhouse, Indeed, "
            "or another commercial ATS."
        )

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="PDF or DOCX only.",
    )

with col2:
    job_description = st.text_area(
        "Paste the job description (optional)",
        height=220,
        placeholder="Paste the target job description here for a more useful keyword and match analysis...",
    )

if uploaded_file:
    st.caption(f"Selected file: **{uploaded_file.name}**")

if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a PDF or DOCX resume first.")
        st.stop()

    try:
        with st.spinner("Extracting resume text..."):
            resume_text = extract_resume_text(uploaded_file)

        if not resume_text.strip():
            st.error(
                "No readable text was found. If this is a scanned/image-only PDF, "
                "convert it to a text-based PDF or DOCX first."
            )
            st.stop()

        with st.spinner("Gemini is analyzing your resume..."):
            result = analyze_resume(
                resume_text=resume_text,
                job_description=job_description,
                model_name=model_name.strip() or DEFAULT_MODEL,
            )

        st.session_state["analysis_result"] = result
        st.session_state["resume_name"] = uploaded_file.name

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    st.divider()
    st.subheader(f"Results — {st.session_state.get('resume_name', 'Resume')}")

    score = result["ats_score"]
    score_col, label_col = st.columns([1, 3])

    with score_col:
        st.metric("ATS Score", f"{score}/100")

    with label_col:
        st.markdown(f"### {score_label(score)}")
        st.progress(score / 100)

    st.markdown("### 🧾 Overall Assessment")
    st.write(result["summary"])

    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Strengths")
        display_bullets(result["strengths"])

    with right:
        st.markdown("### ⚠️ Weaknesses")
        display_bullets(result["weaknesses"])

    st.markdown("### 🎯 Keyword Analysis")
    st.write(result["keyword_analysis"])

    if result["missing_keywords"]:
        st.markdown("**Potentially missing keywords:**")
        display_bullets(result["missing_keywords"])

    if result["recommended_keywords"]:
        st.markdown("**Recommended keywords to consider:**")
        display_bullets(result["recommended_keywords"])

    st.markdown("### 🧩 ATS Formatting Analysis")
    st.write(result["formatting_analysis"])

    st.markdown("### 💼 Job Match Analysis")
    st.write(result["job_match_analysis"])

    st.markdown("### 🛠️ Recommended Improvements")
    display_bullets(result["improvements"])

    st.markdown("### ✍️ Suggested Professional Summary")
    st.write(result["rewritten_summary"])

    st.markdown("### 🚀 Priority Recommendation")
    st.info(result["overall_recommendation"])

    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download analysis as JSON",
        data=result_json,
        file_name="resume_ats_analysis.json",
        mime="application/json",
    )

st.divider()
st.caption(
    "Privacy note: the uploaded resume is processed during the current Streamlit session "
    "and sent to the configured Gemini API for analysis. Do not upload documents containing "
    "information you are not comfortable sending to that service."
)
