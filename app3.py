from flask import Flask, request, render_template
from groq import Groq
import os
import sqlite3
import pickle
import pandas as pd
from contextlib import contextmanager
from dotenv import load_dotenv
import markdown

load_dotenv()

api_key = os.getenv("api_key")
client = Groq(api_key=api_key)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemes.db")

INCOME_BRACKET_FLOOR = {
    "below_1L":   0,
    "1L_2L":      100_001,
    "2L_3L":      200_001,
    "3L_5L":      300_001,
    "5L_8L":      500_001,
    "8L_12L":     800_001,
    "above_12L":  1_200_001,
}

# ─────────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                gender TEXT,
                income_bracket TEXT,
                caste TEXT,
                disability TEXT,
                bpl_card TEXT,
                business_type TEXT,
                state TEXT,
                location TEXT,
                marginalization_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # CREATE TABLE IF NOT EXISTS does nothing if the table already exists
        # with an OLDER schema (e.g. from a previous version of this app that
        # didn't have marginalization_score). Check for and add any missing
        # columns so upgrading never requires deleting your existing data.
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(submissions)")
        }
        required_cols = {
            "marginalization_score": "REAL",
        }
        for col_name, col_type in required_cols.items():
            if col_name not in existing_cols:
                print(f"[database migration] Adding missing column '{col_name}' to submissions table")
                conn.execute(f"ALTER TABLE submissions ADD COLUMN {col_name} {col_type}")

        has_schemes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schemes'"
        ).fetchone()
        if not has_schemes:
            print("[WARNING] No 'schemes' table found. Run database_data_making.py first.")


# ─────────────────────────────────────────────────────────────────
# ML SCORING
# ─────────────────────────────────────────────────────────────────

def predict_score(profile):
    gender_map   = {"female": 1, "male": 0, "other": 1, "prefer_not_to_say": 0}
    caste_map    = {"general": 0, "obc": 1, "sc": 2, "st": 3}
    location_map = {"urban": 0, "rural": 1}

    income_floor = INCOME_BRACKET_FLOOR.get(profile.get("income", "below_1L"), 0)
    gender       = gender_map.get(str(profile.get("gender", "male")).lower(), 0)
    caste        = caste_map.get(str(profile.get("caste", "general")).lower(), 0)
    disability   = 1 if profile.get("disability", "none") != "none" else 0
    location     = location_map.get(str(profile.get("location", "urban")).lower(), 0)
    bpl_card     = 1 if profile.get("bpl_card", "no") == "yes" else 0

    X = pd.DataFrame([{
        "income":     income_floor,
        "gender":     gender,
        "caste":      caste,
        "disability": disability,
        "location":   location,
        "bpl_card":   bpl_card,
    }])
    return round(float(model.predict(X)[0]), 2)


# ─────────────────────────────────────────────────────────────────
# SUBMISSION HELPERS
# ─────────────────────────────────────────────────────────────────

def save_submission(profile, score):
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions
                (full_name, gender, income_bracket, caste, disability,
                 bpl_card, business_type, state, location, marginalization_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.get("full_name"),
                profile.get("gender"),
                profile.get("income"),
                profile.get("caste"),
                profile.get("disability"),
                profile.get("bpl_card"),
                profile.get("business_type"),
                profile.get("state"),
                profile.get("location"),
                score,
            ),
        )
        return cur.lastrowid


def get_submission_by_id(submission_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "full_name":     row["full_name"],
        "gender":        row["gender"],
        "income":        row["income_bracket"],
        "caste":         row["caste"],
        "disability":    row["disability"],
        "bpl_card":      row["bpl_card"],
        "business_type": row["business_type"],
        "state":         row["state"],
        "location":      row["location"],
        "score":         row["marginalization_score"],
    }


# ─────────────────────────────────────────────────────────────────
# SCHEME MATCHING — rewritten as weighted soft scoring
#
# WHY THIS CHANGED: schemes_final_v2.csv stores gender/caste/business_type/
# income/state as a SINGLE representative value per scheme (used only to
# compute that scheme's marginalization_score via the model) -- not as
# multi-value eligibility rules. 27/30 schemes share the exact same
# income=300000, and every scheme has exactly one business_type. Treating
# these as hard filters (exact-match required) meant most real submissions
# got excluded by every single scheme -- zero results. Now every field
# (except bpl_card, which is a genuine yes/no requirement) contributes a
# WEIGHTED BONUS instead of an exclusion, so you always get a ranked list,
# and schemes that genuinely resemble the applicant's profile still rank
# highest.
# ─────────────────────────────────────────────────────────────────

def _normalize_business_type(value):
    """Map form vocabulary ('retail','services') onto CSV vocabulary ('trade','service')."""
    v = str(value or "").strip().lower()
    return {"retail": "trade", "services": "service"}.get(v, v)


def _gender_bonus(scheme_gender, profile_gender):
    """
    'male' in this dataset means the scheme's representative persona is
    male -- NOT male-only. Only give a bonus for an explicit, meaningful
    targeting match (e.g. scheme's persona is 'female' and applicant is
    female); no bonus, and crucially no exclusion, otherwise.
    """
    sg = str(scheme_gender or "").strip().lower()
    pg = str(profile_gender or "").strip().lower()
    if sg == "female" and pg == "female":
        return 10
    return 0


def _caste_bonus(scheme_caste, profile_caste):
    sc = str(scheme_caste or "").strip().lower()
    pc = str(profile_caste or "").strip().lower()
    if sc in ("", "general", "any", "all"):
        return 0  # not targeted at a specific caste
    allowed = {v.strip() for v in sc.split(",") if v.strip()}
    return 10 if pc in allowed else 0


def _location_bonus(scheme_location, profile_location):
    sl = str(scheme_location or "").strip().lower()
    pl = str(profile_location or "").strip().lower()
    if sl in ("", "urban", "both", "all", "all india", "any"):
        return 0  # not a meaningful rural/urban restriction in this dataset
    return 5 if pl == sl else 0


def _state_bonus(scheme_state, profile_state):
    ss = str(scheme_state or "").strip().lower()
    ps = str(profile_state or "").strip().lower()
    if ss in ("", "all india", "all", "any", "both"):
        return 3  # nationwide scheme -- small baseline bonus, applies to everyone
    return 8 if ps == ss else 0  # state-specific scheme the applicant's state matches


def _business_bonus(scheme_biz, profile_biz):
    sb = str(scheme_biz or "").strip().lower()
    pb = _normalize_business_type(profile_biz)
    return 12 if sb == pb else 0


def _income_bonus(scheme_income_limit, income_bracket):
    """Small bonus if the applicant's bracket floor is within the scheme's cap; no exclusion."""
    if not scheme_income_limit:
        return 3
    floor = INCOME_BRACKET_FLOOR.get(income_bracket, 0)
    return 8 if floor <= scheme_income_limit else 0


def get_matching_schemes(profile, entrepreneur_score, limit=10):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM schemes").fetchall()

    results = []
    for row in rows:
        # --- HARD FILTERS ---
        
        # 1. BPL Card Filter
        if int(row["bpl_card"] or 0) == 1 and profile.get("bpl_card") != "yes":
            continue

        # 2. Gender Filter (Female-only schemes)
        scheme_gender = str(row["gender"] or "").strip().lower()
        profile_gender = str(profile.get("gender") or "").strip().lower()
        if scheme_gender == "female" and profile_gender != "female":
            continue

        # 3. Business Type Filter
        scheme_biz = str(row["business_type"] or "").strip().lower()
        profile_biz = _normalize_business_type(profile.get("business_type"))
        if scheme_biz and scheme_biz not in ["", "any", "all"] and scheme_biz != profile_biz:
            continue

        # --- SOFT SCORING FOR RANKING ---
        
        scheme_score = float(row["marginalization_score"] or 50)
        scheme_score = max(0.0, min(100.0, scheme_score))

        # Base relevance
        score_diff = abs(entrepreneur_score - scheme_score)
        base = max(0.0, 100.0 - score_diff)

        # Bonuses are kept in the calculation so that exact matches still bubble 
        # to the top of the eligible list.
        bonus = (
            _gender_bonus(row["gender"], profile.get("gender"))
            + _caste_bonus(row["caste"], profile.get("caste"))
            + _location_bonus(row["location"], profile.get("location"))
            + _state_bonus(row["state"], profile.get("state"))
            + _business_bonus(row["business_type"], profile.get("business_type"))
            + _income_bonus(row["income_limit"], profile.get("income"))
        )

        match_percent = round(min(100.0, base * 0.7 + bonus), 1)

        results.append({
            "id":                    row["id"],
            "name":                  row["name"],
            "description":           row["description"],
            "link":                  row["link"],
            "gender":                row["gender"],
            "caste":                 row["caste"],
            "income_limit":          row["income_limit"],
            "location":              row["location"],
            "age":                   row["age"],
            "bpl_card":              row["bpl_card"],
            "business_type":         row["business_type"],
            "state":                 row["state"],
            "marginalization_score": scheme_score,
            "match_percent":         match_percent,
        })

    results.sort(key=lambda r: r["match_percent"], reverse=True)
    return results[:limit]


def get_scheme_by_id(scheme_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM schemes WHERE id = ?", (scheme_id,)).fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────
# GROQ ELABORATION
# ─────────────────────────────────────────────────────────────────

def build_context(scheme, profile=None):
    lines = [
        f"Scheme name: {scheme.get('name')}",
        f"Description: {scheme.get('description')}",
    ]
    elig = []
    if scheme.get("gender"):        elig.append(f"Gender eligibility: {scheme['gender']}")
    if scheme.get("caste"):         elig.append(f"Caste eligibility: {scheme['caste']}")
    if scheme.get("income_limit"):  elig.append(f"Max annual income: Rs. {int(scheme['income_limit']):,}")
    if scheme.get("location"):      elig.append(f"Location: {scheme['location']}")
    if scheme.get("age"):           elig.append(f"Minimum age: {scheme['age']}")
    if scheme.get("bpl_card"):      elig.append("Requires BPL card")
    if scheme.get("business_type"): elig.append(f"Business type: {scheme['business_type']}")
    if scheme.get("state"):         elig.append(f"State: {scheme['state']}")
    if elig:
        lines.append("Eligibility:\n- " + "\n- ".join(elig))

    if profile:
        prof = []
        if profile.get("full_name"):     prof.append(f"Name: {profile['full_name']}")
        if profile.get("gender"):        prof.append(f"Gender: {profile['gender']}")
        if profile.get("caste"):         prof.append(f"Caste: {profile['caste']}")
        if profile.get("income"):        prof.append(f"Income bracket: {profile['income']}")
        if profile.get("disability"):    prof.append(f"Disability: {profile['disability']}")
        if profile.get("bpl_card"):      prof.append(f"BPL card: {profile['bpl_card']}")
        if profile.get("business_type"): prof.append(f"Business type: {profile['business_type']}")
        if profile.get("state"):         prof.append(f"State: {profile['state']}")
        if profile.get("location"):      prof.append(f"Location: {profile['location']}")
        if profile.get("score"):         prof.append(f"Marginalization score: {profile['score']}/100")
        if prof:
            lines.append("Applicant profile:\n- " + "\n- ".join(prof))

    return "\n\n".join(lines)


def elaborate(context):
    prompt_text = (
        "You are an expert advisor helping an entrepreneur understand a government scheme "
        "they may be eligible for. Below is the scheme's eligibility criteria and the applicant's profile.\n\n"
        f"{context}\n\n"
        "CRITICAL ELIGIBILITY RULES TO APPLY:\n"
        "1. Caste Hierarchy: If a scheme is marked for 'General', it is entirely open to SC, ST, and OBC applicants. "
        "However, schemes specifically marked for 'SC', 'ST', or 'OBC' are restricted and not open to General category applicants.\n"
        "2. Gender Interpretation: If a scheme's target gender is listed as 'Male', it automatically means it is open "
        "to ALL genders. Do not frame it as a male-only scheme.\n\n"
        "Write a personalized explanation detailing:\n"
        "- Why this scheme suits this applicant.\n"
        "- The specific benefits they could receive.\n"
        "- Any strong fit points or potential weak areas in their application.\n\n"
        "FORMATTING INSTRUCTIONS:\n"
        "- Do NOT use markdown tables (no '|' characters).\n"
        "- Strictly present concerns and mitigations using bold bullet points in this exact format:\n"
        "  * **[Area]:** [Concern] - [How to Mitigate]\n"
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{
            "role": "user",
            "content": prompt_text,
        }],
    )
    
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
init_db()


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/explore')
def explore():
    return render_template("explore2.html")

@app.route('/about')
def about():
    return render_template("about2.html")

@app.route('/contact')
def contact():
    return render_template("contact2.html")

@app.route('/how-it-works')
def how():
    return render_template("how-it-works2.html")

@app.route('/resources')
def resources():
    return render_template("resources2.html")

@app.route('/form')
def form():
    return render_template("form2.html")


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        profile = {
            "full_name":     request.form.get("full_name"),
            "gender":        request.form.get("gender"),
            "income":        request.form.get("income"),
            "caste":         request.form.get("caste"),
            "disability":    request.form.get("disability"),
            "bpl_card":      request.form.get("bpl_card"),
            "business_type": request.form.get("business_type"),
            "state":         request.form.get("state"),
            "location":      request.form.get("location"),
        }

        entrepreneur_score = predict_score(profile)
        profile["score"]   = entrepreneur_score
        submission_id      = save_submission(profile, entrepreneur_score)
        matches            = get_matching_schemes(profile, entrepreneur_score)

        return render_template(
            "results.html",
            profile=profile,
            matches=matches,
            submission_id=submission_id,
            entrepreneur_score=entrepreneur_score,
        )

    return render_template("form2.html")



@app.route('/scheme/<int:scheme_id>')
def scheme_detail(scheme_id):
    scheme = get_scheme_by_id(scheme_id)
    if not scheme:
        return "Scheme not found", 404

    submission_id = request.args.get("sub", type=int)
    profile       = get_submission_by_id(submission_id) if submission_id else None
    context       = build_context(scheme, profile)
    
    # Get the raw markdown from the LLM
    raw_elaboration = elaborate(context)
    
    # Convert the markdown to HTML
    elaboration_html = markdown.markdown(raw_elaboration)

    return render_template(
        "scheme_detail.html",
        scheme=scheme,
        elaboration=elaboration_html,  # Pass the parsed HTML to the template
        profile=profile,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
