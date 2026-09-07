# 🏛️ AI-Powered Government Scheme Finder

An intelligent web platform designed to help marginalized entrepreneurs discover, understand, and apply for government business schemes. 

Instead of rigid rule-based filtering that often excludes eligible applicants over technicalities, this platform utilizes a machine learning-based marginalization scorer, a custom weighted matching algorithm, and high-speed Generative AI to provide personalized scheme recommendations and actionable advice.

---

## ✨ Key Features

1. **ML-Powered Marginalization Scoring**
   * Uses an **ExtraTreesRegressor** to evaluate an applicant's socio-economic profile (income, gender, caste, location, disability).
   * **Why ExtraTrees?** It handles categorical/encoded data beautifully, resists overfitting on noisy demographic data by using randomized split thresholds, and executes in fractions of a millisecond for real-time web responsiveness.

2. **Weighted "Soft-Scoring" Matching Algorithm**
   * **The Problem:** Traditional hard-filtering excludes applicants who miss a single minor criterion, resulting in zero matches.
   * **Our Solution:** Built a dynamic scoring system that awards weighted points for matching criteria (e.g., State match = 8 pts, Income bracket fit = 5 pts). This ensures applicants always receive a ranked list of the closest possible schemes, rather than a blank page.

3. **Generative AI Context & Elaboration**
   * Integrates the **Groq API** to generate personalized, easy-to-understand explanations for each scheme.
   * Features a custom context-builder that parses the SQLite database rows and the applicant's profile into a strictly formatted LLM prompt, forcing the AI to output specific risk-mitigation steps for the applicant.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Database:** SQLite (with automatic schema migration)
* **Machine Learning:** Scikit-Learn (ExtraTreesRegressor), Pandas
* **Generative AI:** Groq API (`gpt-oss-120b` via OpenAI SDK format)
* **Frontend:** HTML, CSS, Bootstrap, Markdown (for rendering AI outputs)

---

## 🚧 Challenges We Overcame

Building this required solving several complex data-flow and architectural challenges:

1. **Database Querying & Filtering:** Rather than dumping all data and filtering in Python, we optimized our SQLite queries to fetch only necessary criteria, reducing memory overhead.
2. **Designing the Weighted Scoring System:** Transitioning from strict database `WHERE` clauses to a Python-based soft-scoring algorithm required extensive tuning to balance the weights (e.g., ensuring a BPL card requirement acts as a hard filter, while location acts as a soft bonus).
3. **Prompt Engineering & Context Injection:** Raw database rows cause LLMs to hallucinate. We engineered a dynamic context-builder that translates raw user data and scheme requirements into structured natural language before sending it to Groq, guaranteeing highly accurate, structured outputs.

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
* Python 3.8+
* Ensure you have the pre-trained ML model (`model.pkl`) and the database (`schemes.db`) in the root directory.

### 2. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install flask groq pandas scikit-learn python-dotenv markdown
