"""
AI-Assisted Reading Comprehension Tool
Streamlit Deployment — NLP Project, Binus University @Alam Sutera
"""

import re
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

warnings.filterwarnings("ignore")

import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuizGen AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0f1923 100%); min-height: 100vh; }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; margin-bottom: 1.5rem; }
.hero h1 { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #f0f4f8; margin: 0; letter-spacing: -1px; }
.hero h1 span { color: #4fc3f7; }
.hero p { color: #90a4b4; font-size: 1rem; margin-top: 0.5rem; font-weight: 300; }
.mcq-card { background: linear-gradient(145deg, #1e2f3f, #162333); border: 1px solid #2a3f52; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
.mcq-number { font-size: 0.7rem; font-weight: 600; letter-spacing: 2px; color: #4fc3f7; text-transform: uppercase; margin-bottom: 0.5rem; }
.mcq-type-badge { display: inline-block; background: rgba(79,195,247,0.12); color: #4fc3f7; border: 1px solid rgba(79,195,247,0.3); border-radius: 20px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-left: 0.5rem; vertical-align: middle; }
.mcq-question { font-family: 'DM Serif Display', serif; font-size: 1.05rem; color: #e8f0f7; margin: 0.6rem 0 1rem; line-height: 1.5; }
.option-row { display: flex; align-items: flex-start; padding: 0.45rem 0.8rem; border-radius: 8px; margin-bottom: 0.35rem; font-size: 0.9rem; color: #b0c4d4; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); }
.option-row.correct { background: rgba(79,195,247,0.08); border: 1px solid rgba(79,195,247,0.25); color: #e0f4ff; font-weight: 500; }
.option-label { font-weight: 600; min-width: 22px; color: #4fc3f7; margin-right: 0.5rem; }
.check-icon { margin-left: auto; color: #4fc3f7; font-size: 0.85rem; }
.answer-key { font-size: 0.78rem; color: #4fc3f7; margin-top: 0.8rem; padding-top: 0.6rem; border-top: 1px solid rgba(79,195,247,0.15); }
.stat-box { background: rgba(79,195,247,0.06); border: 1px solid rgba(79,195,247,0.2); border-radius: 12px; padding: 1rem; text-align: center; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #4fc3f7; }
.stat-label { font-size: 0.75rem; color: #7a9aad; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }
section[data-testid="stSidebar"] { background: #111e28 !important; border-right: 1px solid #1e2f3f; }
.stTextArea textarea { background: #162333 !important; border: 1px solid #2a3f52 !important; color: #e8f0f7 !important; border-radius: 12px !important; font-size: 0.95rem !important; }
.stTextArea textarea:focus { border-color: #4fc3f7 !important; box-shadow: 0 0 0 2px rgba(79,195,247,0.15) !important; }
.stButton > button { background: linear-gradient(135deg, #4fc3f7, #0288d1) !important; color: #0a1520 !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important; font-size: 0.95rem !important; width: 100%; }
.stButton > button:hover { opacity: 0.88 !important; }
.stAlert { border-radius: 10px !important; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #c8dce8; margin: 1.5rem 0 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid #1e2f3f; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── NLP Constants ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

NER_Q_MAP = {
    "PERSON": "who", "NORP": "who", "ORG": "who",
    "GPE": "where", "LOC": "where", "FAC": "where",
    "DATE": "when", "TIME": "when",
    "CARDINAL": "how many", "ORDINAL": "which",
    "MONEY": "how much", "PERCENT": "what percent",
    "EVENT": "what", "PRODUCT": "what", "WORK_OF_ART": "what",
    "LAW": "what", "LANGUAGE": "what",
}

BEST_THRESHOLD = 0.30

# ── Load spaCy ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ── Load ML Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        import joblib
        rf = joblib.load("rf_model.joblib")
        with open("feature_cols.pkl", "rb") as f:
            feat_cols = pickle.load(f)
        return rf, feat_cols, True
    except FileNotFoundError:
        return None, None, False

rf_model, FEATURE_COLS, models_loaded = load_models()

# ── Pipeline Functions ────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_sentences(text, min_words=5):
    sents = sent_tokenize(clean_text(text))
    return [s.strip() for s in sents if len(s.split()) >= min_words]


class TextRank:
    def __init__(self, damping=0.85, steps=100, tol=1e-5):
        self.damping = damping
        self.steps = steps
        self.tol = tol
        self._tfidf = TfidfVectorizer(stop_words="english")

    def _sim_matrix(self, sentences):
        if len(sentences) < 2:
            return np.eye(len(sentences))
        try:
            mat = self._tfidf.fit_transform(sentences)
            sim = cosine_similarity(mat)
        except Exception:
            sim = np.eye(len(sentences))
        np.fill_diagonal(sim, 0)
        return sim

    def score(self, sentences):
        if not sentences: return {}
        if len(sentences) == 1: return {0: 1.0}
        G = nx.from_numpy_array(self._sim_matrix(sentences))
        try:
            return nx.pagerank(G, alpha=self.damping, max_iter=self.steps, tol=self.tol)
        except Exception:
            n = len(sentences)
            return {i: 1.0 / n for i in range(n)}

    def top_sentences(self, text):
        sents = get_sentences(text)
        scores = self.score(sents)
        return sents, scores


@st.cache_resource
def get_textrank():
    return TextRank()

tr = get_textrank()


def extract_features(sentence, tr_score=0.0, position=0, total=1):
    doc = nlp(sentence)
    lower = sentence.lower()
    toks = [t for t in word_tokenize(lower) if t.isalpha()]
    labels = [e.label_ for e in doc.ents]
    pos_ = [t.pos_ for t in doc]
    deps = [t.dep_ for t in doc]
    W5H1 = {
        "who": ["who", "whom", "whose"],
        "what": ["what", "which"],
        "when": ["when", "date", "year", "century", "month", "day"],
        "where": ["where", "location", "place", "city", "country"],
        "why": ["why", "because", "reason", "cause"],
        "how": ["how", "method", "way", "process"],
    }
    return {
        "word_count": len(toks),
        "char_count": len(sentence),
        "rel_position": position / max(total - 1, 1),
        "is_first": int(position == 0),
        "is_last": int(position == total - 1),
        "textrank_score": tr_score,
        "ner_count": len(doc.ents),
        "has_person": int(any(l in ["PERSON", "NORP"] for l in labels)),
        "has_location": int(any(l in ["GPE", "LOC", "FAC"] for l in labels)),
        "has_date": int(any(l in ["DATE", "TIME"] for l in labels)),
        "has_org": int("ORG" in labels),
        "has_number": int(any(l in ["CARDINAL", "ORDINAL", "MONEY"] for l in labels)),
        "has_who_kw": int(any(k in lower for k in W5H1["who"])),
        "has_what_kw": int(any(k in lower for k in W5H1["what"])),
        "has_when_kw": int(any(k in lower for k in W5H1["when"])),
        "has_where_kw": int(any(k in lower for k in W5H1["where"])),
        "has_why_kw": int(any(k in lower for k in W5H1["why"])),
        "has_how_kw": int(any(k in lower for k in W5H1["how"])),
        "noun_count": sum(1 for p in pos_ if p in ["NOUN", "PROPN"]),
        "verb_count": sum(1 for p in pos_ if p == "VERB"),
        "has_subject": int(any(d in ["nsubj", "nsubjpass"] for d in deps)),
        "has_object": int(any(d in ["dobj", "pobj", "attr"] for d in deps)),
    }


def get_qtype_and_answer(sentence):
    doc = nlp(sentence)
    priority = {"PERSON": 1, "NORP": 2, "ORG": 3, "GPE": 4, "LOC": 5,
                "FAC": 6, "DATE": 7, "TIME": 8, "CARDINAL": 9, "MONEY": 10}
    ents = sorted(doc.ents, key=lambda e: priority.get(e.label_, 99))
    if ents:
        e = ents[0]
        return NER_Q_MAP.get(e.label_, "what"), e.text, e.label_
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN"]:
            return "what", " ".join(t.text for t in token.subtree), "NOUN"
    return "what", sentence.split()[0], "UNKNOWN"


def form_question(sentence, q_type, answer):
    doc = nlp(sentence)
    root_verb = next((t.lemma_ for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    blanked = sentence.replace(answer, "_____", 1)
    templates = {
        "who": f"Who {root_verb or 'is mentioned'} in this passage?",
        "where": "Where did the events described take place?",
        "when": f"When {blanked.rstrip('.')}?",
        "what": f"What does the passage say about {answer}?",
        "how many": f"How many _____ are mentioned? (clue: {blanked})",
        "how much": f"How much is mentioned in: \"{blanked}\"?",
        "what percent": "What percentage is described in this passage?",
        "which": f"Which {answer} is being referred to?",
    }
    return templates.get(q_type, f"What is meant by \"{blanked}\"?")


def generate_distractors(answer, context, n=3):
    doc_ans = nlp(answer)
    doc_ctx = nlp(context)
    ans_lower = answer.lower().strip()
    tgt_label = doc_ans.ents[0].label_ if doc_ans.ents else None
    distractors = set()
    if tgt_label:
        for ent in doc_ctx.ents:
            if ent.label_ == tgt_label and ent.text.lower() != ans_lower:
                distractors.add(ent.text.strip())
    if len(distractors) < n:
        for chunk in doc_ctx.noun_chunks:
            ct = chunk.text.strip()
            if ct.lower() != ans_lower and 1 < len(ct.split()) <= 5:
                distractors.add(ct)
    result = [d for d in list(distractors) if d.lower() != ans_lower][:n]
    while len(result) < n:
        result.append("None of the above")
    return result[:n]


def generate_mcq(sentence, context):
    q_type, answer, ner_label = get_qtype_and_answer(sentence)
    question = form_question(sentence, q_type, answer)
    distractors = generate_distractors(answer, context, n=3)
    options = [answer] + distractors
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    correct = labels[options.index(answer)]
    return {
        "question": question,
        "options": dict(zip(labels, options)),
        "answer_key": correct,
        "answer_text": answer,
        "type": q_type,
        "ner": ner_label,
    }


def generate_questions(text, n_questions=10):
    sents = get_sentences(text)
    if not sents:
        return []
    _, tr_scores = tr.top_sentences(text)
    rows = []
    for j, sent in enumerate(sents):
        try:
            f = extract_features(sent, tr_scores.get(j, 0.0), j, len(sents))
        except Exception:
            f = {c: 0 for c in FEATURE_COLS}
        rows.append(f)
    X_new = pd.DataFrame(rows)[FEATURE_COLS].fillna(0)
    probs = rf_model.predict_proba(X_new)[:, 1]
    eligible = [i for i, p in enumerate(probs) if p >= BEST_THRESHOLD]
    if len(eligible) < n_questions:
        eligible = list(np.argsort(probs)[::-1][:n_questions + 5])
    eligible_sorted = sorted(eligible, key=lambda i: probs[i], reverse=True)
    mcqs, used = [], set()
    for idx in eligible_sorted:
        if len(mcqs) >= n_questions:
            break
        sent = sents[idx]
        if sent in used:
            continue
        try:
            mcq = generate_mcq(sent, text)
            mcq["rf_prob"] = float(probs[idx])
            mcq["source_sentence"] = sent
            mcqs.append(mcq)
            used.add(sent)
        except Exception:
            continue
    return mcqs


# ── UI: Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>Quiz<span>Gen</span> AI</h1>
<p>AI-Assisted Reading Comprehension Tool &nbsp;·&nbsp; NLP Project &nbsp;·&nbsp; Binus University @Alam Sutera</p>
</div>
""", unsafe_allow_html=True)

# ── UI: Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_questions = st.slider("Number of Questions", 1, 10, 5)
    show_answer = st.toggle("Show Answer Key", value=True)
    show_source = st.toggle("Show Source Sentence", value=False)
    show_prob   = st.toggle("Show RF Confidence", value=False)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if models_loaded:
        st.success("✅ Models loaded")
        st.markdown("- **Classifier:** Random Forest\n- **Threshold:** 0.30\n- **F1 Score:** 0.83\n- **Key Extractor:** TextRank")
    else:
        st.error("❌ Models not found")
        st.markdown("Letakkan `rf_model.joblib` dan `feature_cols.pkl` di folder yang sama dengan `app.py`.")
    st.markdown("---")
    st.markdown("### 👥 Team")
    st.markdown("<p>Catherine Sharon<br>Cherysh Amanda Ovelia<br>Daryl Rafael Wiguna<br>Kezia Gracella Haryono<br>Rebecca Jielian</p>", unsafe_allow_html=True)

# ── UI: Main ──────────────────────────────────────────────────────────────────
if not models_loaded:
    st.warning("⚠️ Model tidak ditemukan. Pastikan `rf_model.joblib` dan `feature_cols.pkl` ada di folder project.", icon="⚠️")
    st.stop()

# Session state untuk example text
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

col_input, col_gap = st.columns([3, 1])

with col_input:
    st.markdown('<div class="section-title">📄 Input Text</div>', unsafe_allow_html=True)

    with st.expander("💡 Try an example text"):
        examples = {
            "🌿 Amazon Rainforest": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in South America. It covers most of the Amazon basin, which spans 7,000,000 square kilometres. Brazil contains 60% of the rainforest, followed by Peru with 13% and Colombia with 10%. The Amazon is home to 10% of all species on Earth and produces roughly 20% of the world's oxygen. Indigenous peoples have lived in the Amazon for at least 11,200 years. In 2019, widespread forest fires attracted global attention to ongoing deforestation of the region.",
            "🚀 History of NASA": "The National Aeronautics and Space Administration, known as NASA, was established in 1958 by President Dwight D. Eisenhower. NASA has led many significant missions, including the Apollo 11 mission in 1969, when Neil Armstrong became the first human to walk on the Moon. The agency is headquartered in Washington, D.C. and operates numerous research centres across the United States. NASA's Mars rovers, including Curiosity and Perseverance, have collected valuable data about the Martian surface. The agency employs more than 17,000 people and has an annual budget of approximately 25 billion dollars.",
            "🌊 The Ocean": "The ocean covers more than 70% of Earth's surface and contains about 97% of all water on the planet. It is divided into five major basins: the Pacific, Atlantic, Indian, Southern, and Arctic Oceans. The Pacific Ocean is the largest and deepest, reaching a maximum depth of about 11,034 metres at the Mariana Trench. Oceans regulate the Earth's climate by absorbing carbon dioxide and distributing heat around the globe. Millions of species, including fish, mammals, and invertebrates, depend on the ocean for their survival. Human activities such as pollution and overfishing pose serious threats to ocean biodiversity.",
        }
        for label, example_text in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state["input_text"] = example_text
                st.rerun()

    input_text = st.text_area(
        label="",
        height=220,
        placeholder="Paste a paragraph or short story here… (min. 5 sentences recommended)",
        label_visibility="collapsed",
        key="input_text",
    )

# ── Generate Button ───────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    generate_btn = st.button(f"✨ Generate {n_questions} Questions")

# ── Generate & Display ────────────────────────────────────────────────────────
if generate_btn:
    current_text = st.session_state.get("input_text", "")
    if not current_text.strip():
        st.warning("Please enter some text first.", icon="✍️")
    elif len(get_sentences(current_text)) < 3:
        st.warning("Text is too short. Please enter at least 3–5 sentences.", icon="📏")
    else:
        with st.spinner("🔍 Analysing text with TextRank + Random Forest…"):
            mcqs = generate_questions(current_text, n_questions=n_questions)

        if not mcqs:
            st.error("Could not generate questions. Try a longer or more informative text.")
        else:
            type_counts = {}
            for m in mcqs:
                type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1
            avg_conf = np.mean([m["rf_prob"] for m in mcqs]) * 100

            st.markdown('<div class="section-title">📝 Generated Questions</div>', unsafe_allow_html=True)

            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{len(mcqs)}</div><div class="stat-label">Questions</div></div>', unsafe_allow_html=True)
            with stat_cols[1]:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{len(type_counts)}</div><div class="stat-label">Types (5W+1H)</div></div>', unsafe_allow_html=True)
            with stat_cols[2]:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{avg_conf:.0f}%</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)
            with stat_cols[3]:
                most_common = max(type_counts, key=type_counts.get).upper()
                st.markdown(f'<div class="stat-box"><div class="stat-value">{most_common}</div><div class="stat-label">Dominant Type</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── MCQ Cards ────────────────────────────────────────────────────
            for i, mcq in enumerate(mcqs, 1):
                # Build options HTML — single line per option (no indentation = no markdown code block)
                options_html = ""
                for lbl, opt in mcq["options"].items():
                    is_correct = (lbl == mcq["answer_key"]) and show_answer
                    css = "option-row correct" if is_correct else "option-row"
                    lbl_css = "option-label correct" if is_correct else "option-label"
                    chk = '<span class="check-icon">✓</span>' if is_correct else ""
                    options_html += f'<div class="{css}"><span class="{lbl_css}">{lbl}.</span><span>{opt}</span>{chk}</div>'

                answer_row = f'<div class="answer-key">🔑 Answer: <strong>{mcq["answer_key"]}. {mcq["answer_text"]}</strong></div>' if show_answer else ""

                source_row = ""
                if show_source:
                    src = mcq.get("source_sentence", "")
                    source_row = f'<div style="font-size:0.75rem;color:#5a7a8a;margin-top:0.5rem;font-style:italic;">Source: {src[:120]}{"…" if len(src)>120 else ""}</div>'

                prob_row = ""
                if show_prob:
                    conf = mcq.get("rf_prob", 0) * 100
                    prob_row = f'<div style="font-size:0.72rem;color:#5a7a8a;margin-top:0.3rem;">RF Confidence: {conf:.1f}%</div>'

                # Single-line f-string to avoid markdown code block interpretation
                card_html = f'<div class="mcq-card"><div><span class="mcq-number">Question {i}</span><span class="mcq-type-badge">{mcq["type"]}</span></div><div class="mcq-question">{mcq["question"]}</div>{options_html}{answer_row}{source_row}{prob_row}</div>'
                st.markdown(card_html, unsafe_allow_html=True)

            # ── Download ──────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            lines = []
            for i, m in enumerate(mcqs, 1):
                lines.append(f"Q{i}. [{m['type'].upper()}] {m['question']}")
                for lbl, opt in m["options"].items():
                    marker = "*" if lbl == m["answer_key"] else " "
                    lines.append(f"  {marker} {lbl}. {opt}")
                lines.append(f"  Answer: {m['answer_key']}. {m['answer_text']}")
                lines.append("")
            export_text = "\n".join(lines)

            _, dl_col, _ = st.columns([1, 2, 1])
            with dl_col:
                st.download_button(
                    label="⬇️ Download Questions (.txt)",
                    data=export_text,
                    file_name="generated_questions.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
