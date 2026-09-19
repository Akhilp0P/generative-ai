"""ESG-aware keyword extraction utilities: TF-IDF, RAKE-style extraction, TextRank and hybrid ranking."""
from __future__ import annotations
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from esg_dictionary import ESG_TERMS, classify_esg, framework_mapping

STOP = set(ENGLISH_STOP_WORDS)
ESG_STOP = STOP | {
    "company", "companies", "group", "business", "businesses", "year", "report", "reports",
    "financial", "performance", "management", "operations", "million", "billion", "including", "according"
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

def load_documents(path="data/esg_documents.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(clean_text(text)) if w.lower() not in ESG_STOP and len(w) > 2]

def esg_domain_score(term: str) -> float:
    term = term.lower().strip()
    return float(any(t == term or t in term for terms in ESG_TERMS.values() for t in terms))

def _split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if s.strip()]
    return sents or [clean_text(text)]

def keywords_tfidf(text: str, n: int = 10) -> list[tuple[str, float]]:
    vectorizer = TfidfVectorizer(stop_words=list(ESG_STOP), ngram_range=(1, 3), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", min_df=1)
    matrix = vectorizer.fit_transform(_split_sentences(text))
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(scores)[::-1]
    return [(str(terms[i]), float(scores[i])) for i in order[:n]]

def keywords_rake(text: str, n: int = 10) -> list[tuple[str, float]]:
    words = [w for w in WORD_RE.findall(clean_text(text).lower()) if len(w) > 2]
    phrases, current = [], []
    for word in words:
        if word in ESG_STOP:
            if current:
                phrases.append(current); current = []
        else:
            current.append(word)
    if current: phrases.append(current)
    phrases = [p[:6] for p in phrases if 1 <= len(p) <= 6]
    frequency, degree = Counter(), defaultdict(int)
    for phrase in phrases:
        for word in phrase:
            frequency[word] += 1
            degree[word] += len(phrase)
    word_score = {w: degree[w] / max(frequency[w], 1) for w in frequency}
    best = {}
    for phrase in phrases:
        candidate = " ".join(phrase)
        score = sum(word_score.get(w, 0.0) for w in phrase) + (1.5 if esg_domain_score(candidate) else 0.0)
        best[candidate] = max(score, best.get(candidate, float("-inf")))
    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:n]

def keywords_textrank(text: str, n: int = 10, window: int = 4, damping: float = 0.85, iterations: int = 40) -> list[tuple[str, float]]:
    toks = tokens(text)
    if len(toks) < 2: return [(w, 1.0) for w in toks[:n]]
    vocab = list(dict.fromkeys(toks)); idx = {w: i for i, w in enumerate(vocab)}
    graph = np.zeros((len(vocab), len(vocab)), dtype=float)
    for i in range(len(toks)):
        for j in range(i + 1, min(i + window, len(toks))):
            a, b = idx[toks[i]], idx[toks[j]]
            if a != b: graph[a, b] += 1; graph[b, a] += 1
    row_sums = graph.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    transition = graph / row_sums; ranks = np.ones(len(vocab)) / len(vocab)
    for _ in range(iterations): ranks = (1 - damping) / len(vocab) + damping * (transition.T @ ranks)
    return [(vocab[i], float(ranks[i] + 0.25 * esg_domain_score(vocab[i]))) for i in np.argsort(ranks)[::-1][:n]]

def _normalize(items):
    if not items: return {}
    vals = np.array([v for _, v in items], dtype=float); lo, hi = vals.min(), vals.max()
    if np.isclose(lo, hi): return {k: 1.0 for k, _ in items}
    return {k: float((v - lo) / (hi - lo)) for k, v in items}

def keywords_hybrid(text: str, n: int = 10, weights=(0.30, 0.25, 0.25, 0.20)):
    tfidf = _normalize(keywords_tfidf(text, max(n * 3, 20)))
    rake = _normalize(keywords_rake(text, max(n * 3, 20)))
    textrank = _normalize(keywords_textrank(text, max(n * 3, 20)))
    candidates = set(tfidf) | set(rake) | set(textrank)
    output = []
    for term in candidates:
        score = weights[0]*tfidf.get(term, 0) + weights[1]*rake.get(term, 0) + weights[2]*textrank.get(term, 0) + weights[3]*esg_domain_score(term)
        output.append((term, float(score), classify_esg(term), framework_mapping(term)))
    return sorted(output, key=lambda x: x[1], reverse=True)[:n]

def extract_company_keywords(df, company=None, year=None, n=10):
    subset = df.copy()
    if company is not None: subset = subset[subset.company.str.lower() == company.lower()]
    if year is not None: subset = subset[subset.year == year]
    rows = []
    for _, row in subset.iterrows():
        for keyword, score, category, frameworks in keywords_hybrid(row.text, n):
            rows.append({"company": row.company, "year": row.year, "document_type": row.document_type, "section": row.section, "keyword": keyword, "score": score, "category": category, "frameworks": ", ".join(frameworks)})
    return pd.DataFrame(rows)

def make_disclosure_profile(extracted):
    if extracted.empty: return pd.DataFrame(columns=["category", "keywords", "share"])
    counts = extracted.category.value_counts(); profile = counts.rename_axis("category").reset_index(name="keywords")
    profile["share"] = profile.keywords / profile.keywords.sum(); return profile
