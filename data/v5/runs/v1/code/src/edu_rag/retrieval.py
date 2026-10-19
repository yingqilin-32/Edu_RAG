"""Dependency-free Chinese lexical BM25 and reciprocal rank fusion."""

from collections import Counter
from math import log1p
import re
import unicodedata

TOKENIZER_VERSION = "nfkc-casefold-cjk-1-2gram-identifiers-v1"
BM25_K1 = 1.2
BM25_B = 0.75
TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOP_TOKENS = frozenset("的 了 呢 吗 请 是 在 这 那 与 和".split())


def tokenize(value: str) -> list[str]:
    """Chinese unigrams/bigrams; retain numbers, decimals and full exercise IDs."""
    value = unicodedata.normalize("NFKC", value).casefold()
    tokens = []
    for match in TOKEN_PATTERN.finditer(value):
        token = match[0]
        if "\u3400" <= token[0] <= "\u9fff":
            tokens.extend(char for char in token if char not in STOP_TOKENS)
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def lexical_text(row):
    # Stored solutions never enter either retrieval index. They are attached
    # only after selection, as in V2. Labels make exercise IDs searchable.
    return "\n".join(str(row.get(key) or "") for key in ("content", "source_title", "knowledge_point_title"))


def bm25_rank(question, rows, limit, *, k1=BM25_K1, b=BM25_B):
    if not rows:
        return []
    terms = set(tokenize(question))
    if not terms:
        return []
    counts = [Counter(tokenize(lexical_text(row))) for row in rows]
    lengths = [sum(counter.values()) for counter in counts]
    average = sum(lengths) / len(lengths)
    if not average:
        return []
    document_frequency = Counter(term for counter in counts for term in counter)
    scored = []
    for row, counter, length in zip(rows, counts, lengths, strict=True):
        score = 0.0
        for term in sorted(terms):
            frequency = counter[term]
            if not frequency:
                continue
            inverse_frequency = log1p((len(rows) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / average))
        # Zero-overlap documents must not become lexical candidates.
        if score > 0:
            scored.append({**row, "bm25_score": score})
    return sorted(scored, key=lambda row: (-row["bm25_score"], str(row["id"])))[:limit]


def unique_hits(hits):
    seen = set()
    for hit in hits:
        identity = str(hit["id"])
        if identity not in seen:
            seen.add(identity)
            yield hit


def rrf_fuse(dense_hits, lexical_hits, limit, *, k=60):
    if k < 1:
        raise ValueError("RRF k must be positive")
    combined = {}
    for branch, hits in (("dense", dense_hits), ("bm25", lexical_hits)):
        for rank, hit in enumerate(unique_hits(hits), 1):
            identity = str(hit["id"])
            if identity not in combined:
                combined[identity] = {**hit, "dense_rank": None, "bm25_rank": None,
                                      "bm25_score": None, "rrf_score": 0.0}
            result = combined[identity]
            result[branch + "_rank"] = rank
            if branch == "bm25":
                result["bm25_score"] = hit["bm25_score"]
            result["rrf_score"] += 1 / (k + rank)
    def ordering(row):
        ranks = [rank for rank in (row["dense_rank"], row["bm25_rank"]) if rank is not None]
        return (-row["rrf_score"], -len(ranks), min(ranks), str(row["id"]))
    return sorted(combined.values(), key=ordering)[:limit]


def rank_candidates(question, rows, *, mode, top_k, candidate_k=20, rrf_k=60):
    dense = sorted(rows, key=lambda row: (-row["similarity"], str(row["id"])))[:candidate_k] if mode != "bm25" else []
    lexical = bm25_rank(question, rows, candidate_k) if mode != "dense" else []
    if mode == "hybrid":
        selected = rrf_fuse(dense, lexical, top_k, k=rrf_k)
    else:
        selected = [{**row, "dense_rank": index if mode == "dense" else None,
                     "bm25_rank": index if mode == "bm25" else None,
                     "bm25_score": row.get("bm25_score"), "rrf_score": None}
                    for index, row in enumerate((dense if mode == "dense" else lexical)[:top_k], 1)]
    trace = {
        "dense": [{"id": str(row["id"]), "rank": index, "score": row["similarity"], "title": row.get("source_title")}
                  for index, row in enumerate(dense, 1)],
        "bm25": [{"id": str(row["id"]), "rank": index, "score": row["bm25_score"], "title": row.get("source_title")}
                 for index, row in enumerate(lexical, 1)],
    }
    return selected, trace


def select_problem_context(ranked, limit):
    """Keep the best knowledge explanation and example when both are available."""
    selected = set()
    if limit >= 2:
        for key in ("document_id", "exercise_id"):
            first = next((row for row in ranked if row.get(key) is not None), None)
            if first is not None:
                selected.add(str(first["id"]))
    for row in ranked:
        if len(selected) >= limit:
            break
        selected.add(str(row["id"]))
    return [row for row in ranked if str(row["id"]) in selected]
