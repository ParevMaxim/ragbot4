from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from config import ENABLE_QUERY_REWRITE, ENABLE_FACTOID_BOOST, ENABLE_MMR
from .llm import embed_texts, rewrite_query, answer_with_context
from .storage import load_kb, kb_file_path


_TOKEN_RE = re.compile(r"[a-zа-я0-9_]+", re.IGNORECASE)

_STOP_RU = {
    "в","во","на","по","о","об","от","до","из","за","для","при","над","под",
    "и","а","но","или","ли","не","ни","это","этот","эта","эти","то","все","всё",
    "как","какой","какая","какие","каком","какого","какую",
    "где","когда","почему","зачем","что","чем","чему","кого","кому","кем",
    "сколько",
}
_STOP_EN = {"the","a","an","in","on","at","to","of","and","or","is","are","was","were","be"}


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower().replace("ё", "е")
    toks = _TOKEN_RE.findall(text)
    return [t for t in toks if t not in _STOP_RU and t not in _STOP_EN]


def _is_factoid_question(q: str) -> bool:
    ql = (q or "").lower()

    # год/дата
    if re.search(r"\b(в каком году|какого года|когда|дата)\b", ql):
        return True

    # цифры: версии/порты/коды/лимиты
    if re.search(r"\d", ql):
        return True

    # документационные маркеры
    if any(x in ql for x in (
        "ошибк", "error", "код", "code", "http", "status",
        "порт", "port", "верси", "version",
        "параметр", "parameter", "arg", "argument",
        "ключ", "key", "флаг", "flag",
        "по умолчанию", "default",
        "timeout", "retry", "limit", "max", "min",
    )):
        return True

    # похоже на код / точные токены
    if any(x in q for x in ("(", ")", "=", "::", "->", ".", "/", "\\", "_")):
        return True

    return False


def _extract_numbers_like(text: str) -> List[str]:
    """
    Вытаскиваем числа и версии вида 1913, 5432, 1.2.3, 10-20.
    """
    return re.findall(r"\b\d+(?:\.\d+){1,3}\b|\b\d{2,}\b", text or "")


@dataclass
class _KBCache:
    mtime: float
    texts: List[str]
    sources: List[str]
    sections: List[str]
    emb_norm: np.ndarray
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]   # токены документов (для overlap rerank)


_KB_CACHE: dict[str, _KBCache] = {}


def _load_cached(kb_name: str) -> Optional[_KBCache]:
    path = kb_file_path(kb_name)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None

    cached = _KB_CACHE.get(kb_name)
    if cached and abs(cached.mtime - mtime) < 1e-6:
        return cached

    kb = load_kb(kb_name)
    if not kb:
        return None

    texts = [ch.text for ch in kb]
    sources = [ch.source for ch in kb]
    sections = [ch.section for ch in kb]

    emb = np.asarray([ch.embedding for ch in kb], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.clip(norms, 1e-8, None)

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    cached = _KBCache(
        mtime=mtime,
        texts=texts,
        sources=sources,
        sections=sections,
        emb_norm=emb_norm,
        bm25=bm25,
        corpus_tokens=corpus_tokens,
    )
    _KB_CACHE[kb_name] = cached
    return cached


def _rrf_fusion(sem_scores: np.ndarray, lex_scores: np.ndarray, k: int = 60, beta_lex: float = 1.0) -> np.ndarray:
    n = sem_scores.size
    sem_order = np.argsort(-sem_scores)
    lex_order = np.argsort(-lex_scores)

    sem_rank = np.empty(n, dtype=np.int32)
    lex_rank = np.empty(n, dtype=np.int32)
    sem_rank[sem_order] = np.arange(1, n + 1)
    lex_rank[lex_order] = np.arange(1, n + 1)

    return (1.0 / (k + sem_rank) + beta_lex * (1.0 / (k + lex_rank))).astype(np.float32)


def _mmr_select(candidates: np.ndarray, relevance: np.ndarray, emb_norm: np.ndarray, top_k: int, lam: float = 0.85) -> List[int]:
    cand = [int(i) for i in candidates.tolist()]
    selected: List[int] = []

    while cand and len(selected) < top_k:
        best_i = None
        best_score = -1e18

        for idx in cand:
            rel = float(relevance[idx])
            if not selected:
                div = 0.0
            else:
                sims = emb_norm[selected] @ emb_norm[idx]
                div = float(np.max(sims))
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best_score = score
                best_i = idx

        selected.append(int(best_i))
        cand.remove(int(best_i))

    return selected


def _factoid_rerank(
    cand: np.ndarray,
    fused: np.ndarray,
    cache: _KBCache,
    query_tokens: List[str],
    query_numbers: List[str],
) -> np.ndarray:
    """
    Универсальный rerank для фактоидных вопросов:
    - token overlap (сколько токенов запроса реально встречается в документе)
    - совпадения чисел/версий (если есть в вопросе)
    """
    qset = set(query_tokens)
    if not qset:
        return cand

    scores = []
    for i in cand.tolist():
        i = int(i)
        doc_toks = cache.corpus_tokens[i]
        doc_set = set(doc_toks)

        overlap = len(qset.intersection(doc_set))

        # совпадение чисел/версий
        txt = cache.texts[i]
        num_match = 0
        if query_numbers:
            for n in query_numbers:
                if n in txt:
                    num_match += 1

        # итоговый скор: fused + небольшой бонус за overlap + бонус за числа
        s = float(fused[i]) + 0.02 * overlap + 0.10 * num_match
        scores.append((s, i))

    scores.sort(reverse=True)
    return np.array([i for _, i in scores], dtype=np.int32)


def answer_question(kb_name: str, question: str, top_k: int = 8) -> str:
    t0 = time.perf_counter()

    search_query = rewrite_query(question) if ENABLE_QUERY_REWRITE else question
    t1 = time.perf_counter()

    cache = _load_cached(kb_name)
    if not cache:
        return (
            "Запрос для поиска по документации:\n"
            f"{search_query}\n\n"
            f"База знаний '{kb_name}' пуста. Сначала проиндексируйте документацию."
        )

    n_docs = len(cache.texts)
    top_k = min(top_k, n_docs)

    # Dense
    q = np.asarray(embed_texts([search_query])[0], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    sem_scores = (cache.emb_norm @ q).astype(np.float32)

    # BM25
    q_tokens = _tokenize(search_query)
    lex_scores = (
        np.asarray(cache.bm25.get_scores(q_tokens), dtype=np.float32)
        if q_tokens else np.zeros(n_docs, dtype=np.float32)
    )

    factoid = ENABLE_FACTOID_BOOST and _is_factoid_question(question)

    beta_lex = 2.8 if factoid else 1.2
    fused = _rrf_fusion(sem_scores, lex_scores, k=60, beta_lex=beta_lex)

    # union кандидатов из лексики и семантики
    pool = min(n_docs, max(50, top_k * 12))
    idx_lex = np.argsort(-lex_scores)[:pool]
    idx_sem = np.argsort(-sem_scores)[:pool]
    cand = np.unique(np.concatenate([idx_lex, idx_sem]))

    # базовая сортировка по fused
    cand_sorted = cand[np.argsort(-fused[cand])]

    if factoid:
        # 1) rerank по overlap/числам
        q_numbers = _extract_numbers_like(search_query)
        cand_sorted = _factoid_rerank(cand_sorted, fused, cache, q_tokens, q_numbers)

        # 2) гарантируем минимум 2 фрагмента из BM25-top (чтобы не потерять “точные строки”)
        min_lex = max(2, (top_k * 2) // 3)  # при 4 -> 2
        selected: List[int] = []
        for i in idx_lex:
            ii = int(i)
            if ii not in selected:
                selected.append(ii)
            if len(selected) >= min_lex:
                break
        for i in cand_sorted:
            ii = int(i)
            if ii not in selected:
                selected.append(ii)
            if len(selected) >= top_k:
                break
        selected = selected[:top_k]
        mmr_used = False
    else:
        # объяснительные вопросы: MMR помогает убрать дубликаты
        if ENABLE_MMR and len(cand_sorted) > top_k:
            selected = _mmr_select(cand_sorted[:min(len(cand_sorted), 200)], fused, cache.emb_norm, top_k=top_k, lam=0.85)
            mmr_used = True
        else:
            selected = [int(i) for i in cand_sorted[:top_k].tolist()]
            mmr_used = False

    hits: List[Dict] = []
    for i in selected:
        hits.append(
            {
                "text": cache.texts[i],
                "source": cache.sources[i],
                "section": cache.sections[i],
                "score": float(fused[i]),
                "index": i,
            }
        )

    t2 = time.perf_counter()
    answer = answer_with_context(question, hits)
    t3 = time.perf_counter()

    print(f"[RAG] rewrite: {t1 - t0:.2f} s (enabled={ENABLE_QUERY_REWRITE})")
    print(f"[RAG] retrieval: {t2 - t1:.2f} s (docs={n_docs}, factoid={factoid}, mmr_used={mmr_used})")
    print(f"[RAG] LLM: {t3 - t2:.2f} s")
    print(f"[RAG] TOTAL: {t3 - t0:.2f} s")

    return (
        "Запрос для поиска по документации:\n"
        f"{search_query}\n\n"
        f"{answer}"
    )


def debug_retrieval(kb_name: str, question: str, top_k: int = 10) -> List[Dict]:
    search_query = rewrite_query(question) if ENABLE_QUERY_REWRITE else question
    cache = _load_cached(kb_name)
    if not cache:
        return []

    n_docs = len(cache.texts)

    q = np.asarray(embed_texts([search_query])[0], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    sem_scores = (cache.emb_norm @ q).astype(np.float32)

    q_tokens = _tokenize(search_query)
    lex_scores = (
        np.asarray(cache.bm25.get_scores(q_tokens), dtype=np.float32)
        if q_tokens else np.zeros(n_docs, dtype=np.float32)
    )

    factoid = ENABLE_FACTOID_BOOST and _is_factoid_question(question)
    beta_lex = 2.8 if factoid else 1.2
    fused = _rrf_fusion(sem_scores, lex_scores, k=60, beta_lex=beta_lex)

    top_k = min(top_k, n_docs)
    order = np.argsort(-fused)[:top_k]

    results: List[Dict] = []
    for idx in order:
        i = int(idx)
        txt = cache.texts[i]
        results.append(
            {
                "index": i,
                "score": float(fused[i]),
                "semantic": float(sem_scores[i]),
                "lexical": float(lex_scores[i]),
                "source": cache.sources[i],
                "section": cache.sections[i],
                "text": txt[:400] + ("..." if len(txt) > 400 else ""),
            }
        )
    return results