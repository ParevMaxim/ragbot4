from __future__ import annotations

from typing import List, Dict, Callable, Optional
import re
import ollama

from config import (
    OLLAMA_HOST,
    EMBEDDING_MODEL,
    DOC_LANGUAGE,
    CHAT_MODEL_MAIN as CFG_CHAT_MODEL_MAIN,
    CHAT_MODEL_SECONDARY as CFG_CHAT_MODEL_SECONDARY,
    REWRITE_MODEL as CFG_REWRITE_MODEL,
    AGGREGATE_MODEL as CFG_AGGREGATE_MODEL,
    LLM_OPTIONS,
    OLLAMA_KEEP_ALIVE,
    ANSWER_WITH_CITATIONS,
    ENABLE_YEAR_GUARD,
)

ollama_client = ollama.Client(host=OLLAMA_HOST)

# Current models (can be changed from UI)
CHAT_MODEL_MAIN = CFG_CHAT_MODEL_MAIN
CHAT_MODEL_SECONDARY = CFG_CHAT_MODEL_SECONDARY
REWRITE_MODEL = CFG_REWRITE_MODEL
AGGREGATE_MODEL = CFG_AGGREGATE_MODEL

MAX_EMBED_CHARS = 4000  # safe for nomic-embed-text


def set_llm_main(model_name: str):
    global CHAT_MODEL_MAIN, CHAT_MODEL_SECONDARY, REWRITE_MODEL, AGGREGATE_MODEL
    CHAT_MODEL_MAIN = model_name
    CHAT_MODEL_SECONDARY = model_name
    REWRITE_MODEL = model_name
    AGGREGATE_MODEL = model_name


def get_llm_main() -> str:
    return CHAT_MODEL_MAIN


def embed_texts(
    texts: List[str],
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    total = len(texts)

    for i, t in enumerate(texts, start=1):
        if len(t) > MAX_EMBED_CHARS:
            t = t[:MAX_EMBED_CHARS]

        try:
            resp = ollama_client.embeddings(
                model=EMBEDDING_MODEL,
                prompt=t,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
        except TypeError:
            # older client
            resp = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=t)

        vectors.append(resp["embedding"])

        if progress:
            try:
                progress(i, total)
            except Exception:
                pass

    return vectors


def _ollama_chat(model_name: str, messages: List[Dict[str, str]]) -> str:
    try:
        resp = ollama_client.chat(
            model=model_name,
            messages=messages,
            options=LLM_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except TypeError:
        resp = ollama_client.chat(model=model_name, messages=messages)
    return resp["message"]["content"].strip()


def rewrite_query(question: str, doc_language: str | None = None) -> str:
    target = (doc_language or DOC_LANGUAGE or "same").lower()

    if target == "same":
        system = (
            "Ты модуль нормализации поисковых запросов.\n"
            "Получаешь вопрос пользователя на ЛЮБОМ языке.\n"
            "Требования к ответу:\n"
            "- ОДНА короткая строка текста (1 предложение).\n"
            "- Без примеров кода, без форматирования, без маркеров списка.\n"
            "- Не используй ``` и переносы строк.\n"
            "- Сохрани исходный язык вопроса.\n"
            "НЕ отвечай на сам вопрос, только переформулируй его как поисковый запрос."
        )
        user = f"Перепиши этот запрос в виде краткого поискового запроса:\n{question}"
    else:
        system = (
            "Ты модуль нормализации и перевода поисковых запросов для RAG-системы.\n"
            f"Документация в основном на языке: {target}.\n"
            "Требования к ответу:\n"
            "- ОДНА короткая строка текста (1 предложение) на языке документации.\n"
            "- Без примеров кода, без форматирования, без маркеров списка.\n"
            "- Не используй ``` и переносы строк.\n"
            "НЕ отвечай на сам вопрос, только переформулируй его как поисковый запрос."
        )
        user = f"Вопрос пользователя:\n{question}\n\nДай итоговый поисковый запрос:"

    rewritten = _ollama_chat(
        REWRITE_MODEL,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    rewritten = rewritten.replace("\n", " ").replace("```", " ").strip()
    if len(rewritten) > 200:
        rewritten = rewritten[:200]
    return rewritten


_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")


def _years_in_text(s: str) -> set[str]:
    return set(_YEAR_RE.findall(s or ""))


def answer_with_context(question: str, context_chunks: List[Dict]) -> str:
    context_text = ""
    for i, ch in enumerate(context_chunks, start=1):
        src = ch.get("source", "")
        sec = ch.get("section", "")
        meta = f"{src}" + (f" — {sec}" if sec else "")
        context_text += f"[Фрагмент {i} — {meta}]\n{ch['text']}\n\n"

    if ANSWER_WITH_CITATIONS:
        system = (
            "Ты помощник по документации.\n"
            "Отвечай на языке пользователя.\n"
            "Строго опирайся ТОЛЬКО на предоставленный контекст.\n"
            "ФОРМАТ ОБЯЗАТЕЛЕН:\n"
            "Ключевые цитаты (1–3 строки): каждая строка вида: (Фрагмент N) «точная цитата из контекста»\n"
            "Ответ: кратко и по делу, после ключевых утверждений указывай (Фрагмент N).\n"
            "Если в контексте нет ответа — напиши: 'В предоставленных фрагментах нет информации, чтобы ответить.'\n"
            "Не добавляй факты, которых нет в цитатах."
        )
    else:
        system = (
            "Ты помощник по документации.\n"
            "Отвечай на языке пользователя.\n"
            "Отвечай строго на основе контекста.\n"
            "Если нужной информации нет — прямо скажи об этом и не выдумывай."
        )

    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context_text}"

    answer = _ollama_chat(
        CHAT_MODEL_MAIN,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # Опциональная защита от "выдуманных годов"
    if ENABLE_YEAR_GUARD:
        ctx_years = _years_in_text(context_text)
        ans_years = _years_in_text(answer)
        extra = sorted(ans_years - ctx_years)
        if extra:
            answer = (
                answer.rstrip()
                + "\n\nПримечание: годы "
                + ", ".join(extra)
                + " не найдены в приведённых фрагментах; проверьте ответ по цитатам/контексту."
            )

    return answer