# 🤖 Become an AI Engineer — By Improving Your Discord RAG Bot

> Based on *AI Engineering* by Chip Huyen · Applied to your [`discord-rag-bot`](https://github.com/prozdrljivac/discord-rag-bot) project

**The idea is simple:** every task below is a real improvement to your app *and* a hands-on lesson in AI Engineering. No tutorial hell — just shipping.

---

## How to Use This Guide

Each section maps a **Chip Huyen concept → a concrete task on your bot**.

> 💡 **Huyen's core rule:** Start simple. Exhaust prompting before RAG. Exhaust RAG before finetuning. You already have RAG — now make it production-grade.

---

## 🟢 Phase 1 — Clean Up (Chapters 1, 5)

*Concept: AI Engineering is about building applications on top of foundation models — not training them. Your first job is getting the basics right.*

### Task 1.1 — Add Type Hints and Structured Output with Pydantic
**What it teaches:** Structured output (Chapter 5 — Prompt Engineering)

Your README already lists this as a TODO. Right now your bot probably returns raw strings from the LLM. Wrap responses in a Pydantic model.

```python
# Before — raw string
response = llm.generate(prompt)

# After — structured output
from pydantic import BaseModel

class BotResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str  # "high" | "medium" | "low"
```

> 💡 **Real-world example:** Every production AI API (OpenAI, Anthropic) has structured output / function calling. This is how you prevent your app from crashing when the LLM changes its response format.

---

### Task 1.2 — Improve the System Prompt
**What it teaches:** System vs. User Prompt, Role Prompting (Chapter 5)

Find where you build the prompt in `main.py` or `embedding.py`. Add a proper system prompt that:
- Defines the bot's role ("You are a helpful assistant specialized in X")
- Sets hard constraints ("Only answer based on the provided context. If the answer is not in the context, say so.")
- Specifies output format

> 💡 **Real-world example:** A customer support bot without a system prompt will happily discuss competitors, share internal info, and go off-topic. A good system prompt is your first guardrail.

---

### Task 1.3 — Add a `.env` Validation Step on Startup
**What it teaches:** Production architecture hygiene (Chapter 10)

Use Pydantic's `BaseSettings` to validate all environment variables at startup. The app should fail fast with a clear error if a required key is missing — not crash mid-request.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    discord_token: str
    openai_api_key: str
    milvus_host: str

    class Config:
        env_file = ".env"
```

---

## 🟡 Phase 2 — Better RAG (Chapter 6)

*Concept: RAG retrieves relevant information from an external knowledge base and injects it into the prompt. "RAG is for facts."*

### Task 2.1 — Make Milvus & Redis Queries Async
**What it teaches:** Inference optimization, latency (Chapter 9)

Your README flags this as a TODO. Blocking I/O in a Discord bot means one slow query blocks all users. Make `db.py` queries async using `asyncio`.

```python
# Before — blocking
def search_vectors(query_embedding):
    return collection.search(...)

# After — non-blocking
async def search_vectors(query_embedding):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, collection.search, ...)
```

> 💡 **Real-world example:** This is exactly what happens at scale — Netflix, Uber, and every high-traffic AI product runs async I/O. A 200ms DB call that blocks the thread will tank your throughput.

---

### Task 2.2 — Improve Chunking Strategy
**What it teaches:** Chunking in RAG (Chapter 6)

Open `populate_db.py` and look at how you split documents. Fixed-size chunking (e.g., every 500 characters) is the naive approach. Try:
- **Sentence-aware chunking** — don't split mid-sentence
- **Overlap chunking** — add 50-token overlap between chunks so context isn't lost at boundaries

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
```

> 💡 **Real-world example:** A legal AI that chunks mid-sentence will retrieve half a clause. Sentence-aware chunking is why legal and medical AI products invest heavily in document parsing.

---

### Task 2.3 — Add Hybrid Retrieval (BM25 + Vector)
**What it teaches:** Retrieval strategies — term-based vs. embedding-based (Chapter 6)

Right now you use pure vector search (Milvus). Add BM25 keyword search alongside it and merge results. This improves recall for exact keyword queries where semantic search fails.

> 💡 **Real-world example:** Elasticsearch + vector search is the standard in production RAG. Pure vector search fails on product codes, names, and acronyms — "GPT-4o" means nothing semantically until the model has seen it.

---

### Task 2.4 — Return Source Citations in Responses
**What it teaches:** Factual consistency, trust (Chapters 6, 10)

After retrieval, include which document chunks were used in the final response. This is a core RAG feature for trust and compliance.

```python
class BotResponse(BaseModel):
    answer: str
    sources: list[str]  # e.g. ["doc_42, chunk_3", "doc_17, chunk_1"]
```

---

## 🟠 Phase 3 — Evaluation (Chapters 3 & 4)

*Concept: "Evaluation is everything. Build your eval pipeline before optimizing anything." — Huyen*

This is the most important phase. Without eval, you are flying blind.

### Task 3.1 — Write a Test Suite for the RAG Pipeline
**What it teaches:** Functional correctness evaluation (Chapter 3)

You have a `tests/` folder — use it. Write tests that:
1. Given a known document in the DB, does the bot retrieve it?
2. Given a question with a known answer, does the response contain it?

```python
def test_retrieval_finds_known_document():
    result = search_vectors(embed("What is X?"))
    assert any("expected_keyword" in r.text for r in result)
```

> 💡 **Real-world example:** Every serious AI team has a "golden dataset" — a fixed set of questions with known correct answers. Any change to the prompt, chunking, or model is validated against it.

---

### Task 3.2 — Add AI-as-a-Judge Evaluation
**What it teaches:** LLM-as-Judge (Chapter 4)

Use a second LLM call to score your bot's responses automatically. This is how you scale evaluation without human raters.

```python
JUDGE_PROMPT = """
Given this question: {question}
And this context retrieved: {context}
And this answer: {answer}

Rate the answer on:
1. Faithfulness (0-5): Is the answer grounded in the context?
2. Relevance (0-5): Does it actually answer the question?

Respond in JSON: {{"faithfulness": N, "relevance": N, "reason": "..."}}
"""
```

> 💡 **Watch out for:** LLM judges prefer longer answers and outputs from their own model family (Huyen's warning). Calibrate your judge against human ratings on a small sample first.

---

### Task 3.3 — Log Every Request and Score It
**What it teaches:** Monitoring and observability (Chapter 10)

Log every Discord query, the retrieved chunks, the response, and the AI-judge score to a file or database. This creates your feedback dataset.

```python
import json
from datetime import datetime

def log_interaction(question, chunks, response, scores):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "retrieved_chunks": chunks,
        "response": response,
        "scores": scores
    }
    with open("interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

---

## 🔴 Phase 4 — Guardrails & Production (Chapters 5, 10)

*Concept: The 5-step production architecture — Enhance Context → Guardrails → Router → Cache → Agents*

### Task 4.1 — Add Input Guardrails
**What it teaches:** Prompt injection defense, guardrails (Chapters 5, 10)

Add a validation step before the prompt reaches the LLM:
- Detect and block prompt injection attempts ("Ignore your previous instructions...")
- Add a topic filter — if the question is off-domain, respond with a polite refusal instead of hallucinating

```python
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore your system prompt",
    "pretend you are",
    "jailbreak"
]

def is_injection_attempt(text: str) -> bool:
    return any(p in text.lower() for p in INJECTION_PATTERNS)
```

> 💡 **Real-world example:** In 2023, a car dealership's AI chatbot was manipulated into agreeing to sell a car for $1 via prompt injection. This is not theoretical.

---

### Task 4.2 — Add Semantic Caching
**What it teaches:** Latency reduction with caches (Chapter 10)

You already have Redis in your stack — use it for semantic caching. If two users ask the same question (or semantically similar ones), return the cached answer instead of calling the LLM again.

```python
import hashlib

def get_cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()

async def cached_query(query: str):
    key = get_cache_key(query)
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)

    response = await full_rag_pipeline(query)
    await redis.setex(key, 3600, json.dumps(response))  # Cache 1hr
    return response
```

> 💡 **Real-world example:** OpenAI charges per token. A Discord bot that answers "What's the pricing?" 500 times/day and caches the answer after the first call saves real money.

---

### Task 4.3 — Write the Docs (Finish README)
**What it teaches:** AI Engineering communication (Chapter 10)

Your README says "Setup Guide: Coming soon..." Fill it in. A good AI project README includes:
- What problem it solves
- Architecture diagram (Retriever → Vector DB → LLM → Response)
- Setup instructions
- How to add new documents to the knowledge base
- Known limitations (what the bot can and can't answer)

---

## 🟣 Phase 5 — Advanced (Chapters 7, 8, 9)

*Only after Phase 1–4 are solid. Huyen's rule: exhaust RAG before finetuning.*

### Task 5.1 — Experiment with a Different Embedding Model
**What it teaches:** Model selection, training data effects (Chapter 2)

Swap out your embedding model and measure the impact on retrieval quality using your eval suite from Phase 3. Try `text-embedding-3-small` vs `text-embedding-ada-002` or an open-source model like `nomic-embed-text`.

---

### Task 5.2 — Evaluate Finetuning vs. RAG for Your Domain
**What it teaches:** RAG vs. Finetuning decision (Chapter 7)

Run this checklist against your bot's specific domain:

| Question | If Yes → |
|----------|----------|
| Does the knowledge change frequently? | Use RAG |
| Do you need source citations? | Use RAG |
| Does the *style* of responses need to change? | Finetuning |
| Is accuracy on a narrow task the goal? | Finetuning |

If finetuning makes sense: try QLoRA on a small open-source model (Llama-3-8B) using your `interactions.jsonl` log from Phase 3 as training data.

---

### Task 5.3 — Add an Agent Pattern
**What it teaches:** Agents, tool use, memory (Chapter 6)

Extend the bot to handle multi-step queries by giving it tools:
- `search_docs(query)` — your existing RAG retrieval
- `get_current_date()` — for time-aware questions
- `summarize_thread(messages)` — summarize a Discord thread

Start with the **Crawl** pattern (Huyen's Crawl–Walk–Run): every action requires a human to approve it in Discord before execution.

---

## 📊 Your Learning Progress Map

```
Phase 1 — Clean Up         [ ] 1.1  [ ] 1.2  [ ] 1.3
Phase 2 — Better RAG       [ ] 2.1  [ ] 2.2  [ ] 2.3  [ ] 2.4
Phase 3 — Evaluation       [ ] 3.1  [ ] 3.2  [ ] 3.3
Phase 4 — Production       [ ] 4.1  [ ] 4.2  [ ] 4.3
Phase 5 — Advanced         [ ] 5.1  [ ] 5.2  [ ] 5.3
```

---

## Key Principles to Internalize (Chip Huyen)

1. **Evaluation is everything.** Build eval before optimizing anything. (Do Phase 3 early if you're impatient.)
2. **Start simple.** Exhaust prompting → then RAG → then finetuning.
3. **RAG is for facts. Finetuning is for form.**
4. **Data is the moat.** Your `interactions.jsonl` log is your most valuable asset.
5. **Agents need guardrails.** More automation = more failure modes.

---

*Based on AI Engineering by Chip Huyen (O'Reilly, 2025) · Applied to [discord-rag-bot](https://github.com/prozdrljivac/discord-rag-bot)*
