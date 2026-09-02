---
title: StayScout
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Agentic RAG stay finder across 15,000 US listings
---

# 🧭 StayScout: Agentic RAG Stay Finder

**Describe your ideal stay in plain English. StayScout turns it into real filters, searches 15,000 listings across 5 US cities and recommends only places that actually exist, with citations.**

[![Live demo](https://img.shields.io/badge/🤗%20Live%20demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/Shashank00164/stayscout)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> “Budget private room near the French Quarter under $90” → `city = New Orleans`, `room_type = Private room`, `price ≤ $90` → 5 cited listings on a map.

---

## ✨ Features

- **Chat-style search with follow-ups.** Ask “cheaper?” or “only entire homes” and the agent refines the previous filters.
- **Listing cards + interactive map.** Every recommendation is a numbered card that matches its `[n]` citation and its pin on the map.
- **Editable filters.** See exactly what the agent understood (city, budget, room type) as chips, and adjust them in one click.
- **Agent trace.** Watch each step live (parse → filter → retrieve → generate) with timings.
- **City insights.** Price distribution, room-type mix, a listing density map and top neighbourhoods per city.
- **Works without an API key.** A rule-based parser and template answer keep the app usable (and still fully grounded) when no Groq key is set.

---

## 🧠 Architecture

```text
User message ("cozy loft in Chicago under $150", or a follow-up like "cheaper?")
   │
   ▼
[1] Intent parser ───────► LLM (gpt-oss-120b on Groq) → {max_price, city, room_type, search_text}
   │                       (rule-based fallback when no key / LLM error; merges previous filters)
   ▼
[2] Metadata filter ─────► ChromaDB `where` clause enforces price / city / room type exactly
   │
   ▼
[3] Semantic retrieval ──► all-MiniLM-L6-v2 embeddings rank the filtered listings
   │
   ▼
[4] Grounded answer ─────► LLM writes a recommendation from the retrieved listings ONLY, citing [1], [2]…
   │
   ▼
Cards + map + filter chips + agent trace in the Streamlit UI
```

Plain vector search ignores numbers: “under $90” is just another token. Pulling hard constraints out first and enforcing them as metadata filters guarantees every result respects the budget, city and room type.

---

## 📁 Project structure

```text
stayscout/
├── app.py                 # Streamlit UI: chat search, cards, map, city insights, how-it-works pages
├── rag.py                 # Core engine: index building, intent parsing, filtered retrieval, grounded generation
├── build_index.py         # Precomputes listing embeddings → embeddings.npy (fast startup)
├── eval.py                # Reproducible evaluation: intent accuracy, constraint satisfaction, faithfulness
├── airbnb_clean.csv       # 15,000 cleaned listings (3,000 each: NYC, LA, SF, Chicago, New Orleans)
├── requirements.txt       # Pinned Python dependencies
├── Dockerfile             # Hugging Face Spaces image (CPU torch, embeddings baked in at build time)
├── .dockerignore
├── .streamlit/
│   └── config.toml        # Theme and server settings
├── .devcontainer/
│   └── devcontainer.json  # GitHub Codespaces / VS Code dev container
├── runtime.txt
├── LICENSE
└── README.md
```

`embeddings.npy` is generated (git-ignored) and `.streamlit/secrets.toml` holds your local API key (git-ignored).

---

## 🛠️ Tech stack

| Component | Tool |
|---|---|
| LLM | `openai/gpt-oss-120b` via Groq |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector database | ChromaDB (in-memory, metadata filtering) |
| Data | Pandas, NumPy |
| UI | Streamlit, pydeck |
| Deployment | Docker on Hugging Face Spaces |

---

## 📊 Evaluation

`eval.py` runs 10 fixed queries through the full pipeline and reports:

| Metric | What it checks |
|---|---|
| Intent accuracy | Extracted city / room type / max price match the expected values |
| Constraint satisfaction | Every retrieved listing respects the extracted filters |
| Citation validity | Every `[n]` in the answer points to a retrieved listing |
| Faithfulness | The cited name and price in the answer exactly match that listing |

```bash
python eval.py                          # rule-based mode, no key needed
GROQ_API_KEY=your_key python eval.py    # full LLM mode
```

Rule-based mode currently scores 1.00 on all four metrics (its answers come from a template, so faithfulness is guaranteed by design). Run LLM mode to measure how faithful the model's free-form answers are.

---

## 🚀 Run locally

```bash
git clone https://github.com/WR-Shashank/hotel-booking-agentic-rag.git
cd hotel-booking-agentic-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python build_index.py      # optional: precompute embeddings (otherwise done on first run)
streamlit run app.py
```

Add a free key from [console.groq.com](https://console.groq.com) in the sidebar, or save it once in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key"
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and choose **Docker → Blank**.
2. In the Space **Settings → Variables and secrets**, add a secret named `GROQ_API_KEY`. Visitors then get AI mode without their own key (capped at 25 AI searches per session; after that the app switches to rule-based mode).
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/Shashank00164/stayscout
   git push space main
   ```
4. The first build takes a few minutes (it installs CPU torch and precomputes embeddings). After that, the app starts in seconds.

---

## 👤 Author

**Shashank Kaushik** · [GitHub @WR-Shashank](https://github.com/WR-Shashank) · MIT License
