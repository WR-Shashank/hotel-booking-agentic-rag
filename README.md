#  Hotel Booking & Travel Assistant — Agentic RAG

An intelligent travel recommendation assistant designed to answer natural-language queries about hotel and Airbnb accommodations across **5 major US metropolitan areas**. It fuses **hybrid semantic retrieval** with **structured metadata query parsing** to deliver **strict, grounded property recommendations with zero hallucinated listings**.

Built as a production-grade applied AI project covering data pipeline processing, vector retrieval, LLM filter extraction, evaluation, and web deployment.

---

## Architecture & Workflow

```text
User Query (e.g. "affordable private room in New Orleans under $90")
   │
   ▼
[1] LLM Filter Extractor ──► Extracts metadata constraints (max_price, city, room_type)
   │
   ▼
[2] Hybrid Vector Search ──► Metadata pre-filtering (ChromaDB) + Semantic dense retrieval (all-MiniLM-L6-v2)
   │
   ▼
[3] Grounded Answer Engine ──► Llama 3.3 70B generates cited recommendation using retrieved context ONLY
   │
   ▼
Final Recommendation with Source Citations
```

---

## ⚙️ Key Technical Features

- **Agentic Constraint Parsing**: Converts plain-English queries into structured metadata filters (e.g., `price <= $90`, `city == New Orleans`, `room_type == Private room`), solving standard RAG issues where vector distance ignores numerical bounds.
- **Hybrid Dense Retrieval**: Combines exact ChromaDB relational filtering with `sentence-transformers` embeddings (`all-MiniLM-L6-v2`).
- **Faithfulness-Hardened Prompting**: Constrains LLM generation strictly to retrieved text, raising evaluation faithfulness scores from **0.60 to 0.92**.
- **Interactive Web Interface**: Streamlit dashboard featuring top-K result tuning, intent extraction inspection, and source context expansion.

---

## 🛠️ Technology Stack

| Component | Framework / Tool |
|---|---|
| **Large Language Model** | Llama 3.3 70B Versatile (via Groq API) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB |
| **Data Engine** | Pandas & NumPy |
| **Web UI** | Streamlit |
| **Indexed Dataset** | 15,000 Cleaned US Accommodation Listings (NYC, LA, SF, Chicago, New Orleans) |

---

## 📊 Evaluation & Performance Metrics

Faithfulness assesses how accurately model outputs strictly mirror retrieved document facts:

| Metric Stage | Faithfulness Score | Notes |
|---|---|---|
| Baseline RAG Prompt | `0.60` | Subject to LLM embellishments and unverified adjectives |
| **Hardened Agentic Prompt** | **`0.92`** | Enforces strict citation rules and factual restriction |

---

## 🚀 Local Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/WR-Shashank/hotel-booking-agentic-rag.git
   cd hotel-booking-agentic-rag
   ```

2. **Create Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application**
   ```bash
   streamlit run app.py
   ```

> **Note**: Obtain a free API key from [console.groq.com](https://console.groq.com) and enter it in the app sidebar to run queries.

---

## 👤 Author & Maintainer

**Shashank Kaushik**
- **GitHub**: [@WR-Shashank](https://github.com/WR-Shashank)
- **License**: MIT License
