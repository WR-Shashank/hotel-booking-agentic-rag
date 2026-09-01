"""Core agentic RAG engine for StayScout.

Pipeline: parse intent (LLM, with a rule-based fallback) -> filtered vector
retrieval in ChromaDB -> grounded, cited answer generation.
"""

import json
import os
import re
import time

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "airbnb_clean.csv"
EMBEDDINGS_PATH = "embeddings.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "stayscout_listings"

CITIES = ["New York City", "Los Angeles", "San Francisco", "Chicago", "New Orleans"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
METADATA_COLUMNS = ["name", "city", "neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews"]

CITY_ALIASES = {
    "New York City": ["new york", "nyc", "manhattan", "brooklyn", "queens", "bronx", "harlem", "soho"],
    "Los Angeles": ["los angeles", "la", "l.a.", "hollywood", "santa monica", "venice beach", "malibu"],
    "San Francisco": ["san francisco", "sf", "frisco", "bay area", "mission district"],
    "Chicago": ["chicago", "chi-town", "wicker park", "lincoln park"],
    "New Orleans": ["new orleans", "nola", "french quarter", "bourbon street"],
}
ROOM_ALIASES = {
    "Private room": ["private room", "private bedroom", "own room"],
    "Shared room": ["shared room", "dorm", "hostel"],
    "Hotel room": ["hotel room"],
    "Entire home/apt": ["entire", "whole place", "whole home", "apartment", "apt", "house", "home", "loft", "condo", "studio"],
}

EMPTY_FILTERS = {"max_price": None, "city": None, "room_type": None, "search_text": ""}


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def load_embeddings(df, embed_model):
    """Load precomputed embeddings, computing and caching them on first run."""
    if os.path.exists(EMBEDDINGS_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] == len(df):
            return embeddings.astype(np.float32)

    embeddings = embed_model.encode(df["document"].tolist(), batch_size=256, show_progress_bar=False)
    try:
        np.save(EMBEDDINGS_PATH, embeddings.astype(np.float16))
    except OSError:
        pass  # read-only filesystem: just keep them in memory
    return embeddings.astype(np.float32)


def build_engine():
    df = pd.read_csv(DATA_PATH)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = load_embeddings(df, embed_model)

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    documents = df["document"].tolist()
    ids = [str(i) for i in df.index]
    metadatas = df[METADATA_COLUMNS].to_dict("records")
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    return df, embed_model, collection


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------

def sanitize_filters(raw, fallback_text):
    """Coerce LLM / rule output into the allowed filter schema."""
    filters = dict(EMPTY_FILTERS)
    try:
        price = raw.get("max_price")
        filters["max_price"] = float(price) if price not in (None, "", "null") and float(price) > 0 else None
    except (TypeError, ValueError):
        pass
    filters["city"] = raw.get("city") if raw.get("city") in CITIES else None
    filters["room_type"] = raw.get("room_type") if raw.get("room_type") in ROOM_TYPES else None
    filters["search_text"] = str(raw.get("search_text") or fallback_text).strip()
    return filters


def _find_alias(text, aliases):
    for canonical, words in aliases.items():
        for word in words:
            if re.search(rf"(?<![\w.]){re.escape(word)}(?![\w])", text):
                return canonical
    return None


def parse_intent_rules(user_query, previous_filters=None):
    """Deterministic parser used when no LLM key is available."""
    text = user_query.lower()
    previous = previous_filters or {}

    price = None
    match = re.search(r"(?:under|below|less than|max(?:imum)?|budget(?: of)?|up to|<=?|within)\s*\$?\s*(\d{2,4})", text)
    if not match:
        match = re.search(r"\$\s*(\d{2,4})", text)
    if match:
        price = float(match.group(1))
    elif previous.get("max_price") and re.search(r"\b(cheaper|less expensive|lower price)\b", text):
        price = round(previous["max_price"] * 0.75)
    elif re.search(r"\b(cheap|budget|affordable)\b", text):
        price = 100.0

    is_follow_up = bool(previous) and len(text.split()) <= 8
    city = _find_alias(text, CITY_ALIASES) or (previous.get("city") if is_follow_up else None)
    room_type = _find_alias(text, ROOM_ALIASES) or (previous.get("room_type") if is_follow_up else None)
    if price is None and is_follow_up:
        price = previous.get("max_price")

    search_text = user_query
    if is_follow_up and previous.get("search_text"):
        search_text = f"{previous['search_text']} {user_query}"

    return sanitize_filters(
        {"max_price": price, "city": city, "room_type": room_type, "search_text": search_text},
        user_query,
    )


def parse_intent_llm(client, user_query, previous_filters=None):
    system_instructions = f"""You are an AI assistant parsing search criteria from travel accommodation queries.
Extract the relevant search constraints and return ONLY a raw JSON object with the following schema:
{{
  "max_price": number or null,
  "city": {" | ".join(f'"{c}"' for c in CITIES)} | null,
  "room_type": {" | ".join(f'"{r}"' for r in ROOM_TYPES)} | null,
  "search_text": string (short semantic summary of location preference or atmosphere)
}}
If PREVIOUS FILTERS are given, the user may be refining an earlier search (e.g. "cheaper", "only private rooms",
"what about Chicago?"). Keep previous values unless the new message changes them. If the message is clearly a brand-new
search, ignore the previous filters.
Do not wrap in Markdown codeblocks or include conversational text."""

    user_content = user_query
    if previous_filters:
        user_content = f"PREVIOUS FILTERS: {json.dumps(previous_filters)}\n\nNEW MESSAGE: {user_query}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    raw_content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return sanitize_filters(json.loads(raw_content), user_query)
    except Exception:
        return parse_intent_rules(user_query, previous_filters)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def build_where_clause(filters):
    conditions = []
    if filters.get("max_price") is not None:
        conditions.append({"price": {"$lte": float(filters["max_price"])}})
    if filters.get("city"):
        conditions.append({"city": {"$eq": filters["city"]}})
    if filters.get("room_type"):
        conditions.append({"room_type": {"$eq": filters["room_type"]}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve(df, embed_model, collection, filters, top_k=5):
    """Return the matching listings as a list of dicts (with coordinates for the map)."""
    query_vector = embed_model.encode([filters.get("search_text") or "comfortable stay"]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k, where=build_where_clause(filters))
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    listings = []
    for listing_id, distance in zip(ids, distances):
        row = df.loc[int(listing_id)]
        listings.append({
            "id": int(listing_id),
            "name": row["name"],
            "city": row["city"],
            "neighbourhood": row["neighbourhood"],
            "room_type": row["room_type"],
            "price": float(row["price"]),
            "minimum_nights": int(row["minimum_nights"]),
            "number_of_reviews": int(row["number_of_reviews"]),
            "availability_365": int(row["availability_365"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "document": row["document"],
            "match": max(0.0, 1.0 - float(distance) / 2.0),
        })
    return listings


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_generation_prompt(user_query, listings):
    context_str = "\n".join(f"[{idx + 1}] {listing['document']}" for idx, listing in enumerate(listings))
    return f"""You are an expert AI Travel Advisor. Provide personalized recommendations based strictly on the retrieved property listings.

STRICT GROUNDING RULES:
1. State ONLY facts explicitly present in the listings (Name, Room Type, City, Neighborhood, Price/night, Min Nights, Review Count).
2. Never invent details or assume unmentioned amenities.
3. Reference each recommended listing with its citation tag [1], [2], etc.
4. Keep tone professional, clean, and helpful.

FORMAT EXPECTED:
Here are the best matching accommodations for your trip:

- **[1] Property Name** — Room type in Neighborhood, City. $X/night (Min N nights, R reviews).
- **[2] Property Name** — Room type in Neighborhood, City. $X/night (Min N nights, R reviews).

Finish with a brief summary recommendation sentence.

RETRIEVED LISTINGS:
{context_str}

USER QUERY: {user_query}"""


def generate_answer_llm(client, user_query, listings):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_generation_prompt(user_query, listings)}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_answer_template(listings):
    """Grounded answer without an LLM: every fact comes straight from the listing."""
    lines = ["Here are the best matching accommodations for your trip:", ""]
    for idx, l in enumerate(listings, start=1):
        lines.append(
            f"- **[{idx}] {l['name']}** — {l['room_type']} in {l['neighbourhood']}, {l['city']}. "
            f"${l['price']:.0f}/night (Min {l['minimum_nights']} nights, {l['number_of_reviews']} reviews)."
        )
    best = max(listings, key=lambda l: l["number_of_reviews"])
    lines += ["", f"The most reviewed option is **{best['name']}** with {best['number_of_reviews']} reviews."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(df, embed_model, collection, user_query, client=None, top_k=5,
              previous_filters=None, override_filters=None, on_step=None):
    """Run the full pipeline. `on_step(label)` is called as each stage starts (for UI tracing)."""
    trace = []

    def step(label, started):
        trace.append({"step": label, "ms": round((time.perf_counter() - started) * 1000)})

    t = time.perf_counter()
    if override_filters is not None:
        on_step and on_step("Applying your adjusted filters")
        filters = sanitize_filters(override_filters, user_query)
        step("Filters adjusted manually", t)
    elif client is not None:
        on_step and on_step("Parsing intent with the LLM")
        filters = parse_intent_llm(client, user_query, previous_filters)
        step("Intent parsed (LLM)", t)
    else:
        on_step and on_step("Parsing intent with rules")
        filters = parse_intent_rules(user_query, previous_filters)
        step("Intent parsed (rule-based)", t)

    t = time.perf_counter()
    on_step and on_step("Searching 15,000 listings")
    listings = retrieve(df, embed_model, collection, filters, top_k=top_k)
    step(f"Retrieved {len(listings)} listings", t)

    if not listings:
        return {
            "answer": "No listings matched those exact criteria. Try raising the budget or broadening the city or room type.",
            "filters": filters,
            "listings": [],
            "trace": trace,
        }

    t = time.perf_counter()
    if client is not None:
        on_step and on_step("Writing a grounded recommendation")
        answer = generate_answer_llm(client, user_query, listings)
        step("Answer generated (LLM)", t)
    else:
        answer = generate_answer_template(listings)
        step("Answer generated (template)", t)

    return {"answer": answer, "filters": filters, "listings": listings, "trace": trace}
