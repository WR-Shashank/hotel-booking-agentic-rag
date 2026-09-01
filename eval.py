"""Reproducible evaluation for StayScout.

Measures, over a fixed set of test queries:
  - intent accuracy:        extracted filters match the expected city / room type / max price
  - constraint satisfaction: every retrieved listing respects the extracted filters
  - citation validity:      every [n] cited in the answer points to a retrieved listing
  - faithfulness:           cited name and price in the answer match the cited listing exactly

Usage:
  python eval.py                 # rule-based mode (no API key needed)
  GROQ_API_KEY=... python eval.py   # full LLM mode
"""

import os
import re

from rag import build_engine, run_agent

TEST_CASES = [
    ("Affordable private room in New Orleans under $90", {"city": "New Orleans", "room_type": "Private room", "max_price": 90}),
    ("Entire apartment in San Francisco under $250", {"city": "San Francisco", "room_type": "Entire home/apt", "max_price": 250}),
    ("Quiet room in Brooklyn under $120", {"city": "New York City", "max_price": 120}),
    ("Beach house in Santa Monica below $400", {"city": "Los Angeles", "room_type": "Entire home/apt", "max_price": 400}),
    ("Shared room in Chicago", {"city": "Chicago", "room_type": "Shared room"}),
    ("Cozy loft in Chicago under $150", {"city": "Chicago", "max_price": 150}),
    ("Private room in Manhattan with lots of reviews", {"city": "New York City", "room_type": "Private room"}),
    ("Somewhere near the French Quarter for a jazz weekend", {"city": "New Orleans"}),
    ("Hollywood apartment under $200", {"city": "Los Angeles", "max_price": 200}),
    ("Budget stay in SF under $100", {"city": "San Francisco", "max_price": 100}),
]

BULLET = re.compile(r"\*\*\[(\d+)\]\s*(.+?)\*\*.*?\$(\d+(?:\.\d+)?)")


def intent_correct(filters, expected):
    for key, value in expected.items():
        got = filters.get(key)
        if key == "max_price":
            if got is None or abs(float(got) - value) > 1:
                return False
        elif got != value:
            return False
    return True


def constraints_hold(filters, listings):
    return all(
        (filters.get("max_price") is None or l["price"] <= filters["max_price"])
        and (not filters.get("city") or l["city"] == filters["city"])
        and (not filters.get("room_type") or l["room_type"] == filters["room_type"])
        for l in listings
    )


def score_answer(answer, listings):
    cited = valid = faithful = 0
    for num, name, price in BULLET.findall(answer):
        cited += 1
        idx = int(num) - 1
        if 0 <= idx < len(listings):
            valid += 1
            listing = listings[idx]
            if name.strip().lower() == listing["name"].strip().lower() and abs(float(price) - listing["price"]) < 1:
                faithful += 1
    return cited, valid, faithful


def main():
    client = None
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    mode = "LLM" if client else "rule-based"

    df, embed_model, collection = build_engine()
    totals = {"intent": 0, "constraints": 0, "cited": 0, "valid": 0, "faithful": 0}

    print(f"Running {len(TEST_CASES)} queries in {mode} mode\n")
    for query, expected in TEST_CASES:
        result = run_agent(df, embed_model, collection, query, client=client, top_k=5)
        ok_intent = intent_correct(result["filters"], expected)
        ok_constraints = constraints_hold(result["filters"], result["listings"])
        cited, valid, faithful = score_answer(result["answer"], result["listings"])
        totals["intent"] += ok_intent
        totals["constraints"] += ok_constraints
        totals["cited"] += cited
        totals["valid"] += valid
        totals["faithful"] += faithful
        print(f"{'✓' if ok_intent else '✗'} intent  {'✓' if ok_constraints else '✗'} constraints  "
              f"{faithful}/{cited} faithful  ·  {query}")

    n = len(TEST_CASES)
    cited = max(totals["cited"], 1)
    print(f"\nIntent accuracy:          {totals['intent'] / n:.2f}")
    print(f"Constraint satisfaction:  {totals['constraints'] / n:.2f}")
    print(f"Citation validity:        {totals['valid'] / cited:.2f}")
    print(f"Faithfulness:             {totals['faithful'] / cited:.2f}")


if __name__ == "__main__":
    main()
