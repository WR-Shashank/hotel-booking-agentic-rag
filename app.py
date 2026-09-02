import html
import os

import pandas as pd
import pydeck as pdk
import streamlit as st
from groq import Groq

from rag import CITIES, LLM_MODEL, ROOM_TYPES, build_engine, run_agent

APP_NAME = "StayScout"
GITHUB_URL = "https://github.com/WR-Shashank/hotel-booking-agentic-rag"
SHARED_KEY_QUERY_LIMIT = 25
ACCENT_RGB = [108, 92, 231]

EXAMPLE_QUERIES = [
    ("🎷", "Budget private room near the French Quarter in New Orleans under $90"),
    ("🌉", "Stylish entire apartment in San Francisco under $250"),
    ("🗽", "Quiet place in Brooklyn for a long stay under $120"),
    ("🌴", "Beach vibes in Santa Monica, entire home under $300"),
]
CITY_ICONS = {"New York City": "🗽", "Los Angeles": "🌴", "San Francisco": "🌉", "Chicago": "🏙️", "New Orleans": "🎷"}

st.set_page_config(page_title=f"{APP_NAME} · AI stay finder", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.hero { padding: 1.6rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(108,92,231,.16), rgba(0,184,148,.12));
        border: 1px solid rgba(108,92,231,.25); }
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 .3rem 0; line-height: 1.15;
           background: linear-gradient(90deg, #6C5CE7, #00B894); -webkit-background-clip: text;
           background-clip: text; color: transparent; }
.hero p { font-size: 1.05rem; opacity: .8; margin: 0 0 .9rem 0; }
.stat-row { display: flex; flex-wrap: wrap; gap: .5rem; }
.stat { padding: .3rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 600;
        background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.2); }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .4rem 0 .8rem 0; }
.chip { padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
        background: rgba(108,92,231,.14); color: inherit; border: 1px solid rgba(108,92,231,.35); }
.chip.muted { background: rgba(128,128,128,.1); border-color: rgba(128,128,128,.25); font-weight: 500; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .8rem; margin: .4rem 0 1rem 0; }
.card { position: relative; padding: 1rem 1rem .9rem 1rem; border-radius: 16px;
        background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.22);
        transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; }
.card:hover { transform: translateY(-3px); border-color: rgba(108,92,231,.6); box-shadow: 0 8px 24px rgba(108,92,231,.15); }
.card .num { position: absolute; top: .8rem; right: .8rem; width: 1.7rem; height: 1.7rem; border-radius: 50%;
             display: flex; align-items: center; justify-content: center; font-size: .8rem; font-weight: 700;
             background: #6C5CE7; color: white; }
.card .title { font-weight: 700; font-size: .98rem; line-height: 1.3; margin-right: 2rem; min-height: 2.5rem; }
.card .where { font-size: .82rem; opacity: .75; margin: .3rem 0 .6rem 0; }
.card .price { font-size: 1.35rem; font-weight: 800; }
.card .price span { font-size: .8rem; font-weight: 500; opacity: .7; }
.card .meta { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .55rem; }
.card .tag { font-size: .72rem; padding: .12rem .5rem; border-radius: 6px; background: rgba(128,128,128,.13); }
.card .tag.room { background: rgba(0,184,148,.16); }
.card .bar { height: 4px; border-radius: 4px; background: rgba(128,128,128,.18); margin-top: .7rem; overflow: hidden; }
.card .bar div { height: 100%; background: linear-gradient(90deg, #6C5CE7, #00B894); }
.card .foot { display: flex; justify-content: space-between; font-size: .72rem; opacity: .7; margin-top: .3rem; }
.card a { color: inherit; }
.step { display: flex; gap: 1rem; padding: 1rem 1.2rem; border-radius: 14px; margin-bottom: .7rem;
        background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.2); }
.step .n { flex: none; width: 2.2rem; height: 2.2rem; border-radius: 50%; background: #6C5CE7; color: white;
           display: flex; align-items: center; justify-content: center; font-weight: 800; }
.step h4 { margin: 0 0 .2rem 0; }
.step p { margin: 0; opacity: .8; font-size: .92rem; }
@media (max-width: 640px) { .hero h1 { font-size: 1.8rem; } .hero { padding: 1.2rem; } }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resources & state
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Warming up the search engine (first load only)…")
def get_engine():
    return build_engine()


def get_shared_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("shared_key_queries", 0)

df, embed_model, collection = get_engine()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## 🧭 {APP_NAME}")
    st.caption("Your AI scout for stays across 5 US cities.")

    shared_key = get_shared_key()
    user_key = st.text_input(
        "Groq API key (optional)" if shared_key else "Groq API key",
        type="password",
        help="Get a free key at console.groq.com. Without one, StayScout runs in rule-based mode.",
    )
    if user_key:
        st.success("Using your key · full AI mode", icon="🔑")
    elif shared_key:
        left = max(0, SHARED_KEY_QUERY_LIMIT - st.session_state.shared_key_queries)
        st.info(f"Demo key active · {left} AI searches left this session", icon="✨")
    else:
        st.warning("No key · running rule-based mode (still grounded, no LLM)", icon="⚙️")

    top_k = st.slider("Results per search", min_value=3, max_value=10, value=5)
    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Built by **Shashank** · [GitHub]({GITHUB_URL})")


def resolve_client():
    """Pick the Groq client for this query, or None for rule-based mode."""
    if user_key:
        return Groq(api_key=user_key), "ai"
    if shared_key and st.session_state.shared_key_queries < SHARED_KEY_QUERY_LIMIT:
        st.session_state.shared_key_queries += 1
        return Groq(api_key=shared_key), "ai"
    return None, "rules"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_hero(title, subtitle, show_stats=True):
    stats = ""
    if show_stats:
        stats = (
            '<div class="stat-row">'
            f'<span class="stat">🏠 {len(df):,} listings</span>'
            f'<span class="stat">📍 {df["city"].nunique()} cities</span>'
            f'<span class="stat">💵 median ${df["price"].median():.0f}/night</span>'
            '<span class="stat">🛡️ zero invented listings</span>'
            '</div>'
        )
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{stats}</div>', unsafe_allow_html=True)


def render_filter_chips(filters, mode):
    chips = []
    if filters.get("city"):
        chips.append(f'{CITY_ICONS.get(filters["city"], "📍")} {filters["city"]}')
    if filters.get("max_price") is not None:
        chips.append(f'💵 ≤ ${filters["max_price"]:.0f}/night')
    if filters.get("room_type"):
        chips.append(f'🛏️ {filters["room_type"]}')
    parts = [f'<span class="chip">{html.escape(c)}</span>' for c in chips] or ['<span class="chip muted">No hard filters · pure semantic search</span>']
    if filters.get("search_text"):
        parts.append(f'<span class="chip muted">🔎 “{html.escape(filters["search_text"])}”</span>')
    parts.append(f'<span class="chip muted">{"🤖 AI mode" if mode == "ai" else "⚙️ Rule mode"}</span>')
    st.markdown(f'<div class="chips">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_cards(listings):
    cards = []
    for idx, l in enumerate(listings, start=1):
        maps_url = f"https://www.google.com/maps?q={l['latitude']},{l['longitude']}"
        cards.append(f"""
<div class="card">
  <div class="num">{idx}</div>
  <div class="title">{html.escape(l['name'])}</div>
  <div class="where">{CITY_ICONS.get(l['city'], '📍')} {html.escape(l['neighbourhood'])}, {html.escape(l['city'])}</div>
  <div class="price">${l['price']:.0f}<span> / night</span></div>
  <div class="meta">
    <span class="tag room">{html.escape(l['room_type'])}</span>
    <span class="tag">⭐ {l['number_of_reviews']} reviews</span>
    <span class="tag">🌙 min {l['minimum_nights']}</span>
  </div>
  <div class="bar"><div style="width:{l['match'] * 100:.0f}%"></div></div>
  <div class="foot"><span>{l['match'] * 100:.0f}% match</span><a href="{maps_url}" target="_blank">📍 map ↗</a></div>
</div>""")
    st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_results_map(listings, key):
    points = pd.DataFrame([
        {"n": str(i), "name": l["name"], "price": f"${l['price']:.0f}", "lat": l["latitude"], "lon": l["longitude"]}
        for i, l in enumerate(listings, start=1)
    ])
    lat_span = points["lat"].max() - points["lat"].min()
    lon_span = points["lon"].max() - points["lon"].min()
    zoom = 3 if max(lat_span, lon_span) > 3 else 10 if max(lat_span, lon_span) > 0.15 else 12
    layers = [
        pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]", get_radius=80, radius_min_pixels=11, radius_max_pixels=16,
                  get_fill_color=ACCENT_RGB + [230], get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=2, pickable=True),
        pdk.Layer("TextLayer", points, get_position="[lon, lat]", get_text="n", get_size=14,
                  get_color=[255, 255, 255], get_alignment_baseline="'center'", font_weight=700),
    ]
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=points["lat"].mean(), longitude=points["lon"].mean(), zoom=zoom),
            tooltip={"html": "<b>[{n}] {name}</b><br/>{price} / night"},
            map_style="road",
        ),
        height=320,
        key=key,
    )


def render_refine_controls(message, idx):
    filters = message["filters"]
    with st.popover("✏️ Adjust filters"):
        st.caption("Not quite what you meant? Override what the agent understood.")
        city_options = ["Any city"] + CITIES
        room_options = ["Any room type"] + ROOM_TYPES
        city = st.selectbox("City", city_options,
                            index=city_options.index(filters["city"]) if filters.get("city") else 0, key=f"city_{idx}")
        room = st.selectbox("Room type", room_options,
                            index=room_options.index(filters["room_type"]) if filters.get("room_type") else 0, key=f"room_{idx}")
        no_cap = st.checkbox("No price cap", value=filters.get("max_price") is None, key=f"nocap_{idx}")
        price = st.slider("Max price / night", 20, 1000, int(filters.get("max_price") or 200), step=10,
                          disabled=no_cap, key=f"price_{idx}")
        vibe = st.text_input("Vibe / location", value=filters.get("search_text", ""), key=f"vibe_{idx}")
        if st.button("Search again", type="primary", key=f"rerun_{idx}", use_container_width=True):
            st.session_state.pending = {
                "query": message["query"],
                "filters": {
                    "city": None if city == "Any city" else city,
                    "room_type": None if room == "Any room type" else room,
                    "max_price": None if no_cap else price,
                    "search_text": vibe,
                },
            }
            st.rerun()


def render_assistant_message(message, idx, is_latest):
    render_filter_chips(message["filters"], message["mode"])
    st.markdown(message["answer"])
    if message["listings"]:
        render_cards(message["listings"])
        render_results_map(message["listings"], key=f"map_{idx}")
    with st.container(horizontal=True):
        if is_latest and message["listings"]:
            render_refine_controls(message, idx)
        with st.popover("🧠 Agent trace"):
            for t in message["trace"]:
                st.markdown(f"✅ {t['step']} · `{t['ms']} ms`")
            st.json(message["filters"])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def find_a_stay():
    render_hero(
        "Find your next stay, just by describing it.",
        "StayScout reads your request, turns it into real filters, searches 15,000 listings and only recommends places that exist.",
    )

    messages = st.session_state.messages
    if not messages:
        st.markdown("##### ✨ Try one of these")
        cols = st.columns(2)
        for i, (icon, query) in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(f"{icon}  {query}", key=f"example_{i}", use_container_width=True):
                st.session_state.pending = {"query": query}
                st.rerun()

    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, message in enumerate(messages):
        with st.chat_message(message["role"], avatar="🧭" if message["role"] == "assistant" else "🙂"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message, i, i == last_assistant)

    typed = st.chat_input("Describe your ideal stay… e.g. “cozy loft in Chicago under $150”")
    pending = st.session_state.pending
    st.session_state.pending = None
    request = {"query": typed} if typed else pending
    if not request:
        return

    query = request["query"]
    override = request.get("filters")
    previous_filters = messages[last_assistant]["filters"] if last_assistant >= 0 else None
    label = query if override is None else f"{query}  ·  _(adjusted filters)_"
    messages.append({"role": "user", "content": label})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(label)

    with st.chat_message("assistant", avatar="🧭"):
        client, mode = resolve_client()
        with st.status("Scouting…", expanded=True) as status:
            try:
                result = run_agent(df, embed_model, collection, query, client=client, top_k=top_k,
                                   previous_filters=previous_filters, override_filters=override,
                                   on_step=lambda s: status.write(f"⏳ {s}…"))
            except Exception as err:
                if client is not None:
                    status.write("⚠️ AI service unavailable, falling back to rule mode…")
                    result = run_agent(df, embed_model, collection, query, client=None, top_k=top_k,
                                       previous_filters=previous_filters, override_filters=override)
                    mode = "rules"
                    if "rate" in str(err).lower():
                        st.toast("Groq rate limit hit, switched to rule mode for this search.", icon="⏱️")
                else:
                    status.update(label="Something went wrong", state="error")
                    st.error(f"Error processing request: {err}")
                    return
            status.update(label=f"Found {len(result['listings'])} stays", state="complete", expanded=False)

    messages.append({"role": "assistant", "query": query, "mode": mode, **result})
    st.rerun()


def city_insights():
    render_hero("City insights", "Explore what the data looks like before you search.", show_stats=False)

    city = st.segmented_control("City", CITIES, default=CITIES[0], format_func=lambda c: f"{CITY_ICONS[c]} {c}",
                                label_visibility="collapsed") or CITIES[0]
    city_df = df[df["city"] == city]

    m = st.columns(4)
    m[0].metric("Listings", f"{len(city_df):,}")
    m[1].metric("Median price", f"${city_df['price'].median():.0f}",
                delta=f"{city_df['price'].median() - df['price'].median():+.0f} vs all cities", delta_color="inverse")
    m[2].metric("Entire homes", f"{(city_df['room_type'] == 'Entire home/apt').mean() * 100:.0f}%")
    m[3].metric("Avg reviews", f"{city_df['number_of_reviews'].mean():.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### 💵 Price per night")
        bins = [0, 50, 100, 150, 200, 300, 500, 1000]
        labels = ["<$50", "$50–100", "$100–150", "$150–200", "$200–300", "$300–500", "$500+"]
        buckets = pd.cut(city_df["price"], bins=bins, labels=labels).value_counts().reindex(labels)
        st.bar_chart(buckets.rename("Listings"), color="#6C5CE7", height=260, sort=False)
    with right:
        st.markdown("##### 🛏️ Room types")
        rooms = city_df.groupby("room_type").agg(listings=("id", "count"), median_price=("price", "median")).sort_values("listings", ascending=False)
        st.bar_chart(rooms["listings"].rename("Listings"), color="#00B894", height=260, horizontal=True, sort=False)

    st.markdown("##### 🗺️ Where the listings are")
    st.caption("Each dot is a listing. Brighter and larger means pricier.")
    map_df = city_df[["latitude", "longitude", "price", "name", "neighbourhood"]].copy()
    map_df["size"] = 30 + map_df["price"].clip(upper=600) / 4
    map_df["alpha"] = (60 + map_df["price"].clip(upper=600) / 600 * 190).astype(int)
    map_df["price_label"] = map_df["price"].map(lambda p: f"${p:.0f}")
    st.pydeck_chart(
        pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", map_df, get_position="[longitude, latitude]", get_radius="size",
                              get_fill_color="[108, 92, 231, alpha]", pickable=True)],
            initial_view_state=pdk.ViewState(latitude=map_df["latitude"].median(), longitude=map_df["longitude"].median(), zoom=10),
            tooltip={"html": "<b>{name}</b><br/>{neighbourhood} · {price_label}/night"},
            map_style="road",
        ),
        height=420,
    )

    st.markdown("##### 🏘️ Top neighbourhoods")
    hoods = (city_df.groupby("neighbourhood")
             .agg(listings=("id", "count"), median_price=("price", "median"), avg_reviews=("number_of_reviews", "mean"))
             .sort_values("listings", ascending=False).head(12).reset_index())
    st.dataframe(
        hoods,
        hide_index=True,
        use_container_width=True,
        column_config={
            "neighbourhood": "Neighbourhood",
            "listings": st.column_config.ProgressColumn("Listings", format="%d", min_value=0, max_value=int(hoods["listings"].max())),
            "median_price": st.column_config.NumberColumn("Median price", format="$%d"),
            "avg_reviews": st.column_config.NumberColumn("Avg reviews", format="%.0f"),
        },
    )


def how_it_works():
    render_hero("How StayScout works", "An agentic RAG pipeline that treats your words as constraints, not just vibes.", show_stats=False)
    steps = [
        ("Understand", f"An LLM ({LLM_MODEL} on Groq) turns your message into structured filters: max price, city and room type, plus a short "
                       "semantic summary. Follow-ups like “cheaper?” update the previous filters. Without an API key, a rule-based parser does this."),
        ("Filter", "Those filters become a ChromaDB metadata query, so “under $90” is enforced exactly instead of being guessed by vector similarity."),
        ("Retrieve", "Inside the filtered set, all-MiniLM-L6-v2 embeddings rank the 15,000 listings by how well they match the vibe you described."),
        ("Ground", "The LLM writes a recommendation using only the retrieved listings, citing each one as [1], [2]… so every claim can be checked "
                   "against the cards and the map."),
    ]
    for i, (title, body) in enumerate(steps, start=1):
        st.markdown(f'<div class="step"><div class="n">{i}</div><div><h4>{title}</h4><p>{html.escape(body)}</p></div></div>',
                    unsafe_allow_html=True)

    st.markdown("##### 🧰 Stack")
    st.markdown(
        f"- **LLM:** `{LLM_MODEL}` via Groq\n"
        "- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2` (precomputed with `build_index.py`)\n"
        "- **Vector store:** ChromaDB with metadata filtering\n"
        "- **UI:** Streamlit + pydeck\n"
        "- **Data:** 15,000 cleaned listings (3,000 each in NYC, LA, SF, Chicago, New Orleans)\n"
        f"- **Source:** [{GITHUB_URL.removeprefix('https://')}]({GITHUB_URL})"
    )


page = st.navigation(
    [
        st.Page(find_a_stay, title="Find a stay", icon="🔎", default=True),
        st.Page(city_insights, title="City insights", icon="📊"),
        st.Page(how_it_works, title="How it works", icon="🧠"),
    ],
    position="top",
)
page.run()
