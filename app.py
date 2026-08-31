import streamlit as st
import pandas as pd
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set page configuration
st.set_page_config(
    page_title="Agentic Travel RAG Assistant",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_initialize_vector_db():
    df = pd.read_csv("airbnb_clean.csv")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize ephemeral ChromaDB client
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("hotel_airbnb_collection")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="hotel_airbnb_collection")
    documents = df["document"].tolist()
    ids = [str(i) for i in df.index]
    metadatas = df[["name", "city", "neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews"]].to_dict("records")
    
    embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
    
    # Add in batches
    BATCH_SIZE = 5000
    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
    return df, embed_model, collection

def parse_user_intent_filters(client, user_query):
    system_instructions = """You are an AI assistant parsing search criteria from travel accommodation queries.
Extract the relevant search constraints and return ONLY a raw JSON object with the following schema:
{
  "max_price": number or null,
  "city": "New York City" | "Los Angeles" | "San Francisco" | "Chicago" | "New Orleans" | null,
  "room_type": "Entire home/apt" | "Private room" | "Shared room" | null,
  "search_text": string (short semantic summary of location preference or atmosphere)
}
Do not wrap in Markdown codeblocks or include conversational text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    
    raw_content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_content)
    except Exception:
        return {"max_price": None, "city": None, "room_type": None, "search_text": user_query}

def execute_hybrid_vector_retrieval(embed_model, collection, query_text, top_k=5, max_price=None, city=None, room_type=None):
    filter_conditions = []
    if max_price is not None:
        filter_conditions.append({"price": {"$lte": float(max_price)}})
    if city is not None:
        filter_conditions.append({"city": {"$eq": str(city)}})
    if room_type is not None:
        filter_conditions.append({"room_type": {"$eq": str(room_type)}})
        
    where_clause = None
    if len(filter_conditions) == 1:
        where_clause = filter_conditions[0]
    elif len(filter_conditions) > 1:
        where_clause = {"$and": filter_conditions}
        
    query_vector = embed_model.encode([query_text]).tolist()
    return collection.query(query_embeddings=query_vector, n_results=top_k, where=where_clause)

def run_agentic_rag(client, embed_model, collection, user_query, top_k=5):
    extracted_filters = parse_user_intent_filters(client, user_query)
    
    search_query = extracted_filters.get("search_text") or user_query
    retrieval_results = execute_hybrid_vector_retrieval(
        embed_model, 
        collection,
        query_text=search_query,
        top_k=top_k,
        max_price=extracted_filters.get("max_price"),
        city=extracted_filters.get("city"),
        room_type=extracted_filters.get("room_type")
    )
    
    retrieved_docs = retrieval_results["documents"][0] if retrieval_results.get("documents") else []
    
    if not retrieved_docs:
        return {
            "answer": "No listings matched your exact search criteria. Try increasing budget or broadening city/room type filters.",
            "filters": extracted_filters,
            "sources": []
        }
        
    context_str = "\n".join([f"[{idx + 1}] {doc}" for idx, doc in enumerate(retrieved_docs)])
    
    generation_prompt = f"""You are an expert AI Travel Advisor. Provide personalized recommendations based strictly on the retrieved property listings.

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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": generation_prompt}],
        temperature=0.2
    )
    
    return {
        "answer": response.choices[0].message.content,
        "filters": extracted_filters,
        "sources": retrieved_docs
    }

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hotel-building.png", width=64)
    st.title("Settings & Config")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Required to run Llama 3.3 model (get free key at console.groq.com)")
    
    st.divider()
    top_k_slider = st.slider("Top Results Count (K)", min_value=3, max_value=10, value=5)
    
    st.markdown("### Supported Cities")
    st.markdown("- 🗽 New York City\n- 🌴 Los Angeles\n- 🌉 San Francisco\n- 🏙️ Chicago\n- 🎷 New Orleans")
    st.divider()
    st.caption("Developed by **Shashank Kaushik**")

# Main Interface Layout
st.markdown('<div class="main-header">🏨 Agentic Travel & Accommodation RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Semantic & Structured Search across 15,000+ US Listings powered by Llama 3.3 & ChromaDB</div>', unsafe_allow_html=True)

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    
    with st.spinner("Initializing ChromaDB vector engine and embedding 15k listings..."):
        df_data, embedding_engine, vector_collection = load_and_initialize_vector_db()
        
    st.success(f"System Ready • {vector_collection.count():,} listings indexed in vector store")
    
    user_input = st.text_input(
        "Describe your ideal stay:",
        placeholder="e.g. Quiet private room in Chicago under $100 near downtown"
    )
    
    if user_input:
        try:
            with st.spinner("Analyzing query intent and executing hybrid retrieval..."):
                result = run_agentic_rag(
                    client=groq_client,
                    embed_model=embedding_engine,
                    collection=vector_collection,
                    user_query=user_input,
                    top_k=top_k_slider
                )
                
            st.subheader("💡 Recommended Stay")
            st.markdown(result["answer"])
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("🔍 Extracted Structured Metadata Filters"):
                    st.json(result["filters"])
            with col2:
                with st.expander("📄 Source Listing Context"):
                    for idx, src in enumerate(result["sources"], start=1):
                        st.markdown(f"**[{idx}]** {src}")
                        
        except Exception as err:
            if "rate_limit" in str(err).lower():
                st.warning("Groq API rate limit reached. Please pause briefly before retrying.")
            else:
                st.error(f"Error processing request: {err}")
else:
    st.info("👋 Please enter your Groq API Key in the sidebar menu to activate the AI agent.")
