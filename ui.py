"""
Streamlit Web UI for the Hybrid Search Engine.

Run with: streamlit run ui.py
"""
import os
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Hybrid Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.search-result {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin-bottom: 1rem;
    background: #fafafa;
}
.score-badge {
    background: #4CAF50;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}
.highlight {
    background: yellow;
    padding: 0 2px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    semantic_weight = st.slider(
        "Semantic Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Weight for semantic (meaning-based) search"
    )
    
    lexical_weight = 1.0 - semantic_weight
    st.write(f"Lexical Weight: {lexical_weight:.1f}")
    
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5
    )
    
    use_learned = st.checkbox(
        "Use Learned Weights",
        help="Use weights optimized from user feedback"
    )
    
    st.divider()
    
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"✅ API Connected")
        st.write(f"Documents: {health.get('indexed_documents', 0)}")
        st.write(f"Watcher: {'Active' if health.get('watcher_active') else 'Inactive'}")
    except:
        st.error("❌ API Unavailable")

# Main content
st.title("🔍 Hybrid Search Engine")
st.write("Combine semantic understanding with keyword matching for better search results.")

# Search input
query = st.text_input(
    "Search Query",
    placeholder="Enter your search query...",
    key="search_query"
)

# Search button
if st.button("Search", type="primary") or query:
    if query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "semantic_weight": semantic_weight if not use_learned else None,
                        "lexical_weight": lexical_weight if not use_learned else None,
                        "use_learned_weights": use_learned
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    weights = data.get("weights_used", {})
                    
                    st.write(f"**{len(results)} results** (semantic: {weights.get('semantic', 0):.1f}, lexical: {weights.get('lexical', 0):.1f})")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            col1, col2 = st.columns([0.9, 0.1])
                            
                            with col1:
                                st.markdown(f"**{i}. Document #{result['doc_id']}**")
                                st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                            
                            with col2:
                                st.metric("Score", f"{result['score']:.3f}")
                                
                                # Feedback buttons
                                col_up, col_down = st.columns(2)
                                with col_up:
                                    if st.button("👍", key=f"up_{i}"):
                                        requests.post(f"{API_URL}/feedback", json={
                                            "query_id": data.get("query_id", 1),
                                            "doc_id": result['doc_id'],
                                            "relevance_score": 5,
                                            "clicked": True
                                        })
                                        st.toast("Feedback recorded!")
                                with col_down:
                                    if st.button("👎", key=f"down_{i}"):
                                        requests.post(f"{API_URL}/feedback", json={
                                            "query_id": data.get("query_id", 1),
                                            "doc_id": result['doc_id'],
                                            "relevance_score": 1,
                                            "clicked": False
                                        })
                                        st.toast("Feedback recorded!")
                            
                            st.divider()
                    
                    if not results:
                        st.info("No results found. Try a different query.")
                
                elif response.status_code == 400:
                    st.warning("No documents indexed. Please index some documents first.")
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Tabs for additional features
tab1, tab2, tab3 = st.tabs(["📁 Index Documents", "📊 Statistics", "⚡ Quick Actions"])

with tab1:
    st.subheader("Index Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Index from Directory**")
        directory = st.text_input("Directory Path", value="data")
        if st.button("Index Directory"):
            with st.spinner("Indexing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/index/directory",
                        json={"directory": directory}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Indexed {data.get('documents_indexed', 0)} documents!")
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(str(e))
    
    with col2:
        st.write("**Add Single Document**")
        doc_content = st.text_area("Document Content", height=150)
        if st.button("Add Document"):
            if doc_content:
                try:
                    response = requests.post(
                        f"{API_URL}/index/add",
                        json={"documents": [{"content": doc_content}]}
                    )
                    if response.status_code == 200:
                        st.success("Document added!")
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(str(e))

with tab2:
    st.subheader("Statistics")
    
    if st.button("Refresh Stats"):
        try:
            response = requests.get(f"{API_URL}/stats")
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Documents", stats.get("total_documents", 0))
                col2.metric("Queries", stats.get("total_queries", 0))
                col3.metric("Unique Queries", stats.get("unique_queries", 0))
                col4.metric("Feedback", stats.get("total_feedback", 0))
                
                if stats.get("avg_relevance"):
                    st.write(f"Average Relevance Score: {stats['avg_relevance']:.2f}")
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**File Watcher**")
        if st.button("Start Watcher"):
            try:
                response = requests.post(f"{API_URL}/watcher/start")
                st.success("Watcher started!")
            except Exception as e:
                st.error(str(e))
        
        if st.button("Stop Watcher"):
            try:
                response = requests.post(f"{API_URL}/watcher/stop")
                st.success("Watcher stopped!")
            except Exception as e:
                st.error(str(e))
    
    with col2:
        st.write("**Documents**")
        if st.button("List Documents"):
            try:
                response = requests.get(f"{API_URL}/documents?limit=10")
                if response.status_code == 200:
                    data = response.json()
                    for doc in data.get("documents", []):
                        st.write(f"**#{doc['doc_id']}**: {doc['content'][:100]}...")
            except Exception as e:
                st.error(str(e))

# Footer
st.divider()
st.caption("Hybrid Search Engine - Semantic + Lexical Search")
