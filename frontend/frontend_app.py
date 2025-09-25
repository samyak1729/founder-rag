import streamlit as st
import requests

# Configuration
BACKEND_URL = "https://founder-rag.onrender.com/search"

# --- Streamlit UI ---

st.set_page_config(page_title="Founder RAG", layout="centered")

st.title("🔎 Founder RAG Search")
st.write("Search for startup founders and professionals from the dataset.")

# Search bar
query = st.text_input(
    "Search query", 
    placeholder="e.g., 'fintech founder in Bangalore'",
    help="Enter your natural language query here."
)

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching..."):
            try:
                # Call the FastAPI backend
                response = requests.post(BACKEND_URL, json={"query": query, "top_k": 5})
                response.raise_for_status()  # Raise an exception for bad status codes

                results = response.json()

                st.divider()

                if results.get("results"):
                    st.success(f"Found {len(results['results'])} results.")
                    for founder in results['results']:
                        st.subheader(founder.get('founder_name', 'N/A'))
                        st.caption(f"{founder.get('role', 'N/A')} at {founder.get('company', 'N/A')} - {founder.get('location', 'N/A')}")
                        
                        with st.expander("Details"):
                            st.markdown(f"**Idea:** {founder.get('idea', 'N/A')}")
                            st.markdown(f"**Bio:** {founder.get('about', 'N/A')}")
                            st.markdown(f"**Keywords:** {founder.get('keywords', 'N/A')}")
                            if founder.get('linked_in'):
                                st.markdown(f"[LinkedIn Profile]({founder.get('linked_in')})")
                        st.divider()
                else:
                    st.info("No relevant results found.")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend. Please ensure it is running. Error: {e}")
