import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(
    page_title="URL Q&A Tool",
    page_icon="🧠",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>📄 Smart Q&A from URLs</h1>", unsafe_allow_html=True)
st.caption("Built by Arihant Jain | Human.AI Assignment")
st.markdown("---")

model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
texts = []

def extract_main_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs])
    except Exception as e:
        return f"Error reading {url}: {e}"

def add_to_vector_store(text):
    global texts
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    for chunk in chunks:
        embedding = model.encode([chunk])[0]
        index.add(np.array([embedding]))
        texts.append(chunk)

with st.container():
    st.subheader("🔗 Enter URLs")
    urls = st.text_area("Paste one or more URLs (one per line):")

with st.container():
    st.subheader("❓ Ask a Question")
    question = st.text_input("What do you want to know from the above content?")

if st.button("Submit"):
    if urls:
        url_list = urls.strip().split('\n')
        with st.spinner("📡 Scraping & Ingesting URLs..."):
            for url in url_list:
                content = extract_main_text(url)
                if content.startswith("Error"):
                    st.error(content)
                else:
                    add_to_vector_store(content)
            st.success("✅ Content ingested successfully!")

    if question:
        with st.spinner("🧠 Thinking..."):
            q_embedding = model.encode([question])[0]
            D, I = index.search(np.array([q_embedding]), k=3)
            answers = [texts[i] for i in I[0]]
            st.markdown("### 📌 Answer:")
            st.write(" ".join(answers))

with st.sidebar:
    st.title("ℹ️ About This App")
    st.write("""
    This tool scrapes content from the given web pages and allows you to ask questions 
    based only on the scraped data. Built using:
    - Streamlit
    - BeautifulSoup
    - FAISS
    - Sentence-Transformers
    """)
    st.markdown("---")
    st.write("✅ No external AI model answers — only your URLs matter.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<small>Made with ❤️ by Arihant Jain — B.Tech Final Year Project</small>"
    "</div>",
    unsafe_allow_html=True
)
