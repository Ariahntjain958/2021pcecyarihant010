# 📄 Smart Q&A from URLs

A simple web app to scrape content from any web page, store it using vector embeddings, and allow users to ask questions strictly based on the scraped content.

## 🔧 Features
- Input one or more webpage URLs
- Scrape and embed main page content
- Ask questions based only on the input pages
- Retrieve precise answers using vector similarity

## 🧠 Tech Stack
- [Streamlit](https://streamlit.io)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author
**Arihant Jain** – Final Year B.Tech Student  
Submitted for Human.AI Assignment