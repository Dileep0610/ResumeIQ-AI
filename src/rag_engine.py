from __future__ import annotations

import os
import time
from typing import List, Tuple
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "career_coach_chroma_db"

@st.cache_resource(show_spinner=False)
def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.2):
    print(f"[{time.strftime('%X')}] Initializing Groq LLM...")
    start_time = time.time()
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Create a .env file and add your Groq API key.")
    llm = ChatGroq(model=model, temperature=temperature)
    print(f"[{time.strftime('%X')}] LLM initialized in {time.time() - start_time:.2f} seconds.")
    return llm


@st.cache_resource(show_spinner=False)
def get_embeddings():
    import time
    start_time = time.perf_counter()
    from langchain_huggingface import HuggingFaceEmbeddings
    # BATCHING OPTIMIZATION: process chunks in batches to speed up CPU inference
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'batch_size': 32}
    )
    load_time = time.perf_counter() - start_time
    return embeddings, load_time


# -----------------------------
# Stage 1: Load Documents
# -----------------------------
def load_text_file(file_path: str, source_name: str, doc_type: str):
    from langchain_core.documents import Document
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "doc_type": doc_type})]


def create_documents(resume_text: str, jd_text: str):
    from langchain_core.documents import Document
    return [
        Document(page_content=resume_text, metadata={"source": "uploaded_resume", "doc_type": "resume"}),
        Document(page_content=jd_text, metadata={"source": "uploaded_job_description", "doc_type": "job_description"}),
    ]


# -----------------------------
# Stage 2: Split Documents
# -----------------------------
def split_documents(docs, chunk_size: int = 800, chunk_overlap: int = 150):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# Stage 3 + 4: Embeddings + Vector DB
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_cached_vectorstore(resume_text: str, jd_text: str, chunk_size: int, chunk_overlap: int):
    import time
    from langchain_chroma import Chroma
    
    t0_total = time.perf_counter()
    
    # 3. Document Creation
    t0 = time.perf_counter()
    docs = create_documents(resume_text, jd_text)
    t_doc_creation = time.perf_counter() - t0
    
    # 4. Chunking
    t0 = time.perf_counter()
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    t_chunking = time.perf_counter() - t0
    
    # 5. HuggingFace Embedding Model Loading
    # (Because get_embeddings is cached, it only takes time on the first run)
    embeddings, t_embed_model_loading = get_embeddings()
    
    # 6. Embedding Generation
    t0 = time.perf_counter()
    texts = [c.page_content for c in chunks]
    embeddings_list = embeddings.embed_documents(texts)
    t_embed_gen = time.perf_counter() - t0
    
    # 7. Chroma Vector Database Creation
    t0 = time.perf_counter()
    vectorstore = Chroma(collection_name="career_coach_rag", embedding_function=embeddings)
    vectorstore.add_texts(texts=texts, metadatas=[c.metadata for c in chunks], ids=[str(i) for i in range(len(chunks))])
    t_vector_db = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t0_total
    
    timings = {
        "Document Creation": t_doc_creation,
        "Chunking": t_chunking,
        "Embedding Model Loading": t_embed_model_loading,
        "Embedding Generation": t_embed_gen,
        "Vector DB Creation": t_vector_db,
        "Total Build Time": t_total
    }
    
    return vectorstore, len(chunks), timings


# -----------------------------
# Stage 5: Retrieve Context
# -----------------------------
def retrieve_context(vectorstore, query: str, k: int = 5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([f"SOURCE: {d.metadata}\nCONTENT:\n{d.page_content}" for d in docs])
    return context, docs


# -----------------------------
# Stage 6: Generate Answer
# -----------------------------
def run_career_coach(vectorstore, resume_text: str, jd_text: str, question: str):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print(f"[{time.strftime('%X')}] Starting LLM generation...")
    start_time = time.time()
    
    llm = get_llm()

    retrieval_query = f"""
    Resume content and job description content relevant to this career coaching question:
    {question}
    """

    context, source_docs = retrieve_context(vectorstore, retrieval_query, k=6)

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach for students, freshers and working professionals.
Use ONLY the given context from the resume and job description.
Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommended Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer simple, actionable and beginner-friendly.
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"[{time.strftime('%X')}] LLM response generated in {time.time() - start_time:.2f} seconds.")
    return answer, source_docs


def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question)
