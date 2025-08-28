#### 📂 `rag_knowledge_agent/README.md`
```markdown
# 🧠 RAG-Based Knowledge Agent

## 📌 Overview
Retrieval-Augmented Generation (RAG) agent that answers IT Ops queries using a vector knowledge base.

## 🎯 Problem
Ops teams need instant troubleshooting knowledge. Static docs are too slow.

## 🛠️ Design
- MockEmbeddingModel + VectorStore
- Knowledge base: CPU, memory, disk, network, Kubernetes, DB, incidents
- Query → retrieval → response generation
- Context window + confidence scoring

## 🚀 How to Run
```bash
python rag_knowledge_agent.py
