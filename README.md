🚀 Vector Database & Retrieval Performance Comparison (ChromaDB vs Pinecone)
📌 Overview
This project provides a practical benchmarking framework for comparing:

Traditional lexical retrieval (BM25)

Vector-based semantic search using HNSW indexing

across:

✅ ChromaDB (local open-source vector database)
✅ Pinecone (managed cloud vector database)

A collection of PDF documents is processed into text chunks, embedded using fast SentenceTransformer models, and evaluated based on:

⏱ Query latency (average response time)

📈 Retrieval quality (heuristic scoring)

The goal is to understand speed vs quality trade-offs between local and cloud vector stores.

📂 Dataset
Source: PDF documents (user-provided folder)

Preprocessing Pipeline:
Text extracted using pypdf

Chunked with overlap for semantic continuity

Embedded using transformer models

Query type: Natural language queries
Top-K results: 5
Similarity metric: Cosine similarity

⚠️ PDFs are excluded from the repository for size reasons.

🧠 Embedding Models Used
Alias	Model Name	Characteristics
MiniLM-L3	paraphrase-MiniLM-L3-v2	Lightweight & fast
DIST	distilbert-base-nli-stsb-mean-tokens	Balanced quality
(Optimized for performance benchmarking)

🗄️ Vector Databases Compared
Database	Type	Deployment
ChromaDB	Open-source	Local persistent storage
Pinecone	Managed	Serverless cloud
🔍 Retrieval Methods
Algorithm	Category
BM25	Lexical keyword search
HNSW	Approximate Nearest Neighbor (vector search)                                                                                 

⚙️ Evaluation Configuration
Queries: 20 natural language questions
Top-K retrieval: 5
Similarity metric: Cosine similarity
Metrics evaluated:
Average response time (ms)
Qualitative retrieval accuracy


 Performance Results
🔹 Benchmark Summary
Configuration	Embedding Model	Avg Latency (ms)	Accuracy Level
BM25	N/A	210.93	Medium
Brute Force	MiniLM	21.43	High
HNSW (ChromaDB)	MiniLM	5.47	High
HNSW (Pinecone)	MiniLM	539.73	Very High


🧠 Key Observations
BM25 is slower and less accurate for semantic queries.
Brute-force vector search improves accuracy but scales poorly.
ChromaDB (HNSW) provides the lowest latency for local workloads.
Pinecone delivers very high retrieval quality but incurs higher latency due to network and serverless overhead.
Both ChromaDB and Pinecone require batched ingestion for large datasets due to internal limits.

🏗️ Project Structure
.
├── ChromaDb.py                  # ChromaDB benchmarking pipeline
├── PineCode.py                  # Pinecone benchmarking pipeline
├── compare_CDB_PI.py            # Aggregates final results
├── chroma_db/                  # Chroma persistent storage
├── chroma_performance_results.pkl
├── pinecone_performance_results.pkl
├── finalresult.pkl
└── README.md


🚀 Conclusion

This project demonstrates that:

👉 Vector-based retrieval systems significantly outperform traditional BM25 for semantic queries
👉 Local vector databases like ChromaDB excel in low-latency scenarios
👉 Cloud platforms like Pinecone offer robust indexing but may introduce network overhead

Overall, HNSW-based vector search is the optimal approach for modern RAG and AI retrieval pipelines.

👨‍💻 Author

Jaya Bharath
