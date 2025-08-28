# RAG-Based Knowledge Agent for ITOps
# Demonstrates retrieval-augmented generation using vector databases

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import math

# Mock dependencies - in real implementation you'd use:
# import openai
# import pinecone
# from sentence_transformers import SentenceTransformer
# import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Query:
    id: str
    text: str
    timestamp: datetime
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class RetrievalResult:
    document: Document
    similarity_score: float
    relevance_score: float

class MockEmbeddingModel:
    """Mock embedding model - in production use SentenceTransformer or OpenAI embeddings"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Simple word-to-vector mapping for demonstration
        self.vocab = {}
        self._build_vocab()
        
    def _build_vocab(self):
        """Build a simple vocabulary with mock embeddings"""
        itops_terms = [
            "server", "cpu", "memory", "disk", "network", "latency", "throughput",
            "monitoring", "alert", "incident", "outage", "performance", "capacity",
            "load", "bandwidth", "timeout", "error", "exception", "log", "metric",
            "kubernetes", "docker", "cloud", "aws", "azure", "gcp", "database",
            "cache", "queue", "api", "microservice", "deployment", "scaling",
            "backup", "restore", "security", "firewall", "ssl", "certificate"
        ]
        
        for i, term in enumerate(itops_terms):
            # Create a simple embedding based on term characteristics
            embedding = np.random.random(self.dimension).tolist()
            # Add some semantic clustering
            if term in ["server", "cpu", "memory", "disk"]:
                embedding[0] += 0.5  # Hardware cluster
            elif term in ["monitoring", "alert", "incident", "metric"]:
                embedding[1] += 0.5  # Monitoring cluster
            elif term in ["kubernetes", "docker", "cloud", "aws"]:
                embedding[2] += 0.5  # Infrastructure cluster
                
            self.vocab[term] = embedding
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        embeddings = []
        
        for text in texts:
            # Simple bag-of-words embedding
            words = re.findall(r'\b\w+\b', text.lower())
            text_embedding = np.zeros(self.dimension)
            word_count = 0
            
            for word in words:
                if word in self.vocab:
                    text_embedding += np.array(self.vocab[word])
                    word_count += 1
                else:
                    # Random embedding for unknown words
                    text_embedding += np.random.random(self.dimension) * 0.1
                    word_count += 1
            
            if word_count > 0:
                text_embedding = text_embedding / word_count
            
            embeddings.append(text_embedding.tolist())
            
        return embeddings

class VectorStore:
    """Mock vector store - in production use Pinecone, Weaviate, or FAISS"""
    
    def __init__(self, embedding_model: MockEmbeddingModel):
        self.embedding_model = embedding_model
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}
        
    async def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = embedding
            
        logger.info(f"Added {len(documents)} documents to vector store")
    
    async def similarity_search(self, query_text: str, k: int = 5, 
                              threshold: float = 0.1) -> List[RetrievalResult]:
        """Perform similarity search"""
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:k]
        
        results = []
        for doc_id, similarity in similarities:
            doc = self.documents[doc_id]
            relevance_score = self._calculate_relevance(query_text, doc, similarity)
            
            result = RetrievalResult(
                document=doc,
                similarity_score=similarity,
                relevance_score=relevance_score
            )
            results.append(result)
            
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def _calculate_relevance(self, query: str, doc: Document, 
                           similarity: float) -> float:
        """Calculate relevance score based on multiple factors"""
        # Base similarity score
        relevance = similarity
        
        # Boost based on metadata
        if doc.metadata.get("priority") == "high":
            relevance += 0.1
        
        if doc.metadata.get("type") == "troubleshooting":
            relevance += 0.05
            
        # Recent documents get slight boost
        age_days = (datetime.now() - doc.created_at).days
        if age_days < 30:
            relevance += 0.02
            
        return min(relevance, 1.0)

class RAGKnowledgeAgent:
    """AI Agent that uses RAG for ITOps knowledge retrieval and Q&A"""
    
    def __init__(self):
        self.name = "RAGKnowledgeAgent"
        self.embedding_model = MockEmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.query_history = []
        self.context_window = []
        
        # Initialize with ITOps knowledge base
        asyncio.create_task(self._initialize_knowledge_base())
    
    async def _initialize_knowledge_base(self):
        """Load ITOps knowledge documents"""
        knowledge_docs = [
            Document(
                id="cpu_troubleshooting_01",
                content="High CPU usage troubleshooting: First check top processes using 'top' or 'htop'. Look for processes consuming >50% CPU. Common causes include infinite loops, inefficient queries, memory leaks causing GC pressure, or legitimate high load. Check system load average and compare to number of CPU cores.",
                metadata={"type": "troubleshooting", "category": "cpu", "priority": "high"}
            ),
            Document(
                id="memory_issues_01", 
                content="Memory usage investigation: Use 'free -h' to check overall memory. Use 'ps aux --sort=-%mem' to find memory-heavy processes. Check for memory leaks by monitoring process memory over time. Java applications: analyze heap dumps. Node.js: use --inspect with Chrome DevTools. Python: use memory_profiler module.",
                metadata={"type": "troubleshooting", "category": "memory", "priority": "high"}
            ),
            Document(
                id="disk_space_01",
                content="Disk space management: Use 'df -h' for filesystem usage and 'du -sh /*' to find large directories. Common space consumers: logs (/var/log), temporary files (/tmp), package caches, core dumps. Set up log rotation with logrotate. Consider disk cleanup scripts for automated maintenance.",
                metadata={"type": "maintenance", "category": "storage", "priority": "medium"}
            ),
            Document(
                id="network_latency_01",
                content="Network latency diagnosis: Use ping for basic connectivity, traceroute for path analysis, and iperf3 for bandwidth testing. Check network interface statistics with 'netstat -i'. Monitor packet loss and retransmissions. Consider network congestion, routing issues, or hardware problems.",
                metadata={"type": "troubleshooting", "category": "network", "priority": "high"}
            ),
            Document(
                id="kubernetes_pods_01",
                content="Kubernetes pod troubleshooting: Use 'kubectl get pods' to check status. Common issues: ImagePullBackOff (wrong image/registry), CrashLoopBackOff (application crashes), Pending (resource constraints). Check events with 'kubectl describe pod <name>'. Review resource requests and limits.",
                metadata={"type": "troubleshooting", "category": "kubernetes", "priority": "high"}
            ),
            Document(
                id="database_performance_01",
                content="Database performance optimization: Monitor slow queries, check connection pool usage, analyze index effectiveness. For MySQL: use EXPLAIN for query analysis, check innodb_buffer_pool_size. For PostgreSQL: analyze with pg_stat_statements, monitor connection counts. Consider query optimization and indexing strategies.",
                metadata={"type": "optimization", "category": "database", "priority": "medium"}
            ),
            Document(
                id="load_balancer_01",
                content="Load balancer configuration: Ensure health checks are properly configured with appropriate timeouts and intervals. Monitor backend server response times and error rates. Configure sticky sessions if needed for stateful applications. Set up proper SSL termination and security headers.",
                metadata={"type": "configuration", "category": "networking", "priority": "medium"}
            ),
            Document(
                id="monitoring_alerts_01",
                content="Alert management best practices: Set meaningful thresholds to avoid alert fatigue. Use alert aggregation and correlation to reduce noise. Implement escalation policies with multiple notification channels. Include runbook links in alert descriptions. Regularly review and tune alert rules based on historical data.",
                metadata={"type": "best_practices", "category": "monitoring", "priority": "high"}
            ),
            Document(
                id="incident_response_01",
                content="Incident response workflow: Acknowledge alert immediately. Assess impact and severity. Create incident ticket with timeline. Implement immediate mitigation if available. Gather relevant logs and metrics. Communicate status to stakeholders. Document root cause analysis and prevention measures.",
                metadata={"type": "process", "category": "incident_management", "priority": "high"}
            ),
            Document(
                id="backup_recovery_01",
                content="Backup and recovery procedures: Implement automated daily backups with retention policy. Test recovery procedures regularly. Store backups in multiple locations (on-site and off-site). Monitor backup job success and validate backup integrity. Document recovery time objectives (RTO) and recovery point objectives (RPO).",
                metadata={"type": "process", "category": "disaster_recovery", "priority": "high"}
            )
        ]
        
        await self.vector_store.add_documents(knowledge_docs)
        logger.info("Knowledge base initialized with ITOps documentation")
    
    async def query(self, query_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a knowledge query using RAG"""
        
        query = Query(
            id=f"query_{len(self.query_history)}",
            text=query_text,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.query_history.append(query)
        
        # Step 1: Retrieve relevant documents
        retrieval_results = await self.vector_store.similarity_search(
            query_text, k=3, threshold=0.2
        )
        
        # Step 2: Generate response using retrieved context
        response = await self._generate_response(query, retrieval_results)
        
        # Step 3: Update context window
        self._update_context(query, response)
        
        self.log_action("query_processed", {
            "query_id": query.id,
            "query_text": query_text,
            "documents_retrieved": len(retrieval_results),
            "response_generated": True
        })
        
        return {
            "query_id": query.id,
            "response": response,
            "retrieved_documents": [
                {
                    "document_id": r.document.id,
                    "similarity_score": r.similarity_score,
                    "relevance_score": r.relevance_score,
                    "metadata": r.document.metadata
                }
                for r in retrieval_results
            ],
            "confidence": self._calculate_confidence(retrieval_results)
        }
    
    async def _generate_response(self, query: Query, 
                               retrieval_results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate response using retrieved documents (mock LLM generation)"""
        
        if not retrieval_results:
            return {
                "answer": "I don't have specific information about that topic in my knowledge base. Could you provide more details or try rephrasing your question?",
                "sources": [],
                "confidence": 0.1
            }
        
        # Extract relevant content
        context_docs = [r.document for r in retrieval_results]
        sources = [doc.id for doc in context_docs]
        
        # Mock response generation based on query type
        answer = self._mock_llm_response(query.text, context_docs)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._calculate_confidence(retrieval_results),
            "retrieval_count": len(retrieval_results)
        }
    
    def _mock_llm_response(self, query_text: str, context_docs: List[Document]) -> str:
        """Mock LLM response generation"""
        query_lower = query_text.lower()
        
        # Analyze query intent
        if any(word in query_lower for word in ["cpu", "high cpu", "cpu usage"]):
            return self._generate_cpu_response(context_docs)
        elif any(word in query_lower for word in ["memory", "ram", "oom"]):
            return self._generate_memory_response(context_docs)
        elif any(word in query_lower for word in ["disk", "storage", "space"]):
            return self._generate_disk_response(context_docs)
        elif any(word in query_lower for word in ["network", "latency", "connectivity"]):
            return self._generate_network_response(context_docs)
        elif any(word in query_lower for word in ["kubernetes", "k8s", "pod"]):
            return self._generate_k8s_response(context_docs)
        elif any(word in query_lower for word in ["database", "db", "sql"]):
            return self._generate_database_response(context_docs)
        elif any(word in query_lower for word in ["incident", "alert", "monitoring"]):
            return self._generate_incident_response(context_docs)
        else:
            return self._generate_general_response(context_docs)
    
    def _generate_cpu_response(self, docs: List[Document]) -> str:
        return """Based on the knowledge base, here's how to troubleshoot high CPU usage:

1. **Immediate Investigation**: Use `top` or `htop` to identify processes consuming >50% CPU
2. **Check System Load**: Compare load average to the number of CPU cores
3. **Common Causes**: Look for infinite loops, inefficient queries, memory leaks causing GC pressure, or legitimate high load
4. **Next Steps**: If you identify problematic processes, consider restarting services, optimizing queries, or scaling resources

Would you like me to help you analyze specific CPU metrics or processes?"""
    
    def _generate_memory_response(self, docs: List[Document]) -> str:
        return """For memory usage investigation, follow these steps:

1. **Check Overall Memory**: Use `free -h` to see total memory usage
2. **Identify Heavy Processes**: Run `ps aux --sort=-%mem` to find memory-intensive processes
3. **Monitor for Leaks**: Track process memory usage over time
4. **Application-Specific Analysis**:
   - Java: Analyze heap dumps
   - Node.js: Use --inspect with Chrome DevTools
   - Python: Use memory_profiler module

Consider implementing memory limits and monitoring to prevent future issues."""
    
    def _generate_disk_response(self, docs: List[Document]) -> str:
        return """For disk space management:

1. **Check Usage**: Use `df -h` for filesystem usage and `du -sh /*` for directory sizes
2. **Common Space Consumers**: 
   - Logs in /var/log
   - Temporary files in /tmp
   - Package caches
   - Core dumps
3. **Cleanup Actions**: Set up log rotation with logrotate, clean temporary files
4. **Prevention**: Implement automated cleanup scripts and disk usage monitoring

Need help with specific directories or cleanup procedures?"""
    
    def _generate_network_response(self, docs: List[Document]) -> str:
        return """For network latency diagnosis:

1. **Basic Tests**: Use ping for connectivity, traceroute for path analysis
2. **Bandwidth Testing**: Use iperf3 to test throughput
3. **Interface Statistics**: Check with `netstat -i` for errors/drops
4. **Monitor Metrics**: Watch for packet loss and retransmissions
5. **Common Issues**: Network congestion, routing problems, hardware failures

What specific network symptoms are you experiencing?"""
    
    def _generate_k8s_response(self, docs: List[Document]) -> str:
        return """For Kubernetes pod troubleshooting:

1. **Check Pod Status**: `kubectl get pods` to see current state
2. **Common Issues**:
   - ImagePullBackOff: Wrong image or registry issues
   - CrashLoopBackOff: Application crashes on startup
   - Pending: Resource constraints or scheduling issues
3. **Detailed Investigation**: `kubectl describe pod <name>` for events
4. **Resource Analysis**: Review CPU/memory requests and limits

Which pods are having issues and what's their current status?"""
    
    def _generate_database_response(self, docs: List[Document]) -> str:
        return """For database performance optimization:

1. **Monitor Slow Queries**: Identify queries taking excessive time
2. **Connection Management**: Check connection pool usage and limits
3. **Index Analysis**: Review index effectiveness and usage
4. **Database-Specific Tools**:
   - MySQL: Use EXPLAIN for query analysis, tune innodb_buffer_pool_size
   - PostgreSQL: Use pg_stat_statements, monitor connections

Consider implementing query optimization and indexing strategies based on your workload patterns."""
    
    def _generate_incident_response(self, docs: List[Document]) -> str:
        return """For effective incident management:

1. **Immediate Response**: Acknowledge alert and assess impact/severity
2. **Documentation**: Create incident ticket with timeline
3. **Mitigation**: Implement immediate fixes if available
4. **Investigation**: Gather relevant logs and metrics
5. **Communication**: Keep stakeholders informed of progress
6. **Follow-up**: Document root cause analysis and prevention measures

Also consider tuning your alert thresholds to reduce noise and implementing proper escalation policies."""
    
    def _generate_general_response(self, docs: List[Document]) -> str:
        if docs:
            doc = docs[0]  # Use the most relevant document
            return f"""Based on the most relevant information in my knowledge base:

{doc.content}

This information comes from our {doc.metadata.get('category', 'general')} documentation. Would you like me to elaborate on any specific aspect or help you with related topics?"""
        else:
            return "I need more specific information to provide a detailed answer. Could you clarify what specific ITOps topic you'd like help with?"
    
    def _calculate_confidence(self, retrieval_results: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not retrieval_results:
            return 0.1
        
        # Base confidence on best similarity score and number of results
        best_score = max(r.similarity_score for r in retrieval_results)
        avg_score = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
        
        confidence = (best_score * 0.6 + avg_score * 0.4)
        
        # Boost confidence if multiple relevant documents found
        if len(retrieval_results) >= 2:
            confidence += 0.1
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _update_context(self, query: Query, response: Dict[str, Any]):
        """Update conversation context window"""
        context_entry = {
            "query": query.text,
            "timestamp": query.timestamp,
            "response_confidence": response.get("confidence", 0),
            "sources_used": response.get("sources", [])
        }
        
        self.context_window.append(context_entry)
        
        # Keep only recent context (last 5 interactions)
        if len(self.context_window) > 5:
            self.context_window.pop(0)
    
    async def get_related_documents(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get documents related to a specific topic"""
        results = await self.vector_store.similarity_search(topic, k=limit, threshold=0.15)
        
        return [
            {
                "id": r.document.id,
                "content": r.document.content,
                "metadata": r.document.metadata,
                "similarity": r.similarity_score
            }
            for r in results
        ]
    
    async def add_knowledge(self, documents: List[Dict[str, Any]]):
        """Add new knowledge documents to the system"""
        new_docs = []
        
        for doc_data in documents:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            new_docs.append(doc)
        
        await self.vector_store.add_documents(new_docs)
        self.log_action("knowledge_added", {"documents_count": len(new_docs)})
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        recent_queries = self.query_history[-limit:]
        
        return [
            {
                "id": q.id,
                "text": q.text,
                "timestamp": q.timestamp.isoformat(),
                "context": q.context
            }
            for q in recent_queries
        ]
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "action": action,
            "details": details or {}
        }
        logger.info(f"[{self.name}] {action}: {details}")

# Demo execution
async def main():
    """Demonstrate the RAG Knowledge Agent"""
    
    print("🧠 Starting RAG-Based Knowledge Agent for ITOps")
    print("=" * 60)
    
    agent = RAGKnowledgeAgent()
    
    # Wait for knowledge base initialization
    await asyncio.sleep(1)
    
    # Sample queries to demonstrate different capabilities
    sample_queries = [
        "How do I troubleshoot high CPU usage on my server?",
        "My application is running out of memory, what should I check?",
        "Kubernetes pod is stuck in CrashLoopBackOff, help!",
        "Database queries are running slowly, how to optimize?",
        "What's the best way to handle incidents and alerts?",
        "How to diagnose network latency issues?"
    ]
    
    print("🔍 Processing sample queries...\n")
    
    for i, query_text in enumerate(sample_queries, 1):
        print(f"Query {i}: {query_text}")
        print("-" * 50)
        
        # Process query
        result = await agent.query(query_text)
        
        # Display results
        print(f"Answer: {result['response']['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sources: {', '.join([d['document_id'] for d in result['retrieved_documents'][:2]]) if result['retrieved_documents'] else 'None'}")
        print(f"Documents Retrieved: {len(result['retrieved_documents'])}")
        print()
    
    print("\n" + "=" * 60)
    print("📊 AGENT STATISTICS")
    print("=" * 60)
    
    history = agent.get_query_history()
    print(f"Total Queries Processed: {len(history)}")
    print(f"Knowledge Documents: {len(agent.vector_store.documents)}")
    print(f"Context Window Size: {len(agent.context_window)}")
    
    print("\n" + "=" * 60)
    print("📚 KNOWLEDGE BASE TOPICS")
    print("=" * 60)
    
    # Show available topics
    topics = set()
    for doc in agent.vector_store.documents.values():
        topics.add(doc.metadata.get("category", "general"))
    
    print(f"Available Topics: {', '.join(sorted(topics))}")
    
    # Demonstrate related document search
    print("\n" + "=" * 60)  
    print("🔗 RELATED DOCUMENTS DEMO")
    print("=" * 60)
    
    related_docs = await agent.get_related_documents("troubleshooting", limit=3)
    print("Documents related to 'troubleshooting':")
    
    for doc in related_docs:
        print(f"- {doc['id']} (similarity: {doc['similarity']:.3f})")
        print(f"  Category: {doc['metadata'].get('category', 'N/A')}")
        print(f"  Content preview: {doc['content'][:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(main())