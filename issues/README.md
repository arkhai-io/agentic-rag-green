# Issues: Haystack <> CoopHive DVD

## Issue 1: Knowledge Graph Component Interaction

### Knowledge Graph Component Interaction Issues

### Problem Statement
We're building a knowledge graph to track components from indexing and retrieval pipelines in Haystack. The main challenge is how retrieval pipelines should interact with document stores that were created during indexing.

### Core Issues

#### 1. Embedding Function Discovery
- **Problem**: How does a retrieval pipeline know which embedding model was used during indexing?
- **Challenge**: Retrieval queries must use the same embedding model as indexing for semantic consistency
- **Current Gap**: No mechanism to store/retrieve embedding model information from document stores

#### 2. Document Store Configuration
- **Problem**: How do we reconstruct the exact document store setup (BM25 vs vector search, connection details, etc.)?
- **Challenge**: Different search strategies (vector, BM25, hybrid) require different component configurations
- **Current Gap**: Document store nodes don't capture retrieval-specific configuration

#### 3. Pipeline Reconstruction
- **Problem**: How do we automatically build retrieval components that match the indexing setup?
- **Challenge**: Users specify "retrieve from store X" but system needs to know HOW to retrieve
- **Current Gap**: No mapping from document stores to required retrieval component chains

### Current Architecture Challenge

```
Indexing Pipeline:
PDF → Chunker → Embedder(model=X) → Writer → DocumentStore

Retrieval Pipeline:
Query → ??? → ??? → DocumentStore
         ↑     ↑
    Which embedder model?
    Which retriever type?
```

### Proposed Solution: DocumentStoreNode as Configuration Store

Store all necessary retrieval information in the `DocumentStoreNode`:

```python
DocumentStoreNode(
    store_name="main_chroma_store",
    store_type="chroma",
    retrieval_components=["query_embedder", "embedding_retriever"],
    # Store configuration needed to reconstruct retrieval
    properties={
        "embedding_model": "all-MiniLM-L6-v2",
        "search_type": "vector",  # or "bm25", "hybrid"
        "connection_string": "...",
        "top_k": 10
    }
)
```

### Architecture Diagram

```mermaid
graph TB
    subgraph "Indexing Pipeline"
        PDF[PDF Converter] --> Chunk[Text Chunker]
        Chunk --> Embed[Text Embedder<br/>model: all-MiniLM-L6-v2]
        Embed --> Writer[Document Writer]
        Writer --> DS[Document Store<br/>ChromaDB]
    end

    subgraph "Knowledge Graph Storage"
        DSNode[DocumentStoreNode<br/>- store_type: chroma<br/>- embedding_model: all-MiniLM-L6-v2<br/>- retrieval_components: [query_embedder, retriever]<br/>- search_type: vector]
    end

    subgraph "Retrieval Pipeline (Auto-Generated)"
        Query[User Query] --> QEmbed[Query Embedder<br/>model: ???]
        QEmbed --> Retriever[Embedding Retriever<br/>document_store: ???]
        Retriever --> Response[Response]
    end

    DS -.-> DSNode
    DSNode -.-> QEmbed
    DSNode -.-> Retriever

    style DSNode fill:#f9f,stroke:#333,stroke-width:2px
    style QEmbed fill:#bbf,stroke:#333,stroke-width:2px
    style Retriever fill:#bbf,stroke:#333,stroke-width:2px
```

### Potential Issues with Proposed Approach

#### 1. Tight Coupling
- **Issue**: Document store node becomes overly complex configuration blob
- **Risk**: Hard to maintain, violates single responsibility principle

#### 2. Multiple Retrieval Patterns
- **Issue**: Same store might need different retrieval strategies for different use cases
- **Risk**: One-size-fits-all configuration doesn't work for diverse retrieval needs

#### 3. Version Conflicts
- **Issue**: What if different pipelines need different embedding models for same store?
- **Risk**: Breaking changes when updating embedding models

#### 4. Dynamic Configuration
- **Issue**: Some retrieval parameters should be query-time decisions (top_k, filters)
- **Risk**: Over-specification at store creation time

---

## Issue 2: Component-Level Caching with Knowledge Graph Gates

### Problem Statement
We want to implement intelligent caching at the component level during retrieval pipeline execution. Before and after each component, we need "gates" that check the knowledge graph to see if the data already exists, fetch it if available, and only process new/missing data through the actual component.

### Core Concept: In/Out Gates

```
Input → [IN GATE] → Component → [OUT GATE] → Output
         ↕                        ↕
    Knowledge Graph         Knowledge Graph
    (Check if data          (Store results
     already exists)         for future use)
```

### Detailed Flow

#### IN GATE (Before Component)
1. **Data Fingerprinting**: Hash/fingerprint incoming data
2. **Knowledge Graph Lookup**: Query if this exact input has been processed by this component before
3. **Cache Hit**: If found, fetch cached result and skip component processing
4. **Cache Miss**: Allow data to flow through to component

#### OUT GATE (After Component)
1. **Result Capture**: Capture component output
2. **Knowledge Graph Storage**: Store input→output mapping with metadata
3. **Forward Results**: Pass results to next component

### Example: PDF Processing Pipeline

```
PDF → [IN GATE] → PDF Converter → [OUT GATE] → [IN GATE] → Chunker → [OUT GATE] → ...
       ↕                           ↕            ↕                      ↕
   Check if PDF              Store converted   Check if chunks    Store chunk
   already converted         markdown         already exist      results
```

### Implementation Challenges

#### 1. Haystack Pipeline.run() Integration
- **Issue**: Current `pipeline.run()` may not support dynamic component skipping
- **Challenge**: How to modify execution flow mid-pipeline
- **Potential Solutions**:
  - Custom pipeline runner that respects gates
  - Wrapper components that encapsulate gate logic
  - Haystack middleware/interceptor pattern

#### 2. Skip Links in Graph
- **Challenge**: How to represent skipped processing in the knowledge graph
- **Considerations**:
  - Should we store "SKIPPED" relationships?
  - How to maintain audit trail of what was/wasn't processed?
  - Performance implications of complex graph queries
