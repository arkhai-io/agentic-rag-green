# Multi-Store Retrieval Pipeline with Auto-Injection

## Problem Statement

Currently, retrieval pipelines cannot properly query multiple indexing pipelines with different embedding models. Each indexing pipeline may use a different embedder (e.g., `all-mpnet-base-v2` for PDFs, `all-MiniLM-L6-v2` for text), creating incompatible embedding spaces.

### Issues with Current Approach

1. **No component substitution**: Writer components are not being substituted with retrievers because `indexing_pipelines` is never set in `PipelineSpec`
2. **Single embedder limitation**: Cannot use one query embedder to query stores with different embedding models
3. **No branching support**: Cannot fan-out to multiple stores and fan-in to merge results
4. **Manual configuration**: Users must manually specify embedders, losing the benefit of stored indexing metadata

### Why Branching is Necessary

Different indexing pipelines may use different embedding models:

```python
# PDF indexing uses model A (768 dimensions)
pdf_indexing = [converter, chunker, embedder(model="all-mpnet-base-v2"), writer]

# Text indexing uses model B (384 dimensions)
text_indexing = [converter, chunker, embedder(model="all-MiniLM-L6-v2"), writer]
```

**You cannot use a single query embedder to query both stores** - the embedding spaces are incompatible!

## Proposed Solution

### 1. Auto-Injection Architecture

The system should automatically:
- Extract embedder configuration from `DocumentStore` metadata
- Create query embedders per branch (matching indexing models)
- Inject retrievers
- Handle branching and merging

### 2. User-Facing Schema

```python
retrieval_spec = {
    "name": "multi_store_qa",
    "type": "retrieval",
    "components": [
        # Shared preprocessing (runs once on query text)
        {"type": "QUERY_EXPANDER"},
        {"type": "QUERY_CLASSIFIER"},

        # 🔑 Injection point - system auto-injects embedders + retrievers here
        {"type": "RETRIEVE_ALL"},

        # Per-branch (runs separately for each store)
        {"type": "RANKER.DIVERSITY"},
        {"type": "GENERATOR.OPENAI"},

        # Merge point
        {"type": "JOINER.ANSWER_JOINER"},

        # Shared postprocessing
        {"type": "EVALUATOR.FAITHFULNESS"}
    ],
    "indexing_pipelines": ["pdf_indexing_pipeline", "text_indexing_pipeline"]
}
```

### 3. Execution Flow

```
Query (text string)
  ↓
query_expander (shared)
  ↓
query_classifier (shared)
  ↓
  ├─ Branch 1 (PDF store)
  │   ├─> text_embedder(model=mpnet)  [auto-injected from DocumentStore]
  │   ├─> retriever_pdf               [auto-injected]
  │   ├─> diversity_ranker_1          [per-branch instance]
  │   ├─> generator_1                 [per-branch instance]
  │   └─> answer_pdf
  │
  └─ Branch 2 (Text store)
      ├─> text_embedder(model=minilm) [auto-injected from DocumentStore]
      ├─> retriever_text              [auto-injected]
      ├─> diversity_ranker_2          [per-branch instance]
      ├─> generator_2                 [per-branch instance]
      └─> answer_text
            ↓
      answer_joiner (merge answers)
            ↓
      evaluator (shared)
            ↓
      final_answer
```

## Implementation Details

### Phase 1: Fix Current Substitution Logic

**File: `storage.py`**

1. **Fix component_name** (line 189):
   ```python
   # WRONG (doesn't match registry):
   component_name = f"{spec.name}_{substitution.target_component}"

   # CORRECT (matches registry):
   component_name = substitution.target_component  # e.g., "chroma_embedding_retriever"
   ```

2. **Copy root_dir config** (lines 191-201):
   ```python
   # Extract root_dir from writer config for retriever
   writer_config = json.loads(comp.get("component_config_json", "{}"))
   retriever_config = {}
   if "root_dir" in writer_config:
       retriever_config["root_dir"] = writer_config["root_dir"]
   config_json = json.dumps(retriever_config)
   ```

3. **Set indexing_pipelines in PipelineFactory** (currently missing):
   ```python
   # In factory.py - look up DocumentStore IDs
   if pipeline_type == PipelineType.RETRIEVAL:
       indexing_stores = {}
       for pipeline_name in indexing_pipeline_names:
           store_id = graph_store.get_document_store_by_pipeline(pipeline_name)
           indexing_stores[pipeline_name] = store_id

       pipeline_spec = PipelineSpec(
           name=pipeline_name,
           components=components,
           indexing_pipelines=indexing_stores  # 🔑 Now set!
       )
   ```

### Phase 2: Add Auto-Injection Logic

**File: `storage.py`**

1. **Detect injection point**:
   ```python
   def find_injection_point(components: List[ComponentSpec]) -> int:
       """Find where to inject embedders + retrievers."""

       # 1. Look for explicit RETRIEVE_ALL marker
       for i, comp in enumerate(components):
           if comp.name == "RETRIEVE_ALL":
               return i

       # 2. Auto-detect: inject before first component needing documents
       NEEDS_DOCUMENTS = {"ranker", "generator", "reader", "summarizer"}

       for i, comp in enumerate(components):
           if any(need in comp.name.lower() for need in NEEDS_DOCUMENTS):
               return i

       # 3. Default: inject at start
       return 0
   ```

2. **Split pipeline into sections**:
   ```python
   injection_point = find_injection_point(spec.components)

   shared_pre = spec.components[:injection_point]  # Before branching
   remaining = spec.components[injection_point:]

   # Find joiner
   joiner_idx = None
   for i, comp in enumerate(remaining):
       if comp.component_type == ComponentType.JOINER:
           joiner_idx = i
           break

   if joiner_idx is not None:
       per_branch = remaining[:joiner_idx]
       shared_post = remaining[joiner_idx:]
   else:
       per_branch = remaining
       shared_post = []
   ```

3. **Create branches with auto-injected embedders**:
   ```python
   for store_name, store_id in spec.indexing_pipelines.items():
       # Get components from DocumentStore
       component_ids = graph_store.get_document_store_component_ids(store_id)
       components = graph_store.get_component_nodes_by_ids(component_ids)

       # Find indexing embedder
       indexing_embedder = None
       for comp in components:
           if comp["component_name"] == "document_embedder":
               indexing_embedder = comp
               break

       if not indexing_embedder:
           raise ValueError(f"No embedder found for store {store_id}")

       # Extract model from indexing embedder config
       embedder_config = json.loads(indexing_embedder["component_config_json"])
       model = embedder_config.get("model")

       # Create query embedder with SAME model
       query_embedder_id = f"{store_name}_query_embedder_for_{spec.name}"
       query_embedder_dict = {
           "id": query_embedder_id,
           "component_name": "embedder",  # Text embedder for queries
           "pipeline_name": spec.name,
           "component_config_json": json.dumps({"model": model}),
           "version": "1.0.0",
           "author": "test_user"
       }

       # Create retriever (substitute writer)
       retriever = substitute_writer_to_retriever(components, store_id, spec.name)

       # Connect: last_shared_pre → query_embedder → retriever
       # Then duplicate per_branch components for this branch
   ```

4. **Handle different merge strategies**:
   ```python
   # If joiner is DocumentJoiner: merge documents, then continue
   if joiner_type == "DOCUMENT_JOINER":
       # retrievers → joiner → generator (single instance)

   # If joiner is AnswerJoiner: generate per branch, then merge
   elif joiner_type == "ANSWER_JOINER":
       # retrievers → generator (per branch) → joiner

   # If no joiner: return all branches separately
   else:
       # Return {"pdf": answer1, "text": answer2, ...}
   ```

### Phase 3: Update PipelineFactory

**File: `factory.py`**

Add method to look up DocumentStore IDs:

```python
def _get_document_store_ids(
    self,
    indexing_pipeline_names: List[str],
    username: str
) -> Dict[str, str]:
    """Look up DocumentStore IDs for indexing pipelines."""

    store_ids = {}

    for pipeline_name in indexing_pipeline_names:
        query = """
            MATCH (ds:DocumentStore {pipeline_name: $pipeline_name})
            RETURN ds.id as store_id
            LIMIT 1
        """

        with self.graph_store.driver.session() as session:
            result = session.run(query, pipeline_name=pipeline_name).single()

            if result:
                store_ids[pipeline_name] = result["store_id"]
            else:
                raise ValueError(f"DocumentStore not found for {pipeline_name}")

    return store_ids
```

## Usage Examples

### Example 1: Simple Multi-Store Query

```python
# Minimal - system handles everything
retrieval_spec = {
    "name": "simple_search",
    "components": [
        {"type": "GENERATOR.OPENAI"}
    ],
    "indexing_pipelines": ["pdf_indexing", "text_indexing"]
}

# System auto-injects:
# → embedder_pdf → retriever_pdf ──┐
# → embedder_text → retriever_text ─┤→ doc_joiner → generator
```

### Example 2: With Preprocessing and Answer Merge

```python
retrieval_spec = {
    "name": "advanced_qa",
    "components": [
        {"type": "QUERY_EXPANDER"},
        {"type": "RETRIEVE_ALL"},
        {"type": "GENERATOR.OPENAI"},
        {"type": "JOINER.ANSWER_JOINER"}
    ],
    "indexing_pipelines": ["pdf_indexing", "text_indexing", "docs_indexing"]
}

# Flow:
# query_expander (shared) →
#   ├─ [embedder_pdf → retriever_pdf → generator_pdf] ──┐
#   ├─ [embedder_text → retriever_text → generator_text] ─┤
#   └─ [embedder_docs → retriever_docs → generator_docs] ─┘
#                                                          ↓
#                                                   answer_joiner
```

### Example 3: Custom Per-Branch Processing

```python
retrieval_spec = {
    "name": "custom_retrieval",
    "components": [
        {"type": "QUERY_CLASSIFIER"},
        {"type": "RETRIEVE_ALL"},
        {"type": "RANKER.DIVERSITY"},
        {"type": "FILTER.RELEVANCE"},
        {"type": "GENERATOR.OPENAI"},
        {"type": "JOINER.ANSWER_JOINER"},
        {"type": "EVALUATOR.FAITHFULNESS"}
    ],
    "indexing_pipelines": ["pdf", "text"]
}

# Each branch gets: retriever → ranker → filter → generator
# Then answers merge and evaluate
```

## Testing Plan

1. **Test substitution fix**: Run `test_multi_store_retrieval.py`
   - Verify retrievers are created (not writers)
   - Verify `component_name` matches registry
   - Verify `root_dir` is copied

2. **Test auto-injection**:
   - Create retrieval pipeline with `RETRIEVE_ALL` marker
   - Verify query embedders match indexing embedders
   - Verify branching structure in Neo4j graph

3. **Test different merge strategies**:
   - Document merge (joiner before generator)
   - Answer merge (joiner after generator)
   - No merge (separate outputs)

4. **Test with different preprocessing**:
   - Query expander + classifier before branching
   - Verify they run once, not per-branch

## Open Questions

1. **Component duplication**: Should per-branch components create new nodes or reuse with different edges?
   - Recommendation: Create new instances with suffix `_{store_name}`

2. **Joiner auto-injection**: Should system auto-add joiner if multiple stores and no joiner specified?
   - Recommendation: Yes, add `DocumentJoiner` by default for multiple stores

3. **Error handling**: What if stores have incompatible schemas (different metadata fields)?
   - Recommendation: Document as limitation, filter at joiner level

4. **Parallel execution**: Should branches run in parallel or sequentially?
   - Recommendation: Sequential for now (simpler), parallel in future optimization

## Benefits

✅ **Automatic compatibility**: Query embedders automatically match indexing embedders
✅ **Flexible branching**: Support preprocessing, per-branch processing, and postprocessing
✅ **Multiple merge strategies**: Merge documents, answers, or return separately
✅ **Clean user API**: Users specify components linearly, system handles complexity
✅ **Metadata-driven**: Uses DocumentStore metadata to ensure correctness

## Related Issues

- [Component-Level Caching](../document-store-duplication/README.md)
- [Retrieval Pipeline Branching](../retrieval-pipeline-branching/README.md)
