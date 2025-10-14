# Retrieval Pipeline Branching - Sub-Pipeline Creation

## Problem

When loading retrieval pipelines that query multiple indexing pipelines (multi-store retrieval), the DFS traversal fetches all connected components from different indexing pipelines, resulting in a flat structure with naming collisions.

### Current Behavior

```
multi_store_retrieval_pipeline:
  - comp_7169d35ed528: SentenceTransformersTextEmbedder (retrieval)
  - comp_0ab48b54c542: SentenceTransformersDocumentEmbedder (text_indexing)
  - comp_4c7ddd4fbdab: DocumentWriter (text_indexing)
  - comp_9319004e0857: SentenceTransformersDocumentEmbedder (pdf_indexing)  ← NAME COLLISION!
  - comp_2bfc95fecbb2: DocumentWriter (pdf_indexing)
```

**Issues:**
- Multiple components with same name (`document_embedder`, `chroma_document_writer`)
- Cannot distinguish which components belong to which branch
- Single flat pipeline doesn't represent the branching structure

### Expected Behavior

Create separate sub-pipelines for each branch:

```
multi_store_retrieval_pipeline:
  Main components: [embedder]

  Sub-pipelines:
    1. multi_store_retrieval_pipeline_text_indexing:
       - embedder (shared from main)
       - text document_embedder
       - text writer
       - text retriever

    2. multi_store_retrieval_pipeline_pdf_indexing:
       - embedder (shared from main)
       - pdf document_embedder
       - pdf writer
       - pdf retriever
```

## Solution Design

### 1. Branch Detection Algorithm

**Identify branching points:**
```python
# A retrieval component branches when its next_components
# have different pipeline_name values

retrieval_components = [
    c for c in components_data
    if c['pipeline_name'] == 'multi_store_retrieval_pipeline'
]

for comp in retrieval_components:
    next_ids = comp['next_components']

    # Group next components by their pipeline_name
    branches = {}
    for next_id in next_ids:
        next_comp = find_component_by_id(next_id)
        branch_name = next_comp['pipeline_name']

        if branch_name != 'multi_store_retrieval_pipeline':
            if branch_name not in branches:
                branches[branch_name] = []
            branches[branch_name].append(next_id)

    # If multiple branches detected
    if len(branches) > 1:
        print(f"Branching at {comp['id']} into: {list(branches.keys())}")
```

### 2. Sub-Pipeline Creation

For each detected branch:

```python
for branch_name, branch_start_ids in branches.items():
    # Create sub-pipeline name
    sub_pipeline_name = f"{main_pipeline_name}_{branch_name}"

    # Components for this sub-pipeline:
    # 1. All main retrieval components up to branching point
    # 2. All components in this specific branch

    sub_pipeline = Pipeline()

    # Add shared components (retrieval components before branch)
    for comp in main_components_before_branch:
        sub_pipeline.add_component(comp_id, component)

    # Add branch-specific components via DFS from branch_start_ids
    branch_components = dfs_from_start_ids(branch_start_ids)
    for comp in branch_components:
        sub_pipeline.add_component(comp_id, component)

    # Connect components
    connect_components_in_sub_pipeline(sub_pipeline, all_components)

    # Store
    sub_pipelines[sub_pipeline_name] = sub_pipeline
```

### 3. Storage Structure

```python
class PipelineRunner:
    def __init__(self):
        # Main pipelines (indexing pipelines)
        self._haystack_pipelines: Dict[str, Pipeline] = {}

        # Sub-pipelines for retrieval branches
        self._retrieval_sub_pipelines: Dict[str, List[Pipeline]] = {}
        # Example: {'multi_store_retrieval_pipeline': [sub_pipeline_1, sub_pipeline_2]}
```

## Implementation Steps

1. **Update `create_haystack_pipeline`** in `runner.py`:
   - Add branch detection logic
   - Identify retrieval vs indexing pipelines
   - For retrieval: detect branches and create sub-pipelines

2. **Create helper methods**:
   - `_detect_branches(components_data, pipeline_name) -> Dict[str, List[str]]`
   - `_create_sub_pipeline(main_components, branch_components, branch_name) -> Pipeline`
   - `_get_components_before_branch(components, branch_point) -> List`

3. **Update storage**:
   - Add `_retrieval_sub_pipelines` dict
   - Store sub-pipelines with naming: `{main}_{branch}`

4. **Update execution**:
   - When running retrieval, execute all sub-pipelines
   - Aggregate results from all branches

## Example Usage

```python
# Load retrieval pipeline
runner = PipelineRunner(
    graph_store=graph_store,
    username="test_user",
    pipeline_names=["multi_store_retrieval_pipeline"]
)

# Check created sub-pipelines
print(runner._retrieval_sub_pipelines)
# {
#   'multi_store_retrieval_pipeline': [
#     'multi_store_retrieval_pipeline_text_indexing',
#     'multi_store_retrieval_pipeline_pdf_indexing'
#   ]
# }

# Run retrieval - executes all sub-pipelines
results = runner.run("multi_store_retrieval_pipeline", type="retrieval", query="test")
# Results aggregated from both branches
```

## Benefits

1. **No naming collisions** - each branch has isolated components
2. **Clear structure** - branches are explicit and separate
3. **Independent execution** - can run branches in parallel
4. **Easier debugging** - can test individual branches
5. **Scalable** - supports any number of branches

## Files to Modify

- `agentic_rag/pipeline/runner.py`:
  - `create_haystack_pipeline()` - add branch detection
  - `_detect_branches()` - new helper method
  - `_create_sub_pipeline()` - new helper method
  - `__init__()` - add `_retrieval_sub_pipelines` storage

## Testing

Create test with multi-store retrieval:
- Load `multi_store_retrieval_pipeline`
- Verify sub-pipelines created for each branch
- Execute retrieval and verify results from all branches
- Check no component naming collisions
