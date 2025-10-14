# Issues

This directory contains detailed documentation of known issues and their proposed solutions.

## Active Issues

### 1. [Document Store Duplication](./document-store-duplication/README.md)
Multiple DocumentStore nodes are created with temporary directories during pipeline creation, causing storage fragmentation and making it difficult to share data between pipelines.

**Status:** Documented
**Priority:** High
**Affects:** Pipeline creation, data persistence, cross-pipeline queries

### 2. [Retrieval Pipeline Branching](./retrieval-pipeline-branching/README.md)
Multi-store retrieval pipelines that query multiple indexing pipelines result in flat component structures with naming collisions. Need to create sub-pipelines for each branch.

**Status:** Documented
**Priority:** High
**Affects:** Retrieval pipelines, multi-store queries, component naming

## How to Use This Directory

Each issue has its own subdirectory with a detailed README containing:
- Problem description
- Current vs expected behavior
- Proposed solution design
- Implementation steps
- Files to modify
- Testing approach

When working on an issue, update its README with progress and implementation notes.
