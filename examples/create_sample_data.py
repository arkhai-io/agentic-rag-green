"""
Create Sample Data

This script generates sample documents for testing the indexing and retrieval pipelines.
It creates text files with content about various technical topics.
"""

from pathlib import Path
from typing import List


def create_sample_documents() -> List[str]:
    """
    Create sample documents in the data directory.

    Creates multiple documents with different topics to demonstrate
    how the indexing pipelines handle diverse content.
    """
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Sample document 1: Machine Learning basics
    ml_content = """# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The core idea is to develop algorithms that can access data and learn from it automatically.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled datasets to train algorithms. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include classification and regression tasks.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find patterns and structures in the data without predefined labels. Clustering and dimensionality reduction are typical applications.

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time.

## Key Concepts

**Training Data**: The dataset used to train the model.
**Features**: Input variables used for making predictions.
**Labels**: Output variables (in supervised learning).
**Model**: The mathematical representation learned from data.
**Overfitting**: When a model performs well on training data but poorly on new data.
**Underfitting**: When a model is too simple to capture patterns in the data.
"""

    # Sample document 2: Neural Networks
    nn_content = """# Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information.

## Architecture Components

### Input Layer
The input layer receives raw data. Each neuron in this layer represents a feature in the dataset.

### Hidden Layers
Hidden layers perform computations and feature extraction. Deep neural networks have multiple hidden layers, enabling them to learn complex patterns.

### Output Layer
The output layer produces the final predictions. The number of neurons depends on the task (e.g., one for binary classification, multiple for multi-class classification).

## Activation Functions

Activation functions introduce non-linearity into the network:
- **ReLU** (Rectified Linear Unit): f(x) = max(0, x)
- **Sigmoid**: f(x) = 1 / (1 + e^(-x))
- **Tanh**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

## Training Process

1. **Forward Propagation**: Input data flows through the network to generate predictions
2. **Loss Calculation**: Measure the difference between predictions and actual values
3. **Backpropagation**: Calculate gradients of the loss with respect to weights
4. **Weight Update**: Adjust weights using optimization algorithms like gradient descent

## Deep Learning Applications

- Computer Vision: Image classification, object detection, segmentation
- Natural Language Processing: Text generation, translation, sentiment analysis
- Speech Recognition: Converting audio to text
- Recommendation Systems: Personalized content suggestions
"""

    # Sample document 3: Transformers and Attention
    transformer_content = """# Transformers and Attention Mechanisms

Transformers revolutionized natural language processing when introduced in 2017. Unlike RNNs, transformers process entire sequences in parallel using attention mechanisms.

## Self-Attention Mechanism

Self-attention allows the model to weigh the importance of different parts of the input when processing each element. For each word, the mechanism computes:

1. **Query (Q)**: What the word is looking for
2. **Key (K)**: What the word offers
3. **Value (V)**: The actual content of the word

Attention scores are calculated as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

## Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel:
- Each head learns different relationships between words
- Outputs are concatenated and linearly transformed
- Enables the model to focus on different aspects simultaneously

## Transformer Architecture

### Encoder
The encoder processes the input sequence:
- Multi-head self-attention layer
- Position-wise feed-forward network
- Residual connections and layer normalization

### Decoder
The decoder generates the output sequence:
- Masked multi-head self-attention (prevents looking ahead)
- Encoder-decoder attention (attends to encoder outputs)
- Position-wise feed-forward network

## Positional Encoding

Since transformers don't have inherent notion of sequence order, positional encodings are added to input embeddings:
- Use sine and cosine functions of different frequencies
- Enable the model to learn relative and absolute positions

## Popular Transformer Models

**BERT** (Bidirectional Encoder Representations from Transformers): Pre-trained on masked language modeling and next sentence prediction. Excellent for understanding tasks.

**GPT** (Generative Pre-trained Transformer): Autoregressive model trained to predict the next token. Excels at text generation.

**T5** (Text-to-Text Transfer Transformer): Frames all NLP tasks as text-to-text problems, enabling unified architecture.
"""

    # Sample document 4: RAG Systems
    rag_content = """# Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with text generation to create more accurate and grounded language model outputs. Instead of relying solely on parametric knowledge, RAG systems access external knowledge bases.

## RAG Architecture

### Retrieval Component
The retrieval component finds relevant documents:
1. **Query Encoding**: Convert user query to embedding vector
2. **Document Search**: Find similar documents using vector similarity
3. **Ranking**: Order documents by relevance score

### Generation Component
The generation component produces answers:
1. **Context Integration**: Combine retrieved documents with query
2. **Prompt Construction**: Format context for the language model
3. **Answer Generation**: Generate response using LLM
4. **Post-processing**: Filter and format the final output

## Vector Databases

Vector databases store and retrieve embeddings efficiently:
- **ChromaDB**: Lightweight embedding database
- **Pinecone**: Managed vector database service
- **Weaviate**: Open-source vector search engine
- **Qdrant**: High-performance vector database

## Chunking Strategies

Breaking documents into chunks is crucial for effective retrieval:

**Fixed-size Chunking**: Split by character count or token count. Simple but may break semantic units.

**Semantic Chunking**: Split at natural boundaries (paragraphs, sections). Preserves context better.

**Recursive Chunking**: Hierarchical splitting with overlapping windows. Balances context and granularity.

## Embedding Models

Different embedding models serve different purposes:
- **Small models** (e.g., all-MiniLM-L6-v2): Fast, good for large-scale applications
- **Large models** (e.g., all-mpnet-base-v2): Better semantic understanding
- **Domain-specific models**: Trained on specialized corpora

## Multi-Index RAG

Using multiple indexes with different strategies:
- One index with small chunks for precise matching
- Another with large chunks for contextual understanding
- Aggregate results for comprehensive answers

## Evaluation Metrics

**Retrieval Metrics**: Precision, recall, MRR, NDCG
**Generation Metrics**: BLEU, ROUGE, BERTScore
**End-to-End Metrics**: Answer accuracy, faithfulness, relevance
"""

    # Write documents to files
    documents = {
        "machine_learning.txt": ml_content,
        "neural_networks.txt": nn_content,
        "transformers.txt": transformer_content,
        "rag_systems.txt": rag_content,
    }

    created_files = []
    for filename, content in documents.items():
        filepath = data_dir / filename
        filepath.write_text(content.strip())
        created_files.append(str(filepath))

    return created_files


if __name__ == "__main__":
    files = create_sample_documents()

    # Display success message
    print("Sample documents created successfully!\n")
    print("Created files:")
    for filepath in files:
        file_size = len(Path(filepath).read_text())
        print(f"  - {filepath} ({file_size:,} characters)")

    print(f"\nTotal files: {len(files)}")
    print("\nYou can now run the indexing pipeline:")
    print("  poetry run python examples/indexing_pipeline_example.py")
