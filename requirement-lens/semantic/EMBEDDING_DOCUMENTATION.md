# Semantic Embedding Documentation for RequirementLens

## Mission 4 - Semantic Embeddings and Vector Retrieval

### What is an Embedding?

An **embedding** is a dense numerical vector representation of textual data in a multi-dimensional continuous vector space. Each dimension captures a latent semantic feature of the text.

**Key characteristics:**
- **Dense**: Information is compressed into a fixed number of dimensions
- **Continuous**: Values are real numbers, not discrete tokens
- **Semantic**: Similar texts map to vectors that are close together in the space
- **Contextual**: Captures meaning beyond simple keyword matching

Example:
- "The system shall authenticate users"
- "Users must log in before accessing data"

These two requirements have different wording but similar meaning - their embeddings will be close in vector space even though they share few exact words.

### Why Sentence Transformers?

Sentence Transformers are a specialized class of models designed specifically for generating sentence/document embeddings. Unlike traditional word embeddings (Word2Vec, GloVe) or MLM models (BERT), sentence transformers are trained with contrastive learning objectives optimized for semantic similarity.

**Advantages for requirement classification:**
1. **Semantic understanding**: Captures meaning beyond keyword overlap
2. **Cross-sentence relationships**: Can identify paraphrases and equivalent requirements
3. **Efficient search**: Enables fast similarity search in large collections
4. **Domain agnostic**: Works well across different software engineering domains

**Architecture:**
```
Text → Tokenizer → Transformer Encoder → Pooling → Normalized Embedding Vector
```

The sentence transformer uses siamese and triplet networks to ensure that semantically similar sentences have similar embeddings, while dissimilar sentences are far apart in vector space.

### Why all-MiniLM-L6-v2?

The **all-MiniLM-L6-v2** model was selected for the following reasons:

**Performance characteristics:**
- **Model size**: 22 MB (lightweight, CPU-friendly)
- **Embedding dimension**: 384 dimensions
- **Speed**: ~300 words/second on CPU
- **Memory footprint**: Minimal, suitable for local deployment

**Quality considerations:**
- Pre-trained on 1B+ sentence pairs using contrastive learning
- Ranked in top 5 of Semantic Textual Similarity (STS) benchmark for its size category
- Excels at paraphrase detection and semantic search
- Balances accuracy with computational efficiency

**Alternatives considered:**
- all-MiniLM-L12-v2: More accurate but 2x slower and larger
- all-MPNet-base-v2: More accurate but 4x larger, slower
- all-distilroberta-v1: Good quality but larger footprint

**Why not BERT/Large models?**
- Too slow for practical semantic search at scale
- Require GPU for reasonable performance
- Overkill for our use case (finding similar guidelines, not deep reasoning)

### Embedding Dimensionality

**all-MiniLM-L6-v2 produces 384-dimensional vectors.**

**What does this mean?**
- Each requirement is represented as a point in 384-dimensional space
- Each dimension captures a different semantic aspect
- Nearby points = semantically similar requirements
- Distant points = semantically different requirements

**Trade-offs of dimensionality:**
- **Higher dimensions** (e.g., 768, 1024):
  - More expressive capacity
  - Better discrimination
  - More memory/CPU requirements
  - Slower search

- **Lower dimensions** (e.g., 384, 128):
  - Faster computation
  - Lower memory usage
  - May lose some fine-grained distinctions
  - Good balance for many applications

**Why 384 is ideal for this project:**
- Large enough to capture nuanced software engineering concepts
- Small enough for fast FAISS indexing and search
- Fits easily in CPU memory even with 10,000+ guidelines
- Compatible with standard FAISS indexing

### Why Normalization is Important

**Normalization** (L2 normalization) means scaling each embedding vector to unit length (norm = 1.0).

**Mathematical definition:**
```
normalized_vector = vector / ||vector||₂
where ||vector||₂ = sqrt(sum(x²))
```

**Why normalize for retrieval?:**

1. **Cosine similarity optimization**:
    - Cosine similarity between vectors v₁ and v₂:
      ```
      cos_sim = (v₁·v₂) / (||v₁|| * ||v₂||)
      ```
    - With normalized vectors: cos_sim = v₁·v₂ (just dot product!)
    - Much faster computation

2. **Magnitude independence**:
    - Length of vector doesn't affect similarity
    - Only direction (semantic meaning) matters
    - Prevents longer texts from artificially appearing more similar

3. **FAISS compatibility**:
    - FAISS FlatIP index works optimally with normalized vectors
    - Inner product search = cosine similarity when vectors are normalized
    - Industry standard for semantic search systems

4. **Semantic retrieval properties**:
    - Similarity scores indicate relative semantic closeness between vectors
    - Scores should not be interpreted as probability or confidence values
    - L2 normalization ensures consistent similarity measurements across all vectors

**Verification:**
```python
embedding = pipeline.embed_text("The system shall authenticate users")
norm = np.linalg.norm(embedding)
# Should be approximately 1.0
print(f"Norm: {norm:.6f}")  # Output: Norm: 1.000000
```

### Embedding Generation Process

1. **Input validation**: Ensure text is non-empty and valid
2. **Tokenization**: Convert text to tokens using MiniLM tokenizer
3. **Encoding**: Pass through 6-layer transformer network
4. **Pooling**: Aggregate token embeddings to sentence embedding
5. **Normalization**: Scale to unit length
6. **Output**: 384-dimensional normalized vector

### Usage Example

```python
from semantic.embedding import EmbeddingPipeline

pipeline = EmbeddingPipeline()
embedding = pipeline.embed_text("The system shall encrypt user passwords")

# embedding is a numpy array of shape (384,)
# with norm ≈ 1.0
```

### Comparison: TF-IDF vs Embeddings

| Aspect | TF-IDF + Logistic Regression | Semantic Embeddings |
|--------|----------------------------|---------------------|
| **Basis** | Keyword frequency | Semantic meaning |
| **Similarity** | Word overlap | Conceptual similarity |
| **Handling paraphrases** | Poor | Excellent |
| **Domain knowledge** | Limited | Rich |
| **Interpretability** | High (feature names) | Low (latent space) |
| **Training required** | Yes | No (pre-trained) |
| **Speed** | Fast classification | Fast search |

**Synergy**: This project uses both:
- Classification for requirement categorization
- Embeddings for finding similar requirements/guidelines

### Limitations

1. **Language specificity**: Optimized for English
2. **Semantic drift**: May not capture highly specialized domain jargon
3. **No reasoning**: Finds similar items but can't reason about them
4. **Training data bias**: Reflects biases in pre-training corpus
5. **Context window**: Limited to ~512 tokens per requirement

### Future Improvements

1. Fine-tune MiniLM on software engineering corpus
2. Use domain-specific sentence transformers
3. Ensemble with TF-IDF for hybrid retrieval
4. Add query expansion for better matching
5. Implement re-ranking with cross-encoder

---

*Generated for RequirementLens Mission 4*
*Baseline ML model unchanged: TF-IDF + Logistic Regression with class_weight="balanced"*
