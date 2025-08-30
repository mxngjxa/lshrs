# Architecture Diagram for LSH Recommender System

## LSH System & Recommendation Engine


```mermaid
---
config:
  layout: dagre
---
graph TD
    subgraph "System Orchestration"
        direction LR
        A[LSHRecommender] -- Manages --> B(RecommendationPipeline)
        C(RecommenderConfig) -- Configures --> A
    end

    subgraph "LSH Processing"
        direction TB
        M[LSH Signatures] --> N[LSH Buckets]
        B -- Finds Candidates --> N
    end

    subgraph "Candidate Search & Recommendation"
        direction TB
        N --> O{BaseSimilarity}
        B -- Computes Similarity --> O
        O -- Similarity Scores --> B
        B -- Top-K --> P[("Recommended Items")]
    end

    subgraph "System Persistence"
        direction TB
        Q{LSHSystemSaver/Loader}
        Q <-- Save/Load --> A
        Q -- Creates/Loads --> R[("lsh_system.tar.gz")]
        subgraph "Archive Contents"
            direction LR
            R1[config.json]
            R2[encoder.joblib]
            R3[hasher.joblib]
            R4[data.pkl]
            R5[signatures.npz]
            R6[manifest.json]
        end
        R --> R1 & R2 & R3 & R4 & R5 & R6
    end

    A -- Contains --> C
    C -- Configures Processing --> M

    style M fill:#e1f5fe
    style P fill:#c8e6c9
```

```{mermaid}
---
config:
  layout: dagre
---
flowchart TD

    A@{ shape: docs, label: "Text documents" }

    A --> B["**DataLoader**
    - Indexing
    - Full Representation
    - Signature
    - Embeddings"]

    B --> C["Preprocessing"]

    subgraph Preprocessing
        direction LR
        C --> D["Tokenize"]
        C --> E["Lemmatize"]
        C --> F["Remove Stopwords"]
        C --> G["Shingling"]
    end

    D & E & F & G --> H["Vectorization"]

    subgraph Vectorization
        direction LR
        H --> I["TF-IDF"]
        H --> J["One-Hot Encoding"]
        H --> K["Embeddings"]
    end

    I --> L["Cosine Similarity"]
    J --> M["Jaccard Similarity"]
    K --> L

    subgraph Hashing
        direction LR
        L --> N["Hyperplane Hashing"]
        M --> O["MinHash"]
        N & O --> P["LSH"]
    end

    P --> Q["Candidate Pairs"]
    Q --> R["Similarity Calculation"]
    R --> S["Top-N Recommendations"]

    S --> T["Output"]
```

## Data Processing Pipeline

```mermaid
---
config:
  layout: dagre
---
graph TD
    subgraph "Data Input & Loading"
        direction TB
        D[("Documents/Text")] --> E{LSHDataLoader}
        F[("Website URLs")] --> G[get_website_content] --> E
        E -- Manages --> H[(Item Data, IDs, Metadata)]
        E -- Optional Save/Load --> I[("dataloader.tar.gz")]
    end

    subgraph "Preprocessing Pipeline"
        direction TB
        J(TextPreprocessor)
        E -- Raw Text --> J
        subgraph "Steps"
            direction LR
            J1[Lemmatize/Stem]
            J2[Remove Stopwords]
            J3[Shingling]
        end
        J --> J1 & J2 & J3
    end

    subgraph "Encoding (Vectorization)"
        direction TB
        K{BaseEncoder}
        J3 -- Preprocessed Text --> K
        subgraph "Encoding Methods"
            direction LR
            K1[TFIDFEncoder]
            K2[EmbeddingEncoder]
            K3[OneHotEncoder]
        end
        K --> K1 & K2 & K3
    end

    subgraph "Hashing & Signature Generation"
        direction TB
        L{BaseHasher}
        K1 -- Vector --> L
        K2 -- Vector --> L
        K3 -- Vector --> L
        subgraph "Hashing Methods"
            direction LR
            L1[HyperplaneLSH]
            L2[MinHash]
        end
        L --> L1 & L2
        L3[OptimalBR] -- Calculates b, r for --> L
        L -- Signatures --> M[LSH Signatures]
    end

    style M fill:#e1f5fe
```