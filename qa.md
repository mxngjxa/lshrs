Here are a few focused questions to better understand how your Redis‑backed LSH package should look before drafting the architecture plan.[1]

1. How do you envision the **public API** of this package: are users calling something like `index(vectors)` / `query(vector, top_k=..., top_p=...)` directly, or do you expect a higher‑level pipeline/recommender object similar to the `RecommendationPipeline`/`LSHRecommender` pattern in your existing codebase?[2]

use a single high level api called `lshrs`. it will have methods like `.ingest()`, `.query()`, `.index()` etc. 

2. What is your **data model** in Redis: do you plan to store only bucket→list-of-indices (and keep vectors/doc metadata in another store), or should this package also manage storage of vectors and/or document payloads in Redis keys/hashes?[3]

bucket->list of indices. for the vectors themselves, the idea is that we want to replace a dedicated vector database and use a generic one. we only need to store the vector data in a database with a index-vector_representation-raw_text format in order for this to work. the idea is that this will actually integrate with whatever people already have, let's say postgres, then all that we will be able to do is take a database that has at least something like: index-vector_representation-raw_document format and our thing is a hash based lookup table. nice and straightforward



3. For the **indexing phase**, will users batch‑load a large corpus once (offline build) and then occasionally append new items, or do you need to support high‑throughput online updates where documents are constantly inserted and maybe deleted?[2]

batch load a large corpus. occasionally insert. consider deletion as well. 

4. On the **query side**, where do you want to run the brute‑force cosine similarity rerank for top‑p / top‑k: inside this package using in‑memory vectors, or by fetching candidate vectors from Redis on demand each time?[3]

brute force cosine sim only for top p. 

my main point of contention is where are we going to consider storing the vectorized forms of the documents. will it be faster storing in redis? faster storing in a database like postgres? I don't know. for integration with existing systems, like they already have a massive postgres database, then we just need a nice and concise redis layer on top 

5. What are the expected **scale and latency** targets (e.g., number of indexed vectors, typical dimensionality, QPS, and acceptable p95 latency), and are you targeting a single Redis instance, Redis Cluster, or letting users decide?[4]

let users decide. the target dataset that we're using is about 6.4 mil datapoints, which is the entire english wikipedia database

6. How much **configuration flexibility** do you want to expose (e.g., number of hash tables/bands, bucket fan‑out, similarity thresholds), and should this follow a typed config model similar to your `RecommenderConfig` / `LSHConfig` style or stay minimal and code‑driven?[4]

it should be minimally but configurable. that means a lot of defualts when defining a function, much like this api interface:

```python
    def __init__(
        self,
        *,
        # All models common arguments
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        torchscript: bool = False,
        dtype: Optional[Union[str, "torch.dtype"]] = None,
        # Common arguments
        pruned_heads: Optional[dict[int, list[int]]] = None,
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        cross_attention_hidden_size: Optional[int] = None,
        add_cross_attention: bool = False,
        tie_encoder_decoder: bool = False,
        # Fine-tuning task arguments
        architectures: Optional[list[str]] = None,
        finetuning_task: Optional[str] = None,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
        task_specific_params: Optional[dict[str, Any]] = None,
        problem_type: Optional[str] = None,
        # Tokenizer kwargs
        tokenizer_class: Optional[str] = None,
        prefix: Optional[str] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        **kwargs,
    ):
```

7. Do you want this package to include **error and health reporting** (custom exceptions, instrumentation, logging) in a style similar to your existing `exceptions.py` and logging in the pipeline, or should it be very lightweight and let host applications handle that?[5][2]

let host handle that. we should throw minimal errors