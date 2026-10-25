# RAG & Vector Search — Interview Prep for Beginners

**Prerequisites:** documents 01 and 02 of this set (embeddings, cosine similarity, fine-tuning).

**What this covers:** a production RAG system from raw documents to grounded answer — parsing, chunking, embedding, BM25, hybrid search, RRF, fuzzy matching, reranking, ANN indexes, quantization, evaluation.

Every formula here is worked through with real numbers, because "I know the BM25 formula" and "I can compute a BM25 score" are very different answers in an interview.

> **The one idea to carry through everything:** retrieval quality is a *pipeline* problem. A world-class embedding model cannot compensate for a PDF parser that destroyed your tables, a chunking strategy that split answers in half, or a missing reranker.

---

# PART 1 — The Whole System at a Glance

```text
                    ┌──────────────────────────────┐
                    │        DATA INGESTION        │   (offline, batch)
                    │  Parse / OCR → Normalize     │
                    │  → Structure → Chunk         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     RETRIEVAL ARTIFACTS      │   (the stuff you store)
                    │  Dense vectors (embeddings)  │
                    │  Sparse / BM25 index         │
                    │  Metadata + raw content      │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │      ONLINE INFERENCE        │   (per query, milliseconds)
                    │  Validate → Route/Rewrite    │
                    │  → Dense + BM25 + Fuzzy      │
                    │  → Fuse → Rerank             │
                    │  → LLM + Citations           │
                    └──────────────────────────────┘
```

**Three layers, three different engineering problems:**

| Layer | Runs | Optimised for | Fails by |
|---|---|---|---|
| Ingestion | Offline, batch | Correctness and structure preservation | Silently destroying information |
| Artifacts | Storage | Memory footprint, query speed | Getting too expensive |
| Inference | Online, per query | Latency and relevance | Being slow or returning junk |

Most beginners spend 90% of their attention on layer 3 and lose most of their quality in layer 1.

---

# PART 2 — Data Ingestion

## 2.1 Parsing and digitization

The goal is not "turn the file into a string." It's **turn the file into a structured representation**.

### Three categories of input

**Text-native** (the text is already in the file):

- `PyMuPDF` / `pdfplumber` for PDFs
- `python-docx` for DOCX
- `BeautifulSoup` for HTML

**Scanned images** (the text is pixels, you need OCR):

- Tesseract (free, decent, no layout)
- AWS Textract, Azure Document Intelligence, Google Document AI (paid, returns layout: blocks, tables, coordinates, reading order)

> Basic OCR gives you characters. It does **not** reliably give you reading order, table structure or which text is a header. If your documents have any layout complexity, plain Tesseract will quietly scramble them.

**Complex layouts** (multi-column, tables, forms, footnotes): you need layout-aware processing that emits typed elements.

```text
Title · Heading · Paragraph · ListItem · Table · Figure · Header · Footer
```

The exact taxonomy depends on the parser.

### A normalized representation to aim for

```json
{
  "document_id": "doc_123",
  "pages": [
    {
      "page_number": 4,
      "elements": [
        {"type": "heading",   "text": "Refund Policy",           "bbox": [72, 110, 480, 140]},
        {"type": "paragraph", "text": "Customers can request...", "bbox": [72, 150, 520, 250]}
      ]
    }
  ]
}
```

The `bbox` (bounding box) matters more than you'd expect — it lets you highlight the exact source region in a citation, which is what makes users trust the system.

### The interview takeaway

> **Digitization quality sets a hard ceiling on retrieval quality.** If a two-column PDF is read straight across so each line mixes two unrelated sentences, no embedding model on earth will fix it. If a table is flattened into `Revenue 2023 2024 120 145`, no reranker can tell you which number belongs to which year.

**Diagnostic you should always suggest:** before blaming retrieval, take 20 failed queries, find the chunk that *should* have answered them, and read it. Half the time the answer isn't in the index at all, or is in the index as garbage.

---

## 2.2 Chunking — where most quality is lost

### Why fixed-size chunking fails

Take this source document:

```markdown
## Refund Policy

Customers may request a refund within 14 days of purchase.

### Eligibility
The item must be unused and in original packaging. Digital goods
are non-refundable once downloaded.

### Processing time
Approved refunds are processed within 5-7 business days to the
original payment method.
```

Chunked naively into fixed 25-token windows, you might get:

```text
CHUNK 1: "## Refund Policy Customers may request a refund within 14 days of
          purchase. ### Eligibility The item must be unused and in"

CHUNK 2: "original packaging. Digital goods are non-refundable once downloaded.
          ### Processing time Approved refunds are processed within"

CHUNK 3: "5-7 business days to the original payment method."
```

Now ask: **"How long do refunds take to process?"**

Chunk 3 is the answer. But look at chunk 3 — it never contains the word "refund." Its embedding is about business days and payment methods. The query embedding, which is about refunds, won't match it well.

This is the single most common cause of bad RAG retrieval: **a chunk loses the context that makes it findable.**

### The fix: contextual chunking

Prepend the document's structural breadcrumb to every chunk:

```text
CHUNK 3: "[Refund Policy > Processing time] Approved refunds are processed
          within 5-7 business days to the original payment method."
```

Now it embeds as being about refunds *and* processing time. Same content, dramatically better retrieval. This costs almost nothing to implement and is often worth more than upgrading your embedding model.

### Chunking strategies

| Strategy | How it works | Good for |
|---|---|---|
| **Heading/section based** | Split at document structure boundaries | Manuals, policies, wikis — anything with real headings |
| **Recursive splitting** | Try paragraph → sentence → word boundaries until under the limit | General-purpose fallback (LangChain's default) |
| **Table-aware** | Keep tables intact; if too large, split by row groups and repeat headers | Financial, specification documents |
| **Semantic chunking** | Detect topic shifts using sentence-to-sentence similarity | Long narrative text with no headings |
| **Sliding window** | Fixed size with overlap | Simple baseline, dense prose |
| **Parent–child** | Embed a small child chunk for precision; return the larger parent for generation | Long technical documents — often the best of both |

### The parent–child pattern, explained

This deserves its own paragraph because it resolves the core tension.

```text
SEARCH over small chunks (200 tokens)   → precise matching, focused embeddings
RETURN the parent section (1500 tokens) → the LLM gets enough context to answer
```

You store both; you index only the children; you return the parents. It's the standard answer to "small chunks match better but don't contain enough to answer."

### The chunk-size trade-off

There is **no universal optimal chunk size**. But the failure modes are predictable:

```text
TOO SMALL (< 100 tokens)
  • fragments that match but don't answer
  • "5-7 business days" retrieved with no idea what takes 5-7 days

TOO LARGE (> 1000 tokens)
  • multiple topics averaged into one vector → matches nothing precisely
  • wastes LLM context window
  • harder to cite precisely

OVERLAP
  • helps preserve context across boundaries
  • but duplicates candidates in results and inflates index size
  • 10-20% overlap is a common starting point
```

A common starting range is 300–800 tokens with 10–15% overlap, but **tune it against your evaluation set**. Treating any number as a universal rule is a weak interview answer.

### Metadata to attach to every chunk

```text
source_document_id      ← for citation
page_number             ← for citation
section / breadcrumb    ← for context and filters
chunk_id                ← unique key
chunk_index             ← position within the document
source_type             ← pdf / confluence / email / ticket
created_at / version    ← for freshness filters and re-indexing
parent_chunk_id         ← for parent-child retrieval
access_control_labels   ← for multi-tenant filtering
```

That last one matters more than it looks. **Access control must be applied as a filter during retrieval, not after.** If you retrieve the top 10 and then filter out 8 for permissions, you return 2 results when 10 valid ones existed. Worse, you may leak information through the fact that a result *existed*.

---

## 2.3 Text, tables and images

| Content | Handling |
|---|---|
| **Text** | Preserve reading order, headings, breadcrumbs. Chunk on structure. |
| **Tables** | Preserve row/column relationships. Serialize to Markdown or HTML. For large tables, split by row groups and **repeat the header row in every split**. |
| **Images / figures** | Store the image reference and page metadata. Generate a caption or description for text retrieval, or use a multimodal embedding model. |
| **Charts** | Keep the image, and also extract the textual facts and axis labels where possible. |

### The table trick

Don't only embed the raw table. Numeric cells and terse column headers embed terribly — a table of quarterly revenue is semantically almost identical to a table of quarterly headcount.

```text
Raw table      → stored, used for generation and citation (exact numbers preserved)
Table summary  → additionally indexed as a retrieval representation
                 "Quarterly revenue by region for FY2024, covering
                  APAC, EMEA and Americas, in USD millions"
```

The summary makes the table *findable*. The raw table makes the answer *correct*. **The summary must never replace the source table** when exact numbers matter — an LLM summarising "revenue grew significantly" will then be your only record of a number.

### The image trick

A text-only embedding model cannot see pixels. Two options:

1. Generate a caption with a vision model, index the caption text.
2. Use a multimodal embedding model that puts images and text in the same vector space.

Option 1 is usually more practical and more debuggable.

---

# PART 3 — Retrieval Artifacts

## 3.1 The embedding service

### Batching

Embed many chunks per request. The difference between 1-per-request and 256-per-request is often 10–50× throughput, because you amortise both network round-trips and GPU kernel launches.

### Caching

Hash the normalized chunk text. If the hash is unchanged, skip re-embedding. On an incremental re-index of a corpus where 2% of documents changed, this turns a $100 job into a $2 job.

### Versioning — the one that bites people

```text
Store the embedding model NAME and VERSION in your index metadata.
```

**Vectors from different models are not comparable.** A 768-dim BGE vector and a 768-dim E5 vector have the same shape and completely unrelated meaning. Comparing them produces numbers, and the numbers are noise.

Consequences:

- Changing embedding models requires **re-embedding the entire corpus**. There is no incremental path.
- A mixed index (some chunks from model A, some from model B) is silently broken. Nothing errors; results just get worse.

### Dimensionality

Higher is not automatically better. It increases storage linearly and search cost roughly linearly. Matryoshka-trained models (document 01 §2.6) let you truncate with graceful degradation.

### Query/document consistency

```text
same model, same version
same preprocessing (including prompt prefixes!)
same dimension
same similarity metric
same normalization (normalized or not — pick one)
```

Any mismatch here degrades retrieval with no error message.

---

## 3.2 Choosing a vector database

| Option | Notes |
|---|---|
| **FAISS** | A similarity-search *library*, not a database. Excellent inside a custom service. No persistence, filtering or replication out of the box. |
| **Pinecone** | Managed, dense + sparse/hybrid. Low ops burden, higher cost. |
| **Milvus / Zilliz** | Full vector database, many index types, scales well. More operational complexity. |
| **Weaviate** | Vector DB with integrated keyword and hybrid retrieval. |
| **Qdrant** | Dense + sparse, strong filtering, multiple fusion options. Good Rust performance. |
| **pgvector** | Vector search inside PostgreSQL. Compelling when relational data and vectors should live together and you already run Postgres. |
| **Elasticsearch / OpenSearch** | Mature BM25 plus vector support. Often the pragmatic choice if you already run it. |

### How to actually decide

```text
corpus size and growth rate
query latency target (p50 and p99)
filtering requirements (how complex? how selective?)
does it need native hybrid dense+sparse?
update/delete patterns (append-only? frequent updates?)
operational complexity your team can absorb
replication / sharding / HA requirements
cost
```

> **Never say "X is the best vector DB."** Say "for a 5M-chunk corpus with heavy metadata filtering and an existing Postgres deployment, I'd start with pgvector and only move to a dedicated vector DB if we outgrow it — because the operational cost of one fewer system is real."

**A practical bias worth stating:** if your corpus is under ~1M chunks and you already run Postgres or Elasticsearch, adding a dedicated vector database is often premature. Under about 100k vectors, a brute-force index in memory is fine and gives you exact results.

---

# PART 4 — Online Inference

## 4.1 Validate the query

```text
empty / malformed query
maximum length
rate limiting and abuse
prompt-injection or policy checks
tenant and access-control checks
```

> Distinguish **application security** from **retrieval logic**. Authentication and authorization are enforced before any protected data is returned — not as a post-filter on results.

## 4.2 Classification and routing

A production system may decide:

- **Which corpus / tenant** should be searched
- **Query type**: lookup, comparison, summarization, aggregation
- **Metadata filters**: date, product, department, document type

### Query rewriting and expansion

| Technique | What it does | Cost |
|---|---|---|
| **Rewriting** | "when did they ship it" → "when did Acme Corp ship order #4471" using conversation history | One LLM call of latency |
| **Multi-query expansion** | Generate 3 paraphrases, retrieve for each, union the results | 3× retrieval cost |
| **HyDE** | Ask an LLM to *hallucinate* an ideal answer, embed that instead of the question — a fake answer is lexically closer to a real answer than a question is | One LLM call |
| **Decomposition** | Split "compare X and Y policies" into two sub-queries | Multiple retrievals |

All of these increase recall and add latency, and all risk **query drift** — the rewritten query asks something subtly different. Always measure whether rewriting actually helps *your* workload before shipping it.

## 4.3 Retrieve candidates

```text
User query
   │
   ├── Dense vector retrieval    → semantic similarity, paraphrases
   ├── BM25 / sparse retrieval   → exact tokens, IDs, codes, rare terms
   └── Fuzzy / entity retrieval  → typos, OCR noise, name variants
             │
             ▼
       candidate union (dedupe)
             │
             ▼
       RRF or calibrated score fusion
             │
             ▼
        cross-encoder reranker
             │
             ▼
       top K → LLM with citations
```

---

# PART 5 — BM25, Worked by Hand

## 5.1 What BM25 is

BM25 is a **lexical** ranking function from the 1990s. It scores documents by token overlap — no semantics, no embeddings. It is still essential, and it will still beat your embedding model on certain queries.

$$
\text{score}(D,Q)=\sum_{q_i \in Q} \text{IDF}(q_i)\cdot
\frac{f(q_i,D)\,(k_1+1)}{f(q_i,D)+k_1\left(1-b+b\frac{|D|}{\text{avgdl}}\right)}
$$

| Symbol | Meaning |
|---|---|
| `f(qᵢ,D)` | how many times query term `qᵢ` appears in document `D` |
| `IDF(qᵢ)` | inverse document frequency — rare terms score higher |
| `\|D\|` | length of this document in tokens |
| `avgdl` | average document length in the corpus |
| `k₁` | term-frequency saturation (typically 1.2–2.0) |
| `b` | length normalization strength (typically 0.75) |

## 5.2 The three intuitions inside the formula

**1. IDF — rare words matter more.** In a corpus about software, "the" appears everywhere and tells you nothing; "kubernetes" narrows things down; "CVE-2024-3094" narrows things to one document.

**2. `k₁` — term frequency saturates.** A document mentioning "refund" 50 times is not 50× more relevant than one mentioning it once. Probably 3× at most. The `k₁` term creates diminishing returns.

**3. `b` — long documents are penalised.** A 10,000-word document contains "refund" by accident more often than a 200-word one. Dividing by `|D|/avgdl` corrects for that.

## 5.3 Full worked example

**Setup:**

```text
Query:  "refund policy"
Corpus: N = 1,000,000 documents
        avgdl = 100 tokens
        "refund" appears in n = 10,000 documents
        "policy" appears in n = 100,000 documents
Params: k₁ = 1.2, b = 0.75

Document A: 50 tokens long
            contains "refund" 3 times, "policy" 1 time
```

### Step 1 — IDF for each term

Using the Lucene/Robertson form: `IDF(q) = ln(1 + (N − n + 0.5)/(n + 0.5))`

```text
IDF("refund") = ln(1 + (1,000,000 − 10,000 + 0.5) / (10,000 + 0.5))
              = ln(1 + 990,000.5 / 10,000.5)
              = ln(1 + 98.995)
              = ln(99.995)
              = 4.6051

IDF("policy") = ln(1 + (1,000,000 − 100,000 + 0.5) / (100,000 + 0.5))
              = ln(1 + 8.99996)
              = ln(9.99996)
              = 2.3026
```

Notice: "refund" is 10× rarer than "policy" and gets **2× the IDF weight** (logarithmic, not linear).

### Step 2 — TF component for "refund"

```text
numerator   = f × (k₁ + 1)         = 3 × (1.2 + 1)       = 3 × 2.2 = 6.6

denominator = f + k₁ × (1 − b + b × |D|/avgdl)
            = 3 + 1.2 × (1 − 0.75 + 0.75 × 50/100)
            = 3 + 1.2 × (0.25 + 0.375)
            = 3 + 1.2 × 0.625
            = 3 + 0.75
            = 3.75

TF component = 6.6 / 3.75 = 1.76
```

### Step 3 — TF component for "policy"

```text
numerator   = 1 × 2.2 = 2.2
denominator = 1 + 0.75 = 1.75
TF component = 1.2571
```

### Step 4 — Total

```text
score = IDF("refund") × 1.76    +  IDF("policy") × 1.2571
      = 4.6051 × 1.76           +  2.3026 × 1.2571
      = 8.105                   +  2.895
      = 11.00
```

### Step 5 — Now watch `b` do its job

Same document content, but the document is **200 tokens** instead of 50:

```text
denominator = 3 + 1.2 × (0.25 + 0.75 × 200/100)
            = 3 + 1.2 × (0.25 + 1.5)
            = 3 + 2.1 = 5.1

TF component = 6.6 / 5.1 = 1.294
"refund" contribution = 4.6051 × 1.294 = 5.959   (was 8.105)
```

The longer document scores **26% lower** for the same three occurrences. That's length normalization working.

> **Interview note:** exact IDF formulas differ slightly across implementations (Lucene, Elasticsearch, `rank_bm25`) because of smoothing details. Focus on the intuition — rarity, saturation, length — not on memorising one variant.

## 5.4 When BM25 beats dense retrieval

```text
"E402"                  ← a Python lint code
"ISO 27001"             ← a standard
"RTX 4090"              ← a product SKU
"ABC-123"               ← an order ID
"customer_id_9812"      ← an identifier
"CVE-2024-3094"         ← a vulnerability ID
```

Embedding models are trained to find *similar* things. For an exact identifier, "similar" is exactly wrong — you want *that one*, not one that looks like it. An embedding model will happily return CVE-2024-3095.

## 5.5 When BM25 fails

```text
Query:    "automobile maintenance"
Document: "car servicing"
Overlap:  zero tokens
BM25:     0
```

The meanings are the same and BM25 cannot see it. This is exactly what dense retrieval fixes.

> **This complementarity is the entire argument for hybrid search.** They fail on different queries.

---

# PART 6 — Hybrid Search and RRF

## 6.1 The problem

Three retrievers return scores on wildly different scales:

```text
Dense (cosine):     0.83,  0.79,  0.78, ...    range roughly [0, 1]
BM25:              14.2,  11.8,  10.1, ...     unbounded, corpus-dependent
Fuzzy (RapidFuzz):   96,    89,    84, ...     range [0, 100]
```

You cannot do this:

```text
0.83 + 14.2 + 96 = 111.03      ← meaningless. BM25 and fuzzy swamp the dense signal entirely.
```

## 6.2 Approach A — score fusion (harder than it looks)

$$
\text{HybridScore}(d)=\alpha \cdot S_{\text{dense}}(d)+(1-\alpha)\cdot S_{\text{sparse}}(d)
$$

This requires normalizing each signal first. Options: min-max normalization over the result set, z-score standardization, or learned calibration.

**Why it's fragile:** min-max normalization depends on which documents happened to be in *this* result set. A query where the top BM25 score is 40 and a query where it's 4 will be normalized to the same range, so your `α` means different things on different queries.

## 6.3 Approach B — Reciprocal Rank Fusion

RRF sidesteps calibration entirely by throwing away the scores and keeping only the **ranks**.

$$
\text{RRF}(d)=\sum_{r \in R}\frac{w_r}{k+\text{rank}_r(d)}
$$

| Symbol | Meaning |
|---|---|
| `R` | the set of ranked lists (dense, BM25, fuzzy) |
| `rank_r(d)` | position of document `d` in list `r`, starting at 1 |
| `w_r` | optional weight for that retriever |
| `k` | smoothing constant, conventionally **60** |

### What `k` does

`k = 60` means rank 1 contributes `1/61 = 0.0164` and rank 2 contributes `1/62 = 0.0161`. The gap is tiny.

Without `k` (i.e. `k = 0`), rank 1 gives 1.0 and rank 2 gives 0.5 — a 2× drop. That would let a single retriever's top result dominate everything.

**So `k` controls how much the very top ranks dominate.** Large `k` → flatter, more democratic, rewards appearing in *many* lists. Small `k` → the #1 result of any retriever wins.

### Worked example

```text
Dense results          BM25 results           Fuzzy results
rank 1: D2             rank 1: D1             rank 1: D1
rank 3: D1             rank 2: D2             rank 2: D4
rank 5: D4             rank 4: D3             rank 3: D2
```

With `k = 60` and all weights = 1:

```text
D1:  dense rank 3 → 1/63 = 0.01587
     BM25  rank 1 → 1/61 = 0.01639
     fuzzy rank 1 → 1/61 = 0.01639
     ────────────────────────────────
     RRF(D1) = 0.04866   ★ winner

D2:  dense rank 1 → 1/61 = 0.01639
     BM25  rank 2 → 1/62 = 0.01613
     fuzzy rank 3 → 1/63 = 0.01587
     ────────────────────────────────
     RRF(D2) = 0.04840

D4:  dense rank 5 → 1/65 = 0.01538
     fuzzy rank 2 → 1/62 = 0.01613
     (absent from BM25 → contributes 0)
     ────────────────────────────────
     RRF(D4) = 0.03151

D3:  BM25  rank 4 → 1/64 = 0.01563
     ────────────────────────────────
     RRF(D3) = 0.01563

Final ranking: D1 > D2 > D4 > D3
```

**The insight:** D1 wins even though the dense retriever ranked it *third*. Appearing near the top of **all three** lists beat being #1 in one. That's consensus, and consensus is usually a better relevance signal than any single retriever's confidence.

### Weighted RRF

```text
Dense weight = 1.0
BM25  weight = 1.2     ← offline eval showed BM25 is more reliable in this domain
Fuzzy weight = 0.6
```

Use this when your evaluation data shows one signal is systematically more trustworthy.

### RRF vs score fusion vs learned ranking

| Method | Idea | Main challenge |
|---|---|---|
| Weighted score fusion | Combine normalized scores | Calibration is fragile across queries |
| **RRF** | Combine rank positions | **Discards score magnitude** — a document that scored 0.99 and one that scored 0.71 look identical if both ranked 1st |
| Learned ranker (LTR) | Learn how signals predict relevance | Needs training data, monitoring, and retraining |

> **The honest limitation of RRF:** it cannot distinguish "confidently the best" from "narrowly the best." If your dense retriever returns a top result at 0.31 cosine similarity — meaning nothing in the corpus is really relevant — RRF still treats it as rank 1 and gives it full weight. That's why a good pipeline also has an **absolute score floor** to detect "no good results," independent of fusion.

### What RRF is NOT

```text
WRONG:  RRF = dense_score + bm25_score

RIGHT:  retrieve ranked lists from each retriever
            ↓
        convert each document's RANK into 1/(k + rank)
            ↓
        sum those contributions per document
            ↓
        sort by the fused score
```

---

# PART 7 — Fuzzy Matching

## 7.1 What it's for

Fuzzy matching compares two strings by **character or token similarity**, not meaning.

```text
"Rajesh Kumar"   vs  "Rajesh Kumer"      ← typo
"Micro soft"     vs  "Microsoft"          ← tokenization noise
"INV-2024-0Ol2"  vs  "INV-2024-0012"      ← OCR confusing O/0 and l/1
"Bengaluru"      vs  "Bangalore"          ← name variants
```

None of these are handled well by embeddings (which see them as different words) or BM25 (which sees them as different tokens).

## 7.2 The metrics

**1. Character-level ratio** — based on Levenshtein edit distance, normalized to 0–100.

```text
"Microsoft" → "Micrsoft"  needs 1 deletion out of 9 characters
ratio ≈ 94
```

**2. Token-aware matching** — sorts or sets the tokens before comparing.

```text
token_sort_ratio("Kumar Rajesh", "Rajesh Kumar") = 100
plain ratio for the same pair ≈ 45
```

**3. Partial matching** — finds the best-matching substring, so a short query can match inside a long field.

```python
from rapidfuzz import fuzz, process

choices = ["Rajesh Kumar", "Ramesh Kumar", "Rajesh Kumaran"]
print(process.extract("Rajesh Kumer", choices, scorer=fuzz.WRatio, limit=3))
```

## 7.3 Where it fits — and where it doesn't

```text
USE fuzzy as a retrieval leg when:
  • the corpus has entity names, IDs, SKUs, part numbers
  • OCR noise is present
  • users misspell things frequently (they do)
  • you have a bounded candidate set to match against

DO NOT:
  • brute-force fuzzy match across millions of rows — it's O(N) string comparisons per query
  • use it as a substitute for semantic retrieval
```

**How to make it scale:** narrow the candidate set first. Use BM25 or an n-gram index to get 1,000 candidates, *then* fuzzy-score those 1,000. Or index trigrams (Postgres `pg_trgm` does this well) so the database can prune before you score.

## 7.4 Fuzzy scores and RRF

A fuzzy score of 96 and a cosine similarity of 0.83 are not comparable. Rank the fuzzy results, feed the *ranks* into RRF. This is exactly the problem RRF exists to solve.

---

# PART 8 — Reranking

## 8.1 Bi-encoder vs cross-encoder

```text
BI-ENCODER (the retriever)               CROSS-ENCODER (the reranker)

query   → [encoder] → vector ─┐          [query [SEP] passage] → [encoder] → score
                               ├→ cos
passage → [encoder] → vector ─┘          the two texts attend to EACH OTHER

✓ document vectors precomputed offline   ✗ nothing precomputable
✓ search 10M docs in milliseconds        ✗ one forward pass per candidate pair
✗ query and passage never interact       ✓ far more accurate
```

**Why the cross-encoder is more accurate:** in a bi-encoder, the passage is compressed into a fixed vector *before ever seeing the query*. That compression has to preserve everything anyone might ask. A cross-encoder reads both at once, so attention can check "does this passage contain the specific number this query asks for?" — a question the bi-encoder's vector was never able to represent.

**The cost:** reranking 100 candidates = 100 forward passes. At maybe 5–20ms each on GPU, that's 0.5–2 seconds unless you batch them. Reranking 10M documents is not a thing you can do.

## 8.2 The production pattern

```text
10,000,000 documents
       ↓  bi-encoder ANN search (~10ms)
      500 candidates
       ↓  RRF fusion with BM25 and fuzzy
      100 candidates
       ↓  cross-encoder rerank (~100ms batched)
       10 passages
       ↓  LLM
     answer + citations
```

Each stage is cheaper per item than the next but processes more items. That's the whole design principle: **wide and cheap, then narrow and expensive.**

## 8.3 How rerankers are trained

Data: `(query, passage, relevance_label)`.

**Hard negatives are essential.** A reranker trained only on random negatives learns to distinguish "about the right topic" from "completely unrelated" — but by the time a candidate reaches the reranker, everything is already about the right topic. The reranker must be trained on exactly the confusions it will face in production: mine its negatives from the *retriever's actual top-k output*.

**Losses:** `BinaryCrossEntropyLoss` for pairwise relevance; listwise losses like `LambdaLoss` when you have graded judgments per query. (Full detail in document 02 §11.)

**Distillation:** a common trick is to use a large cross-encoder or an LLM as a teacher to produce soft relevance scores, then train a small fast reranker to match them. You get most of the quality at a fraction of the latency.

## 8.4 When to skip the reranker

Rerankers cost latency on every single query, forever. Skip it when:

- Recall@10 is already high and the top result is usually correct
- Your latency budget is under ~150ms end-to-end
- The corpus is small and homogeneous

---

# PART 9 — Approximate Nearest Neighbour Indexes

## 9.1 Why ANN exists

Brute-force search over `N` vectors of dimension `d` costs `O(N·d)` per query.

```text
10,000,000 vectors × 768 dims = 7.68 billion multiply-adds per query
```

That's viable on a GPU with batching. On a CPU serving 1,000 queries per second, it is not. ANN trades a little recall for enormous speed.

## 9.2 Flat (brute force)

- **Build:** none
- **Search:** compare against everything
- **Recall:** 100%, exact
- **Use:** small corpora (<100k), and — critically — as the **ground truth** you measure ANN recall against

> Always build a Flat index over a sample of your corpus. Without it you cannot say "our HNSW config achieves 0.97 recall" — you'd have nothing to compare to.

## 9.3 LSH (Locality Sensitive Hashing)

Hash vectors so that similar ones land in the same bucket, e.g. by recording which side of each of several random hyperplanes a vector falls on.

```text
vectors → hash into buckets → probe likely buckets → exact-rescore candidates
```

Conceptually important, historically significant, but modern production usually uses graph-based or inverted-file methods instead. **Don't claim LSH is obsolete** — it's still used where the metric, update pattern or memory profile suits it.

## 9.4 IVF (Inverted File Index)

**Build:**
1. Run k-means over the vectors to create `nlist` centroids.
2. Assign each vector to its nearest centroid's list.

**Query:**
1. Find the `nprobe` nearest centroids to the query.
2. Search only those lists.
3. Return the best candidates.

**Strength:** good speed/recall trade-off, pairs beautifully with quantization (see IVF-PQ).
**Weakness:** if the true nearest neighbour sits in an unprobed cell, you simply never see it.

### `nlist` and `nprobe`, worked

```text
N = 10,000,000 vectors
nlist = 10,000 clusters

Average vectors per list = 10,000,000 / 10,000 = 1,000

With nprobe = 10:
   vectors examined ≈ 10 × 1,000 = 10,000
   instead of 10,000,000
   → roughly 1000× fewer comparisons
```

That's a first-order estimate. Real clusters are unbalanced, so a probed list may hold 3,000 vectors or 200.

| Parameter | Set at | Controls | Changing it requires |
|---|---|---|---|
| `nlist` | Build time | How finely the space is partitioned | **Rebuilding the index** |
| `nprobe` | Query time | How many partitions you search | Nothing — tune it live |

**Rule of thumb:** `nlist ≈ 4√N` to `16√N`. For N = 10M, √N ≈ 3,162, so nlist between roughly 12,000 and 50,000.

```text
nprobe = 1        →  fastest, lowest recall
nprobe ↑          →  recall ↑, latency ↑
nprobe = nlist    →  searches everything ≈ exhaustive
```

> **The key interview point:** `nprobe` is the knob you tune *in production*, because it trades latency for recall without rebuilding anything. `nlist` is a build-time commitment.

## 9.5 HNSW (Hierarchical Navigable Small World)

A multi-layer proximity graph.

```text
Layer 2 (sparse):    A ──────────────── F        long-range hops
                     │                  │
Layer 1:             A ──── C ───── E ──F        medium hops
                     │      │       │   │
Layer 0 (all nodes): A─B─C─D─E─F─G─H─I─J         every vector, dense local links
```

Search enters at the top layer, greedily walks toward the query, then descends. Upper layers cover distance fast; lower layers refine.

| Parameter | Meaning | Effect |
|---|---|---|
| `M` | Max links per node | Higher = better recall, more memory, slower build |
| `ef_construction` | Candidate list size during build | Higher = better graph quality, slower build |
| `ef_search` | Candidate list size at query time | **Higher = better recall, higher latency** — your live tuning knob |

`ef_search` is HNSW's `nprobe`: adjustable at query time without rebuilding.

**Trade-offs:** HNSW is memory-hungry (the graph itself is significant overhead on top of the vectors) and deletes are awkward — most implementations tombstone rather than truly remove, so a high-churn index needs periodic rebuilds.

## 9.6 Comparison

| Index | Build | Search | Memory | Recall | Best for |
|---|---|---|---|---|---|
| Flat | None | Slow at scale | Vectors only | Exact | <100k vectors, ground truth |
| LSH | Fast | Fast for suitable workloads | Can be high | Tuning-dependent | Specific metric/update patterns |
| IVF | Moderate (k-means) | Fast, tunable | Moderate | Tunable via `nprobe` | Large corpora, especially with PQ |
| HNSW | Slow | Very fast | **High** | Usually very high | Low-latency production, moderate churn |

```text
NEVER SAY:  "HNSW is always best."
            "IVF is always fastest."
            "LSH is obsolete."

SAY:        "HNSW for low-latency serving when memory allows.
             IVF-PQ when the corpus is too large to keep in RAM.
             Flat for the ground-truth baseline."
```

---

# PART 10 — Similarity Metrics

## Cosine similarity

$$\cos(A,B)=\frac{A\cdot B}{\|A\|\,\|B\|}$$

Measures **angle only**. Range `[-1, 1]`. The default for text embeddings.

## L2 / Euclidean distance

$$L2(A,B)=\sqrt{\sum_i (a_i-b_i)^2}$$

Measures straight-line distance. **Smaller is closer.** Magnitude-sensitive. It's the objective k-means optimises, which is why IVF (built on k-means) has a natural affinity for it.

## Dot product

$$A\cdot B=\sum_i a_i b_i$$

Direction **and** magnitude.

## The identity you must know

If both vectors are L2-normalized (length 1), then `‖A‖ = ‖B‖ = 1` and:

$$A\cdot B = \cos(A,B)$$

So: **normalize once at index time, then use dot product, and get cosine ranking for free at lower cost.** This is what nearly every production system does.

There's a second identity worth knowing:

```text
For unit vectors:   L2(A,B)² = 2 − 2·cos(A,B)
```

L2 is a monotonically *decreasing* function of cosine similarity on unit vectors — so **on normalized vectors, cosine, dot product and L2 all produce the identical ranking.** They differ only in cost and in the numbers they report.

| Metric | Magnitude-sensitive? | Typical use |
|---|---|---|
| Cosine | No | Semantic similarity — the default |
| L2 | Yes | Euclidean geometry, k-means-based indexes |
| Dot product | Yes (unless normalized) | Efficient cosine on normalized vectors; also used when magnitude encodes confidence |

> **Careful with "dot product is faster."** It is one fewer operation, but real performance depends on the library, SIMD support, whether normalization is precomputed, and the index implementation. The honest answer is "dot product on pre-normalized vectors avoids computing norms at query time, which is why it's the standard choice — but I'd benchmark rather than assume."

---

# PART 11 — Large-Scale Storage and Compression

## 11.1 The memory problem

```text
Per vector of D dimensions:
  float32  ≈ 4 × D bytes
  float16  ≈ 2 × D bytes
  int8     ≈ 1 × D bytes
  binary   ≈ D / 8 bytes

100 million vectors at 768 dims, float32:
  100,000,000 × 768 × 4 = 307 GB          ← before index overhead
```

307 GB does not fit on one machine's RAM at reasonable cost. This is why compression exists.

## 11.2 Scalar quantization

Map each float to an int8. Roughly **4× smaller**, conceptually simple, and it usually preserves more accuracy than PQ or binary. Often the best first move.

(The full math of how a float becomes an int8 is in document 04 §5 — read it, because interviewers ask.)

## 11.3 Product Quantization (PQ)

**The idea:** don't quantize each number independently. Split the vector into chunks and replace each chunk with the ID of the nearest entry in a learned codebook.

**Steps:**
1. Split a `D`-dimensional vector into `m` sub-vectors.
2. Run k-means on each subspace independently to learn a codebook (typically 256 centroids).
3. Replace each sub-vector with its nearest centroid's ID.
4. Store the IDs, not the floats.

With 256 centroids per subspace, each ID fits in **8 bits = 1 byte**.

### Worked example

```text
Original: 128-dim float32 vector = 128 × 4 = 512 bytes

With m = 16 sub-vectors (each 8 dims), 256 centroids each:
  each sub-vector → 1 byte
  16 sub-vectors  → 16 bytes

512 → 16 bytes = 32× reduction
```

(Ignoring the codebooks, which are shared across all vectors and therefore amortise to nothing at scale.)

### ADC — Asymmetric Distance Computation

Here's why PQ search is fast. At query time:

1. Keep the **query** in full precision.
2. Precompute the distance from each query sub-vector to all 256 centroids in its subspace → a 16 × 256 lookup table.
3. For each stored vector, its distance is 16 table lookups plus 15 additions.

No decompression, no float multiplication. Just table lookups. This is why PQ is fast as well as small.

## 11.4 Binary quantization

One bit per dimension — keep only the sign.

```text
768 dims × 4 bytes = 3,072 bytes  →  768 bits = 96 bytes     32× smaller
```

Comparison becomes **Hamming distance**: XOR the two bit strings and count the 1s. That's one or two CPU instructions per 64 dimensions.

Quality drops meaningfully. The standard fix:

```text
binary search over the whole corpus
        ↓
   top 500 candidates
        ↓
rescore with full-precision (or int8) vectors
        ↓
   top 50
```

You only need full-precision vectors available for the rescore — they can live on disk, since you only fetch 500.

## 11.5 OPQ — Optimized Product Quantization

PQ assumes the sub-vectors are independent, which real embeddings aren't. OPQ learns a **rotation matrix** applied before splitting, so that the resulting subspaces are more independent and each quantizes with less error. Same compression ratio, lower error. Usually worth the extra build step.

## 11.6 IVF-PQ

The workhorse for very large corpora:

```text
IVF → prune the search space (only look at nprobe clusters)
PQ  → compress the vectors inside those clusters
```

100M vectors at 768 dims:
```text
float32 flat:    307 GB      ← doesn't fit
IVF-PQ (m=96):   ~9.6 GB     ← fits on one machine
```

Plus typically a rescore stage using full vectors on disk for the final top-k.

## 11.7 Disk-based indexes and sharding

**Disk-based** (e.g. DiskANN-style): keep a compressed index in RAM for navigation, fetch full vectors from SSD only for the candidates you actually visit.

**Sharding:**

```text
query
  ↓  fan out to all shards in parallel
each shard returns its local top-K
  ↓  merge
global top-K
```

Challenges: p99 latency is set by your **slowest** shard, not the average (tail amplification). Hot shards. Rebalancing when data grows unevenly.

## 11.8 Trade-off summary

| Method | Compression | Advantage | Cost |
|---|---|---|---|
| Full precision | 1× | Exact | Highest memory |
| Float16 | 2× | Nearly free quality-wise | Minor |
| Scalar (int8) | 4× | Simple, good accuracy | Small quantization error |
| PQ | 16–64× | Strong compression, fast ADC | Approximate distances, codebook training |
| Binary | 32× | Extreme compression, Hamming speed | Real recall loss — needs rescoring |
| IVF-PQ | 16–64× + pruning | Handles billions of vectors | Most tuning complexity |

```text
THE UNIVERSAL PATTERN
  cheap approximate retrieval → small candidate set → precise rescore/rerank
```

---

# PART 12 — End-to-End Example

**Query:** `"Why did invoice INV-2024-0012 from Achme Corp get rejcted?"`

Note: two typos ("Achme" for "Acme", "rejcted" for "rejected"), plus an exact identifier. This query is designed to break a pure-dense pipeline.

### Step 1 — Validate and normalize
Length OK, tenant verified, access labels resolved. Lowercase, strip punctuation. Entity extraction spots `INV-2024-0012` as an invoice ID.

### Step 2 — Route
Classified as a **lookup** query against the invoicing corpus. Filter: `tenant_id = acme_corp`.

### Step 3 — Dense retrieval
Embed the query, ANN search, top 50. This handles "why was it rejected" → chunks about rejection reasons and validation failures. It will probably **miss** the specific invoice, because embeddings don't do exact IDs.

### Step 4 — BM25
Top 50. `INV-2024-0012` is an extremely rare token, so its IDF is enormous and any document containing it rockets to rank 1. **This is the leg that finds the right document.**

### Step 5 — Fuzzy / entity retrieval
`"Achme"` vs the customer table: `Acme Corp` scores ~90. Resolves the entity so we can filter or boost correctly. Also catches OCR-style ID variants if the source was scanned.

### Step 6 — Union and RRF
Merge the three lists (~120 unique documents after dedupe), fuse with RRF, keep the top 100.

### Step 7 — Cross-encoder rerank
Score all 100 `(query, passage)` pairs jointly. The reranker now sees the full query *with* the passage and can tell "rejection reason for invoice INV-2024-0012" apart from "general invoice rejection policy." Keep the top 5.

### Step 8 — Generate
Feed the top 5 chunks with their metadata to the LLM:

```text
System: Answer ONLY from the provided context. Cite the chunk id for
        every claim. If the context does not contain the answer, say so.

Context:
[chunk_8871 | invoices.pdf p.12 | Invoices > Rejections]
  Invoice INV-2024-0012 was rejected on 2024-03-14: PO number mismatch.
...
```

### Step 9 — Verify and return
Check that cited chunk IDs actually exist in the context (catches fabricated citations). Return the answer with links to the source pages.

**Which leg saved us?** BM25 found the document. Fuzzy resolved the customer name. Dense understood "why was it rejected." The reranker picked the right chunk among 100. Remove any one and the answer degrades.

---

# PART 13 — Evaluation

## 13.1 Retrieval metrics

### Recall@K — the most important one for RAG

```text
                 queries where relevant evidence appears in the top K
Recall@K  =  ─────────────────────────────────────────────────────────
                          total queries
```

**Why it's the most important:** it's a hard ceiling. If the right chunk isn't in the top K that reaches your reranker, no reranker and no LLM can recover it. Everything downstream is bounded by this number.

### Precision@K

Fraction of the top K that are relevant. Matters when you're paying LLM tokens for every chunk you pass along.

### MRR — Mean Reciprocal Rank

$$MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\text{rank}_i}$$

Only the **first** relevant result counts.

```text
Query 1: first relevant at rank 1  →  1/1  = 1.000
Query 2: first relevant at rank 3  →  1/3  = 0.333
Query 3: first relevant at rank 5  →  1/5  = 0.200
                                       ─────────────
MRR = (1.000 + 0.333 + 0.200) / 3 = 0.511
```

### NDCG@K — worked in full

Use this when relevance is **graded** (2 = directly answers, 1 = related, 0 = irrelevant) rather than binary.

$$DCG@k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$$

The `log₂(i+1)` denominator discounts lower positions — a relevant document at rank 5 is worth less than the same document at rank 1.

**Ranking A** — relevance grades by position: `[3, 2, 0, 1, 0]`

```text
position 1, rel=3:  (2³−1)/log₂(2) = 7 / 1.0000 = 7.0000
position 2, rel=2:  (2²−1)/log₂(3) = 3 / 1.5850 = 1.8928
position 3, rel=0:  (2⁰−1)/log₂(4) = 0 / 2.0000 = 0.0000
position 4, rel=1:  (2¹−1)/log₂(5) = 1 / 2.3219 = 0.4307
position 5, rel=0:                                0.0000
                                                 ────────
DCG@5(A) = 9.3235
```

**Ideal ranking** — the same documents sorted best-first: `[3, 2, 1, 0, 0]`

```text
7/1.0000 = 7.0000
3/1.5850 = 1.8928
1/2.0000 = 0.5000
0
0
        ────────
IDCG@5  = 9.3928
```

```text
NDCG@5(A) = 9.3235 / 9.3928 = 0.993     ← nearly perfect
```

**Ranking B** — same documents, badly ordered: `[0, 2, 3, 0, 1]`

```text
position 1, rel=0:  0/1.0000      = 0.0000
position 2, rel=2:  3/1.5850      = 1.8928
position 3, rel=3:  7/2.0000      = 3.5000
position 4, rel=0:                  0.0000
position 5, rel=1:  1/2.5850      = 0.3868
                                   ────────
DCG@5(B) = 5.7796
NDCG@5(B) = 5.7796 / 9.3928 = 0.615
```

**Same five documents. 0.993 vs 0.615.** NDCG measures the ordering, which is exactly what a reranker changes.

## 13.2 Beyond retrieval

```text
retrieval recall          ← is the evidence even there?
       ↓
reranker quality          ← is it at the top?
       ↓
answer faithfulness       ← does the answer follow from the evidence?
       ↓
answer relevance          ← does it address the question?
       ↓
citation accuracy         ← do the cited chunks actually support the claims?
       ↓
latency + cost            ← can we afford it?
```

A system can have Recall@10 = 0.95 and still give bad answers — because the reranker picked the wrong 3 chunks, or the LLM ignored the evidence and used its parametric knowledge instead.

**Faithfulness** (also called groundedness) is the one people forget. Measure it by checking whether every claim in the answer is supported by a retrieved chunk. Common approach: an LLM judge scoring claim-by-claim against the provided context, calibrated against a human-labelled sample.

---

# PART 14 — Common Interview Mistakes

| # | The mistake | The corrected answer |
|---|---|---|
| 1 | "RAG is just embeddings and a vector database." | "RAG is a pipeline: parsing, chunking, indexing, multi-signal retrieval, fusion, reranking, generation, evaluation. Most quality is won or lost in parsing and chunking." |
| 2 | "I'd use cosine similarity because it's the most accurate." | "On L2-normalized vectors, cosine, dot product and L2 give identical rankings. I'd normalize at index time and use dot product for efficiency." |
| 3 | "Bigger chunks are better because they have more context." | "Bigger chunks average multiple topics into one vector and match nothing precisely. I'd use small chunks for matching and parent–child retrieval to give the LLM context." |
| 4 | "Just add dense + BM25 + fuzzy scores together." | "Their scales are incompatible — BM25 is unbounded, cosine is [0,1], fuzzy is [0,100]. Either normalize and calibrate, or use RRF on ranks." |
| 5 | "HNSW is the best index." | "It depends on corpus size, memory budget, latency target and update rate. HNSW for low-latency in-memory serving; IVF-PQ when the corpus exceeds RAM; Flat for ground truth." |
| 6 | "Use the cross-encoder for retrieval, it's more accurate." | "A cross-encoder can't precompute document vectors, so scoring a full corpus needs one forward pass per document. It's strictly a second-stage reranker over a retriever's top-k." |
| 7 | "We just swap the embedding model when a better one comes out." | "Different models produce incomparable vector spaces, so a swap means re-embedding the whole corpus and rebuilding the index. I'd build the new index in parallel, shadow-test it, and cut over behind a flag." |
| 8 | "Recall@10 is 0.9, so the system works." | "Retrieval recall is a ceiling, not an outcome. I'd also measure reranker quality, answer faithfulness, citation accuracy and end-to-end latency." |
| 9 | "Gradient accumulation gives more negatives." | "It doesn't — each forward pass still sees only its own batch. Use CachedMultipleNegativesRankingLoss for a larger effective negative pool." |
| 10 | "We filter by permissions after retrieval." | "That silently shrinks result sets and can leak existence information. Access control must be a filter *inside* the retrieval query." |

---

# PART 15 — Interview Scenario Bank

### Scenario 1 — Design from scratch

> "Design a RAG system for 500,000 internal company documents — PDFs, Confluence pages, Slack exports. 200 employees will use it."

**Strong answer structure:**

"I'd start with questions: what latency do users expect, what's the access-control model, and how often does content change? Those three answers determine most of the architecture.

Assuming standard answers — sub-second, per-team permissions, daily updates — here's my design.

**Ingestion:** different parsers per source. PDFs need layout-aware parsing because internal PDFs are full of tables; Confluence has an API that gives me structure for free; Slack is conversational and needs thread-level rather than message-level chunking. I'd chunk on structure with breadcrumbs prepended, 400–600 tokens, parent–child so I can return wider context.

**Indexing:** dense embeddings plus BM25 from day one. 500k documents is maybe 3M chunks, which at 768 dims float32 is about 9 GB — that fits in RAM, so HNSW is fine and I don't need quantization yet. Every chunk carries team/permission labels as filterable metadata.

**Retrieval:** dense and BM25 in parallel, top 50 each, RRF fusion, then a cross-encoder rerank to top 8. Access filters applied *inside* both queries, not after.

**Generation:** strict grounding prompt, citations to chunk IDs, and a verification pass that the cited IDs exist.

**Evaluation:** before any of this ships, 150 queries collected from employees with labelled relevance. I measure Recall@50 to check the retriever's ceiling, NDCG@10 to check ranking, and a faithfulness score on generated answers.

**Where I expect to lose quality:** Slack. Conversational text with no structure and heavy context dependence is the hardest source here, and I'd consider excluding it from v1 rather than letting it drag down the whole system."

---

### Scenario 2 — Debug bad retrieval

> "Users say the answers are wrong. Where do you start?"

**Strong:** "I'd localise the failure before changing anything, because 'answers are wrong' has at least four distinct causes with different fixes.

Take 30 failed queries and for each one ask, in order:

1. **Does the answer exist in the corpus at all?** Surprisingly often it doesn't, and we're debugging a content problem as if it were a retrieval problem.
2. **Is it in the index correctly?** Find the chunk and read it. If the table was flattened or the reading order scrambled, that's a parsing bug.
3. **Is it retrieved in the top 50?** If not, that's a retriever problem — chunking, embedding model, or a missing BM25 leg.
4. **Is it in the top 5 after reranking?** If it was in the top 50 but not the top 5, that's a reranker problem.
5. **Did the LLM use it?** If the right chunk was in context and the answer is still wrong, that's a prompting or grounding problem.

Each of those five buckets has a completely different fix, and the counts tell me where to spend effort. Without this breakdown I'd just be randomly changing things."

---

### Scenario 3 — Scale

> "The corpus is growing from 1M to 500M chunks. What changes?"

**Strong:** "Almost everything about the storage layer; almost nothing about the retrieval logic.

At 1M chunks × 768 dims × 4 bytes, I'm at about 3 GB — HNSW in RAM, no compression needed. At 500M that's 1.5 TB of raw vectors, which doesn't fit anywhere sensible.

So: IVF-PQ instead of flat HNSW. With m=96 subquantizers that's roughly 48 GB, which fits. I'd add a rescore stage that fetches full-precision vectors from disk for the top few hundred candidates, which recovers most of the quantization loss. I'd also shard — probably by tenant or by time, whichever gives better locality — and accept that p99 latency is now set by the slowest shard, so I need per-shard timeouts and partial-result handling.

BM25 at that scale is a serious index in its own right; I'd use Elasticsearch or OpenSearch rather than anything hand-rolled.

Ingestion becomes a distributed pipeline problem: 500M chunks means embedding throughput is the bottleneck, so batching, GPU nodes, and a queue with resumability. And I'd need incremental re-indexing, because a full rebuild will take days.

The retrieval *logic* — hybrid, fuse, rerank — stays identical. That's the nice thing about the architecture."

---

### Scenario 4 — Latency

> "P99 is 2 seconds. Get it under 300ms."

**Strong:** "First I need the breakdown by stage, because the fixes are unrelated. A typical split might be: query embedding 50ms, ANN search 30ms, BM25 40ms, reranking 1,400ms, LLM time-to-first-token 400ms.

If that's the shape, reranking is 70% of the budget and everything else is noise. Fixes in order of preference: reduce the candidate set from 100 to 30 (measure the NDCG cost — it's often small), use a smaller distilled reranker, batch the pairs properly on GPU if we aren't already, and serve it with ONNX or TensorRT.

If ANN search dominates instead, I lower `ef_search` or `nprobe` and measure the recall cost.

Two things I'd check regardless: are we running dense and BM25 sequentially when they could be parallel — that's free latency — and are we paying a network round-trip to a hosted embedding API per query, which caching common queries would largely remove.

And I'd be explicit about the trade: getting to 300ms will cost some quality. I'd want the product owner to see the quality-vs-latency curve and choose the point, rather than me picking silently."

---

### Scenario 5 — The exact-match failure

> "Users search for product codes like 'XR-4471-B' and get irrelevant results. The embedding model is state of the art."

**Strong:** "This is expected behaviour, not a bug in the model. Embedding models are trained to map *similar* things close together. For an identifier, similarity is precisely the wrong objective — 'XR-4471-B' and 'XR-4472-B' are maximally similar as strings and completely different as products. Worse, the tokenizer probably splits the code into fragments that carry no meaning.

The fix is BM25, where a rare token like that has a huge IDF and dominates the score. I'd add a lexical leg and fuse with RRF. I'd go further and add an entity-detection step in query processing: if the query matches a product-code pattern, I'd route it as a structured lookup against the product table entirely, and skip semantic search for that part.

I'd also add fuzzy matching, because users typo codes constantly and OCR'd documents contain O/0 and I/1 confusions.

The general lesson: dense retrieval and lexical retrieval fail on different query types, which is why production search is always hybrid."

---

### Scenario 6 — Freshness

> "Documents are updated hourly. Users complain they see stale answers."

**Strong:** "Several possible causes and I'd check them in order.

First, is the index actually updating? An hourly job that silently fails is the most common version of this problem. I'd check ingestion lag as a monitored metric.

Second, deletes. Many ANN indexes — HNSW especially — tombstone deleted vectors rather than removing them, so an updated document can leave its old version retrievable. I'd verify the delete path works and schedule periodic compaction.

Third, caching. If we cache query results or embeddings, a stale cache looks exactly like a stale index.

Fourth, versioning in the data itself: if the corpus contains both the old and new version of a document and nothing distinguishes them, the retriever may legitimately return the old one. I'd add a `version` and `is_current` field and filter on it at query time.

Architecturally, for a genuinely high-churn corpus I'd consider a two-tier index: a large periodically-rebuilt main index plus a small fresh index over recent changes, searching both and merging. That decouples freshness from rebuild cost."

---

# Cheat Sheet

```text
PIPELINE
  ingest → parse/OCR → preserve structure → chunk + metadata → embed + BM25 index
  query  → validate → route/rewrite → dense + BM25 + fuzzy → RRF → rerank → LLM + cite

BM25          lexical. TF saturation (k₁), length norm (b), rarity (IDF).
              Wins on IDs, codes, rare exact terms.
DENSE         semantic. Wins on paraphrase and synonym.
FUZZY         character-level. Wins on typos, OCR noise, name variants.
HYBRID        because they fail on DIFFERENT queries.

RRF(d) = Σ  w_r / (k + rank_r(d))          k ≈ 60
          r
  Combines RANKS, not scores. Rewards consensus. Discards magnitude.

RERANKER      cross-encoder reads (query, passage) together. Accurate, not precomputable.
              Second stage only. Ceiling = the retriever's Recall@k.

ANN
  Flat   exact, ground truth, <100k
  IVF    nlist = build-time partitions (≈4√N to 16√N); nprobe = query-time knob
  HNSW   M, ef_construction = build; ef_search = query-time knob; memory-hungry

METRICS ON NORMALIZED VECTORS
  cosine = dot product;  L2² = 2 − 2·cos  → all three give the SAME ranking

COMPRESSION
  float16   2×    scalar int8  4×    PQ  16-64×    binary  32×
  IVF-PQ = prune + compress. Always pair aggressive compression with a rescore stage.

EVALUATION
  Recall@K   a CEILING — the reranker and LLM cannot recover what wasn't retrieved
  MRR        1/rank of the first relevant result, averaged
  NDCG@K     Σ (2^rel − 1)/log₂(i+1), normalized by the ideal ordering
  plus:      faithfulness, citation accuracy, latency, cost

THE PATTERN THAT REPEATS EVERYWHERE
  retrieve wide and cheap → narrow → score precisely and expensively
```

---

## One-line interview summary

> "Use structure-aware ingestion to create context-rich chunks, combine dense and lexical retrieval because they fail on different queries, add fuzzy matching when entities are noisy, fuse heterogeneous rankings with RRF, rerank a small candidate set with a cross-encoder, and generate only from verified retrieved evidence — then measure Recall@K as a ceiling and faithfulness as the outcome."

---

## Where to go next

- **Document 01** — choosing the embedding model.
- **Document 02** — fine-tuning the retriever and the reranker.
- **Document 04** — what's inside the models: transformer encoders/decoders, quantization math, LoRA and QLoRA.
