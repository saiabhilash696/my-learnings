# Fine-Tuning Embedding Models — A Beginner's Guide (Interview Edition)

**Prerequisite:** document 01 of this set (what an embedding is, cosine similarity, MTEB).

**What you'll be able to do after this:** explain what fine-tuning an embedder actually changes, compute a contrastive loss by hand, choose the right loss for the data you have, mine hard negatives, and answer the standard interview question *"how would you fine-tune an embedding model for RAG?"*

Library used throughout: [`sentence-transformers`](https://sbert.net/).

```bash
pip install -U sentence-transformers datasets accelerate scikit-learn
```

---

# PART 1 — The Mental Model

## 1.1 What does fine-tuning an embedder actually do?

An embedding model maps text to points in space. Fine-tuning **moves the points**.

Before fine-tuning, a general model has learned general English. It knows "dog" and "puppy" are close. It does *not* know that at your company:

```text
"P1 incident"  and  "Sev-1 outage"      mean the same thing
"churn"        and  "logo retention"     are opposites
"the platform" means your internal deployment system, not a train station
```

Fine-tuning takes pairs of texts *you* say should be close, and pulls them together. It takes texts that should be apart, and pushes them apart.

```text
BEFORE                              AFTER
                                    
   "P1 incident"                       "P1 incident"
        •                                   •• ← now adjacent
                                    
              •  "Sev-1 outage"       
                                       
   •  "billing question"                •  "billing question"
```

That's it. Everything else in this document is machinery for doing that well.

> **Key point for interviews:** fine-tuning an embedder is *far* cheaper and easier than fine-tuning an LLM. The models are small (often 110M–350M parameters), the objective is stable, and a few thousand examples is enough. On a specialised corpus it usually beats swapping to a bigger off-the-shelf model.

## 1.2 What you need before you start

```text
1. A frozen evaluation set        ← build this FIRST, before any training
2. A baseline number on it        ← the un-fine-tuned model's score
3. Training pairs                 ← (query, relevant passage)
```

If you skip steps 1 and 2, you will have no way to tell whether the fine-tune helped. This is the most common beginner mistake, and interviewers ask about it deliberately.

---

# PART 2 — The Math, Worked by Hand

This section is the difference between "I ran the training script" and "I understand what happened." Interviewers can tell.

## 2.1 The default retrieval loss: MultipleNegativesRankingLoss (MNRL)

Also called **InfoNCE** or **in-batch contrastive loss**. Here is how it works.

You feed the trainer a batch of positive pairs:

```text
a1 = "How do I reset my password?"       p1 = "Go to Settings > Security > Reset Password."
a2 = "What is the refund window?"         p2 = "Refunds are issued within 14 days of purchase."
a3 = "How do I change my email alerts?"   p3 = "Go to Settings > Notifications to manage alerts."
```

The model embeds all six texts and computes a **similarity matrix** — every anchor against every positive:

```text
          p1      p2      p3
  a1    0.72    0.15    0.58
  a2    0.11    0.81    0.09
  a3    0.55    0.12    0.77
```

The diagonal (0.72, 0.81, 0.77) is the correct pairing. Everything off the diagonal is a **free negative**: `p2` and `p3` are not the answer to `a1`, so they act as negatives at zero labelling cost.

The loss then says: **treat each row as a classification problem where the correct class is the diagonal.**

### Step 1 — Scale the similarities

Cosine similarities live in `[-1, 1]`. That range is far too narrow for a softmax to produce a useful gradient. So `sentence-transformers` multiplies by a **scale** (default `20.0`, equivalent to a temperature of 1/20 = 0.05).

```text
Row a1, raw:     [0.72,  0.15,  0.58]
Row a1, ×20:     [14.4,   3.0,  11.6]
```

### Step 2 — Softmax

```text
Subtract the max (14.4) for numerical stability:  [0, -11.4, -2.8]
exp:                                              [1.0000, 0.0000112, 0.0608]
sum:                                              1.0608
probabilities:                                    [0.9427, 0.0000105, 0.0573]
```

### Step 3 — Cross-entropy loss for this row

```text
loss(a1) = -ln(probability of the correct answer)
         = -ln(0.9427)
         = 0.0590
```

The batch loss is the average across all rows. Low loss = the model already ranks the true positive above the in-batch negatives.

### Why the scale matters — see it for yourself

Run the same row with `scale = 1`:

```text
raw:       [0.72, 0.15, 0.58]
exp:       [2.054, 1.162, 1.786]
sum:       5.002
p(correct) = 0.411
loss       = -ln(0.411) = 0.89
```

Loss of 0.89 instead of 0.059 — and critically, the gradient barely distinguishes 0.72 from 0.58. Without the scale, the model can't tell "clearly right" from "close enough." **The temperature is what sharpens the contrast.**

> **Interview question you can now answer:** *"Why does contrastive loss need a temperature?"* — "Because cosine similarity is bounded to [-1,1], which is too compressed for softmax to separate. Scaling by ~20 (temperature 0.05) sharpens the distribution so the gradient concentrates on the hardest negatives instead of spreading uniformly."

### Why hard negatives matter — the same math explains it

Look at row `a1` again: `[0.72, 0.15, 0.58]`.

- `p2` ("refund window") scores 0.15. Its softmax probability is 0.00001. **It contributes essentially zero gradient.** It's an easy negative — the model already knows it's wrong, so there's nothing to learn.
- `p3` ("Settings > Notifications") scores 0.58. Its probability is 0.057. **This is where the gradient goes.** The model is genuinely confused because both texts contain "Go to Settings >".

**That is the entire justification for hard-negative mining.** Easy negatives waste batch slots. Hard negatives carry the learning signal.

### Why batch size matters for MNRL

Every extra item in the batch is one more negative for every anchor. Batch size 32 gives you 31 negatives per anchor; batch size 256 gives you 255. More negatives = a harder, more informative classification problem.

But note carefully:

> **Gradient accumulation does NOT increase the number of in-batch negatives.** `gradient_accumulation_steps=8` runs eight separate forward passes and sums their gradients. Each forward pass still only sees its own batch's negatives. If you need a bigger negative pool and you're memory-limited, use `CachedMultipleNegativesRankingLoss`, which uses gradient caching (GradCache) to get a large effective batch without holding every activation in memory.

This is a very common interview trap. People say "I'll just use gradient accumulation" and it's wrong.

### The false-negative danger

In-batch negatives are assumed to be negative. If two queries in the same batch genuinely share a relevant passage, you are actively training the model to push apart two things that belong together.

Mitigation: `batch_sampler=BatchSamplers.NO_DUPLICATES`, which prevents duplicate texts landing in the same batch.

---

# PART 3 — Pick the Loss First; the Data Format Follows

The loss and the data shape must agree. Decide the objective **before** you collect annotations, or you will annotate the wrong thing.

| Loss | Training data | Columns | Use when |
|---|---|---|---|
| `MultipleNegativesRankingLoss` (InfoNCE) | Positive pairs, optional explicit negatives | `anchor`, `positive`, [`negative`, `negative_1`…] | **Default for retrieval / RAG.** In-batch negatives are free. |
| `CachedMultipleNegativesRankingLoss` | Same | Same | Same, but you need a much larger effective batch than memory allows. |
| `GISTEmbedLoss` / `CachedGISTEmbedLoss` | Same | Same | Large or noisy corpora where in-batch false negatives are likely. A frozen "guide" model screens out in-batch items that are actually relevant before they're used as negatives. |
| `TripletLoss` | Explicit triplets | `anchor`, `positive`, `negative` | You already have curated triplets and want margin-based training. |
| `ContrastiveLoss` | Binary-labelled pairs | `sentence1`, `sentence2`, `label` (0/1) | Duplicate detection, "are these the same thing?" |
| `CosineSimilarityLoss` / `CoSENTLoss` | Graded pairs | `sentence1`, `sentence2`, `score` (float) | STS-style regression with real-valued similarity targets. `CoSENTLoss` usually outperforms `CosineSimilarityLoss` on the same data. |
| `BatchHardSoftMarginTripletLoss` | Single texts with class IDs | `sentence`, `label` | Classification-oriented embeddings and supervised clustering. |
| `SoftmaxLoss` | Labelled sentence pairs | `sentence1`, `sentence2`, `label` | Pair classification (NLI). **Legacy** — see §7.3. |
| `MatryoshkaLoss` | *Wraps another loss* | Same as the wrapped loss | Embeddings that stay useful after truncation. |

### The one-line decision rule

```text
Do you have (query, relevant_passage) pairs?        → MultipleNegativesRankingLoss
Do you have (text, class_id)?                       → BatchHardSoftMarginTripletLoss
Do you have (text_a, text_b, 0/1)?                  → ContrastiveLoss
Do you have (text_a, text_b, 0.0-1.0 score)?        → CoSENTLoss
Do you have unlabelled domain text only?            → SimCSE-style MNRL (§8.3)
Do you need a reranker, not a retriever?            → CrossEncoder + BinaryCrossEntropyLoss (§9)
```

---

# PART 4 — Data Formats

## 4.1 Positive pairs (the normal case)

**`train_pairs.jsonl`**

```json
{"anchor": "How do I reset my password?", "positive": "To reset your password, go to Settings > Security > Reset Password."}
{"anchor": "What's the refund policy?", "positive": "Refunds are issued within 14 days of purchase if the item is unused."}
```

No explicit negatives needed. The batch supplies them.

## 4.2 Explicit hard negatives

**`train_triplets.jsonl`**

```json
{"anchor": "How do I reset my password?", "positive": "Go to Settings > Security > Reset Password.", "negative": "Go to Settings > Notifications to manage email alerts."}
```

This also works with `MultipleNegativesRankingLoss` — the explicit negative is added *on top of* the in-batch negatives, not instead of them.

For multiple negatives, `mine_hard_negatives(..., output_format="n-tuple")` produces `negative_1`, `negative_2`, … By default, `mine_hard_negatives` returns the simpler three-column triplet.

## 4.3 Binary pairs

```json
{"sentence1": "The cat sat on the mat.", "sentence2": "A cat was sitting on a mat.", "label": 1}
{"sentence1": "The cat sat on the mat.", "sentence2": "The stock market fell today.", "label": 0}
```

## 4.4 Graded pairs

```json
{"sentence1": "A man is playing guitar.", "sentence2": "A person plays a musical instrument.", "score": 0.85}
{"sentence1": "A man is playing guitar.", "sentence2": "A woman is cooking dinner.", "score": 0.05}
```

The score range depends on your labelling scheme. Normalise consistently; here it's 0.0–1.0.

## 4.5 A detail that confuses everyone

Sentence Transformers does **not** match columns by name. It treats columns named `label` / `labels` / `score` / `scores` as labels, and passes **all remaining columns to the model in order**.

So a dataset with columns `question | answer` works fine with `MultipleNegativesRankingLoss` — the first column is the anchor, the second is the positive, purely by position.

Use descriptive names (`anchor`, `positive`, `negative`) anyway, because it makes your dataset self-documenting and prevents a silent column-order bug six months later.

> **A silent bug worth knowing:** if you accidentally swap the column order in a triplet dataset, training will run happily and produce a model that's worse than the baseline. Nothing errors. This is why you always measure against the frozen eval set.

---

# PART 5 — Prompt Conventions (the silent point-killer)

Many embedding models were trained with a specific query/document prefix. If you use the wrong one — or use them at training time but forget at inference time — you lose several points of retrieval quality with no error message.

> **The rule: use the exact convention documented by the specific checkpoint, and keep it identical from training through to production.**

| Model / family | Convention | Example |
|---|---|---|
| **BGE v1.5** | Query gets a retrieval instruction; passages do not. | Query: `Represent this sentence for searching relevant passages: How do I reset my password?` <br> Passage: raw text |
| **E5** | `query:` and `passage:` prefixes for asymmetric retrieval. | `query: How do I reset my password?` <br> `passage: Go to Settings > Security > Reset Password.` |
| **Qwen3-Embedding** | Query-side instruction via named prompts; documents plain. | `model.encode(queries, prompt_name="query")` <br> `model.encode(documents)` |
| **Jina Embeddings v3/v4** | Task-specific adapters rather than string prefixes. | `task="retrieval.query"` / `task="retrieval.passage"` |
| **Everything else** | **Do not generalise by family name.** Read the model card. | — |

> **Do not copy one model's convention to another.** E5's `query:` is not BGE's instruction, and applying E5 prefixes to a BGE model actively hurts. Nomic, GTE, MiniLM and others each have their own rules — or none.

### Why "asymmetric" matters

In retrieval, the query ("how do I reset my password") and the document ("To reset your password, navigate to...") look different. Queries are short and interrogative; documents are long and declarative. Models handle this by encoding them slightly differently — hence the prefixes. This is **asymmetric retrieval**.

In duplicate detection ("are these two support tickets the same?") both sides are the same kind of text. That's **symmetric**, and prefixes usually aren't used.

---

# PART 6 — The Training Script

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# 1. Base model. Start small — bge-base is 110M params and trains in minutes.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 2. Data. Each row = one positive pair.
train_dataset = load_dataset("json", data_files="train_pairs.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="eval_pairs.jsonl",  split="train")

# 3. Loss.
loss = MultipleNegativesRankingLoss(model)

# 4. Training configuration.
args = SentenceTransformerTrainingArguments(
    output_dir="output/bge-finetuned",

    num_train_epochs=3,
    per_device_train_batch_size=32,     # more negatives per anchor = better signal
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                 # a starting point, not a magic number
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    # Prevents duplicate texts in a batch becoming accidental false negatives.
    batch_sampler=BatchSamplers.NO_DUPLICATES,

    eval_strategy="epoch",
    save_strategy="epoch",              # must match eval_strategy for load_best_model_at_end
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=25,

    # Pick ONE precision mode your hardware supports.
    # fp16 for older NVIDIA GPUs (T4, V100); bf16 for Ampere+ (A100, RTX 30xx/40xx).
    fp16=True,
)

# 5. Train.
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()

# 6. Save.
model.save_pretrained("output/bge-finetuned/final")
```

### A deliberate design choice in this script

The trainer evaluates using the **training loss** on the held-out set. Retrieval quality is measured **separately** with `InformationRetrievalEvaluator` (§7).

This avoids a mistake you'll see in a lot of blog posts: feeding `[1.0] * len(eval_dataset)` into `EmbeddingSimilarityEvaluator` and reporting the resulting correlation. If every label is 1.0, there is no variance, and the correlation is meaningless. `EmbeddingSimilarityEvaluator` is for genuine graded similarity scores only.

### Hyperparameters: what to actually do

| Parameter | Starting point | How to think about it |
|---|---|---|
| `learning_rate` | `2e-5` | Too high → the model forgets its general knowledge. Too low → nothing changes. Try `1e-5`, `2e-5`, `5e-5`. |
| `num_train_epochs` | 1–3 | Small datasets overfit fast. Watch the held-out retrieval metric, not the loss. |
| `per_device_train_batch_size` | as large as fits | Directly controls the number of in-batch negatives for MNRL. |
| `warmup_ratio` | 0.1 | Prevents a large early update from wrecking the pretrained weights. |
| Precision | `bf16=True` on Ampere+ | `bf16` has the same exponent range as fp32, so it's less prone to overflow than `fp16`. |

---

# PART 7 — Evaluation (do not skip this)

## 7.1 The right question

Training loss going down means the model got better at the training objective. That is **not** the same as "retrieval improved."

The real question:

> **Does the right chunk appear in the top-k for a realistic query?**

## 7.2 InformationRetrievalEvaluator

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = {
    "q1": "How do I reset my password?",
    "q2": "What's the refund policy?",
}

corpus = {
    "d1": "To reset your password, go to Settings > Security > Reset Password.",
    "d2": "Refunds are issued within 14 days of purchase if the item is unused.",
    "d3": "Go to Settings > Notifications to manage email alerts.",
}

relevant_docs = {
    "q1": {"d1"},
    "q2": {"d2"},
}

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="domain-eval",
    accuracy_at_k=[1, 5, 10],
    ndcg_at_k=[10],
    map_at_k=[100],
)

results = ir_evaluator(model)
print(results)
```

In reality your corpus should be your *actual* corpus (thousands of chunks), not three toy documents. A tiny corpus makes retrieval look artificially easy.

## 7.3 The metrics, in plain language

| Metric | Question it answers | When to use it |
|---|---|---|
| **Recall@k** | Did a relevant document make it into the top k at all? | The most important metric for RAG — if the chunk isn't in the candidate set, the reranker and the LLM can't recover it. |
| **MRR** | How high was the *first* relevant result? | When users only look at the top answer. |
| **NDCG@k** | How good is the whole ranking, weighting by graded relevance? | When you have graded labels (2/1/0) rather than binary. |
| **MAP@k** | How well are *all* relevant docs ranked? | When multiple documents are relevant per query. |

## 7.4 The protocol that makes results trustworthy

```text
1. Build the eval set BEFORE training.
2. FREEZE it. Never train on it, never tune hyperparameters against it and then report it.
3. Measure the BASE model on it. Write the number down.
4. Train.
5. Measure the FINE-TUNED model on the same set.
6. Report the delta.
```

If you can't state "Recall@10 went from 0.61 to 0.78 on a frozen 200-query held-out set," you don't know whether the fine-tune worked.

---

# PART 8 — Hard Negative Mining

## 8.1 Why

From §2.1: easy negatives contribute almost no gradient. If your batch is full of obviously-unrelated text, the model learns very little per step.

A hard negative is a document that **looks plausible but is wrong**.

```text
Query:            "How do I reset my password?"
Positive:         "Go to Settings > Security > Reset Password."
HARD negative:    "Go to Settings > Notifications to manage email alerts."   ← similar words, wrong answer
EASY negative:    "Our Q3 revenue grew 12% year over year."                  ← model already knows this is wrong
FALSE negative:   "Navigate to Security settings to change your password."   ← actually correct! DO NOT USE
```

That third category is the danger. Training on a false negative teaches the model that a correct answer is wrong.

## 8.2 The pipeline

```text
positive pairs
     ↓
embed the corpus with a strong base model
     ↓
for each query, retrieve the top-N most similar documents
     ↓
exclude the known positive
     ↓
FILTER: drop candidates that are suspiciously similar to the positive (likely false negatives)
     ↓
train with MNRL + explicit hard negatives
```

## 8.3 Code

```python
from sentence_transformers.util import mine_hard_negatives

hard_negative_dataset = mine_hard_negatives(
    dataset=train_dataset,
    model=model,                  # use a STRONG model for mining, not a weak one
    num_negatives=5,
    range_min=10,                 # skip the top 10 — they're probably true positives
    range_max=50,                 # only consider ranks 10-50 as negative candidates
    relative_margin=0.05,         # negative must be at least 5% less similar than the positive
    sampling_strategy="top",
    batch_size=128,
    use_faiss=True,
    output_format="n-tuple",      # -> anchor, positive, negative_1 ... negative_5
)
```

## 8.4 What each parameter is protecting you from

**`range_min=10`** — The top-ranked "negatives" are frequently just other correct answers your annotators didn't label. Skipping the top 10 is cheap insurance against false negatives.

**`range_max=50`** — Beyond rank 50 the candidates get easy, so there's no point mining them.

**`relative_margin=0.05`** — Requires `sim(negative) < sim(positive) × (1 − 0.05)`. As a *fraction* of the positive score.

**`absolute_margin`** — The alternative: `sim(negative) < sim(positive) − absolute_margin`. A fixed gap.

> **Which margin to use:** prefer `absolute_margin` when your positive similarities cluster in a narrow band (a fixed gap is easier to reason about). Prefer `relative_margin` when positive similarities vary a lot across examples. You can set both — a candidate must clear whichever margins you configured.

**`sampling_strategy="top"`** — Takes the hardest candidates in the allowed window. `"random"` samples randomly among those that pass the filters, which gives more diverse but easier negatives.

**`use_faiss=True`** — Uses FAISS for the similarity search. Matters a lot above ~100k documents.

> **The single most important hard-negative rule: a false negative is worse than an easy negative.** An easy negative wastes a batch slot. A false negative actively teaches the model something wrong. When in doubt, mine more conservatively.

### Scenario

> **Interviewer:** "You mined hard negatives and your retrieval got *worse*. What happened?"
>
> **Strong answer:** "Most likely false negatives. If the mining window starts at rank 1, the top candidates are usually just other genuinely relevant passages that weren't in the labelled set — especially if the corpus has near-duplicate documents, which is very common when the same policy appears in an FAQ, a help article and a PDF. I'd set `range_min` to 10 or higher and add a margin filter so anything close in similarity to the positive is rejected. I'd also manually inspect 50 mined negatives — that takes twenty minutes and usually makes the problem obvious. Second possibility: the negatives are fine but I'm training too long or at too high a learning rate, and the model is overfitting to the mined distribution rather than generalising."

---

# PART 9 — Matryoshka Fine-Tuning

If you need smaller vectors later, train for it now. `MatryoshkaLoss` teaches the model that the first 512, 256, 128 or 64 dimensions of its output should each be a usable embedding on their own.

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss

base_loss = MultipleNegativesRankingLoss(model)

loss = MatryoshkaLoss(
    model,
    base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
)
```

Then pass `loss` to the trainer exactly as before.

**Two things people get wrong:**

1. `MatryoshkaLoss` is **not a data format**. It's a wrapper around another loss. Your dataset must match the *wrapped* loss.
2. The dimensions must be valid for your model. Don't copy `[768, 512, ...]` onto a model whose native dimension is 384.

**Why it's worth it:** it costs you one flag at training time and gives you a permanent 4× or 12× storage lever later. See document 01 §2.6 for the storage math.

---

# PART 10 — Classification and Clustering

The same fine-tuned encoder can serve retrieval, classification and clustering. What changes is the loss you train with and the algorithm you put on top.

```text
The loss shapes the geometry of the vector space.
The final application algorithm is a separate, swappable layer.
```

## 10.1 Classification, embedding-first

Two different things get called "classification":

1. **Embedding-first:** train an encoder whose space separates classes, then put a cheap classifier (logistic regression, k-NN, nearest centroid) on top. The embedding stays reusable.
2. **Direct:** train a dedicated classifier. Simpler if you only need the label and will never need the vectors.

This guide covers embedding-first.

**Data — `train_classification.jsonl`:**

```json
{"sentence": "My debit card was declined at checkout.", "label": 0}
{"sentence": "The card payment did not go through.", "label": 0}
{"sentence": "How can I update my login password?", "label": 1}
{"sentence": "I forgot my password and need to change it.", "label": 1}
{"sentence": "How do I get a refund for an order?", "label": 2}
{"sentence": "I want my purchase money back.", "label": 2}
```

```text
0 = payment_failed    1 = password_reset    2 = refund
```

**Loss:** `BatchHardSoftMarginTripletLoss` — for each anchor in the batch, it finds the hardest same-class example (the one furthest away) and the hardest different-class example (the one closest), and pushes accordingly.

```text
same class      → pull closer
same class      → pull closer
                ↘
                 anchor
                ↗
different class → push away
```

Requirements: at least two examples per class, and `BatchSamplers.GROUP_BY_LABEL` so batches actually contain multiple examples of each class (otherwise there is no same-class example to pull toward).

**Sibling losses worth knowing** — all take the same `(sentence, label)` data:

- `BatchHardTripletLoss` — same hardest-in-batch mining, but with a fixed numeric `margin`. Use when you want direct control.
- `BatchAllTripletLoss` — averages over *all* valid triplets instead of only the hardest. Noisier gradient, sometimes more stable on small or messy label sets.
- `BatchSemiHardTripletLoss` — restricts to semi-hard triplets (negative further than positive, but within the margin). A middle ground.

`BatchHardSoftMarginTripletLoss` is a good default because it removes the margin hyperparameter entirely. If it's unstable, the other three are your first alternatives.

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses,
)
from sentence_transformers.training_args import BatchSamplers

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train_dataset = load_dataset("json", data_files="train_classification.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="eval_classification.jsonl",  split="train")

loss = losses.BatchHardSoftMarginTripletLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="output/classification-encoder",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    batch_sampler=BatchSamplers.GROUP_BY_LABEL,   # ← essential for this loss
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
)

trainer = SentenceTransformerTrainer(
    model=model, args=args,
    train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss,
)
trainer.train()
model.save_pretrained("output/classification-encoder/final")
```

**Then put a classifier on top:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = SentenceTransformer("output/classification-encoder/final")

X_train = model.encode(train_dataset["sentence"], normalize_embeddings=True)
X_eval  = model.encode(eval_dataset["sentence"],  normalize_embeddings=True)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_dataset["label"])
preds = clf.predict(X_eval)

print("Accuracy:", accuracy_score(eval_dataset["label"], preds))
print("Macro-F1:", f1_score(eval_dataset["label"], preds, average="macro"))
print(classification_report(eval_dataset["label"], preds))
```

> Use **macro-F1** when classes are imbalanced and you want each class to count equally. Accuracy on a dataset that's 90% one class will look great while the model ignores the other classes entirely.

**Zero-training alternative — nearest centroid.** With a well-separated space you may not need a trained classifier at all:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("output/classification-encoder/final")

class_examples = {
    0: ["My debit card was declined.", "The payment failed at checkout."],
    1: ["I forgot my password.", "How can I reset my login password?"],
    2: ["I want a refund.", "How can I get my purchase refunded?"],
}

centroids = {}
for label, texts in class_examples.items():
    vectors = model.encode(texts, normalize_embeddings=True)
    centroid = vectors.mean(axis=0)
    centroids[label] = centroid / np.linalg.norm(centroid)   # re-normalise after averaging

text = "My card payment was rejected."
vector = model.encode([text], normalize_embeddings=True)[0]

scores = {label: float(vector @ c) for label, c in centroids.items()}
print(max(scores, key=scores.get), scores)
```

This shines when you need to add a new class without retraining — just add its examples and compute a centroid.

## 10.2 Clustering

Sentence Transformers does not replace K-Means. It produces the **space** that K-Means operates on.

**With labels:** identical setup to classification above — use the class labels to shape the space, then cluster.

**Then cluster:**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

embeddings = model.encode(texts, normalize_embeddings=True)
cluster_ids = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(embeddings)

print("ARI:", adjusted_rand_score(true_labels, cluster_ids))    # needs ground truth
print("Silhouette:", silhouette_score(embeddings, cluster_ids, metric="cosine"))  # doesn't
```

**Choosing the algorithm:**

| Situation | Start with |
|---|---|
| You know `k` | K-Means |
| You don't know `k`, want a hierarchy | Agglomerative (practical up to a few thousand items) |
| Irregular cluster shapes, noise points | HDBSCAN |
| Very large corpus, want near-duplicate groups | `sentence_transformers.util.community_detection` |

```python
from sentence_transformers import util

embeddings = model.encode(large_corpus, batch_size=128, convert_to_tensor=True)

# Returns communities (lists of corpus indices), largest first.
# The first index of each community is its most central point.
clusters = util.community_detection(
    embeddings,
    threshold=0.75,          # cosine cutoff for "same community"
    min_community_size=10,
)
```

This handles tens of thousands of sentences in seconds without needing `k`, at the cost of some precision. Validate `threshold` on a labelled sample.

## 10.3 No labels at all: SimCSE domain adaptation

If you have lots of domain text but no labels, you can still adapt the model to your domain. **Unsupervised SimCSE** encodes the *same sentence twice* — dropout makes the two passes slightly different, and those two views are treated as a positive pair.

```python
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

sentences = [
    "The customer could not complete the card payment.",
    "The user forgot the account password.",
    "The order was delivered two days late.",
    # ... many thousands of unlabelled domain sentences ...
]

# Same sentence in BOTH columns. Dropout creates the two views.
train_dataset = Dataset.from_dict({"sentence1": sentences, "sentence2": sentences})

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="output/domain-adapted-encoder",
    num_train_epochs=1,
    per_device_train_batch_size=128,
    fp16=True,
)

trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
trainer.train()
```

> **Be honest about what this does.** SimCSE adapts the representation to your *vocabulary and style*. It does not discover your business's "correct" clusters — it has no idea what those are. You still need a clustering algorithm afterwards, and you should still inspect the result.

## 10.4 What about SoftmaxLoss?

`SoftmaxLoss` was the *original* Sentence-BERT objective, for sentence **pairs** with a categorical relationship — NLI (entailment / neutral / contradiction).

**It is now a legacy option.** The modern approach for NLI-style data is to reshape the triples and use MNRL, treating entailment as the positive and contradiction as a hard negative:

```python
from datasets import Dataset
from sentence_transformers.losses import MultipleNegativesRankingLoss

train_dataset = Dataset.from_dict({
    "anchor":        ["A soccer game is being played."],
    "entailment":    ["People are playing football."],
    "contradiction": ["The field is completely empty."],
})

loss = MultipleNegativesRankingLoss(model)   # columns read positionally: anchor, positive, negative
```

This produces stronger general-purpose embeddings than the softmax classifier head.

Reach for `SoftmaxLoss` only when you specifically need a classifier head that outputs a *category* rather than a reusable embedding — and even then, a small classifier on top of MNRL-trained embeddings (§10.1) is usually more flexible.

**The distinction to hold in your head:**

```text
Single-text topic classification          Pair classification / NLI
(sentence, class_id)                      (sentence1, sentence2, relationship_id)
        ↓                                         ↓
batch-label metric learning               pair-classification loss
        ↓                                         ↓
embedding + light classifier              relationship prediction
```

> **Interview answer:** "For topic classification with one text and one label, I'd shape the embedding space with class-label metric learning and put a lightweight classifier on top, which keeps the embeddings reusable. For pairwise NLI-style classification, the modern approach is to reshape into anchor/entailment/contradiction and train with MNRL rather than the legacy SoftmaxLoss."

---

# PART 11 — Rerankers (Cross-Encoders)

Everything above trains a **bi-encoder**: query and passage are embedded *separately*, so document vectors can be precomputed and indexed. That's what makes searching millions of documents cheap.

A **cross-encoder** feeds `(query, passage)` through the model **together**, so attention can directly compare the two texts token by token.

```text
BI-ENCODER (retriever)                  CROSS-ENCODER (reranker)

query  → [encoder] → vector             query  ─┐
                        ↓                        ├→ [encoder] → relevance score
passage → [encoder] → vector            passage ─┘
                        ↓
                  cosine similarity

✓ precompute all document vectors       ✗ nothing can be precomputed
✓ millions of docs, milliseconds        ✗ one forward pass per (query, doc) pair
✗ query and doc never "see" each other  ✓ far more accurate
```

**Never use a cross-encoder for first-stage retrieval.** Scoring a query against 2 million documents means 2 million forward passes — seconds to minutes per query.

The production pattern is two stages: bi-encoder retrieves the top 50–200 cheaply, cross-encoder re-scores just those.

## 11.1 Training data

| Use case | `num_labels` | Data shape |
|---|---|---|
| **Reranking** (binary or graded relevance) | `1` | `(query, passage, label)` — label is 0/1 or a graded float |
| **Pair classification** (NLI) | number of classes | `(sentence1, sentence2, class_id)` |

Mixing these up is a common source of confusing loss curves.

**`train_reranker.jsonl`:**

```json
{"query": "How do I reset my password?", "passage": "Go to Settings > Security > Reset Password.", "label": 1}
{"query": "How do I reset my password?", "passage": "Go to Settings > Notifications to manage email alerts.", "label": 0}
```

**Good news:** if you already have `(anchor, positive)` pairs from §4.1, `mine_hard_negatives()` can emit exactly this format. No re-annotation needed.

## 11.2 Training

```python
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.util import mine_hard_negatives

# Rerankers don't need to be large — they only compare a handful of candidates.
model = CrossEncoder("microsoft/mpnet-base", num_labels=1)   # num_labels=1 → reranking head

train_dataset = load_dataset("json", data_files="train_pairs.jsonl", split="train")

# Reuse the same mining pipeline; just change the output format.
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train_dataset = mine_hard_negatives(
    train_dataset, embedding_model,
    num_negatives=5, range_min=10, range_max=100,
    sampling_strategy="top", batch_size=128, use_faiss=True,
    output_format="labeled-pair",       # → (query, passage, label)
)

# pos_weight upweights the rare positives when negatives greatly outnumber them.
loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(5))

args = CrossEncoderTrainingArguments(
    output_dir="output/reranker-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
)

evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=32)

trainer = CrossEncoderTrainer(
    model=model, args=args, train_dataset=train_dataset, loss=loss, evaluator=evaluator,
)
trainer.train()
model.save_pretrained("output/reranker-finetuned/final")
```

## 11.3 Listwise ranking with LambdaLoss

If your data is naturally "one query, many candidates with graded relevance" (click logs, human relevance judgments), a listwise learning-to-rank loss often beats pairwise BCE.

```python
from sentence_transformers.cross_encoder.losses import LambdaLoss

# labeled-list groups all candidates per query:
# {"query": ..., "docs": [d1, d2, ...], "labels": [1, 0, 0, ...]}
train_dataset = mine_hard_negatives(
    train_dataset, embedding_model,
    num_negatives=10, output_format="labeled-list", use_faiss=True,
)

loss = LambdaLoss(model)
# NOTE: each sample now contains N docs, so use a smaller batch size (e.g. 16).
```

Other listwise/pairwise losses consuming the same `labeled-list` shape: `RankNetLoss`, `ListNetLoss`, `ListMLELoss`, `PListMLELoss`. The library's own MS MARCO write-up found `LambdaLoss` and `BinaryCrossEntropyLoss` strongest, anecdotally `LambdaLoss > PListMLELoss > ListNetLoss > RankNetLoss > ListMLELoss` — treat that as a starting point to validate, not a guarantee.

## 11.4 Evaluating a reranker

```python
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

samples = [
    {
        "query": "How do I reset my password?",
        "positive": ["Go to Settings > Security > Reset Password."],
        "negative": ["Go to Settings > Notifications to manage email alerts."],
    },
    # ...
]

results = CrossEncoderRerankingEvaluator(samples=samples, name="domain-reranking-eval")(model)
```

> **The critical point:** measure the **full pipeline** (retriever → reranker) end to end, not each component alone. A brilliant reranker sitting behind a retriever that rarely surfaces the right passage in its top-50 is useless — the reranker can only reorder what it's given. If pipeline Recall@50 is 0.60, your reranker's ceiling is 0.60 and you should invest in the retriever instead.

---

# PART 12 — Efficient Fine-Tuning with LoRA

Everything above does **full fine-tuning**: every weight in the model is updated. For a 110M-parameter encoder that's fine. For a 7B-parameter embedding model like Qwen3-Embedding-8B, it isn't — full fine-tuning needs well over 100 GB of GPU memory.

**LoRA** (Low-Rank Adaptation) freezes the base model and trains small adapter matrices instead. Sentence Transformers supports PEFT adapters directly:

```python
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, TaskType

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,                 # rank — the size of the adapter
    lora_alpha=32,        # scaling factor, commonly 2×r
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],   # which layers get adapters
)

model.add_adapter(peft_config)
# then train exactly as before with SentenceTransformerTrainer
```

Benefits beyond memory: adapters are a few megabytes, so you can keep one per customer or per domain and hot-swap them at inference time, all sharing one copy of the base model in memory.

**The math of why this works — including what `r` and `lora_alpha` actually do, and what QLoRA adds — is document 04 of this set.** Read it before you use this in an interview answer; "I'd use LoRA" without being able to explain the rank decomposition is a weak answer.

---

# PART 13 — Managed / API Fine-Tuning

Hosted fine-tuning changes more often than open-weight workflows. Treat provider support as a provider-specific feature, not something that maps onto the `sentence-transformers` API.

- **Cohere** documents fine-tuning for several model/task families and separately offers a dedicated **Rerank** fine-tuning product — useful if you'd rather not self-host a cross-encoder. Confirm current data-format requirements in their docs; the local JSONL shapes above don't transfer directly.
- **OpenAI** — do not assume embedding models are fine-tunable. Check the current documentation for the specific model family.
- **Voyage AI** documents fine-tuned embeddings through its platform as a managed workflow.

> **When to self-host:** if you need control over data, training code, checkpoints, prompts and evaluation — or if your data cannot leave your network — a self-hosted open-weight model is clearest. For a managed service, follow that provider's schema rather than trying to reuse local code.

---

# PART 14 — Practical Rules That Actually Matter

### 1. Data quality beats clever hyperparameters
A clean set of 2,000 genuine `(query, positive)` pairs beats 50,000 noisy ones. Spend your time on the data.

### 2. Hard negatives must be genuinely negative
The goal isn't "as similar as possible." It's **non-relevant documents a retrieval system would realistically confuse with the positive**.

### 3. Batch size matters for MNRL — but gradient accumulation isn't a substitute
See §2.1. Use `CachedMultipleNegativesRankingLoss` when memory is the constraint.

### 4. Start conservative on learning rate and epochs
`2e-5` and 1–3 epochs are starting points, not magic numbers. Sweep nearby values and watch a held-out *retrieval* metric.

### 5. Never mix prompt conventions
E5's format isn't BGE's. Keep the convention identical from training through production.

### 6. Benchmark the real use case
Not "did cosine similarity go up?" but **"does the right chunk appear in the top-k?"**

### 7. Watch for catastrophic forgetting
Train too hard on a narrow domain and the model loses general competence. Symptom: your domain eval improves while performance on general queries collapses. Fix: lower learning rate, fewer epochs, or mix ~10–20% general-purpose pairs into the training data.

### 8. Tune a two-stage pipeline as one system
A reranker fine-tuned in isolation under-delivers if the retriever rarely surfaces the right passage. Measure end-to-end.

### 9. Log your experiments
Model, dataset version, loss, learning rate, epochs, batch size, and the resulting eval numbers. Three weeks in you will not remember which run produced the good checkpoint.

---

# PART 15 — Interview Scenario Bank

### Scenario 1 — The standard opener

> "How would you fine-tune an embedding model for RAG?"

**The 10-step answer:**

```text
1.  Define the retrieval task and build a FROZEN evaluation set first.
2.  Measure the base model on it. That's the number to beat.
3.  Collect high-quality (query, relevant passage) pairs.
4.  Use MultipleNegativesRankingLoss as the baseline objective.
5.  Set BatchSamplers.NO_DUPLICATES so in-batch negatives aren't accidentally positive.
6.  Mine hard negatives with a strong model; filter false negatives with range_min and a margin.
7.  Fine-tune with a conservative learning rate (2e-5) and 1-3 epochs.
8.  Evaluate Recall@k / NDCG@k before vs after on the frozen set.
9.  Keep the model's prompt convention identical at training and inference.
10. If vector size matters, wrap the loss in MatryoshkaLoss.
```

Natural follow-up: *"and how would you add a reranker?"* — reuse the same mined `(query, positive, hard-negative)` data, feed it to a `CrossEncoder` with `BinaryCrossEntropyLoss`, and evaluate the retriever+reranker pipeline together.

---

### Scenario 2 — No labelled data

> "We have 500,000 internal documents and zero labelled queries. Now what?"

**Strong:** "Three options, in increasing order of cost and quality. Cheapest: unsupervised domain adaptation with SimCSE — it adapts vocabulary and style but won't teach task-specific relevance. Middle: synthetic query generation — use an LLM to generate 2–3 plausible questions per document chunk, which gives me `(query, positive)` pairs for MNRL essentially for free. I'd hand-verify a sample of maybe 200 to check the generated questions are actually answerable from the chunk. Best: mine real queries from logs if any search or support system exists, even a bad one — real user language is different from LLM-generated language in ways that matter. Whichever I pick, I still need a small human-labelled eval set — maybe 100 queries — or I can't tell if any of it worked. That's non-negotiable; it's the cheapest part and the one that makes everything else measurable."

---

### Scenario 3 — Loss selection

> "Here's my data: pairs of support tickets labelled 'duplicate' or 'not duplicate'. Which loss?"

**Strong:** "That's binary-labelled pairs, so `ContrastiveLoss` is the natural fit — it pulls label-1 pairs together and pushes label-0 pairs apart beyond a margin. But I'd check the class balance first: duplicate-detection datasets are usually overwhelmingly negative, and if so the model can trivially minimise loss by pushing everything apart. I'd either rebalance the sampling or consider reframing it: take just the duplicate pairs as `(anchor, positive)` and train with MNRL, letting in-batch negatives supply the negative signal. That reframing usually works better in practice, and it also gives me a reusable embedding space instead of a space tuned only for one threshold. At deployment I'd tune the similarity threshold on a validation set rather than guessing 0.8."

---

### Scenario 4 — It got worse

> "After fine-tuning, training loss dropped a lot but Recall@10 went down. Explain."

**Strong:** "Loss and retrieval quality measure different things, so this is a known pattern with several causes. Most likely overfitting — the model memorised the training pairs rather than learning generalisable relevance, especially if the dataset is small relative to the number of epochs. Second: false negatives in the mined data, teaching the model that correct answers are wrong. Third, and easy to miss: a prompt-convention mismatch, where I trained with prefixes but evaluated without them, or vice versa. Fourth: catastrophic forgetting from too high a learning rate. I'd diagnose by checking the eval loss curve for the classic divergence from training loss, inspecting 50 mined negatives by hand, and verifying the encode calls in my eval script match my training preprocessing exactly."

---

### Scenario 5 — The budget question

> "Do we fine-tune the embedding model or add a reranker? We can only do one."

**Strong:** "It depends on where the failure is, and I can measure that before deciding. I'd look at Recall@50 from the retriever. If Recall@50 is high — say 0.90 — the right chunks are being found but ranked badly, so a reranker will help a lot and fine-tuning the retriever won't add much. If Recall@50 is low — say 0.55 — the right chunks aren't being retrieved at all, and a reranker is worthless because it can only reorder what it receives. In that case I fine-tune the retriever. As a rough prior, a reranker is faster to add and usually gives the bigger immediate jump, but it costs latency on every query forever, whereas fine-tuning costs once at training time. If the domain vocabulary is very unusual, fine-tuning tends to win."

---

### Scenario 6 — The math question

> "Explain what MultipleNegativesRankingLoss computes."

**Strong:** "For a batch of `(anchor, positive)` pairs, it computes the full similarity matrix between every anchor and every positive, multiplies by a scale factor of about 20 — equivalent to a temperature of 0.05 — and applies cross-entropy per row, where the correct class is the diagonal. So each anchor is doing an N-way classification: 'which of these N passages is mine?' The off-diagonal entries are free negatives from the other pairs in the batch. The scale matters because cosine similarities are bounded to [-1,1], which is too compressed for softmax to separate meaningfully. And because softmax concentrates gradient on the highest-scoring wrong answer, easy negatives contribute almost nothing — which is exactly why hard negative mining works."

---

# Quick Reference

```text
DEFAULT RECIPE FOR RAG
  data:   (anchor, positive) pairs in JSONL
  loss:   MultipleNegativesRankingLoss
  sampler: BatchSamplers.NO_DUPLICATES
  lr:     2e-5, cosine schedule, warmup_ratio 0.1
  epochs: 1-3
  batch:  as large as fits (more in-batch negatives)
  eval:   InformationRetrievalEvaluator, Recall@k + NDCG@k, base vs fine-tuned

THE THREE NEGATIVES
  easy   → no gradient, wasted batch slot
  hard   → where the learning happens
  false  → actively harmful; filter with range_min + margin

TRAPS
  gradient accumulation ≠ more in-batch negatives  (use CachedMNRL)
  MatryoshkaLoss is a wrapper, not a data format
  prompt convention must match training ↔ inference
  columns are read POSITIONALLY, not by name
  never use a cross-encoder for first-stage retrieval

ALWAYS
  freeze the eval set before training
  record the base model's score
  report the delta
```

---

## Where to go next

- **Document 01** — choosing which model to fine-tune in the first place.
- **Document 03** — the full RAG pipeline this model lives inside.
- **Document 04** — transformer encoders/decoders, quantization math, LoRA and QLoRA in depth.

## Official references

- Sentence Transformers — Training Overview: https://sbert.net/docs/sentence_transformer/training_overview.html
- Losses: https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- Loss overview (task → loss mapping): https://sbert.net/docs/sentence_transformer/loss_overview.html
- Hard negative mining: https://www.sbert.net/docs/package_reference/util/hard_negatives.html
- Evaluation: https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- Cross-Encoder training: https://sbert.net/docs/cross_encoder/training_overview.html
- Cross-Encoder losses: https://sbert.net/docs/package_reference/cross_encoder/losses.html
- Clustering examples: https://sbert.net/examples/sentence_transformer/applications/clustering/README.html
- SimCSE example: https://www.sbert.net/examples/sentence_transformer/unsupervised_learning/SimCSE/README.html
- Model cards: [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5) · [E5](https://huggingface.co/intfloat/e5-base-v2) · [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B) · [Jina v3](https://huggingface.co/jinaai/jina-embeddings-v3)
