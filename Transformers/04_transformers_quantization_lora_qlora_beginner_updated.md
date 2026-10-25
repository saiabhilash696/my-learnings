# Transformers, Quantization, LoRA and QLoRA — From Zero

**Who this is for:** you don't know what quantization is. You've heard "encoder" and "decoder" and "LoRA" in job descriptions. You want to understand them well enough to answer interview questions, not just recite definitions.

**What we'll do:** build up from "how does text become numbers" to "how does a 65-billion-parameter model fit on one GPU", computing every important step by hand with real numbers on real sentences.

**Nothing is assumed** except basic arithmetic and the idea that a matrix is a grid of numbers.

---

# Table of Contents

**Part 1 — From text to numbers** (tokenization, embeddings, position)
**Part 2 — Self-attention, computed by hand**
**Part 3 — A full transformer block**
**Part 4 — Encoder vs decoder vs encoder-decoder**
**Part 5 — Quantization from zero, with the math**
**Part 6 — Why fine-tuning is expensive** (the memory arithmetic)
**Part 7 — LoRA**
**Part 8 — QLoRA**
**Part 9 — Inference-time caching** (KV cache, prompt/context cache, and others)
**Part 10 — Attention variants for serving** (global, local, paged, MQA/GQA)
**Part 11 - Temperature, Top_k, and Top_p — Explained With One Example
**Part 12 — Interview scenario bank**
**Part 13 — Cheat sheet**

---

# PART 1 — From Text to Numbers

A neural network cannot read. It multiplies numbers. So the first job is turning `"The cat sat on the mat"` into numbers.

## 1.1 Tokenization

Text is split into **tokens** — pieces that are usually smaller than words and bigger than letters.

```text
"The cat sat on the mat"
       ↓ tokenizer
["The", " cat", " sat", " on", " the", " mat"]
       ↓ lookup in vocabulary
[464, 3797, 3332, 319, 262, 2603]
```

Those integers are just row numbers in a vocabulary table (typically 30,000–150,000 entries).

### Why subwords, not words?

If you used whole words, you'd need a vocabulary entry for every word that exists, including every typo and every proper noun — impossible. If you used characters, sequences would be enormously long.

Subword tokenization (BPE, WordPiece, SentencePiece) splits rare words into common pieces:

```text
"unhappiness"   →  ["un", "happiness"]           (rare word, common pieces)
"cat"           →  ["cat"]                        (common word, one token)
"Bengaluru"     →  ["Ben", "gal", "uru"]          (rare proper noun, fragments)
"XR-4471-B"     →  ["XR", "-", "44", "71", "-", "B"]   (identifier, meaningless fragments)
```

> **Connect this to document 03:** that last line is *exactly* why embedding models fail on product codes and IDs. The tokenizer shreds `XR-4471-B` into pieces that carry no meaning, so the model has nothing semantic to work with. `XR-4472-B` shreds into almost the same pieces. That's why you need BM25 for identifiers.

**Practical consequence:** token counts differ by language. English averages roughly 1.3 tokens per word. Hindi, Telugu, Arabic and Chinese often need 2–4× more tokens per word with an English-centric tokenizer, which means higher API cost and less effective context for the same text.

## 1.2 The embedding table

Each token ID looks up a row in a big matrix called the **embedding table**.

```text
Embedding table: shape (vocab_size, d_model) = (50,000, 768)

token 3797 ("cat") → row 3797 → [0.021, -0.114, 0.307, ..., 0.045]   (768 numbers)
```

These 768 numbers are **learned during training**. Before training they're random; after training, tokens with related meanings have similar rows.

`d_model` is the model's width. Common values: 384 (small), 768 (base), 1024 (large), 4096 (7B LLM), 8192 (70B LLM).

So our sentence is now a matrix:

```text
"The cat sat on the mat"  →  6 tokens × 768 dimensions
```

## 1.3 Position — the thing attention forgets

Here's a problem. The attention mechanism (Part 2) treats its input as a **set**, not a sequence. To it:

```text
"The dog bit the man"     and     "The man bit the dog"
```

are the same bag of tokens. Clearly they don't mean the same thing.

The fix: add position information to each token's vector.

### Sinusoidal positional encoding (the original method)

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Worked with `d = 4` so you can see actual numbers:

```text
For i=0: denominator = 10000^(0/4) = 1
For i=1: denominator = 10000^(2/4) = 100

position 0:  [sin(0/1),  cos(0/1),  sin(0/100),  cos(0/100)]  = [0.0000,  1.0000, 0.0000, 1.0000]
position 1:  [sin(1/1),  cos(1/1),  sin(1/100),  cos(1/100)]  = [0.8415,  0.5403, 0.0100, 0.9999]
position 2:  [sin(2/1),  cos(2/1),  sin(2/100),  cos(2/100)]  = [0.9093, -0.4161, 0.0200, 0.9998]
```

Each position gets a unique signature. Early dimensions oscillate fast (fine-grained local position), later dimensions oscillate slowly (coarse global position) — like the hands of a clock at different speeds.

This vector is **added** to the token embedding:

```text
final input to the model = token_embedding + positional_encoding
```

**Modern alternatives:** learned positional embeddings (BERT), **RoPE** (rotary position embeddings — used by Llama, Qwen, most modern LLMs; rotates the query and key vectors by an angle proportional to position), and **ALiBi** (adds a distance-based penalty directly to attention scores). RoPE is the current default because it extrapolates better to longer sequences than the model was trained on.

---

# PART 2 — Self-Attention, Computed by Hand

This is the core of the transformer. Once you can compute it by hand, everything else is bookkeeping.

## 2.1 The intuition first

Take the sentence:

```text
"The animal didn't cross the street because it was too tired."
```

What does **"it"** refer to? The animal, not the street. To represent "it" properly, the model needs to pull in information from "animal."

Self-attention does exactly this: **every token looks at every other token and builds a weighted mixture of their information.**

## 2.2 Query, Key, Value

A useful analogy: attention is a soft database lookup.

| Term | Role | Analogy |
|---|---|---|
| **Query (Q)** | What this token is looking for | Your search query |
| **Key (K)** | What each token advertises about itself | The index/title of each document |
| **Value (V)** | The information each token actually carries | The document's contents |

Each is produced by multiplying the input by a learned weight matrix:

```text
Q = X · W_Q      K = X · W_K      V = X · W_V
```

`W_Q`, `W_K`, `W_V` are learned during training. This matters for LoRA later — **these are exactly the matrices LoRA usually adapts.**

## 2.3 The formula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$

Four steps: score, scale, normalize, mix.

## 2.4 Full worked example

Sentence: **"the cat sat"**. We'll use 2 dimensions so the arithmetic is visible, and set `W_Q = W_K = W_V = I` (the identity matrix) so Q, K and V are just the input. Real models learn these matrices; this simplification lets you see the mechanism.

```text
x_the = [1, 0]
x_cat = [0, 1]
x_sat = [1, 1]
```

We'll compute the new representation of **"sat"**.

### Step 1 — Score: how relevant is each token to "sat"?

Dot product of the query for "sat" against every key:

```text
q_sat = [1, 1]

q_sat · k_the = (1×1) + (1×0) = 1
q_sat · k_cat = (1×0) + (1×1) = 1
q_sat · k_sat = (1×1) + (1×1) = 2
```

### Step 2 — Scale by √d_k

`d_k = 2`, so `√2 = 1.4142`:

```text
1 / 1.4142 = 0.7071
1 / 1.4142 = 0.7071
2 / 1.4142 = 1.4142
```

**Why divide by √d_k?** Dot products grow with dimension. With d_k = 64 and typical values, raw scores can reach the hundreds. Feed a large number into softmax and it saturates — one weight becomes 1.0, everything else 0.0, and the gradient vanishes. Dividing by √d_k keeps the variance of the scores at roughly 1 regardless of dimension.

> This is a favourite interview question and most candidates answer "for normalization." The real answer is "to prevent softmax saturation and vanishing gradients as dimension grows."

### Step 3 — Softmax: turn scores into weights that sum to 1

```text
exp(0.7071) = 2.0281
exp(0.7071) = 2.0281
exp(1.4142) = 4.1132
                ────────
sum          = 8.1694

weight("the") = 2.0281 / 8.1694 = 0.2483
weight("cat") = 2.0281 / 8.1694 = 0.2483
weight("sat") = 4.1132 / 8.1694 = 0.5035
                                  ──────
                                  1.0001  ✓ (rounding)
```

### Step 4 — Mix the values using those weights

```text
output_sat = 0.2483 × [1, 0]     ("the")
           + 0.2483 × [0, 1]     ("cat")
           + 0.5035 × [1, 1]     ("sat")

           = [0.2483 + 0.0000 + 0.5035,  0.0000 + 0.2483 + 0.5035]
           = [0.7518, 0.7518]
```

**What just happened:** "sat" started as `[1, 1]` and became `[0.7518, 0.7518]` — a blend that now carries a little information from "the" and "cat". Its representation is **contextual**. In a real model with learned weight matrices, this is how "bank" gets a different vector in "river bank" than in "savings bank."

This computation runs for every token simultaneously, which is why transformers parallelise so well — unlike RNNs, which had to process tokens one at a time.

## 2.5 Multi-head attention

One attention computation learns one kind of relationship. Real models run several in parallel — **heads** — each with its own `W_Q`, `W_K`, `W_V`.

```text
d_model = 768, num_heads = 12   →   each head works in 768/12 = 64 dimensions

head 1 might learn:  subject → verb agreement
head 2 might learn:  pronoun → antecedent ("it" → "animal")
head 3 might learn:  adjective → noun
...
all 12 outputs concatenated back to 768, then multiplied by W_O
```

Total parameter count is roughly the same as one big head — you split the width rather than adding capacity. The gain is **diversity of relationships**, not size.

## 2.6 The cost you must know

Attention compares every token to every token. For sequence length `n`:

```text
n × n attention matrix → O(n²) time and memory
```

```text
n = 512    →      262,144 scores
n = 2,048  →    4,194,304 scores   (16× more for 4× the length)
n = 32,768 →  1,073,741,824 scores (per head, per layer)
```

**This quadratic cost is why context windows were historically small**, and why long-context models need tricks: FlashAttention (same math, memory-efficient implementation), sliding-window attention, sparse attention, or linear-attention approximations.

> Connect to document 01 §2.4: this is the real reason a 32K-token embedding model is expensive, and part of why "just use a huge context window" is rarely the right answer.

---

# PART 3 — A Full Transformer Block

Attention is one component. A transformer **block** wraps it with three more.

```text
           input x
              │
       ┌──────┴──────┐
       │             │
       │      Multi-Head Attention
       │             │
       └──────►(+)◄──┘          ← residual connection: output = x + attention(x)
              │
          LayerNorm
              │
       ┌──────┴──────┐
       │             │
       │     Feed-Forward Network
       │             │
       └──────►(+)◄──┘          ← residual connection
              │
          LayerNorm
              │
           output
```

Stack this block 12 times (BERT-base), 24 times (BERT-large), or 32–80 times (modern LLMs).

## 3.1 The feed-forward network (FFN)

Attention mixes information *between* tokens. The FFN processes each token *independently*, adding non-linear transformation capacity.

```text
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂

W₁ : d_model → 4 × d_model      (768 → 3072)   expand
W₂ : 4 × d_model → d_model      (3072 → 768)   contract
```

**Where the parameters actually are** — this surprises people:

```text
Per block, d_model = 768:
  Attention (W_Q, W_K, W_V, W_O):  4 × 768 × 768   =  2,359,296
  FFN (W₁, W₂):                    2 × 768 × 3072  =  4,718,592
                                                      ──────────
                                   FFN is ~67% of the block's parameters
```

Yet almost all LoRA tutorials only adapt the attention matrices. That's a deliberate trade-off — attention adaptation is cheaper and usually sufficient — but adapting the FFN too can help on harder tasks. Worth knowing when someone asks "which modules would you target?"

## 3.2 Residual connections

```text
output = x + sublayer(x)
```

The `+ x` gives gradients a direct path back through the network. Without it, 32-layer models simply don't train — the gradient signal degrades to nothing by the time it reaches the early layers.

## 3.3 Layer normalization

Normalizes each token's vector to mean 0 and variance 1, then applies learned scale and shift:

$$\text{LayerNorm}(x)=\gamma \cdot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

Keeps activations in a stable range so training doesn't diverge.

> **Pre-norm vs post-norm:** the original transformer applied LayerNorm *after* the residual add (post-norm). Modern models apply it *before* the sublayer (pre-norm), which trains more stably at depth. Most current LLMs also use **RMSNorm**, a cheaper variant that skips the mean subtraction.

---

# PART 4 — Encoder vs Decoder

This is the section interviewers ask about most, and where beginners are most often vague.

## 4.1 The one-sentence difference

```text
ENCODER: each token sees ALL tokens — before and after it.       (bidirectional)
DECODER: each token sees ONLY tokens before it.                  (causal / autoregressive)
```

Same mathematics. The only structural difference is a **mask**.

## 4.2 The encoder

```text
Input:  "The cat [MASK] on the mat"
                    ↓
    Every token attends to every other token
    "[MASK]" sees "The cat" AND "on the mat"
                    ↓
Output: a contextual vector for every token
        prediction for [MASK]: "sat"
```

**Why bidirectional helps:** to fill in `[MASK]`, knowing what comes *after* is enormously informative. "The cat ___ on the mat" is much easier than "The cat ___".

**Training objective — Masked Language Modelling (MLM):** randomly hide ~15% of tokens and train the model to predict them.

```text
Original:  The movie was absolutely terrible and I want a refund
Masked:    The movie was [MASK] terrible and I [MASK] a refund
Predict:   [MASK]₁ = "absolutely",  [MASK]₂ = "want"
```

**Encoders produce representations, not text.** That's what they're for.

**Examples:** BERT, RoBERTa, DeBERTa, ELECTRA, ModernBERT — and **every embedding model in documents 01–03** (BGE, E5, GTE, MiniLM are all BERT-family encoders).

## 4.3 The decoder

```text
Input:  "The cat sat on the"
                    ↓
    Each token attends ONLY to itself and earlier tokens
                    ↓
Output: probability distribution over the next token
        most likely: "mat"
```

**Training objective — Causal Language Modelling (CLM):** predict the next token. No masking of random positions needed — the causal mask means every position is simultaneously a prediction target.

### The causal mask, computed by hand

Reuse the "the cat sat" example, but now compute the representation of **"cat"** (position 2). It may see "the" and "cat", but **not** "sat".

```text
q_cat = [0, 1]

q_cat · k_the = (0×1) + (1×0) = 0
q_cat · k_cat = (0×0) + (1×1) = 1
q_cat · k_sat = MASKED → set to −∞
```

Scale by √2:

```text
0 / 1.4142 = 0.0000
1 / 1.4142 = 0.7071
           = −∞
```

Softmax (`exp(−∞) = 0`, so the future token gets exactly zero weight):

```text
exp(0.0000) = 1.0000
exp(0.7071) = 2.0281
exp(−∞)     = 0.0000
              ───────
sum         = 3.0281

weight("the") = 1.0000 / 3.0281 = 0.3302
weight("cat") = 2.0281 / 3.0281 = 0.6698
weight("sat") = 0.0000           ← cannot see the future
```

```text
output_cat = 0.3302 × [1, 0] + 0.6698 × [0, 1] = [0.3302, 0.6698]
```

**In practice** this is implemented as a triangular mask added to the score matrix before softmax:

```text
        the    cat    sat
the  [   0    −∞     −∞  ]
cat  [   0     0     −∞  ]
sat  [   0     0      0  ]
```

**Why the mask is mandatory:** without it, predicting the next token would be trivial — the model could just look at the answer. It would score perfectly in training and be useless at generation time, when the future genuinely doesn't exist yet.

**Examples:** GPT family, Llama, Mistral, Qwen, Claude, Gemini — essentially every chat model.

## 4.4 Encoder-decoder

Two stacks. The encoder reads the input; the decoder writes the output while looking at the encoder's representation.

```text
 "Translate to French: the cat sat on the mat"
                    │
            ┌───────▼────────┐
            │    ENCODER     │   bidirectional; reads the whole input
            │  (6-24 blocks) │
            └───────┬────────┘
                    │  encoder output (one vector per input token)
                    │
            ┌───────▼────────┐
            │    DECODER     │   causal; generates one token at a time
            │  (6-24 blocks) │
            │                │
            │  self-attn ────┼──► masked, over what it has generated so far
            │  cross-attn ───┼──► Q from decoder, K and V from the ENCODER
            │  FFN           │
            └───────┬────────┘
                    ▼
     "le chat s'est assis sur le tapis"
```

### Cross-attention — the bridge

The decoder block has **two** attention layers:

1. **Masked self-attention** over what it has generated so far.
2. **Cross-attention**: the **Query comes from the decoder**, the **Key and Value come from the encoder**.

That means: *"I'm about to write the next French word — which English words should I look at?"* It's exactly the attention mechanism from Part 2, with Q and K/V coming from different places.

**Examples:** T5, BART, mT5, the original 2017 "Attention Is All You Need" transformer, most speech and translation models.

## 4.5 The comparison table (memorise this)

| | **Encoder-only** | **Decoder-only** | **Encoder-decoder** |
|---|---|---|---|
| Attention | Bidirectional | Causal (masked) | Bidirectional + causal + cross |
| Training | Masked LM | Next-token prediction | Denoising / seq2seq |
| Output | Vector per token | Next-token distribution | Generated sequence |
| Can generate text? | No | Yes | Yes |
| Best at | Understanding, classification, **embeddings, retrieval** | Open-ended generation, chat, few-shot | Transformation tasks: translation, summarization |
| Typical size | 33M–1B | 1B–500B+ | 60M–11B |
| Examples | BERT, RoBERTa, DeBERTa, ModernBERT, **BGE/E5/GTE** | GPT, Llama, Mistral, Qwen, Claude | T5, BART, mT5 |

## 4.6 Why embedding models are encoders

A retrieval system needs **one vector per document**. Encoders are the natural fit:

1. Bidirectional attention means every token sees full context — better representations.
2. They're small and fast, which matters when you're embedding millions of chunks.
3. They were never designed to generate, and you don't want generation.

**Pooling** turns the per-token outputs into a single vector:

```text
"The cat sat on the mat" → encoder → 6 vectors of 768 dims
                                          ↓  pooling
                                   1 vector of 768 dims
```

| Pooling | How | Notes |
|---|---|---|
| **Mean pooling** | Average all token vectors | Most common; the `sentence-transformers` default |
| **CLS pooling** | Use the special `[CLS]` token's vector | BERT's original design; BGE uses this |
| **Last-token pooling** | Use the final token's vector | Used by decoder-based embedding models |
| **Max pooling** | Element-wise maximum | Rare |

> **Must match your model.** Using mean pooling on a model trained with CLS pooling degrades quality silently. `sentence-transformers` reads the right setting from the model config — which is one reason to load models through it rather than raw `transformers`.

### The rise of decoder-based embedders

Newer models (Qwen3-Embedding, e5-mistral, llama-embed-nemotron) are **decoder** models adapted for embedding, usually with last-token pooling. They benefit from the enormous scale of LLM pretraining and often score higher on benchmarks — at the cost of being 8B parameters instead of 110M, making them far more expensive to run over a large corpus.

> **Interview answer:** "Embedding models are traditionally encoders because bidirectional attention gives better representations and the models are small enough to embed millions of documents cheaply. But the strongest models on MTEB now are decoder-based LLMs adapted with last-token pooling and contrastive fine-tuning — they trade a lot of inference cost for higher quality. Which is right depends on corpus size and latency budget."

---

# PART 5 — Quantization, From Zero

You said you don't know what quantization is. Here it is completely.

## 5.1 The one-line version

> **Quantization = storing numbers with fewer bits, accepting a small loss of precision in exchange for a large saving in memory and a speedup in computation.**

That's it. Everything else is detail about *how* to pick the fewer bits.

## 5.2 How a computer stores a decimal number

Model weights are decimal numbers like `0.0342`, `-1.827`, `0.0001`. How many bits does one take?

### FP32 (single precision) — 32 bits, the default

```text
[ 1 bit sign ][ 8 bits exponent ][ 23 bits mantissa ]
     ±              scale              precision

Range:     roughly ±3.4 × 10³⁸
Precision: about 7 decimal digits
Memory:    4 bytes per number
```

### FP16 (half precision) — 16 bits

```text
[ 1 sign ][ 5 exponent ][ 10 mantissa ]

Range:     ±65,504  ← small!
Precision: about 3 decimal digits
Memory:    2 bytes
```

**The problem with FP16:** the range is narrow. During training, gradients can be tiny (like `1e-8`) and **underflow to zero**, or activations can exceed 65,504 and **overflow to infinity**. This is why FP16 training needs "loss scaling" — multiply the loss by a big constant to push gradients into representable range, then divide it back out.

### BF16 (brain float) — 16 bits, arranged differently

```text
[ 1 sign ][ 8 exponent ][ 7 mantissa ]
             ↑
        SAME as FP32

Range:     same as FP32 (±3.4 × 10³⁸)
Precision: about 2 decimal digits (worse than FP16)
Memory:    2 bytes
```

BF16 trades precision for range. **For neural networks that's the right trade** — networks tolerate imprecise weights far better than they tolerate infinities and zeros. This is why BF16 is the default for modern training on Ampere-generation GPUs and later.

> **Interview question:** *"FP16 or BF16?"* — "BF16 if the hardware supports it. Same memory, but it keeps FP32's exponent range, so you avoid overflow and underflow and don't need loss scaling. FP16 has more mantissa bits, so it's slightly more precise, but range matters more than precision in training. On older GPUs without BF16 support, FP16 with loss scaling."

### INT8 — 8 bits, integers only

```text
Range:  −128 to 127  (signed)
Memory: 1 byte
```

There are only **256 possible values**. To store `0.0342` we must map the real numbers onto those 256 slots. That mapping *is* quantization.

### Summary

| Format | Bits | Bytes | Range | Notes |
|---|---|---|---|---|
| FP32 | 32 | 4 | ±3.4e38 | Baseline |
| FP16 | 16 | 2 | ±65,504 | Overflow risk; needs loss scaling |
| BF16 | 16 | 2 | ±3.4e38 | Training default on modern GPUs |
| FP8 | 8 | 1 | varies | Newest hardware (H100+) |
| INT8 | 8 | 1 | −128..127 | 256 levels; needs a scale factor |
| INT4 / NF4 | 4 | 0.5 | 16 levels | QLoRA territory |

## 5.3 Symmetric absmax quantization — worked completely

This is the simplest scheme and the one to know cold.

**Take a real row of weights:**

```text
W = [0.21, -0.85, 1.42, -0.07, 0.63, -1.10, 0.38, 0.95]
```

### Step 1 — Find the largest absolute value

```text
absmax = max(|0.21|, |−0.85|, |1.42|, ..., |0.95|) = 1.42
```

### Step 2 — Compute the scale factor

We're mapping `[−1.42, +1.42]` onto `[−127, +127]`:

```text
scale = absmax / 127 = 1.42 / 127 = 0.0111811
```

**What the scale means:** one integer step equals `0.0111811` in the original units. This single number is stored alongside the quantized values — it's how we get back.

### Step 3 — Quantize (divide by scale, round)

```text
w        w / scale     rounded (INT8)
─────────────────────────────────────
 0.21  →   18.782   →     19
-0.85  →  -76.021   →    -76
 1.42  →  127.000   →    127
-0.07  →   -6.261   →     -6
 0.63  →   56.346   →     56
-1.10  →  -98.381   →    -98
 0.38  →   33.986   →     34
 0.95  →   84.965   →     85
```

We now store `[19, -76, 127, -6, 56, -98, 34, 85]` (8 bytes) plus one FP32 scale (4 bytes) = **12 bytes**, instead of `8 × 4 = 32 bytes`. And at scale, one scale factor covers a whole block of values, so the overhead vanishes.

### Step 4 — Dequantize (multiply by scale)

```text
INT8    × scale       recovered      original     error
────────────────────────────────────────────────────────
  19  × 0.0111811  =   0.212441  vs   0.21   →  0.002441
 -76  × 0.0111811  =  -0.849764  vs  -0.85   →  0.000236
 127  × 0.0111811  =   1.420000  vs   1.42   →  0.000000
  -6  × 0.0111811  =  -0.067087  vs  -0.07   →  0.002913
  56  × 0.0111811  =   0.626142  vs   0.63   →  0.003858
 -98  × 0.0111811  =  -1.095748  vs  -1.10   →  0.004252
  34  × 0.0111811  =   0.380157  vs   0.38   →  0.000157
  85  × 0.0111811  =   0.950394  vs   0.95   →  0.000394
```

**Maximum error: 0.00425.** The values span 2.52 (from −1.10 to +1.42), so the worst error is about **0.17% of the range**. We used a quarter of the memory.

**That is quantization.** Everything else is refinement.

## 5.4 The outlier problem — why naive quantization breaks

Now add one unusual value to the same row:

```text
W = [0.21, -0.85, 1.42, -0.07, 0.63, -1.10, 0.38, 0.95, 17.50]
                                                          ↑ outlier
```

```text
absmax = 17.50
scale  = 17.50 / 127 = 0.137795     ← 12× larger than before
```

Requantize the first value:

```text
0.21 / 0.137795 = 1.524 → rounds to 2
2 × 0.137795 = 0.27559
error = |0.27559 − 0.21| = 0.06559
```

**Error went from 0.0024 to 0.0656 — 27× worse.** One outlier destroyed the precision of every other value in the block, because they all now share a scale calibrated for the outlier.

**This is the central practical problem in quantizing large models**, and it's why real transformer activations — which have well-documented outlier dimensions — are hard to quantize naively.

### The three standard fixes

**1. Smaller blocks.** Instead of one scale per tensor, use one scale per group of 64 or 128 values. An outlier then only damages its own block.

```text
Per-tensor:   1 scale for 4096 × 4096 = 16.7M values   ← outliers are catastrophic
Per-channel:  1 scale per row (4096 scales)            ← much better
Per-group:    1 scale per 64 values (262,144 scales)   ← best, small overhead
```

**2. Keep outliers in high precision.** LLM.int8() detects outlier *dimensions* and runs those in FP16 while everything else runs in INT8.

**3. Smooth the outliers away.** SmoothQuant mathematically shifts difficulty from activations into weights (which are easier to quantize) by scaling both so the product is unchanged.

## 5.5 Asymmetric (affine) quantization

Symmetric quantization assumes values are centred on zero. Activations often aren't — after a ReLU, everything is ≥ 0, so half your integer range is wasted.

Asymmetric quantization adds a **zero-point**:

```text
scale      = (max − min) / 255
zero_point = round(−min / scale)

quantize:   q = round(x / scale) + zero_point
dequantize: x = (q − zero_point) × scale
```

### Worked example

```text
Activations range from −0.5 to 6.0.

scale      = (6.0 − (−0.5)) / 255 = 6.5 / 255 = 0.0254902
zero_point = round(0.5 / 0.0254902) = round(19.61) = 20

Quantize x = 2.3:
  2.3 / 0.0254902 = 90.23 → 90
  90 + 20 = 110                        ← stored value

Dequantize:
  (110 − 20) × 0.0254902 = 2.29412
  error = 0.00588
```

The zero-point means the real value `0.0` maps exactly to the integer `20`, so the full integer range is used.

| | Symmetric | Asymmetric |
|---|---|---|
| Parameters | scale only | scale + zero-point |
| Zero maps to | 0 | zero_point |
| Faster? | Yes (simpler arithmetic) | Slightly slower |
| Good for | **Weights** (roughly zero-centred) | **Activations** (often one-sided) |

## 5.6 Why quantization makes things *faster*, not just smaller

This is the part people miss.

A neural network layer is `y = W·x`. If both `W` and `x` are quantized:

```text
W ≈ s_w × W_int
x ≈ s_x × x_int

y ≈ (s_w × s_x) × (W_int · x_int)
                   └─────┬──────┘
                    integer dot product — accumulated in INT32
```

**You do the entire matrix multiplication in integer arithmetic, then apply one floating-point rescale at the end.** Integer multiply-accumulate is dramatically cheaper than floating point, and modern hardware has dedicated INT8 paths (NVIDIA tensor cores, CPU VNNI instructions).

### Worked example

```text
W_int = [19, -76, 127]      s_w = 0.0111811
x_int = [50, -30,  10]      s_x = 0.01

Integer dot product:
  (19 × 50) + (−76 × −30) + (127 × 10)
  = 950 + 2280 + 1270
  = 4500                                ← all integer arithmetic

Rescale once:
  4500 × 0.0111811 × 0.01 = 0.50315
```

Check against full precision:

```text
w ≈ [0.2124, -0.8498, 1.4200]
x =  [0.50,  -0.30,   0.10  ]
  (0.2124 × 0.50) + (−0.8498 × −0.30) + (1.4200 × 0.10)
  = 0.1062 + 0.25494 + 0.1420
  = 0.50314          ← matches to 4 decimal places
```

Three floating-point multiplies replaced by three integer multiplies and one float multiply. Across billions of operations, that's the speedup.

**The three benefits, stated separately:**

1. **Memory:** 4× smaller (INT8) or 8× smaller (INT4).
2. **Bandwidth:** moving weights from memory to the compute units is often the real bottleneck; 4× smaller means 4× less to move.
3. **Compute:** integer arithmetic units are faster and more numerous.

## 5.7 When quantization happens: PTQ vs QAT

| | **Post-Training Quantization (PTQ)** | **Quantization-Aware Training (QAT)** |
|---|---|---|
| When | After training is finished | During training |
| How | Convert weights; optionally calibrate scales on a sample of data | Simulate quantization in the forward pass so the model learns to be robust to it |
| Cost | Minutes | A full training run |
| Quality | Good at 8-bit, degrades at 4-bit | Better, especially at 4-bit and below |
| Use when | You have a trained model and want it smaller — **the common case** | Accuracy loss from PTQ is unacceptable |

**Calibration** in PTQ means running a few hundred sample inputs through the model to observe the actual range of activations, so you can pick sensible scales. Without it you'd have to guess.

## 5.8 The named methods you'll be asked about

| Method | What's distinctive |
|---|---|
| **LLM.int8()** | INT8 matmul with outlier dimensions kept in FP16. Near-lossless, moderate speedup. |
| **GPTQ** | One-shot 4-bit PTQ using second-order (Hessian) information to decide how to round each weight so overall output error is minimised. Needs calibration data. |
| **AWQ** (Activation-aware Weight Quantization) | Observes that ~1% of weights matter disproportionately, identified by *activation* magnitude, and protects those. Often faster than GPTQ at similar quality. |
| **GGUF / llama.cpp** | A file format plus a family of schemes (`Q4_K_M`, `Q5_K_S`, …) aimed at CPU and consumer-GPU inference. |
| **NF4** | 4-bit NormalFloat. Non-uniformly spaced levels chosen to be information-theoretically optimal for normally-distributed data. **This is QLoRA's format** — see Part 8. |

## 5.9 What quantization costs you

Be honest about this in interviews.

```text
FP32 → BF16/FP16     essentially free. Always do this.
FP16 → INT8          small loss, usually under 1% on downstream tasks with good calibration
INT8 → INT4          noticeable loss; needs GPTQ/AWQ/NF4-class methods to stay usable
Below 4 bits         significant degradation without specialised techniques
```

**Smaller models suffer more.** A 70B model quantized to 4 bits often outperforms a 13B model at FP16 of the same family. A 1B model quantized to 4 bits can fall apart. The rule of thumb: **prefer a bigger model quantized over a smaller model at full precision**, up to about 4 bits.

---

# PART 6 — Why Fine-Tuning Is Expensive

Before LoRA makes sense, you need to feel the problem.

## 6.1 The memory arithmetic for full fine-tuning

Take a **7-billion-parameter** model, standard mixed-precision training with the Adam optimizer:

```text
1. Model weights (BF16)              7B × 2 bytes  =  14 GB
2. Gradients (BF16)                  7B × 2 bytes  =  14 GB
3. Adam momentum m (FP32)            7B × 4 bytes  =  28 GB
4. Adam variance v (FP32)            7B × 4 bytes  =  28 GB
5. FP32 master copy of weights       7B × 4 bytes  =  28 GB
                                                     ───────
                                                     112 GB
   + activations (depends on batch size and sequence length)
```

**Roughly 16 bytes per parameter.**

```text
A100 80GB:      not enough for a 7B full fine-tune
2× A100 80GB:   workable with sharding
70B model:      ~1,120 GB → you need a cluster
```

## 6.2 Where the memory actually goes

```text
Weights:            14 GB  ███
Gradients:          14 GB  ███
Optimizer + master: 84 GB  ██████████████████
```

**The optimizer states and the master copy are 75% of the cost**, and they exist purely to update weights. If you could avoid updating most of the weights, they'd vanish.

That observation is LoRA.

---

# PART 7 — LoRA (Low-Rank Adaptation)

## 7.1 The key insight

When you fine-tune a pretrained model, the change to each weight matrix — call it `ΔW` — turns out to be **low rank**. The model isn't learning arbitrary new behaviour; it's making a structured, constrained adjustment.

A low-rank matrix can be written as the product of two much smaller matrices. So instead of learning `ΔW` directly, learn its two factors.

## 7.2 The math

Normally a layer computes:

```text
h = W₀ · x
```

Full fine-tuning updates `W₀` itself. LoRA **freezes `W₀`** and adds a trainable detour:

$$h = W_0 x + \frac{\alpha}{r}BAx$$

```text
W₀ ∈ ℝ^(d×k)    frozen, the original pretrained weights
A  ∈ ℝ^(r×k)    trainable — initialised with small random Gaussian values
B  ∈ ℝ^(d×r)    trainable — initialised to ZERO
r               the rank: a small number like 4, 8, 16, 32, 64
α               a scaling constant, commonly 2r
```

```text
        x
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 [W₀]      [A]  (k → r)   ← squeeze down to r dimensions
frozen       │
   │         ▼
   │       [B]  (r → d)   ← expand back up
   │         │
   │      × (α/r)
   │         │
   └───►(+)◄─┘
        │
        h
```

### Two initialisation details that matter

**`B` starts at zero**, so `BA = 0` and `h = W₀x` exactly. **The model's behaviour is completely unchanged at step 0.** Training starts from the pretrained model, not from a perturbed version of it. If `B` were random, you'd corrupt the model before learning anything.

**`A` starts random** because if both were zero, the gradient would be zero forever and nothing would ever train.

### What `α/r` does

It decouples the learning rate from the rank. Without it, doubling `r` roughly doubles the magnitude of the update and you'd have to retune your learning rate every time. With `α` fixed (commonly `α = 2r`), you can change `r` and keep the same learning rate.

## 7.3 The parameter saving, computed

Take one attention projection in a 7B model: `d = k = 4096`.

```text
FULL fine-tuning of this matrix:
  4096 × 4096 = 16,777,216 parameters

LoRA with r = 8:
  A: 8 × 4096    =  32,768
  B: 4096 × 8    =  32,768
                   ────────
                    65,536 parameters

Ratio: 16,777,216 / 65,536 = 256× fewer
As a percentage: 0.39%
```

### For the whole model

Llama-2-7B: 32 layers. Apply LoRA with `r=8` to `q_proj` and `v_proj` in each layer:

```text
Per layer:   2 matrices × 65,536 = 131,072
× 32 layers:                       4,194,304  ≈ 4.2M trainable parameters

4.2M / 7,000M = 0.06% of the model
```

### The memory consequence

```text
FULL FINE-TUNE (7B)                    LoRA (7B, r=8, q+v)
  weights BF16          14 GB            frozen weights BF16    14 GB
  gradients             14 GB            gradients (4.2M)        8 MB
  Adam m + v            56 GB            Adam m + v (4.2M)      34 MB
  FP32 master           28 GB            FP32 master (4.2M)     17 MB
  ──────────────────────────             ───────────────────────────
  112 GB                                 ~14.1 GB
```

**112 GB → 14 GB.** A 7B fine-tune now fits on a single 24GB consumer GPU (with the remaining headroom going to activations).

> **Careful:** LoRA reduces *optimizer and gradient* memory. It does **not** reduce the frozen base weights (still 14 GB) and it does not reduce **activation** memory, because you still forward and backward through the full network. That's what QLoRA and gradient checkpointing address.

## 7.4 The hyperparameters

| Parameter | Typical | What it controls |
|---|---|---|
| `r` (rank) | 8–64 | Capacity. Higher = more expressive, more memory, more overfitting risk. |
| `lora_alpha` | 2 × r | Scaling. `α/r` is the effective multiplier. |
| `lora_dropout` | 0.05–0.1 | Regularization on the adapter path. |
| `target_modules` | `["q_proj", "v_proj"]` | Which matrices get adapters. |

### How to choose `r`

```text
r = 4-8      simple style or format adaptation; small datasets
r = 16-32    domain adaptation, most tasks — a good default
r = 64-128   substantially new capabilities, large datasets
```

Start at 16. If the model underfits (training loss plateaus high), raise it. If it overfits (eval loss diverges from training loss), lower it or add dropout.

### Which modules to target

```text
["q_proj", "v_proj"]                                    ← the original paper; cheapest
["q_proj", "k_proj", "v_proj", "o_proj"]                ← all attention
["q_proj","k_proj","v_proj","o_proj",
 "gate_proj","up_proj","down_proj"]                     ← attention + FFN; most capacity
```

Remember from Part 3 that the FFN holds ~⅔ of the parameters. Targeting everything gives the most capacity but costs the most memory. The QLoRA paper found that **targeting all linear layers matters more than using a high rank** — if you have to choose, go wide rather than deep.

## 7.5 Merging: zero inference overhead

After training you can fold the adapter into the base weights:

```text
W_merged = W₀ + (α/r) · B · A
```

`W_merged` has exactly the same shape as `W₀`. **Inference is then identical in speed to the original model** — no extra matrices, no extra latency.

Or keep them separate and swap adapters at runtime:

```text
                      ┌── adapter_legal.safetensors      (20 MB)
base_model (14 GB) ───┼── adapter_medical.safetensors    (20 MB)
                      └── adapter_support.safetensors    (20 MB)
```

One base model in GPU memory, many specialised behaviours. This is how multi-tenant LLM services work, and it's a genuinely strong interview point.

## 7.6 Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,          # α = 2r
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243

# ... train with the usual HF Trainer ...

model.save_pretrained("./my-adapter")     # a few MB, not 14 GB

# Later, to merge for deployment:
# merged = model.merge_and_unload()
# merged.save_pretrained("./merged-model")
```

## 7.7 Variants worth naming

| Variant | Idea |
|---|---|
| **DoRA** | Decomposes the update into magnitude and direction, adapting each separately. Often closes more of the gap to full fine-tuning. |
| **LoRA+** | Uses a higher learning rate for `B` than for `A`. A one-line change, measurably better convergence. |
| **rsLoRA** | Rank-stabilised scaling (`α/√r` instead of `α/r`), which makes high ranks behave better. |
| **AdaLoRA** | Allocates rank adaptively across layers instead of using the same `r` everywhere. |

---

# PART 8 — QLoRA

## 8.1 The remaining problem

LoRA got us from 112 GB to 14 GB for a 7B model. But 14 GB of that is the **frozen base model in BF16**, and it's now essentially all of the cost.

For a 70B model, the frozen base alone is 140 GB. LoRA doesn't help.

**QLoRA's question:** the base model is frozen — we never update it. So why store it at full precision?

## 8.2 The answer

```text
Quantize the frozen base model to 4 bits.
Keep the LoRA adapters in BF16.
Backpropagate THROUGH the quantized weights into the adapters.
```

```text
         x
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
[W₀ in NF4]  [A] BF16
 4-bit,        │
 frozen        ▼
    │        [B] BF16
    │          │
    │       × (α/r)
    │          │
    └───►(+)◄──┘
         │
         h
```

During the forward pass, the 4-bit weights are dequantized to BF16 **one block at a time, on the fly**, used for the multiplication, and discarded. You never hold the full BF16 model in memory.

## 8.3 Innovation 1 — NF4 (4-bit NormalFloat)

4 bits means **16 possible values**. How should they be spaced?

**Uniform spacing (INT4)** puts levels at equal intervals. But neural network weights are approximately **normally distributed** — clustered near zero, thin in the tails. Uniform spacing wastes most levels on tail regions where almost no weights live.

```text
Weight distribution (roughly normal):

                    ████████
                  ████████████
                ████████████████
              ████████████████████
    ────────────────────────────────────
   -3σ      -1σ        0        1σ      3σ

INT4 (uniform):   |    |    |    |    |    |    |    |
                  wasted levels out here ↑        ↑ wasted

NF4 (normal-optimal):        ||||||||||||
                             dense where the weights actually are
```

**NF4** places its 16 levels at the quantiles of a standard normal distribution, so each level receives roughly an equal *number of weights*. It is information-theoretically optimal for normally-distributed data.

NF4 is also **symmetric with an exact zero**, which matters because zero weights (from pruning or padding) must stay exactly zero.

## 8.4 Innovation 2 — Double quantization

Quantization needs a scale factor per block. QLoRA uses a block size of 64.

```text
4 bits per weight
+ one FP32 scale per 64 weights = 32/64 = 0.5 bits per weight
                                  ──────────────────────────
= 4.5 bits per weight
```

Those scale constants are 11% of the total. So QLoRA quantizes **the scale factors themselves** to 8 bits, with a second-level scale per 256 blocks:

```text
4 bits per weight
+ 8-bit scale per 64 weights          = 8/64        = 0.125 bits
+ FP32 second-level scale per 256 blocks = 32/(64×256) = 0.002 bits
                                          ─────────────────────────
= 4.127 bits per weight
```

**Saving: 0.373 bits per parameter.** On a 65B model that's about 3 GB — the difference between fitting on a 48GB GPU and not.

## 8.5 Innovation 3 — Paged optimizers

Memory usage spikes during gradient checkpointing on long sequences, causing out-of-memory crashes at random points in training.

QLoRA uses NVIDIA unified memory to **page optimizer states out to CPU RAM** when the GPU is under pressure, and page them back when needed — exactly like OS virtual memory. Training slows briefly instead of crashing.

## 8.6 The memory comparison

**7B model:**

| Approach | Base weights | Gradients | Optimizer | Total (excl. activations) |
|---|---|---|---|---|
| Full fine-tune | 14 GB (BF16) | 14 GB | 84 GB | **112 GB** |
| LoRA r=16 | 14 GB (BF16) | 17 MB | 100 MB | **~14.1 GB** |
| **QLoRA r=16** | **3.6 GB (NF4)** | 17 MB | 100 MB | **~3.8 GB** |

```text
112 GB  →  14 GB  →  3.8 GB
           LoRA      QLoRA
```

The QLoRA paper's headline result: **fine-tune a 65B model on a single 48GB GPU**, reaching quality comparable to full 16-bit fine-tuning.

## 8.7 The costs

Be honest about these:

- **Slower.** Dequantizing blocks on the fly costs time — typically 20–40% slower per step than plain LoRA.
- **Small quality gap.** Usually tiny, occasionally noticeable on hard tasks.
- **Merging is awkward.** You can't cleanly merge a BF16 adapter into 4-bit base weights. You either dequantize the base to BF16 and merge (giving up the memory saving at inference) or keep the adapter separate.

## 8.8 Code

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4, not plain int4
    bnb_4bit_use_double_quant=True,         # quantize the scale factors too
    bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to BF16 for the actual matmul
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # QLoRA paper finding: target ALL linear layers rather than using a high rank
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### The four config lines, explained

| Line | Why |
|---|---|
| `bnb_4bit_quant_type="nf4"` | Uses the normal-distribution-optimal levels rather than uniform INT4 |
| `bnb_4bit_use_double_quant=True` | Quantizes the block scales — saves ~0.37 bits/param |
| `bnb_4bit_compute_dtype=torch.bfloat16` | Storage is 4-bit; **computation** happens in BF16 after on-the-fly dequantization |
| `prepare_model_for_kbit_training` | Casts layer norms to FP32 for stability and enables gradient checkpointing |

---

# PART 9 — Inference-Time Caching (KV Cache, Prompt Cache, and Other Caches)

Everything so far explains what a transformer computes. This part explains what production systems do to avoid computing the same thing twice. It's a different skill — interviewers use it to separate "understands the model" from "has actually served one."

## 9.1 Why caching exists: the redundant-compute problem

A decoder generates one token, appends it to the sequence, and generates the next one. Naively, that means re-running the *entire* sequence through the model at every step:

```text
Without caching:
step 1: process [t1]                    → predict t2
step 2: process [t1, t2]                → predict t3        (t1 redone, for nothing)
step 3: process [t1, t2, t3]            → predict t4        (t1, t2 redone, for nothing)
...
```

`t1`'s key and value vectors don't change between steps — the weight matrices `W_K`, `W_V` are fixed, and `t1`'s embedding is fixed, so `K_1 = t1 · W_K` is the same number every single time. Recomputing it at every step is pure waste, and it's O(n²) waste on top of the O(n²) attention cost Part 2.6 already flagged.

**KV caching removes exactly this waste**: compute each token's K and V once, store them, and reuse them for every later step.

## 9.2 KV Cache — the one every interview asks about

What's cached: **K and V only, never Q** — a new token needs its own fresh Q to ask "what am I looking for," but the K/V of every past token is already settled information.

Two distinct phases:

```text
PREFILL (once, at the start):
  the whole prompt goes through in parallel, one pass
  → K, V computed and stored for every prompt token

DECODE (once per generated token):
  only the ONE new token goes through
  → compute Q, K, V for just that token
  → append the new K, V to the cache
  → attend the new Q against ALL cached K, V (old + new)
```

That's the whole trick: prefill is compute-heavy but happens once; decode is cheap per step because only one new K/V pair is ever produced, but the attention step still has to look at every K/V produced so far.

### Memory math

```text
KV cache size (bytes) = 2 (K and V) × batch × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element
```

Worked example — a 7B-class model (32 layers, 32 heads, head_dim 128), full multi-head attention, FP16, batch 1, 4096-token context:

```text
2 × 1 × 32 × 32 × 128 × 4096 × 2 bytes
= 2 × (32×32×128×4096) × 2
= 2 × 536,870,912 × 2
= 2,147,483,648 bytes  ≈  2 GiB — for ONE request
```

That's per request, and it grows *linearly* with sequence length while weights stay fixed. This is why long-context serving is memory-bound, not compute-bound, and why the KV cache — not the weights — is usually what limits how many users one GPU can serve concurrently. (Cut `num_kv_heads` from 32 to 8 with GQA — Part 10.4 — and this drops to ~512 MB. That connection is the whole reason GQA exists.)

### Pseudocode

```python
class KVCache:
    """Minimal illustrative KV cache for ONE layer, ONE head.
    A real model keeps one of these per layer and stacks the heads."""

    def __init__(self, max_seq_len, head_dim, dtype="float16"):
        self.k = np.zeros((max_seq_len, head_dim), dtype=dtype)
        self.v = np.zeros((max_seq_len, head_dim), dtype=dtype)
        self.length = 0                       # positions filled so far

    def append(self, k_new, v_new):
        """k_new, v_new: shape (head_dim,) — ONE new token."""
        self.k[self.length] = k_new
        self.v[self.length] = v_new
        self.length += 1

    def get(self):
        return self.k[:self.length], self.v[:self.length]


def decode_step(x_new, W_q, W_k, W_v, cache: KVCache, d_k):
    """One decode step for one new token x_new (shape: d_model,)."""
    q = x_new @ W_q                           # only for the NEW token
    k_new = x_new @ W_k                       # only for the NEW token
    v_new = x_new @ W_v

    cache.append(k_new, v_new)                # O(1) append, not recomputation
    K, V = cache.get()                        # (seq_len_so_far, d_k)

    scores = (q @ K.T) / np.sqrt(d_k)         # (seq_len_so_far,)
    weights = softmax(scores)
    return weights @ V                        # (d_k,)
```

**Interview line:** "Without the cache, `decode_step` would recompute K and V for every past token, every step — O(n²) waste for O(n) tokens generated. The cache turns the K/V projection into O(1) work per step; only the attention dot product against a growing K still grows, O(n) per step, which is unavoidable with vanilla attention but far cheaper than recomputing projections too."

## 9.3 Prompt Cache / Context Cache (what people loosely call "the LM's cache")

This is a **different thing** from the KV cache above, and the distinction is exactly what a good interviewer probes for.

- **KV cache** — lives for the duration of *one* generation request, private to it, thrown away when the request ends.
- **Prompt / context cache** — a serving-layer feature that keeps a *prefix's* KV state alive **across** requests, so a later call sharing the same prefix (same system prompt, same tool schema, same long document) skips prefill for that prefix entirely.

How it works: on the first call, the server hashes the prompt into chunks, runs prefill normally, and stores the resulting K/V blocks keyed by the exact token-id hash of the prefix — not the raw text, so a single differing token is a cache miss. On a later call, the matching prefix's K/V is read back instead of recomputed; only the new suffix goes through prefill.

- Anthropic, OpenAI and Google all expose this as an API feature ("prompt caching" / "context caching"). Cache-hit input tokens are typically billed far below normal price, time-to-first-token drops sharply because the expensive prefill pass is skipped, and entries expire on a short TTL unless refreshed.
- Self-hosted engines implement the same idea two ways: **vLLM** hashes fixed-size token blocks (16–32 tokens) and matches on block hash — this is the same block structure PagedAttention already uses (Part 10.3). **SGLang** builds a radix tree over token sequences ("RadixAttention") so many requests can share overlapping prefixes automatically, which suits branching multi-turn or agent workloads better.

| | KV Cache | Prompt / Context Cache |
|---|---|---|
| Scope | one request | across requests |
| Lives in | GPU memory, per active request | server-side store, keyed by prefix |
| Triggered by | the next token in the *same* sequence | a *new* request sharing a prefix with a past one |
| Cleared by | end of that request | any earlier token in the prefix changing, or TTL expiry |
| Exists to avoid | recomputing K/V within a generation | re-running prefill across repeated prompts/docs |

**Design rule:** put the stable part of a prompt first (system instructions, tool definitions, retrieved documents) and the varying part last (the actual user turn). Caching is prefix-based — changing anything invalidates everything that comes after it in the sequence.

### Pseudocode

```python
import hashlib

class PromptCache:
    """Server-side cache: reuse KV state for a repeated prompt PREFIX."""

    def __init__(self):
        self.store = {}                       # hash(prefix_tokens) -> kv_blocks

    def _key(self, token_ids):
        return hashlib.sha256(bytes(token_ids)).hexdigest()

    def longest_cached_prefix(self, token_ids):
        for end in range(len(token_ids), 0, -1):
            key = self._key(token_ids[:end])
            if key in self.store:
                return end, self.store[key]
        return 0, None

    def save(self, token_ids, kv_blocks):
        self.store[self._key(token_ids)] = kv_blocks


def prefill_with_cache(token_ids, cache: PromptCache, run_prefill_fn):
    hit_len, cached_kv = cache.longest_cached_prefix(token_ids)
    new_tokens = token_ids[hit_len:]           # only compute what's NOT cached
    new_kv = run_prefill_fn(new_tokens, prior_kv=cached_kv)
    full_kv = merge(cached_kv, new_kv)
    cache.save(token_ids, full_kv)             # so future requests can reuse it too
    return full_kv
```

## 9.4 Other caches you'll hear about — and shouldn't confuse with the above

- **Semantic cache** (e.g. GPTCache): embeds the *whole incoming query* and, if a past query's embedding is close enough, returns that **past response** directly — a full response cache, not a KV cache; the model isn't even called on a hit. Risk: two different-but-similar queries ("cancel my order" vs. "cancel my subscription") can collide in embedding space and return the wrong cached answer, so semantic caches need a conservative similarity threshold and are best kept to low-stakes, high-repetition traffic like FAQ deflection.
- **Session cache**: per-user/per-conversation state kept alive between turns of one chat so the server doesn't rebuild everything from scratch each turn — the simplest form of prompt caching, evicted when the session ends or times out.
- **KV cache offload** (CPU/NVMe): when GPU memory is under pressure, cold KV blocks for paused or lower-priority requests get paged out to CPU RAM or disk and paged back in when that request resumes — the same idea as QLoRA's paged optimizers (Part 8.5), applied to KV cache instead of optimizer state.

### Quick disambiguation for interviews

> "What's the difference between KV cache and prompt caching?"

**Strong:** "KV cache is what stops autoregressive decoding from recomputing K and V for old tokens on every new token — it's per-request and basically mandatory for efficient generation. Prompt caching sits on top of that at the serving layer: when two *different* requests share an identical prefix, the server reuses the first request's KV state instead of re-running prefill for the second. KV cache is about not recomputing within one generation; prompt caching is about not recomputing across generations. You can have a KV cache with no prompt caching — that's most self-hosted setups by default — but you can't have prompt caching without a KV cache underneath it, because the thing being cached and reused *is* the KV state."

---

# PART 10 — Attention Variants for Serving

## 10.1 Self-attention and cross-attention — quick recap

Both already covered end-to-end: self-attention in Part 2, cross-attention in Part 4.4.

| Kind | Q from | K/V from | Used for |
|---|---|---|---|
| Self-attention | same sequence | same sequence | every transformer block |
| Masked self-attention | same sequence, future masked | same sequence, future masked | decoders, causal generation |
| Cross-attention | decoder | encoder | translation, encoder-decoder models |

Everything below is the *same formula* — `softmax(QKᵀ/√d_k)V` — with a different choice of which K/V a given Q is allowed to see, or a different way of scheduling that matmul in memory. Not new math, new plumbing.

## 10.2 Global attention vs. local (sliding-window) attention

**Global attention** is the default computed by hand in Part 2: every token's query attends to every token's key. Cost O(n²) in sequence length — already flagged in Part 2.6 as the reason long context is expensive.

**Local / sliding-window attention**: each token attends only to a fixed-size window of nearby tokens.

```text
Global:                token i attends to: [1, 2, 3, ..., i]     → grows with i
Local (window w=3):    token i attends to: [i-2, i-1, i]         → fixed size, always 3
```

Cost drops from O(n²) to O(n·w) — linear in sequence length. Mistral 7B popularised this for LLMs (window ≈ 4096 tokens); Longformer and BigBird mix local windows with a handful of "global tokens" that every position can see and that can see everything, keeping some long-range connectivity without paying full O(n²).

Trade-off: within a single layer, a token genuinely cannot see outside its window. Long-range dependencies still propagate — just layer by layer, the same way a stack of small convolution kernels builds a large receptive field: by layer 2, a token's window already contains information that layer 1 mixed in from beyond its own original window.

### Worked example — same sentence, two different outputs

Reuse Part 1's sentence, **"the cat sat on the mat"** (6 tokens), and Part 2.4's simplification: 2 dimensions, `W_Q = W_K = W_V = I`, so Q, K, V are just the token vectors (positional encoding omitted for clarity, same as Part 2.4).

```text
x_the(1) = [1, 0]     x_cat = [0, 1]     x_sat = [1, 1]
x_on     = [1,-1]     x_the(5) = [1, 0]  x_mat  = [0,-1]
```

We'll compute the new representation of the **last token, "mat"** (position 6), two ways.

**Global — "mat" scores against all 6 tokens:**

```text
q_mat = [0,-1]

q_mat·k_the(1) = 0     q_mat·k_cat = -1     q_mat·k_sat = -1
q_mat·k_on     = 1     q_mat·k_the(5) = 0   q_mat·k_mat = 1

scaled by √2 = 1.4142:   0, -0.7071, -0.7071, 0.7071, 0, 0.7071

softmax weights:  the(1)=0.1420  cat=0.0700  sat=0.0700  on=0.2880  the(5)=0.1420  mat=0.2880

output_mat (global) = Σ weight × value = [0.6420, -0.4359]
```

**Local, window = 3 — "mat" can only see positions 4, 5, 6 ("on", "the", "mat"):**

```text
scaled scores (only 3 now):   on = 0.7071   the(5) = 0   mat = 0.7071

softmax weights:  on = 0.4011   the(5) = 0.1978   mat = 0.4011

output_mat (local) = [0.5989, -0.8022]
```

**Compare the two outputs for the exact same token in the exact same sentence:**

```text
global output: [ 0.642, -0.436]
local  output: [ 0.599, -0.802]
```

They're different, and that's the whole trade-off in one number: with a window of 3, "mat" never gets to look at "the," "cat," or "sat" at all — their weights (0.1420, 0.0700, 0.0700 — 28% of the global attention mass) are simply gone, and the remaining three weights get rescaled to still sum to 1. Cheaper to compute, but literally less information mixed in, at this layer.

```python
def sliding_window_mask(seq_len, window):
    """True = allowed to attend, False = masked out."""
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        lo = max(0, i - window + 1)
        mask[i, lo:i + 1] = True               # causal: current + past, within window
    return mask

# scores  = (Q @ K.T) / sqrt(d_k)
# scores  = np.where(mask, scores, -np.inf)
# weights = softmax(scores, axis=-1)
```

## 10.3 Paged attention — fixing the KV cache's memory fragmentation

Not a different attention formula — a different way of **storing** the KV cache, introduced by vLLM.

The problem: a naive KV cache reserves one contiguous block per request, sized for its *maximum possible* length. Most requests never reach that max, so memory sits reserved-but-unused (internal fragmentation), and different-sized contiguous reservations fragment the space between them too. vLLM's own measurements found naive serving wastes roughly 60–80% of KV cache memory this way.

**PagedAttention** borrows OS virtual memory: the KV cache is split into small fixed-size blocks (typically 16 tokens), stored anywhere in physical GPU memory, with a per-request **block table** mapping logical block index → physical block address — exactly like a page table.

```text
Request's logical KV cache:   [block 0]  [block 1]  [block 2]  [block 3] ...
Block table:                      ↓          ↓          ↓          ↓
Physical GPU memory:          [phys 91]  [phys 14]  [phys 55]  [phys 3] ...   (not contiguous)
```

Two wins fall out of this:

1. **Near-zero waste** — blocks are allocated only as the sequence actually grows; at most `block_size − 1` tokens are wasted, in the last block, instead of an entire unused reserved tail.
2. **Copy-on-write sharing** — requests that share a prefix (parallel sampling of one prompt, beam search, or two users with the same system prompt) can point their block tables at the *same* physical blocks for the shared part. A private copy is only made when one request diverges. This is what makes prompt caching (Part 9.3) cheap to implement underneath.

### Worked example — two prompts, one shared prefix

Use a tiny `block_size = 2` (real vLLM uses 16 — small here so the tokens actually fit on the page) and two requests:

```text
Request A: "the cat sat on the mat"   → tokens: [the, cat, sat, on, the, mat]
Request B: "the cat sat on the rug"   → tokens: [the, cat, sat, on, the, rug]
```

The first **5 of 6** tokens are identical — only the last word differs. Split each into logical blocks of 2:

```text
Request A logical blocks:   block0=[the,cat]   block1=[sat,on]   block2=[the,mat]
Request B logical blocks:   block0=[the,cat]   block1=[sat,on]   block2=[the,rug]
                                  ↑ identical        ↑ identical      ↑ differs
```

The KV cache stores *numbers*, not words — but identical tokens in the identical position with the identical prior context produce identical K/V numbers, so block0 and block1 genuinely are the same data for both requests. The block table only needs to point both requests at the same physical memory for those two blocks, and give block2 its own copy:

```text
Physical GPU memory:   [phys 7]        [phys 2]        [phys 9]    [phys 5]
                        [the,cat] K/V   [sat,on] K/V   [the,mat] K/V [the,rug] K/V

Request A's block table:   block0 → phys 7   block1 → phys 2   block2 → phys 9
Request B's block table:   block0 → phys 7   block1 → phys 2   block2 → phys 5
                                 ↑ SAME pointer   ↑ SAME pointer   ↑ own block
```

Two-thirds of request B's KV cache was never recomputed or duplicated — it's the same physical memory request A is already using. Only the last block, where the sentences actually differ, gets its own space. Scale this up to a shared 2,000-token system prompt across a thousand concurrent users and the saving is the entire point of PagedAttention.

```python
BLOCK_SIZE = 16

class PagedKVCache:
    def __init__(self, num_physical_blocks, head_dim):
        self.free_blocks = list(range(num_physical_blocks))
        self.physical_k = np.zeros((num_physical_blocks, BLOCK_SIZE, head_dim))
        self.physical_v = np.zeros((num_physical_blocks, BLOCK_SIZE, head_dim))
        self.block_tables = {}                 # request_id -> [physical_block_idx, ...]

    def alloc_block(self, request_id):
        phys_idx = self.free_blocks.pop()
        self.block_tables.setdefault(request_id, []).append(phys_idx)
        return phys_idx

    def append(self, request_id, position, k_vec, v_vec):
        logical_block, offset = position // BLOCK_SIZE, position % BLOCK_SIZE
        table = self.block_tables[request_id]
        if logical_block == len(table):        # need a fresh block
            self.alloc_block(request_id)
        phys_idx = table[logical_block]
        self.physical_k[phys_idx, offset] = k_vec
        self.physical_v[phys_idx, offset] = v_vec

    def share_prefix(self, new_request_id, from_request_id, num_blocks):
        """Copy-on-write: alias the block table, copy no data."""
        self.block_tables[new_request_id] = list(
            self.block_tables[from_request_id][:num_blocks]
        )
```

## 10.4 Other attention variants you'll hear about

| Variant | K/V heads vs. Q heads | Idea | Used by |
|---|---|---|---|
| **Multi-Head Attention (MHA)** | equal — every Q head has its own K/V head | the original design (Part 2.5) | GPT-2/3, BERT |
| **Multi-Query Attention (MQA)** | ALL Q heads share ONE K/V head | shrinks KV cache by roughly num_heads×, at some quality cost | original PaLM, Falcon |
| **Grouped-Query Attention (GQA)** | Q heads split into groups, each group shares one K/V head | most of MQA's memory saving, most of MHA's quality | Llama 2 70B, Llama 3, Mistral, Qwen — today's default |
| **FlashAttention** | same heads as MHA | not fewer FLOPs — a fused, tiled *implementation* that never materialises the full n×n score matrix in slow GPU memory, only in fast on-chip SRAM | almost every serving stack today |
| **Sparse attention** | full heads | each token attends to a fixed, structured subset of positions (strided, block-local, or learned) instead of everything | Sparse Transformer, some long-context research models |

Why GQA matters for the memory math in 9.2: cache size scales with `num_kv_heads`, not `num_q_heads`, so cutting 32 KV heads to 8 divides the KV cache by 4 directly — exactly the "≈512 MB" note in 9.2.

### Worked example — the same query head, three different answers

This is the number that makes "quality cost" concrete instead of just a phrase. Say head 3 is computing attention for the word **"sat"**, with its own learned query `q3 = [1, 2]` (head_dim = 2), and we run the *exact same query* through three setups that differ only in which K/V it's allowed to read.

**MHA — head 3 uses its own, individually-learned K/V for "cat" and "sat":**

```text
K_cat=[2,0]  K_sat=[0,3]        V_cat=[5,1]  V_sat=[1,5]

scores = q3·K = [2, 6]      scaled by √2 = [1.4142, 4.2426]
weights = softmax(...) = [0.0558, 0.9442]

output = 0.0558×[5,1] + 0.9442×[1,5] = [1.223, 4.777]
```

**MQA — head 3 is forced to use head 1's K/V instead (the only K/V that exists):**

```text
K_cat=[1,0]  K_sat=[0,1]        V_cat=[1,0]  V_sat=[0,1]     ← someone else's learned keys

scores = q3·K = [1, 2]      scaled = [0.7071, 1.4142]
weights = softmax(...) = [0.3302, 0.6698]

output = 0.3302×[1,0] + 0.6698×[0,1] = [0.330, 0.670]
```

**GQA — head 3 is grouped with head 4, sharing a *dedicated* K/V (not head 1's):**

```text
K_cat=[2,1]  K_sat=[1,3]        V_cat=[4,1]  V_sat=[2,4]     ← its own group's keys, not identical to head 3's,
                                                                 but not a stranger's either

scores = q3·K = [4, 7]      scaled = [2.8284, 4.9497]
weights = softmax(...) = [0.1070, 0.8930]

output = 0.1070×[4,1] + 0.8930×[2,4] = [2.214, 3.679]
```

**Line all three up:**

```text
MHA output (own K/V):        [1.223, 4.777]
GQA output (shared, 2 heads): [2.214, 3.679]   ← distance from MHA: 1.48
MQA output (shared, all heads):[0.330, 0.670]   ← distance from MHA: 4.20
```

Same query, three different answers — because the *K/V it's allowed to read* changed, not the query itself. GQA's output stays noticeably closer to what head 3 "actually meant" (MHA) than MQA's does, which is the real, numeric version of the table's claim that GQA keeps "most of MHA's quality." MQA wanders further because it's reading a K/V that a *different* head learned for a *different* purpose.

```python
def attention_heads(X, W_q, W_k, W_v, num_q_heads, num_kv_heads, head_dim):
    """num_kv_heads == num_q_heads      -> MHA
       num_kv_heads == 1                -> MQA
       1 < num_kv_heads < num_q_heads   -> GQA"""
    Q = split_heads(X @ W_q, num_q_heads, head_dim)     # (num_q_heads, seq, head_dim)
    K = split_heads(X @ W_k, num_kv_heads, head_dim)    # (num_kv_heads, seq, head_dim)
    V = split_heads(X @ W_v, num_kv_heads, head_dim)

    group_size = num_q_heads // num_kv_heads
    outputs = []
    for q_head_idx in range(num_q_heads):
        kv_head_idx = q_head_idx // group_size          # <- the only line that differs from MHA
        scores = (Q[q_head_idx] @ K[kv_head_idx].T) / np.sqrt(head_dim)
        weights = softmax(scores)
        outputs.append(weights @ V[kv_head_idx])
    return concat(outputs)
```

**Interview line:** "GQA is a one-line change at inference — several query heads look up the same K/V head instead of each having its own — but it's why every serving-cost-conscious open model since Llama 2 70B uses it: it shrinks the KV cache, which is what actually limits concurrent requests per GPU, without a full retrain from scratch — usually just a short continued-pretraining phase called 'uptraining'."

---


# PART 11 - Temperature, Top_k, and Top_p — Explained With One Example
**The running example:** the model is completing *"My favorite fruit is ___"*.
Before any sampling settings are applied, the model has these **raw logits**
(scores that can be any number, positive or negative):

| Word | Raw logit |
|---|---|
| apple | 2.31 |
| banana | 1.61 |
| mango | 1.10 |
| durian | 0.34 |
| kiwi | -0.51 |

Every technique below starts from these same five logits. Nothing is ever
added to this list — words only ever get **reshaped** (temperature) or
**deleted** (top_k, top_p).

---

## Step 1: Temperature — reshape the logits into probabilities

Temperature divides each logit by `T` **before** the softmax conversion:

$$p_i = \frac{e^{\text{logit}_i / T}}{\sum_j e^{\text{logit}_j / T}}$$

| Word | T = 0.5 (low) | T = 1.0 (baseline) | T = 1.2 (slightly high) | T = 2.0 (high) |
|---|---|---|---|---|
| apple | 73.6% | 50.2% | 45.2% | 34.9% |
| banana | 18.1% | 24.9% | 25.2% | 24.6% |
| mango | 6.5% | 14.9% | 16.5% | 19.0% |
| durian | 1.4% | 7.0% | 8.7% | 13.0% |
| kiwi | 0.3% | 3.0% | 4.3% | 8.5% |

**Low T** → gaps widen → apple dominates.
**High T** → gaps shrink → underdogs (durian, kiwi) get real odds.

All five words survive at every temperature. Temperature never deletes
anything — it only shifts how much of the pie each word gets. That's the
job top_k and top_p do instead.

We'll use the **T = 1.2** column as the working example for the rest of
this document, since it's the most interesting case for showing cutoffs.

---

## Step 2: Top_k — delete everything except the K highest-probability words

Rule: keep only the top **K** words by probability, discard the rest, then
**renormalize** the survivors so they sum to 100% again.

**Applying top_k = 4** to the T = 1.2 distribution:

| Word | Before top_k | Kept? | After renormalizing |
|---|---|---|---|
| apple | 45.2% | ✅ | 45.2 / 95.6 = **47.3%** |
| banana | 25.2% | ✅ | 25.2 / 95.6 = **26.4%** |
| mango | 16.5% | ✅ | 16.5 / 95.6 = **17.3%** |
| durian | 8.7% | ✅ | 8.7 / 95.6 = **9.1%** |
| kiwi | 4.3% | ❌ deleted | — |

kiwi is thrown out completely (0% chance) regardless of what temperature
said its odds were. The remaining 4 words' probabilities (which summed to
95.6%) are rescaled to sum to 100% again.

**Key trait of top_k:** the cutoff is a **fixed count**. It always keeps
exactly K words, no matter how confident or uncertain the model is.

---

## Step 3: Top_p (nucleus sampling) — delete by cumulative probability budget

Rule: sort words by probability, keep adding them **until the running total
reaches P**, then discard everything after that point, and renormalize.

**Applying top_p = 0.9** to the *same* T = 1.2 distribution (starting fresh
from temperature, not from the top_k result yet):

| Word | Probability | Running total | Kept? |
|---|---|---|---|
| apple | 45.2% | 45.2% | ✅ (under 90%, keep going) |
| banana | 25.2% | 70.4% | ✅ (under 90%, keep going) |
| mango | 16.5% | 86.9% | ✅ (under 90%, keep going) |
| durian | 8.7% | 95.6% | ✅ (this crossed 90% — include it, then stop) |
| kiwi | 4.3% | — | ❌ excluded |

Pool = {apple, banana, mango, durian}. Renormalize by dividing by 95.6%:

| Word | After renormalizing |
|---|---|
| apple | 45.2 / 95.6 = **47.3%** |
| banana | 25.2 / 95.6 = **26.4%** |
| mango | 16.5 / 95.6 = **17.3%** |
| durian | 8.7 / 95.6 = **9.1%** |

**Key trait of top_p:** the cutoff is a **probability budget**, so the pool
size *changes* depending on how spread out the distribution is. If the
model were more confident, top_p might keep only 2 words; if it's more
uncertain, it might keep 15.

---

## Step 4: Combining all three — the real pipeline

In practice, an API applies these **in sequence**, each one narrowing the
field further:

```
raw logits → [temperature] → [top_k] → [top_p] → sample randomly
```

Let's run the full funnel on our example: **T = 1.2, top_k = 4, top_p = 0.9**

**① Temperature (T=1.2)** reshapes the 5 logits into probabilities:

| apple | banana | mango | durian | kiwi |
|---|---|---|---|---|
| 45.2% | 25.2% | 16.5% | 8.7% | 4.3% |

**② Top_k = 4** deletes the lowest word (kiwi) and renormalizes:

| apple | banana | mango | durian |
|---|---|---|---|
| 47.3% | 26.4% | 17.3% | 9.1% |

**③ Top_p = 0.9** now runs on *this* 4-word set, not the original 5:

| Word | Probability | Running total | Kept? |
|---|---|---|---|
| apple | 47.3% | 47.3% | ✅ |
| banana | 26.4% | 73.7% | ✅ |
| mango | 17.3% | 91.0% | ✅ (crossed 90%, include, then stop) |
| durian | 9.1% | — | ❌ excluded — even though top_k had kept it |

**④ Final renormalized pool for sampling** (sum = 91.0 → rescale to 100%):

| Word | Final probability |
|---|---|
| apple | 47.3 / 91.0 = **52.0%** |
| banana | 26.4 / 91.0 = **29.0%** |
| mango | 17.3 / 91.0 = **19.0%** |

**The model will now randomly sample from just these 3 words.** durian and
kiwi — 2 of the original 5 fruits — never had a chance, even though durian
still had 8.7% probability after temperature alone.

---

## Side-by-side: what survives at each stage

| Word | After temp (T=1.2) | After temp → top_k=4 | After temp → top_k=4 → top_p=0.9 |
|---|---|---|---|
| apple | 45.2% | 47.3% | **52.0%** |
| banana | 25.2% | 26.4% | **29.0%** |
| mango | 16.5% | 17.3% | **19.0%** |
| durian | 8.7% | 9.1% | ❌ excluded |
| kiwi | 4.3% | ❌ excluded | ❌ excluded |

Each stage only ever **removes options or reshapes existing ones** — never
adds new ones. The pipeline gets stricter left to right.

---

## Cheat sheet: what each knob actually does

| Setting | What it controls | Effect of a **low** value | Effect of a **high** value |
|---|---|---|---|
| **Temperature** | How sharply probability favors the top word(s) | Very predictable, repetitive, "confident" | More diverse, surprising, sometimes incoherent |
| **Top_k** | Fixed number of candidates allowed | Small, safe vocabulary (e.g. k=1 = always the top word) | Wide vocabulary, rarely restrictive |
| **Top_p** | Probability budget for candidates | Small pool when model is confident, shrinks further | Almost no cutoff — nearly all words allowed |

**Typical defaults people use:** temperature 0.7–1.0, top_p 0.9–0.95, top_k
either unset or high (40–100) — since top_p already does adaptive
filtering, many people don't bother setting top_k at all.


# PART 12 — Interview Scenario Bank

### Scenario 1 — The foundational question

> "Explain the difference between an encoder and a decoder."

**Strong:** "The mathematics is identical — the difference is a mask. In an encoder, every token attends to every other token in both directions, so the representation of a token incorporates context from before *and* after it. That's ideal for understanding tasks: classification, NER, and embeddings, because a bidirectional representation is richer.

In a decoder, a triangular mask sets attention scores to negative infinity for all future positions, so after softmax they get exactly zero weight. Each token only sees itself and what came before. That's mandatory for generation — the model is trained to predict the next token, and if it could see the answer, training would be trivial and generation would fail.

An encoder-decoder has both, plus cross-attention in the decoder, where the query comes from the decoder and the keys and values come from the encoder. That's the bridge that lets the generator look at the input.

Practically: BERT and every embedding model are encoders. GPT, Llama and Claude are decoders. T5 and BART are encoder-decoders."

---

### Scenario 2 — The scaling question

> "Why do we divide by √d_k in attention?"

**Strong:** "For numerical stability in the softmax. The dot product of two vectors with roughly unit-variance components has variance proportional to d_k, so the scores grow as the dimension grows. With d_k = 64, raw scores can easily reach magnitudes where softmax saturates — one weight becomes essentially 1.0 and the rest essentially 0.0. At that point the gradient of the softmax is near zero, so the model stops learning. Dividing by √d_k normalises the variance back to roughly 1 regardless of head dimension, keeping the softmax in its responsive range."

*(The weak answer, which most people give, is "to normalize it.")*

---

### Scenario 3 — Quantization from scratch

> "What is quantization and why does it work?"

**Strong:** "It's storing weights with fewer bits. Concretely, for INT8 symmetric quantization you take the maximum absolute value in a block, divide by 127 to get a scale factor, then divide every weight by that scale and round to an integer. To use the weight you multiply the integer back by the scale. So a weight of 0.21 with an absmax of 1.42 becomes the integer 19, and 19 × (1.42/127) recovers 0.2124 — an error of 0.24%.

It works because neural networks are massively over-parameterised and robust to small weight perturbations. The error behaves like noise, and networks trained with dropout and other regularisation are already noise-tolerant.

Three benefits, and people usually only mention the first: memory is 4× smaller, memory *bandwidth* is 4× lower, which matters because moving weights is often the real bottleneck, and the matmul itself runs in integer arithmetic with a single float rescale at the end, which is faster on hardware with INT8 tensor cores.

The main failure mode is outliers. One large value in a block inflates the scale and destroys precision for every other value — I've seen a single outlier increase quantization error 27×. The fixes are smaller quantization blocks, keeping outlier dimensions in FP16 like LLM.int8() does, or smoothing the outliers into the weights like SmoothQuant."

---

### Scenario 4 — Why LoRA works

> "Why can LoRA train only 0.1% of parameters and still match full fine-tuning?"

**Strong:** "The hypothesis in the paper is that the weight *update* during fine-tuning has low intrinsic rank. You're not teaching the model new language — the pretrained model already has that. You're making a structured adjustment: shifting its style, its domain vocabulary, its output format. That kind of change lives in a low-dimensional subspace, and a low-rank matrix product can represent it.

Mechanically, instead of learning a 4096×4096 update — 16.8 million parameters — you learn two matrices, 8×4096 and 4096×8, which is 65,536 parameters. 256× fewer, and their product has the same shape as the original.

The memory saving is bigger than the parameter saving suggests, because optimizer state is the dominant cost. Adam keeps two FP32 moments plus an FP32 master copy, so full fine-tuning is about 16 bytes per parameter. Making 99.9% of parameters non-trainable eliminates all of that for them.

Two details that matter: B is initialised to zero so the model is unchanged at step zero and training starts from the exact pretrained model, and the α/r scaling decouples the effective update magnitude from the rank so you don't have to retune the learning rate when you change r.

The limit is that LoRA is worse when you genuinely need new *knowledge* rather than new behaviour — a new language, a new modality. Those aren't low-rank adjustments."

---

### Scenario 5 — Choosing an approach

> "We need a model that writes support tickets in our house style. 10,000 examples. What do you do?"

**Strong:** "I'd try things in increasing order of cost, because the cheapest option often works and I'd rather find that out in an afternoon.

First: few-shot prompting with 5–10 examples in the system prompt. Style transfer is exactly what prompting is good at. If it works, I ship it today with zero training infrastructure and I can change the style by editing text.

If prompting isn't consistent enough, QLoRA on a 7B or 8B model. 10,000 examples is a good size — enough to learn a style, small enough to overfit if I'm careless, so I'd use r=16, one or two epochs, and hold out 1,000 examples to watch for divergence. It fits on a single consumer GPU.

I would not full fine-tune. It needs over 100 GB for a 7B model, it risks catastrophic forgetting of general ability, and for a style task the update is exactly the kind of low-rank adjustment LoRA handles well. I'd also not reach for a bigger model first — a 7B fine-tuned on in-domain style usually beats a 70B generic model at this specific task, and costs a fraction to serve.

One thing I'd insist on: an evaluation set before training. Style is subjective, so I'd get a human to rate 100 outputs from each approach blind. Otherwise 'it seems better' is all I'll have."

---

### Scenario 6 — Debugging

> "You quantized a model to 4 bits and quality collapsed. Diagnose it."

**Strong:** "First question: what method? If it's naive uniform INT4 with per-tensor scales, that's the likely cause on its own — 4 bits is 16 levels, and one scale across an entire 16-million-parameter matrix means outliers destroy everything. I'd move to per-group scales with a block size of 64 or 128, and use a proper method — GPTQ, AWQ or NF4 — rather than round-to-nearest.

Second: model size. Small models degrade far more under 4-bit quantization than large ones. If this is a 1B model, 4 bits may just be too aggressive and 8 bits is the right answer. A useful reframing: a 13B model at 4 bits usually beats a 7B model at 8 bits for the same memory.

Third: what exactly got quantized. Quantizing the embedding table, the output head, or the layer norms is much more damaging than quantizing the transformer body. Most good implementations leave those in higher precision — I'd check that they did.

Fourth: calibration data. GPTQ and AWQ pick their rounding based on a calibration set. If that set doesn't resemble the deployment distribution — say it's English Wikipedia and we're serving code — the scales will be wrong for our actual traffic.

For diagnosis I'd measure perplexity on held-out in-domain text at FP16 and at 4-bit, layer by layer if needed, to localise where the error is introduced rather than guessing."

---

### Scenario 7 — Bringing it together

> "Explain how LoRA and quantization combine in QLoRA."

**Strong:** "They solve different halves of the memory problem, which is why they compose so well.

For a 7B model, full fine-tuning needs about 112 GB, and roughly 84 GB of that is optimizer state and the FP32 master copy. LoRA freezes the base model and trains small rank-decomposition adapters, so all of that optimizer memory disappears for 99.9% of the parameters. You drop to about 14 GB — but now essentially all of it is the frozen base model in BF16.

QLoRA attacks that remainder. The base is frozen, so it never needs update precision. Quantize it to 4-bit NF4 and you get about 3.6 GB. The adapters stay in BF16, and during the forward pass the 4-bit blocks are dequantized to BF16 on the fly, one block at a time, so the full-precision model never exists in memory. Gradients flow through the dequantized weights into the adapters.

Three specific tricks make it work: NF4 spaces its 16 levels at normal-distribution quantiles rather than uniformly, because weights are roughly Gaussian, so each level gets a similar share of the weights. Double quantization quantizes the block scale factors themselves, saving another 0.37 bits per parameter. And paged optimizers spill state to CPU RAM during memory spikes instead of crashing.

Net effect: 112 GB to about 3.8 GB, roughly 30× less, with quality close to full 16-bit fine-tuning. The costs are about 20–40% slower training and awkward merging, since you can't cleanly fold a BF16 adapter into 4-bit weights."

---

### Scenario 8 — Connecting to retrieval

> "How does any of this apply to our RAG system?"

**Strong:** "In three places.

The embedding model is an encoder — a BERT-family model with a pooling layer on top. Understanding that tells me why it produces one vector per chunk, why the pooling method has to match the model's training, and why tokenization shredding product codes into meaningless fragments is a structural weakness of dense retrieval rather than a fixable bug.

Quantization appears twice and people confuse them. Quantizing the *model weights* makes the encoder faster and smaller — INT8 on CPU gives a substantial speedup for embedding a large corpus. Quantizing the *output vectors* shrinks the vector database — int8 gives 4×, binary gives 32×. Those are independent decisions with different trade-offs.

LoRA matters if we ever fine-tune a large embedding model. For a 110M-parameter BGE model, full fine-tuning is cheap and LoRA is unnecessary. For an 8B embedding model like Qwen3-Embedding, LoRA or QLoRA is the only practical option. And adapters being a few megabytes means we could keep one per customer, hot-swapped over a single shared base model, which is a genuinely useful multi-tenant architecture."

---

### Scenario 9 — Serving and memory

> "Why does KV cache size limit how many users a GPU can serve at once, and what would you do about it?"

**Strong:** "Model weights are a fixed cost, loaded once. KV cache is a per-request cost that scales with batch size and sequence length, and it's what eats the GPU memory left over after weights: on a 7B model with full multi-head attention, a single 4K-token request can hold roughly 2 GB of KV cache, and that multiplies by however many requests are concurrent and however long their context gets. Weights don't grow with traffic; KV cache does, so it's usually the thing that runs out first.

Three levers, in the order I'd reach for them. First, grouped-query attention — if I control the model, sharing K/V heads across groups of query heads cuts cache size by however many heads are in a group, 4× or more, for a small and recoverable quality cost. Second, PagedAttention-style serving — vLLM's block-based KV storage removes the internal fragmentation that wastes 60–80% of naive allocations, and lets identical prefixes share physical memory instead of duplicating it, which is close to a free win since it changes nothing about the model. Third, quantizing the KV cache itself to INT8 or lower, the same idea as weight quantization but applied to the cache, once memory is still the bottleneck after the first two.

I wouldn't start with quantizing the KV cache — it's the option that touches numerical precision directly during generation, so I'd exhaust the architecture-level and serving-level wins first."

---

# PART 13 — Cheat Sheet

```text
═══ TEXT → NUMBERS ═══
tokenize → integer IDs → embedding table lookup → + positional encoding
Subword tokenization shreds IDs/codes into meaningless pieces → why BM25 exists

═══ ATTENTION ═══
                    softmax( Q·Kᵀ / √d_k ) · V

Q = what I'm looking for   K = what I advertise   V = what I carry
√d_k prevents softmax saturation as dimension grows (NOT just "normalization")
Cost is O(n²) in sequence length → why long context is expensive

═══ TRANSFORMER BLOCK ═══
x → attention → (+x) → LayerNorm → FFN → (+x) → LayerNorm
FFN holds ~2/3 of the block's parameters (expand 4×, contract)
Residuals make deep stacks trainable

═══ ENCODER vs DECODER ═══
ENCODER   bidirectional   masked-LM      vectors    BERT, BGE, E5    → embeddings
DECODER   causal mask     next-token     text       GPT, Llama       → generation
ENC-DEC   both + cross-attention                    T5, BART         → translation
Cross-attention: Q from decoder, K/V from encoder

═══ NUMBER FORMATS ═══
FP32  1+8+23   4 bytes   baseline
FP16  1+5+10   2 bytes   range only ±65,504 → overflow risk, needs loss scaling
BF16  1+8+7    2 bytes   FP32's range, less precision → TRAINING DEFAULT
INT8           1 byte    256 levels, needs a scale
NF4            0.5 byte  16 levels at normal quantiles → QLoRA

═══ QUANTIZATION MATH ═══
SYMMETRIC (weights):
  scale = absmax / 127
  q     = round(w / scale)
  w'    = q × scale

  0.21 with absmax 1.42 → scale 0.01118 → q=19 → 0.2124  (0.24% error)

ASYMMETRIC (activations):
  scale = (max − min)/255 ;  zp = round(−min/scale)
  q = round(x/scale) + zp ;  x' = (q − zp) × scale

FASTER because:  y ≈ (s_w · s_x) × (W_int · x_int)
                 whole matmul in INT, one float rescale at the end

OUTLIERS are the enemy: one big value inflates the scale and
wrecks precision for the whole block (27× worse error in the worked example).
Fixes: smaller blocks · outliers in FP16 (LLM.int8) · SmoothQuant

PTQ = quantize after training (fast, common)
QAT = simulate during training (better at 4-bit, expensive)
Methods: LLM.int8() · GPTQ (Hessian) · AWQ (activation-aware) · GGUF · NF4

═══ FINE-TUNING MEMORY (7B) ═══
FULL:   weights 14 + grads 14 + Adam 56 + master 28 = 112 GB   (~16 bytes/param)
LoRA:   frozen 14 + tiny adapter state              ≈  14 GB
QLoRA:  NF4 base 3.6 + tiny adapter state           ≈ 3.8 GB

═══ LoRA ═══
             h = W₀x + (α/r)·B·A·x

W₀ frozen · A random init · B ZERO init (model unchanged at step 0)
r = rank (8-64) · α usually 2r · α/r decouples LR from rank

4096×4096 full = 16,777,216 params
4096×4096 r=8  =     65,536 params   →  256× fewer (0.39%)

Merge for zero inference overhead:  W_merged = W₀ + (α/r)·B·A
Or keep separate → one base model, many hot-swappable adapters
Variants: DoRA · LoRA+ · rsLoRA · AdaLoRA

═══ QLoRA = 4-bit frozen base + BF16 LoRA adapters ═══
1. NF4       16 levels at normal-distribution quantiles (weights are Gaussian)
2. Double quantization    quantize the scales too → saves 0.37 bits/param
                          4.5 bits/param → 4.127 bits/param
3. Paged optimizers       spill to CPU RAM on memory spikes instead of OOM

Cost: ~20-40% slower · merging is awkward · small quality gap

═══ DECISION ORDER ═══
prompting → few-shot → RAG → LoRA/QLoRA → full fine-tune
(cheapest and most reversible first)

LoRA is for BEHAVIOUR (style, format, domain vocabulary)
RAG is for KNOWLEDGE (facts that change, facts you must cite)
Full fine-tuning is for genuinely new CAPABILITIES (a new language, a new modality)

═══ INFERENCE-TIME CACHING ═══
KV CACHE (per request): store K,V per layer/head for every past token
  prefill = whole prompt, parallel, one pass   decode = 1 new token, O(1) append + O(n) attend
  size = 2 · batch · layers · kv_heads · head_dim · seq_len · bytes   (grows linearly → GPU-memory bound)
  7B model, MHA, FP16, 4K context, batch 1 ≈ 2 GiB per request

PROMPT / CONTEXT CACHE (across requests): reuse prefill of a repeated PREFIX
  keyed by exact token-id hash (or block hash / radix tree) · short TTL · steep discount on hit
  put STABLE content first, VARYING content last — cache is prefix-based, one change invalidates the rest

OTHER CACHES:
  semantic cache  → caches whole RESPONSES by query-embedding similarity, no KV reuse, collision risk
  session cache   → per-conversation state kept alive between turns, evicted on timeout
  KV offload      → cold KV blocks paged to CPU RAM/NVMe under GPU memory pressure (cf. paged optimizers)

═══ ATTENTION VARIANTS FOR SERVING ═══
GLOBAL   every token sees every token        O(n²)     the Part 2 default
LOCAL    fixed window of nearby tokens only  O(n·w)    Mistral (sliding window), Longformer/BigBird
PAGED    not a formula — KV cache stored in fixed-size non-contiguous blocks, OS-style paging
         near-zero fragmentation + copy-on-write prefix sharing → vLLM's PagedAttention

MHA    every Q head has its own K/V head        baseline
MQA    all Q heads share ONE K/V head           ~num_heads× smaller KV cache, quality cost
GQA    groups of Q heads share a K/V head       today's default (Llama 3, Mistral, Qwen)
FLASH ATTENTION   same math, tiled/fused so the n×n matrix never hits slow GPU memory
```

---

## Where to go next

- **Document 01** — choosing an embedding model (uses Part 4 and Part 5 of this doc)
- **Document 02** — fine-tuning embedding models (uses Part 7)
- **Document 03** — the RAG pipeline these models live inside

## Papers worth reading, in this order

1. **Attention Is All You Need** (2017) — the transformer. Read sections 3.1–3.3.
2. **BERT** (2018) — the encoder and masked language modelling.
3. **LoRA: Low-Rank Adaptation of Large Language Models** (2021) — short and very readable.
4. **LLM.int8()** (2022) — the outlier problem, explained clearly.
5. **QLoRA** (2023) — NF4, double quantization, paged optimizers.
