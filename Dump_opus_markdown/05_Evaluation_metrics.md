# The Complete Beginner's Guide to ML Evaluation Metrics

*Assumes you know nothing. No prior statistics. Every formula is explained symbol by symbol, every number is worked out by hand, and every metric comes with real scenarios.*

---

## Table of Contents

**Part I — Foundations**
1. [Why metrics exist](#1-why-metrics-exist)
2. [How to read the math](#2-how-to-read-the-math-dont-skip-this)
3. [The vocabulary](#3-the-vocabulary)
4. [Splitting your data properly](#4-splitting-your-data-properly)

**Part II — The Two Classic Problems**

5. [Regression metrics](#5-regression-metrics-predicting-a-number)
6. [Classification metrics](#6-classification-metrics-predicting-a-category)

**Part III — Specialized Tasks**

7. [Ranking, search & embedding retrieval](#7-ranking-search--embedding-retrieval)
8. [Time series forecasting](#8-time-series-forecasting)
9. [Clustering](#9-clustering-grouping-without-labels)

**Part IV — Language & Modern AI**

10. [NLP text generation](#10-nlp-text-generation-metrics)
11. [Summarization](#11-summarization)
12. [RAG evaluation & Ragas](#12-rag-evaluation--ragas)
13. [LLM evaluation](#13-llm-evaluation)

**Part V — Everything Else & Interview Prep**

14. [Other tasks](#14-other-tasks)
15. [The master cheat sheet](#15-the-master-cheat-sheet)
16. [Scenario drills](#16-scenario-drills-test-yourself)
17. [Interview questions & model answers](#17-interview-questions--model-answers)
18. [Mistakes that get you rejected](#18-mistakes-that-get-you-rejected)

---

# PART I — FOUNDATIONS

## 1. Why metrics exist

A model makes guesses. A metric is **one number that scores how good those guesses were**.

That sounds simple, and mechanically it is. The hard part — and the entire substance of an ML interview on this topic — is that **there is no universally best metric.** The right choice depends on what it costs you to be wrong in different ways.

### Three scenarios where "being wrong" means completely different things

**Scenario A — A hospital screening for breast cancer.**
Two ways to fail. You could tell a healthy woman she might have cancer, causing weeks of fear and an unnecessary biopsy. Or you could tell a woman with a tumour that she's fine, and she comes back two years later with stage 4 disease. These are not equally bad. The first is distressing; the second can be fatal. Any metric that treats them as equal is the wrong metric for this problem.

**Scenario B — Gmail's spam filter.**
Also two ways to fail. Spam reaches your inbox — mildly annoying, you delete it in two seconds. Or your job offer letter goes to the spam folder and you never see it — a genuine life event you missed. Here the *opposite* error is the expensive one compared to Scenario A.

**Scenario C — Netflix recommending a movie.**
A bad recommendation costs almost nothing; you scroll past it. Missing a movie you'd have loved costs almost nothing either; there are ten thousand others. Here neither error is expensive, but *ordering* matters enormously — a good movie at position 1 is worth far more than the same movie at position 40, because nobody scrolls that far.

Same mathematics underneath all three. Completely different metrics.

> **The thing interviewers are actually testing:** when they ask "which metric would you use?", they are asking *"do you understand what this business loses when the model is wrong?"* Candidates who recite formulas fail. Candidates who say "it depends on the relative cost of a false alarm versus a miss, and here's how I'd estimate those" pass.

### The three layers of evaluation

People use the word "metric" for three different things. Knowing the distinction makes you sound experienced.

| Layer | What it is | Must it be smooth? | Examples |
|---|---|---|---|
| **Loss function** | What the model minimizes *while training*. The optimizer needs to compute a gradient (a slope) to know which direction to adjust the weights, so this must be a smooth, differentiable function. | **Yes** | Mean Squared Error, Cross-Entropy |
| **Offline metric** | What you compute on held-out data to compare candidate models. No gradients needed, so it can be anything — even a discontinuous, jumpy function. | No | F1, NDCG, ROUGE, mAP |
| **Online / business metric** | What the company actually cares about, measured in production with real users. | No | Revenue, retention, support tickets, click-through rate |

**Why the loss function and the metric are often different:** F1 score is what you care about, but it's built from counting — you can't take its derivative, so you can't train on it directly. So you train on cross-entropy (smooth) and *evaluate* with F1 (meaningful). This mismatch is normal and worth mentioning.

**The strongest answers always connect layer 2 to layer 3:**

> "I'd optimize for recall at 95% precision. We estimated a missed fraud case costs us about ₹35,000 in chargebacks, while a false alarm costs about ₹250 in analyst review time. That's a 140:1 ratio, so we should be willing to tolerate a lot of false alarms — but not so many that we drown the review team, which is why precision is constrained rather than ignored."

---

## 2. How to read the math (don't skip this)

If mathematical notation makes your eyes glaze over, this section fixes that. Every symbol you'll see in this document is explained here. It takes five minutes and it makes the rest of the guide readable.

### Σ — the summation sign (Greek capital sigma)

**Σ means "add all of these up."** That's it. It's shorthand so you don't have to write `x₁ + x₂ + x₃ + ... + x₁₀₀₀`.

The full form looks intimidating:

```
  n
  Σ  xᵢ
 i=1
```

Read it as three separate instructions:
- `i = 1` at the bottom → "start counting at item number 1"
- `n` at the top → "keep going until item number n" (n = how many items you have)
- `xᵢ` → "and add up this thing, where `i` is the item number"

So if your data is `x = [4, 7, 2]`, then `n = 3` and:

```
Σ xᵢ  =  x₁ + x₂ + x₃  =  4 + 7 + 2  =  13
```

**In code it's a for-loop:**
```python
total = 0
for i in range(n):
    total = total + x[i]
```

That's all Σ ever means. When you see `Σ (ŷᵢ − yᵢ)²`, read it as: *"for every data point, compute (prediction minus actual), square it, and add all those squared numbers together."*

### Subscripts — the little number below

`xᵢ` means "the i-th item of x." If `x = [4, 7, 2]`, then `x₁ = 4`, `x₂ = 7`, `x₃ = 2`.

It's just array indexing. `xᵢ` is `x[i]`.

### y and ŷ — actual vs predicted

- **`y`** (just "y") = the **true** answer. The real house price. The actual label.
- **`ŷ`** (pronounced **"y-hat"**, the little triangle on top) = what the **model predicted**.

The hat always means "estimated" or "predicted." When you see both in a formula, it's comparing truth to guess.

- **`ȳ`** (pronounced **"y-bar"**, a flat line on top) = the **average** of all the y values.

So three similar symbols with very different meanings: `y` = truth, `ŷ` = guess, `ȳ` = average of the truths.

### |x| — absolute value

**Vertical bars mean "throw away the minus sign."**

```
|5|   = 5
|−5|  = 5
|0|   = 0
```

**Why we need it:** suppose you make two errors, `+10` and `−10`. If you just average them: `(10 + (−10))/2 = 0`. Zero error! But you were wrong twice. The minus signs cancelled out and hid your mistakes. Absolute value stops that: `(|10| + |−10|)/2 = 10`. Honest.

### x² and √x — squaring and square roots

`x²` means `x × x`. `5² = 25`. `(−5)² = 25` — **squaring also kills the minus sign**, which is the other way to stop errors cancelling.

`√x` (square root) undoes squaring. `√25 = 5`.

**Why squaring matters so much in metrics:** it doesn't just remove signs, it **amplifies big numbers disproportionately**.

| Error | Absolute value | Squared |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 10 | 10 | **100** |
| 100 | 100 | **10,000** |

An error of 100 is 10× bigger than an error of 10 in absolute terms — but **100× bigger** when squared. So any metric built on squares is saying *"I consider large mistakes catastrophically worse than small ones."* That's a design choice you're making, whether you realize it or not.

### log — the logarithm

The one symbol beginners find genuinely confusing. Here's the clean version.

**`log₂(x)` asks: "how many times do I multiply 2 by itself to get x?"**

```
log₂(2)   = 1    because 2 = 2
log₂(4)   = 2    because 2 × 2 = 4
log₂(8)   = 3    because 2 × 2 × 2 = 8
log₂(16)  = 4    because 2 × 2 × 2 × 2 = 16
```

**The crucial property: logs grow extremely slowly.**

```
log₂(1,000)          ≈ 10
log₂(1,000,000)      ≈ 20
log₂(1,000,000,000)  ≈ 30
```

Multiply the input by a *thousand*, the log only goes up by *ten*. This "squashing" behaviour is exactly why logs appear in metrics — they let you handle numbers that span huge ranges, and they create gentle decay curves.

**Three more facts you'll need:**

1. `log(1) = 0` — in any base. (Multiply 2 by itself zero times → you get 1.)
2. **The log of a number between 0 and 1 is negative.** `log₂(0.5) = −1`, `log₂(0.25) = −2`, `log₂(0.01) ≈ −6.6`.
3. As the input gets closer to 0, the log plunges toward **negative infinity**. `log₂(0.0001) ≈ −13.3`. It never stops falling.

Fact 3 is the entire reason log loss (Section 6.13) punishes confident wrong answers so brutally. A probability is always between 0 and 1, so its log is negative, and if the model gave the correct answer a probability near zero, that log is enormously negative.

**`ln(x)` or "natural log"** is the same idea with base `e ≈ 2.718` instead of 2. Why that odd number? Because it makes calculus clean. For your purposes, `ln` and `log` behave identically — they just differ by a constant multiplier.

### e^x — the exponential

`e^x` is the reverse of `ln`. If `ln(a) = b`, then `e^b = a`. You'll see it in perplexity, where we undo a log to get back to a readable number.

### Mean, median, and why the difference matters

- **Mean** = add everything up, divide by how many. The everyday "average."
- **Median** = sort everything, take the middle value.

**They can be wildly different, and this matters for metrics.** Salaries at a 5-person startup:

```
₹6L, ₹7L, ₹8L, ₹9L, ₹400L (the founder)

Mean   = (6 + 7 + 8 + 9 + 400) / 5 = 430/5 = ₹86L
Median = middle value when sorted    = ₹8L
```

The mean says the "average" employee earns ₹86 lakh, which describes literally nobody. **The mean is dragged around by extreme values; the median ignores them.** Any metric built on means (MAE, MSE, RMSE) inherits that sensitivity. Metrics built on medians don't.

### Harmonic mean vs arithmetic mean

You'll meet this in F1 score.

- **Arithmetic mean** of a and b: `(a + b) / 2`
- **Harmonic mean** of a and b: `2ab / (a + b)`

**Why the harmonic mean exists:** it gets dragged toward the *smaller* number, hard.

| a | b | Arithmetic mean | Harmonic mean |
|---|---|---|---|
| 0.8 | 0.8 | 0.80 | 0.80 |
| 0.9 | 0.7 | 0.80 | 0.79 |
| 1.0 | 0.6 | 0.80 | 0.75 |
| 1.0 | 0.1 | 0.55 | **0.18** |
| 1.0 | 0.01 | 0.505 | **0.0198** |

Look at the last row. Arithmetic mean says 0.505 — "about average." Harmonic mean says 0.02 — "this is terrible." When one component has collapsed, the harmonic mean refuses to let the other one cover for it. That's exactly what you want when combining precision and recall.

### Cosine similarity

Appears constantly in embeddings and retrieval. **It measures the angle between two vectors, ignoring their length.**

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

You don't need to compute it by hand. You need to know what the output means:

| Value | Meaning |
|---|---|
| **1.0** | Vectors point in exactly the same direction → meanings are identical |
| **0.0** | Vectors are perpendicular → completely unrelated |
| **−1.0** | Opposite directions → opposite meanings |

Why "ignoring length" is the point: the sentence "cat" and the sentence "a cat, a very fine cat indeed" produce vectors of different magnitudes but similar direction. Cosine similarity says they're about the same thing. Plain distance would say they're far apart.

---

## 3. The vocabulary

Six terms, then we start.

**Ground truth** (also called *label*, *actual*, *target*, written `y`)
The correct answer. What you wish the model had said. Where does it come from? Sometimes it's automatic (did the customer actually churn? we know 90 days later). Sometimes humans label it by hand, expensively. Sometimes it's genuinely ambiguous — two doctors may disagree on whether an X-ray shows pneumonia, and then your "ground truth" has noise in it. **A metric can never be more reliable than the labels it's computed against.**

**Prediction** (`ŷ`)
What the model actually said.

**Error / residual**
The gap between them. For numbers, `ŷ − y`. Positive means you over-predicted, negative means you under-predicted.

**Held-out data**
Data the model never saw during training. **Every metric in this document must be computed on held-out data.** Measuring on training data is like grading a student using the exact exam they were given the answer key to — they'll score 100% and you'll have learned nothing about whether they understand the subject.

**Overfitting**
When a model memorizes the training data instead of learning general patterns. The telltale sign: training score excellent, held-out score poor. A student who memorized last year's answer key but can't solve a new problem.

**Baseline**
The dumbest reasonable alternative, which your model must beat to be worth anything.

| Task | The baseline to beat |
|---|---|
| Regression | Always predict the mean of the training targets |
| Classification | Always predict the most common class |
| Time series | "Tomorrow = today" (naive) or "this Monday = last Monday" (seasonal naive) |
| Search / retrieval | BM25 keyword search |
| Summarization | Just take the first 3 sentences ("Lead-3") |
| Recommendation | Recommend the most popular items to everyone |

> **"My model got 0.82" is not a result. "My model got 0.82 versus 0.61 for the baseline" is a result.** Forgetting the baseline is the single most common way beginners present meaningless work. In summarization specifically, Lead-3 beats a surprising number of published neural models — because the first three sentences of a news article really are a decent summary.

---

## 4. Splitting your data properly

### The three-way split

```
Full dataset (100%)
│
├── Training set    ~70%  → the model learns its parameters from this
├── Validation set  ~15%  → you compare models and tune settings using this
└── Test set        ~15%  → touched EXACTLY ONCE, at the very end
```

**Why not just two sets?** Suppose you try 50 model configurations and pick whichever scores best on validation. You have now, indirectly, *fit to the validation set through the act of selecting.* Some of that 50-model winner's advantage is real skill and some is luck on that particular slice of data. The validation score is therefore optimistic. The test set — untouched during all that searching — gives you an honest final number.

Think of it as: training set = homework, validation set = practice exams you take repeatedly, test set = the actual final exam you sit once.

### Cross-validation — when you don't have enough data

If you only have 1,000 rows, carving out 15% for testing leaves a test set of 150, and a score computed on 150 examples is noisy. **K-fold cross-validation** reuses the data more efficiently.

**5-fold cross-validation, step by step:**

```
Split all data into 5 equal chunks (folds): A B C D E

Round 1: train on B,C,D,E → test on A → score₁
Round 2: train on A,C,D,E → test on B → score₂
Round 3: train on A,B,D,E → test on C → score₃
Round 4: train on A,B,C,E → test on D → score₄
Round 5: train on A,B,C,D → test on E → score₅

Final result: mean(score₁..score₅)  ±  standard deviation
```

Every data point gets used for testing exactly once and for training four times.

**The standard deviation is the underrated half of this.** Compare:

- Model A: scores 0.81, 0.82, 0.80, 0.83, 0.81 → **0.814 ± 0.011**
- Model B: scores 0.95, 0.71, 0.88, 0.65, 0.91 → **0.820 ± 0.128**

Model B has a marginally higher average. But its performance swings from 0.65 to 0.95 depending on which slice of data it sees. That's an unstable model — in production it might land anywhere. **Model A is the safer choice despite the lower mean.** Reporting mean ± std instead of a bare number is a small thing that signals real experience.

**Stratified k-fold:** if your data is 95% class A and 5% class B, a random split might create a fold with almost no class B examples in it, producing a wild score. Stratified splitting preserves the class ratio inside every fold. **Always use stratified splitting for imbalanced classification.**

**Group k-fold:** if you have 10,000 medical images from 500 patients (20 images each), a random split puts some of a patient's images in training and others in test. The model can recognize the patient rather than the disease and score brilliantly on something useless. Group k-fold keeps all of one patient's data inside a single fold.

### Data leakage — the silent killer

**Leakage is when information that wouldn't be available at prediction time sneaks into training.** Your offline scores look spectacular and production is a disaster.

**Example 1 — preprocessing leakage.** You standardize features (subtract the mean, divide by standard deviation) using the whole dataset, *then* split.

Why it's wrong: the mean you subtracted was computed partly from test-set rows. Your model has absorbed a whisper of information about the test data. The fix: fit the scaler on training data only, then apply those same numbers to validation and test.

```python
# WRONG
X_scaled = scaler.fit_transform(X)          # sees everything
X_train, X_test = split(X_scaled)

# RIGHT
X_train, X_test = split(X)
X_train = scaler.fit_transform(X_train)     # learns from train only
X_test  = scaler.transform(X_test)          # applies train's numbers
```

**Example 2 — target leakage.** You're predicting whether a customer will churn, and one of your features is `cancellation_reason_code`. That field is only filled in *after* the customer has already churned. Your model achieves 99.8% accuracy and is completely useless, because at real prediction time the field is empty.

**How to catch it:** if accuracy is suspiciously high, be suspicious. Ask of every feature: *"would I actually have this value at the moment I need to make the prediction?"* If not, drop it.

**Example 3 — temporal leakage.** Covered in Section 8. Shuffling time series data lets the model train on Friday to predict Wednesday.

**Example 4 — duplicate leakage.** Near-identical rows appear in both train and test (common with scraped data, or with augmented images). The model has effectively memorized the test set. Deduplicate before splitting.

---

# PART II — THE TWO CLASSIC PROBLEMS

## 5. Regression metrics (predicting a number)

**Regression** = the model outputs a continuous number. Not a category — a quantity.

| Problem | Output |
|---|---|
| What will this house sell for? | ₹ 78,50,000 |
| How long until this delivery arrives? | 34 minutes |
| How many units will we sell next month? | 12,480 |
| What will tomorrow's peak temperature be? | 38.2 °C |
| How much is this customer worth over 3 years? | ₹ 1,45,000 |

### Our running example

You've built a model to predict house prices in a city. You test it on 5 houses it has never seen. Prices in lakhs of rupees (₹1 lakh = 100,000).

| House | Description | Actual price (y) | Model predicted (ŷ) | Error (ŷ − y) |
|---|---|---|---|---|
| A | 2BHK, suburb | 200 | 210 | **+10** (over by 10) |
| B | 2BHK, near metro | 250 | 240 | **−10** (under by 10) |
| C | 3BHK, mid-city | 300 | 320 | **+20** |
| D | 3BHK, good school zone | 400 | 380 | **−20** |
| E | 5BHK villa, lakefront | 1000 | 600 | **−400** ← disaster |

Four predictions are reasonable. House E — a luxury villa — was badly underestimated by ₹4 crore. The model has probably never seen many villas.

**Now watch how differently each metric reacts to House E.** That difference *is* the lesson of this section.

---

### 5.1 MAE — Mean Absolute Error

**The question it answers:** *On average, how far off am I?*

**Formula:**
```
MAE = (1/n) × Σ |ŷᵢ − yᵢ|
```

**Reading it piece by piece:**
1. `ŷᵢ − yᵢ` → for house i, subtract the true price from the predicted price. That's the error.
2. `| ... |` → drop the minus sign, so over-predictions and under-predictions don't cancel out.
3. `Σ` → add all five of those up.
4. `(1/n) ×` → divide by 5 to get the average.

**Worked out:**
```
Step 1 — errors:           +10,  −10,  +20,  −20,  −400
Step 2 — absolute values:   10,   10,   20,   20,   400
Step 3 — sum:               10 + 10 + 20 + 20 + 400  =  460
Step 4 — divide by n=5:     460 / 5  =  92

MAE = 92
```

**What this number means:** "On average, my price prediction is off by ₹92 lakh."

**MAE's superpower is that it's in the same unit as your target.** You can say that sentence to a property developer with no maths background and they'll understand it instantly. Most metrics can't be explained in one sentence to a non-technical person; MAE can.

**Range:** 0 to ∞. Lower is better. 0 = perfect.

**When to use it:**
- You need a number a business person can act on.
- Your data contains genuine extreme values (real villas, real billionaires) that you don't want dominating the score.
- All errors are roughly equally bad to you. Being off by ₹200 lakh once is exactly as bad as being off by ₹100 lakh twice.

**Scenario where MAE is right:** A food delivery app predicting arrival time. Being 10 minutes late twice is about as annoying as being 20 minutes late once. Errors accumulate linearly in customer frustration. MAE matches that.

**Scenario where MAE is wrong:** A hospital predicting how many ICU beds will be needed tonight. Being off by 1 bed is a scheduling nuisance. Being off by 20 beds is a crisis. Errors do *not* scale linearly here. Use RMSE.

---

### 5.2 MSE — Mean Squared Error

**The question it answers:** *On average how far off am I — but with big mistakes punished disproportionately?*

**Formula:**
```
MSE = (1/n) × Σ (ŷᵢ − yᵢ)²
```

Identical to MAE except step 2 uses squaring instead of absolute value. Squaring also removes minus signs, but it does something extra: **it magnifies large errors.**

**Worked out:**
```
Step 1 — errors:        +10,      −10,      +20,      −20,      −400
Step 2 — squared:        100,      100,      400,      400,      160,000
Step 3 — sum:            100 + 100 + 400 + 400 + 160,000  =  161,000
Step 4 — divide by 5:    161,000 / 5  =  32,200

MSE = 32,200
```

**Now look at what happened.** House E's squared error alone is 160,000 out of a total of 161,000.

```
House E's share of the total error = 160,000 / 161,000 = 99.38%
```

**One house out of five accounts for 99.4% of the score.** The other four houses — where the model did fine — are essentially invisible. Under MSE, this model is judged almost entirely on how it handles villas.

Is that good or bad? **It depends entirely on your problem.** If mispricing a ₹10 crore property is genuinely catastrophic for your business, MSE is correctly telling you where the pain is. If villas are rare and you mostly sell 2BHKs, MSE is distorting your view.

**The problem with MSE as a reported number:** the unit is "squared lakhs of rupees." What is a squared rupee? Nothing. You cannot explain 32,200 to anybody. Which leads straight to:

**When MSE is genuinely used:** as a **training loss**, not a reported metric. It's smooth and differentiable everywhere, which gradient descent needs. MAE has a kink at zero where the derivative is undefined, making it slightly awkward to optimize. So: train on MSE, report RMSE and MAE.

---

### 5.3 RMSE — Root Mean Squared Error

**Formula:** `RMSE = √MSE`

Take MSE and square-root it, returning to interpretable units.

```
RMSE = √32,200 ≈ 179.4
```

### The MAE vs RMSE comparison — this is the part interviewers like

```
MAE  =  92.0
RMSE = 179.4
```

RMSE is nearly **double** MAE. That gap is not noise. It's a diagnostic.

**Mathematical fact: RMSE ≥ MAE always.** They're equal only when every single error is identical in size.

**So the ratio tells you about the shape of your error distribution:**

| RMSE / MAE ratio | What it means |
|---|---|
| ≈ 1.0 | All errors are about the same size. Consistent, boring, predictable model. |
| 1.2 – 1.5 | Normal, healthy variation in error size. |
| > 2.0 | **A few enormous errors are dominating.** Go investigate those specific cases. |

Our ratio is `179.4 / 92.0 = 1.95` — high. It's telling us, without us needing to look at the data, that a small number of predictions went badly wrong. And indeed: House E.

**Demonstrating it with a counter-example.** Suppose a different model made these errors: `−90, +90, −95, +95, −90`. Same MAE (92), but:

```
Squared: 8100, 8100, 9025, 9025, 8100  → sum 42,350 → MSE 8,470 → RMSE ≈ 92.0
Ratio = 92.0 / 92.0 = 1.0
```

Identical MAE, RMSE drops from 179 to 92. **Two models with the same average error but completely different risk profiles**, and MAE alone cannot distinguish them. That's why you report both.

> **A line worth memorizing:** "I report RMSE and MAE together. RMSE being roughly double MAE told me a handful of outliers were driving the error, so I segmented the test set and found the model was failing specifically on luxury properties — which we then fixed with a separate model for that segment."

**Scenario where RMSE is right:** An airline predicting how many passengers will show up, to decide how much to overbook. Being off by 3 costs a few vouchers. Being off by 40 means bumping forty furious people off a flight and paying massive compensation. The cost curve is steeply nonlinear, so square the errors.

**Scenario where RMSE is wrong:** Predicting individual customers' next purchase amounts, where a handful of wholesale buyers order 500× the typical volume. RMSE will make your model obsess over five wholesale accounts and ignore the ten thousand retail customers who are the actual business. Use MAE, or model the two segments separately.

---

### 5.4 MAPE — Mean Absolute Percentage Error

**The question it answers:** *On average, what percentage off am I?*

**Formula:**
```
MAPE = (100 / n) × Σ [ |ŷᵢ − yᵢ| / |yᵢ| ]
```

New step: divide each absolute error by the true value, converting it to a fraction, then × 100 for a percentage.

**Worked out:**

| House | Absolute error | ÷ actual | = percentage |
|---|---|---|---|
| A | 10 | 10 / 200 | 5.00% |
| B | 10 | 10 / 250 | 4.00% |
| C | 20 | 20 / 300 | 6.67% |
| D | 20 | 20 / 400 | 5.00% |
| E | 400 | 400 / 1000 | **40.00%** |

```
Sum  = 5.00 + 4.00 + 6.67 + 5.00 + 40.00 = 60.67
MAPE = 60.67 / 5 = 12.13%
```

**Why people love it:** "We're 12% off" is **scale-free**. You can compare a house-price model to a sales-volume model to a temperature model, because the unit cancels out. Executives understand percentages instinctively.

**Notice something interesting:** by MAPE, House E contributes 40% of 60.67 ≈ **66%** of the total — significant but not overwhelming. Under MSE it was 99.4%. Dividing by the actual value *softens* the penalty for errors on large-valued items, because being off by 400 on a house worth 1000 is proportionally less shocking than being off by 400 on a house worth 200.

### MAPE's three serious problems

**Problem 1 — it explodes near zero.**

Suppose you're forecasting daily sales of a niche product:

| Day | Actual sales | Predicted | MAPE contribution |
|---|---|---|---|
| Mon | 100 | 110 | 10 / 100 = 10% |
| Tue | 1 | 3 | 2 / 1 = **200%** |
| Wed | 0 | 2 | 2 / 0 = **∞ / crash** |

Tuesday's prediction was off by *two units* — trivially small in the real world — and MAPE calls it a 200% error. Wednesday divides by zero and your evaluation script crashes. In demand forecasting, zeros are everywhere (weekends, discontinued items, new products). **MAPE is close to unusable there.**

**Problem 2 — it's asymmetric and quietly biases your model.**

Suppose the truth is 100.

| Prediction | Error | MAPE |
|---|---|---|
| 150 (over by 50) | 50 | 50% |
| 50 (under by 50) | 50 | 50% |
| 200 (over by 100) | 100 | 100% |
| 0 (under by 100) | 100 | 100% |

So far symmetric. But notice: **you can never get worse than 100% by under-predicting**, because the lowest you can predict is 0. Over-prediction, however, is unbounded — predict 1000 and you get 900% error.

**Consequence:** if a model is trained or tuned to minimize MAPE, it learns that under-predicting is the safer mistake. It will systematically forecast low. In inventory planning, that means chronic stockouts. This is a real, documented failure mode, and mentioning it in an interview shows you've read past the textbook.

**Problem 3 — it's meaningless when values can be negative.** Profit/loss, temperature in Celsius, net change in anything. Dividing by −5 produces nonsense.

### The fixes

**sMAPE (symmetric MAPE)** divides by the average of actual and predicted:
```
sMAPE = (100/n) × Σ [ |ŷᵢ − yᵢ| / ((|yᵢ| + |ŷᵢ|) / 2) ]
```
Reduces the asymmetry. Doesn't eliminate it, and introduces quirks of its own. Better than MAPE, not great.

**WAPE (Weighted Absolute Percentage Error)** — the one you should actually use:
```
WAPE = Σ|ŷᵢ − yᵢ| / Σ|yᵢ|
```

**The key difference: sum first, then divide — instead of dividing each row then averaging.**

```
Our example:  WAPE = 460 / 2150 = 21.4%
```

Why this fixes everything:
- **Zeros are harmless** — a single zero in the denominator doesn't matter because you're dividing by the *total*, which isn't zero.
- **It's weighted by volume** — an error on a high-volume product counts more than an error on a low-volume one, which is exactly right for a business.
- **No asymmetry problem.**

WAPE is the standard in retail and supply-chain forecasting for precisely these reasons. Note our WAPE (21.4%) is much higher than our MAPE (12.13%) — because WAPE weights by value, and our biggest error was on our most valuable house. WAPE is being more honest about business impact.

---

### 5.5 R² — R-squared (Coefficient of Determination)

**The question it answers:** *How much better is my model than just guessing the average every time?*

This is the only metric so far with a **built-in baseline comparison**, which is why people like it.

**Formula:**
```
R² = 1 − (SS_res / SS_tot)

where:
  SS_res = Σ (yᵢ − ŷᵢ)²      ← squared errors of YOUR model
  SS_tot = Σ (yᵢ − ȳ)²       ← squared errors of a "always predict the mean" model
```

("SS" = Sum of Squares. "res" = residual, meaning leftover error. "tot" = total.)

**The logic:** `SS_tot` is how much error you'd have if you gave up on modelling entirely and just predicted the average price for every house. `SS_res` is how much error you actually have. The ratio `SS_res / SS_tot` is *"what fraction of the dumb model's error did I fail to eliminate?"* Subtracting from 1 flips it to *"what fraction did I eliminate?"*

**Worked out:**

**Step 1 — find the mean actual price:**
```
ȳ = (200 + 250 + 300 + 400 + 1000) / 5 = 2150 / 5 = 430
```

**Step 2 — compute SS_tot** (how wrong the "always predict 430" model would be):

| House | Actual | Actual − 430 | Squared |
|---|---|---|---|
| A | 200 | −230 | 52,900 |
| B | 250 | −180 | 32,400 |
| C | 300 | −130 | 16,900 |
| D | 400 | −30 | 900 |
| E | 1000 | +570 | 324,900 |

```
SS_tot = 52,900 + 32,400 + 16,900 + 900 + 324,900 = 428,000
```

**Step 3 — SS_res** (our model's squared errors, already computed in Section 5.2):
```
SS_res = 161,000
```

**Step 4 — combine:**
```
R² = 1 − (161,000 / 428,000)
   = 1 − 0.3762
   = 0.6238
```

**What 0.624 means, in three equivalent phrasings:**
- "My model explains 62.4% of the variation in house prices."
- "My squared error is only 37.6% of what you'd get by blindly guessing the average."
- "I eliminated 62.4% of the error a no-model approach would have."

**Range: −∞ to 1.**

| R² | Meaning |
|---|---|
| **1.0** | Perfect predictions, zero error |
| **0.8** | Strong model |
| **0.3** | Weak, but may still be valuable if the problem is genuinely hard |
| **0.0** | Exactly as good as predicting the mean — completely useless |
| **−0.5** | **Worse than predicting the mean.** Your model is actively harmful. |

**Yes, R² can be negative, and interviewers ask this constantly.** It happens when `SS_res > SS_tot` — your predictions have *more* squared error than just saying "430" every time. Common causes: severe overfitting (great on train, catastrophic on test), a bug in your pipeline, or evaluating on data from a different distribution than you trained on.

### The R² trap: it always goes up

**Critical property:** adding *any* feature to a linear model — even a column of pure random noise — will never decrease training R², and will usually increase it slightly. The model can always find some spurious pattern to exploit.

So you cannot use R² to answer "is this feature useful?"

**Adjusted R² fixes this** by charging a penalty per feature:
```
Adjusted R² = 1 − [ (1 − R²) × (n − 1) / (n − p − 1) ]

n = number of data points
p = number of features
```

**Worked intuition.** Suppose n = 100 data points and R² = 0.60.

| Features (p) | Adjusted R² | Calculation |
|---|---|---|
| 5 | 0.579 | 1 − (0.40 × 99 / 94) |
| 20 | 0.499 | 1 − (0.40 × 99 / 79) |
| 50 | 0.191 | 1 − (0.40 × 99 / 49) |

Same raw R², but adjusted R² collapses as you add features. **If you add a feature and adjusted R² goes down, that feature is not earning its keep.** Use adjusted R² whenever comparing models with different feature counts.

**One more caution:** R² is only really meaningful for linear-ish models on the data they were built for. Comparing R² across completely different datasets is misleading — an R² of 0.3 predicting human behaviour is impressive, while an R² of 0.95 predicting a physics experiment might be disappointing. **Always interpret R² relative to what's achievable in your domain.**

---

### 5.6 The rest of the regression toolkit

#### Median Absolute Error

```
MedAE = median( |ŷᵢ − yᵢ| )
```

**Our example:** absolute errors are `10, 10, 20, 20, 400`. Sorted, the middle one is **20**.

Compare: MAE = 92, RMSE = 179, MedAE = **20**.

The median completely ignores House E. **Interpretation: "half my predictions are within ₹20 lakh."** That's a genuinely useful statement about typical performance.

**Use when:** you want to describe the *typical* case and you have known outliers you've decided not to worry about. Report it alongside MAE, never instead of it — on its own it would let you hide a model that's catastrophically wrong 49% of the time.

#### RMSLE — Root Mean Squared Log Error

```
RMSLE = √[ (1/n) × Σ (log(1 + ŷᵢ) − log(1 + yᵢ))² ]
```

Apply `log(1+x)` to both prediction and actual, *then* do RMSE. (The `1+` is there so that a value of 0 doesn't break the log.)

**The problem it solves.** Imagine predicting YouTube video view counts. Your test set contains a video with 100 views and a video with 10,000,000 views.

| Video | Actual | Predicted | Absolute error | Relative error |
|---|---|---|---|---|
| Small channel | 100 | 200 | 100 | 100% wrong |
| Viral video | 10,000,000 | 10,100,000 | 100,000 | 1% wrong |

Under RMSE, the viral video's error (100,000) is 1,000× more important than the small channel's (100). But intuitively the small channel's prediction is *far worse* — you doubled it. RMSLE fixes this by working on the log scale, where what matters is the **ratio**, not the absolute gap.

```
log(1+200) − log(1+100)          = log(201/101) = log(1.99)   → substantial
log(1+10.1M) − log(1+10M)        = log(1.01)                  → tiny
```

**Second property: RMSLE penalizes under-prediction more than over-prediction.** Because the log curve is steeper near zero, predicting 50 when the truth is 100 hurts more than predicting 200 when the truth is 100.

**Scenario where this is exactly what you want:** forecasting demand for a product. Over-forecast → some extra stock sits in a warehouse. Under-forecast → you run out, the customer buys from a competitor and maybe never comes back. The asymmetry is real, and RMSLE encodes it for free.

**Use when:** your target spans orders of magnitude (1 to 10,000,000), or relative error matters more than absolute, or under-prediction is worse than over-prediction.

#### Huber Loss — the compromise

```
For error e = (ŷ − y), and a threshold δ you choose:

if |e| ≤ δ:   loss = ½ e²                  ← behaves like MSE
if |e| >  δ:  loss = δ(|e| − ½δ)           ← behaves like MAE
```

**In plain English:** for small errors it squares them (smooth, good for training). Once an error gets bigger than δ, it switches to linear growth so a single wild outlier can't dominate.

**Scenario:** you're training a model on sensor data that occasionally produces garbage readings from a faulty device. MSE would let those broken readings hijack the training. MAE would be robust but is awkward to optimize. Huber gives you MSE's smooth training behaviour on the 99% of good data while refusing to be bullied by the 1% of garbage. `δ` is typically set around the 90th percentile of your typical errors.

#### Quantile / Pinball Loss — predicting a range

Everything so far predicts a *single best guess*, which is usually an estimate of the **mean** outcome. Often you want something else entirely.

**Scenario:** you manage a warehouse. You ask for a demand forecast and get "500 units." But the true demand varies — it might be 400, might be 650. If you stock 500 and demand is 650, you lose 150 sales. What you actually want to know is: **"what stock level will cover me 90% of the time?"** That's the 90th percentile, not the mean.

**Formula** for quantile τ (tau), e.g. τ = 0.9:
```
if y > ŷ  (you UNDER-predicted):  loss = τ × (y − ŷ)         = 0.9 × gap
if y ≤ ŷ  (you OVER-predicted):   loss = (1 − τ) × (ŷ − y)   = 0.1 × gap
```

**Worked example** with τ = 0.9 and a gap of 100 units:
```
Under-predicted by 100:  loss = 0.9 × 100 = 90
Over-predicted by 100:   loss = 0.1 × 100 = 10
```

**Under-predicting costs 9× more.** A model minimizing this loss will therefore learn to predict high — and mathematically, the value that minimizes this exact loss is precisely the 90th percentile of the outcome distribution. That's the elegant bit.

**Symmetric case:** τ = 0.5 makes both penalties equal, and the minimizer is the **median**. (Pinball loss at τ=0.5 is just MAE / 2.)

**Use when:** you need prediction intervals, inventory safety stock, capacity planning, or any situation where "how bad could it get?" matters more than "what's the typical outcome?"

---

### 5.7 Scenario-based practice: choosing a regression metric

**Scenario 1 — Zomato predicting delivery time.**
*Considerations:* Customers see a countdown. Being 5 minutes late is mildly annoying; being 25 minutes late is 5× as annoying, roughly linearly. Times are always positive and in a narrow range (10–60 min), so no scale problem. Outliers exist (a delivery stuck in a flood) but are rare and not the model's fault.
*Answer:* **MAE as the headline** ("we're typically 4 minutes off"), plus **bias** (`mean(ŷ − y)`) to make sure you're not systematically optimistic — a model that's always 4 minutes optimistic is worse than one that's 4 minutes off in random directions, because it breaks trust every single time. Also check **P90 absolute error** to understand the tail.

**Scenario 2 — A power utility forecasting tomorrow's peak electricity demand.**
*Considerations:* If you under-forecast, the grid can't meet demand and there are blackouts affecting millions. If you over-forecast, you spin up an extra gas turbine and waste money. Under-prediction is vastly worse, and large errors are catastrophically worse than small ones.
*Answer:* **RMSE** as the primary metric (large errors squared), plus an **asymmetric quantile loss** targeting a high quantile like τ = 0.95 so the model deliberately errs on the high side. Report **peak-hour errors separately** from off-peak, because 3 AM accuracy is irrelevant.

**Scenario 3 — A bank estimating customer lifetime value across 10 million customers.**
*Considerations:* CLV is wildly skewed — most customers are worth ₹5,000, a few private-banking clients are worth ₹5 crore. Values span four orders of magnitude.
*Answer:* **RMSLE** (handles the scale span, cares about relative error), plus **MAE computed separately per value-segment** so you can see whether the model works for both mass-market and high-net-worth customers. A single overall number would be dominated by whichever segment has more variance and would hide failure in the other.

**Scenario 4 — A startup predicting monthly revenue for its own board deck.**
*Considerations:* Revenue can be compared across months of different sizes. The board wants a percentage. Revenue is never near zero.
*Answer:* **MAPE or WAPE** for communication, **RMSE** for internal model selection. And absolutely report **bias** — a model that consistently over-forecasts revenue to a board is a career-limiting model.

### 5.8 Regression decision tree

```
START: what matters most?

Do you need one number a non-technical person can act on?
└── → MAE ("we're typically off by X")

Are large errors disproportionately costly?
└── → RMSE   (and check the RMSE/MAE ratio to see if outliers dominate)

Does your target span orders of magnitude (1 to 1,000,000)?
└── → RMSLE

Do you need a scale-free percentage?
├── Values comfortably above zero, no zeros → MAPE
└── Zeros present, or business-weighted     → WAPE  ← usually the better choice

Is your training data contaminated with bad outliers?
└── → Huber loss for training, MAE for reporting

Do you need a RANGE not a point?
└── → Pinball loss (+ check interval coverage, Section 8.5)

Do you need to prove the model beats "do nothing"?
└── → R² (+ Adjusted R² if comparing different feature sets)

ALWAYS ALSO REPORT: bias = mean(ŷ − y). A metric that takes absolute
values can't tell you the model is systematically wrong in one direction,
and systematic bias is usually a bigger business problem than random error.
```

---

## 6. Classification metrics (predicting a category)

**Classification** = the model outputs a category, not a number.

| Problem | Possible outputs |
|---|---|
| Is this email spam? | spam / not spam (**binary** — 2 classes) |
| Is this transaction fraud? | fraud / legitimate (binary) |
| What animal is in this photo? | cat / dog / bird / horse (**multi-class** — pick one of many) |
| What topics does this article cover? | {politics, economy, Europe} (**multi-label** — several at once) |

This section is the longest in the guide because it's the most asked-about in interviews, and because nearly every metric here is built from the same four numbers. Get those four numbers solid and everything else follows.

---

### 6.1 Building the confusion matrix from absolute zero

Let's set up a concrete scenario and never leave it.

**The scenario:** you've built a spam filter for a company's email system. You run it over **1,000 emails** whose true status you already know.

Some vocabulary first, because this trips up every beginner:

> **The "positive" class is the thing you're trying to detect.** It does *not* mean "good." In a cancer detector, cancer is the positive class. In a fraud detector, fraud is the positive class. Here, **spam is the positive class** — spam is what the filter is hunting for.

**The truth about our 1,000 emails:**
- **50 are genuinely spam** (the positive class)
- **950 are genuine legitimate emails** (the negative class)

**What our filter did:**
- It flagged **40 emails** as spam.
- Of those 40, **30 really were spam** — good catches.
- The other **10 were legitimate emails** it wrongly flagged — those went to the spam folder by mistake.
- That means **20 spam emails slipped through** (50 real spam − 30 caught = 20 missed).
- And **940 legitimate emails** were correctly left alone (950 − 10 wrongly flagged = 940).

**Those four numbers — 30, 10, 20, 940 — are the confusion matrix.** Here it is:

|  | **Model says: SPAM** | **Model says: NOT SPAM** | Row total |
|---|---|---|---|
| **Truth: SPAM** | **TP = 30** ✅ | **FN = 20** ❌ | 50 actual spam |
| **Truth: NOT SPAM** | **FP = 10** ❌ | **TN = 940** ✅ | 950 actual legit |
| **Column total** | 40 flagged | 960 not flagged | **1,000 total** |

### How to never forget the four names

The names are two words. **Read them backwards.**

> **The SECOND word tells you what the MODEL SAID.**
> - "...Positive" → the model said SPAM
> - "...Negative" → the model said NOT SPAM
>
> **The FIRST word tells you whether the model was RIGHT.**
> - "True..." → yes, it was correct
> - "False..." → no, it was wrong

Apply it:

| Name | Model said | Was it right? | So what actually happened |
|---|---|---|---|
| **True Positive (30)** | "spam" | ✅ yes | It really was spam. **Good catch.** |
| **False Positive (10)** | "spam" | ❌ no | It was a real email. **False alarm** — your job offer went to the spam folder. |
| **False Negative (20)** | "not spam" | ❌ no | It really was spam. **Missed it** — spam landed in the inbox. |
| **True Negative (940)** | "not spam" | ✅ yes | It really was legitimate. **Correctly left alone.** |

**Two everyday names for the two error types, used constantly in industry:**
- **False Positive = false alarm.** The fire alarm went off and there was no fire.
- **False Negative = a miss.** There was a fire and the alarm stayed silent.

### Type I and Type II errors

Statisticians call them:
- **Type I error = False Positive**
- **Type II error = False Negative**

**Memory hook — the boy who cried wolf, in story order:**
1. **First**, the boy cried "wolf!" when there was no wolf → **false alarm → Type I**.
2. **Then**, a real wolf came and nobody believed him → **a miss → Type II**.

The story happens in that order, so the numbering follows the story.

### Reading a confusion matrix at a glance

Once you're fluent, you can diagnose a model just by looking at the four cells:

- **FP is large, FN is small** → the model is trigger-happy. It flags too aggressively. Raise the threshold.
- **FN is large, FP is small** → the model is too cautious. It's missing real cases. Lower the threshold.
- **Both large** → the model genuinely can't tell the classes apart. Threshold tuning won't save you; you need better features or a better model.
- **TN is enormous compared to everything else** → your data is imbalanced, and you should immediately be suspicious of any accuracy number you compute.

---

### 6.2 Accuracy — and the trap it sets

**The question it answers:** *What fraction of all predictions did I get right?*

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (correct predictions) / (all predictions)
```

**Worked out:**
```
Accuracy = (30 + 940) / 1000
         = 970 / 1000
         = 0.97  →  97%
```

97%! You'd put that on a slide. You'd feel good.

### Now let's break it

Consider a competing "model" that is one line of code:

```python
def predict(email):
    return "not spam"     # that's the whole model
```

It never flags anything. Ever. Let's fill in its confusion matrix:

|  | Says SPAM | Says NOT SPAM |
|---|---|---|
| **Truth: SPAM** | TP = 0 | FN = 50 |
| **Truth: NOT SPAM** | FP = 0 | TN = 950 |

```
Accuracy = (0 + 950) / 1000 = 0.95  →  95%
```

**A model that does absolutely nothing scores 95%.**

Suddenly our real model's 97% looks very different. It's not "97% good." It's "2 percentage points better than doing nothing." That's the honest framing.

### The accuracy paradox, stated properly

> **When one class heavily outnumbers the other, accuracy is dominated by the majority class and tells you almost nothing about performance on the class you actually care about.**

And the class you care about is *almost always the rare one*. Nobody builds a fraud detector to identify legitimate transactions.

**Real-world base rates that make accuracy useless:**

| Problem | Positive rate | "Always say no" accuracy |
|---|---|---|
| Credit card fraud | ~0.1% | **99.9%** |
| Rare disease screening | ~0.5% | **99.5%** |
| Ad click prediction | ~1% | **99%** |
| Manufacturing defects | ~2% | **98%** |
| Email spam (corporate, filtered) | ~5% | **95%** |

A fraud model with 99.9% accuracy might be catching **zero** frauds.

### What to do instead

**Always, always compute the majority-class baseline first.** One line:

```python
baseline_accuracy = max(class_counts) / total_count
```

Then report your model *relative to it*. If your model gets 99.91% and the baseline is 99.90%, you have essentially nothing, and saying so honestly is far better than being caught.

**When accuracy IS appropriate:**
- Classes are roughly balanced (say 40/60 or better), **and**
- both error types cost about the same, **and**
- you've reported the confusion matrix alongside so nobody can hide behind the single number.

That combination is rarer than beginners assume.

---

### 6.3 Precision — "when I raise an alarm, am I right?"

**Formula:**
```
Precision = TP / (TP + FP)
```

**Reading the denominator:** `TP + FP` is everything the model **flagged as positive** — the correct flags plus the false alarms. So precision asks: *of everything I flagged, what fraction was correct?*

**Worked out:**
```
Precision = 30 / (30 + 10)
          = 30 / 40
          = 0.75  →  75%
```

**Translation into a sentence you could say out loud:**
> "When my filter puts an email in the spam folder, it's right 75% of the time. **One in four emails in the spam folder is a legitimate email.**"

That second sentence is what makes it real. If you send 400 emails to the spam folder a day, 100 of them are real emails your employees never see.

**Precision is entirely about False Positives.** Look at the formula — FN doesn't appear anywhere. Precision does not care how much spam you missed. It only cares whether your alarms are trustworthy.

**High precision = few false alarms = "when this thing fires, believe it."**

### Scenarios where precision is the metric

**Scenario A — Automatically suspending accounts for policy violations.**
A false positive means you've locked a paying customer out of their account and they're furious on Twitter within the hour. A false negative means one bad actor operates a bit longer. You would happily miss ten violations to avoid one wrongful ban. **Optimize precision. Set a very high threshold and route the uncertain cases to human review.**

**Scenario B — A recommendation carousel on an e-commerce homepage.**
You have six slots. Every irrelevant product in those slots costs you screen real estate and erodes trust in the recommendations. There are 50,000 products you could have shown — "missing" one costs you nothing because there are thousands of equally good alternatives. **Precision@6 is what matters. Recall is nearly meaningless here.**

**Scenario C — A drug with severe side effects.**
You only prescribe if you're confident the patient has the condition. Prescribing to a healthy person causes real harm. **Optimize precision.**

---

### 6.4 Recall — "of everything I should have caught, how much did I catch?"

**Formula:**
```
Recall = TP / (TP + FN)
```

Also called **Sensitivity** (medical literature) or **True Positive Rate / TPR** (signal detection, ROC curves). Three names, one formula — know all three because interviewers switch between them.

**Reading the denominator:** `TP + FN` is everything that was **genuinely positive** — the ones you caught plus the ones you missed. So recall asks: *of all the spam that existed, what fraction did I find?*

**Worked out:**
```
Recall = 30 / (30 + 20)
       = 30 / 50
       = 0.60  →  60%
```

**Translation:**
> "My filter catches 60% of all spam. **Four out of every ten spam emails still land in people's inboxes.**"

**Recall is entirely about False Negatives.** FP doesn't appear in the formula. Recall does not care how many legitimate emails you wrongly flagged. It only cares whether anything got past you.

**High recall = few misses = "if it exists, we'll find it."**

### Scenarios where recall is the metric

**Scenario A — Screening mammograms for breast cancer.**
A false negative means a woman with a tumour is told she's fine and comes back two years later with metastatic disease. A false positive means an anxious three weeks and a needle biopsy that turns out fine. These are not comparable. Screening programmes deliberately run at **very high recall (>95%) and accept low precision** — most people recalled for further testing turn out healthy, and that's the intended design. **Optimize recall.**

**Scenario B — Detecting money laundering for a regulator.**
Missing a real case means regulatory fines in the hundreds of crores plus criminal liability. A false positive means an analyst spends 20 minutes reviewing a clean transaction. **Optimize recall hard**, then make the review queue as efficient as possible.

**Scenario C — First-stage retrieval in a search pipeline.**
Your system retrieves 1,000 candidate documents, then a slower, smarter reranker picks the best 10. The first stage only needs to *not lose* the right document — the reranker will sort out the noise. **Optimize recall at stage 1, precision at stage 2.** This is the standard architecture for search and RAG, and being able to articulate it is a strong signal.

---

### 6.5 The precision–recall tradeoff, demonstrated

Here's the thing nobody tells beginners: **most classifiers don't output "spam" or "not spam." They output a probability.** 0.93, 0.41, 0.07. You then choose a **threshold** and call everything above it spam.

**That threshold is not part of the model. It's a business decision you make afterwards** — and it's what moves precision and recall around.

### A full worked demonstration

Ten emails. The model gives each a spam-probability. We know the truth.

| Email | Model's spam score | True label |
|---|---|---|
| E1 | 0.95 | **SPAM** |
| E2 | 0.90 | **SPAM** |
| E3 | 0.85 | legit |
| E4 | 0.80 | **SPAM** |
| E5 | 0.65 | **SPAM** |
| E6 | 0.55 | legit |
| E7 | 0.45 | **SPAM** |
| E8 | 0.30 | legit |
| E9 | 0.20 | legit |
| E10 | 0.10 | legit |

5 spam, 5 legit. Now let's sweep the threshold and recompute everything each time.

---

**Threshold = 0.90** → flag E1, E2

```
Flagged: E1(spam), E2(spam)                   → TP = 2, FP = 0
Missed spam: E4, E5, E7                       → FN = 3
Precision = 2/2 = 1.000   ← every flag correct!
Recall    = 2/5 = 0.400   ← but 3 spam got through
F1        = 2×(1.0×0.4)/(1.0+0.4) = 0.8/1.4 = 0.571
```

---

**Threshold = 0.80** → flag E1, E2, E3, E4

```
Flagged: E1(spam), E2(spam), E3(LEGIT), E4(spam)  → TP = 3, FP = 1
Missed spam: E5, E7                                → FN = 2
Precision = 3/4 = 0.750
Recall    = 3/5 = 0.600
F1        = 2×(0.75×0.60)/(1.35) = 0.90/1.35 = 0.667
```

---

**Threshold = 0.60** → flag E1–E5

```
Flagged: E1,E2,E4,E5 (spam) + E3 (LEGIT)      → TP = 4, FP = 1
Missed spam: E7                                → FN = 1
Precision = 4/5 = 0.800
Recall    = 4/5 = 0.800
F1        = 2×(0.8×0.8)/(1.6) = 1.28/1.6 = 0.800
```

---

**Threshold = 0.40** → flag E1–E7

```
Flagged: E1,E2,E4,E5,E7 (spam) + E3,E6 (LEGIT)  → TP = 5, FP = 2
Missed spam: none                                → FN = 0
Precision = 5/7 = 0.714
Recall    = 5/5 = 1.000   ← caught everything!
F1        = 2×(0.714×1.0)/(1.714) = 1.428/1.714 = 0.833
```

---

**Threshold = 0.05** → flag everything

```
TP = 5, FP = 5, FN = 0
Precision = 5/10 = 0.500
Recall    = 5/5  = 1.000
F1        = 2×(0.5×1.0)/1.5 = 0.667
```

---

### The full picture

| Threshold | Flagged | TP | FP | FN | **Precision** | **Recall** | **F1** |
|---|---|---|---|---|---|---|---|
| 0.90 | 2 | 2 | 0 | 3 | **1.000** | 0.400 | 0.571 |
| 0.80 | 4 | 3 | 1 | 2 | 0.750 | 0.600 | 0.667 |
| 0.60 | 5 | 4 | 1 | 1 | 0.800 | 0.800 | 0.800 |
| **0.40** | 7 | 5 | 2 | 0 | 0.714 | **1.000** | **0.833** ← best F1 |
| 0.05 | 10 | 5 | 5 | 0 | 0.500 | 1.000 | 0.667 |

**Read down the columns.** As the threshold falls: precision generally drifts **down**, recall marches **up**. They pull in opposite directions. That's the tradeoff, made concrete.

### The two degenerate extremes

You can always max out either metric by cheating:

- **Flag everything** → Recall = 100% guaranteed. Precision collapses to the base rate.
- **Flag only the single email you're most certain about** → Precision = 100% likely. Recall collapses to near zero.

**This is why precision or recall alone is never a valid claim.** "Our model has 99% recall!" means nothing until you say what the precision was at that operating point. Anyone can get 100% recall.

> **What to say in an interview:** "The threshold is not a property of the model — it's a business decision. I'd plot precision and recall across thresholds, attach a rupee cost to a false positive and a false negative, and pick the operating point that minimizes expected cost. And I'd report the metric *at a constrained operating point* — for example 'recall at 95% precision' — so two models can be compared on an axis the business can actually live with."

### Making the cost calculation concrete

Say for a fraud system: a missed fraud costs **₹8,000** on average; a false alarm costs **₹40** of analyst time.

At a given threshold you get FN = 120 missed frauds and FP = 9,000 false alarms:
```
Cost = (120 × 8,000) + (9,000 × 40)
     = 9,60,000 + 3,60,000
     = ₹13,20,000
```

Lower the threshold: FN drops to 60, FP rises to 22,000:
```
Cost = (60 × 8,000) + (22,000 × 40)
     = 4,80,000 + 8,80,000
     = ₹13,60,000   ← slightly worse
```

Now the threshold choice has an answer rather than a vibe. **This is what separating a metric from a decision looks like, and it's the kind of reasoning that gets you hired.**

---

### 6.6 F1 Score — combining precision and recall

Since you need both numbers, it's convenient to have one that summarizes them.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This is the **harmonic mean** (Section 2) of precision and recall.

**Worked out for our original 1,000-email example** (Precision = 0.75, Recall = 0.60):
```
Step 1 — multiply:   0.75 × 0.60 = 0.45
Step 2 — add:        0.75 + 0.60 = 1.35
Step 3 — divide:     0.45 / 1.35 = 0.3333
Step 4 — double:     2 × 0.3333  = 0.6667

F1 = 0.667
```

### Why harmonic and not just the average?

Compare the two means on a model with **perfect precision but useless recall** — precision = 1.00, recall = 0.01 (it flags one email, correctly, and misses 99% of spam):

```
Arithmetic mean:  (1.00 + 0.01) / 2 = 0.505    "hmm, about average"
Harmonic mean:    2 × (1.00 × 0.01) / (1.01)
                = 0.02 / 1.01
                = 0.0198                        "this is garbage"
```

**The arithmetic mean lets a perfect score cover for a catastrophic one. The harmonic mean refuses.** It's dragged toward whichever number is smaller.

Here's the behaviour across a range, so you can see the shape:

| Precision | Recall | Arithmetic | **F1** |
|---|---|---|---|
| 0.8 | 0.8 | 0.80 | **0.80** |
| 0.9 | 0.7 | 0.80 | **0.79** |
| 1.0 | 0.6 | 0.80 | **0.75** |
| 1.0 | 0.3 | 0.65 | **0.46** |
| 1.0 | 0.1 | 0.55 | **0.18** |

Every row has decent precision. F1 punishes the imbalance progressively. **You can only score a high F1 if both components are genuinely high.**

**Range:** 0 to 1. Higher is better.

### What F1 does NOT do

**F1 completely ignores True Negatives.** Look at the formula — TN appears nowhere in precision or recall.

Usually that's a feature: in imbalanced problems the huge pile of true negatives is uninformative, and excluding it stops the metric being inflated. But it means F1 is **asymmetric** — it evaluates your performance on the positive class only. If you swap which class you call "positive," F1 changes. Accuracy and MCC don't have this property.

**Practical consequence:** if both classes genuinely matter to you (say, classifying documents as "urgent" vs "routine," where both need to be right), F1 on one class is an incomplete picture. Use macro-F1 across both classes, or MCC.

---

### 6.7 F-beta — when precision and recall aren't equally important

F1 weights precision and recall **equally**. As Sections 6.3 and 6.4 showed, they're rarely equally important.

**Formula:**
```
Fβ = (1 + β²) × (Precision × Recall) / ((β² × Precision) + Recall)
```

`β` (beta) is a dial you set.

| β | Name | Effect | Use for |
|---|---|---|---|
| **0.5** | F0.5 | Precision counts **2× more** | Spam, recommendations, auto-moderation |
| **1** | F1 | Equal | Default when you have no information |
| **2** | F2 | Recall counts **2× more** | Cancer screening, fraud, safety |

**Memory hook:** β > 1 favours **R**ecall (a bigger β gives more **R**each).

### Worked example with our numbers (P = 0.75, R = 0.60)

**F2** (recall matters double):
```
β = 2, so β² = 4

Numerator   = (1 + 4) × (0.75 × 0.60) = 5 × 0.45 = 2.25
Denominator = (4 × 0.75) + 0.60 = 3.00 + 0.60 = 3.60

F2 = 2.25 / 3.60 = 0.625
```

**F0.5** (precision matters double):
```
β = 0.5, so β² = 0.25

Numerator   = (1 + 0.25) × 0.45 = 1.25 × 0.45 = 0.5625
Denominator = (0.25 × 0.75) + 0.60 = 0.1875 + 0.60 = 0.7875

F0.5 = 0.5625 / 0.7875 = 0.714
```

**Comparing all three:**
```
F0.5 = 0.714   (rewards us — our precision, 0.75, is the stronger number)
F1   = 0.667
F2   = 0.625   (punishes us — our recall, 0.60, is the weaker number)
```

The scores separate exactly as expected. Our model is precision-leaning, so the precision-weighted metric flatters it and the recall-weighted one doesn't.

**Practical interview point:** picking β is equivalent to declaring a cost ratio. Saying "we use F2" is saying "a miss is roughly twice as costly as a false alarm." If you can't defend that ratio, you can't defend the β.

---

### 6.8 Specificity — the forgotten fourth metric

**Formula:**
```
Specificity = TN / (TN + FP)
```

Also called **True Negative Rate (TNR)**.

*Of everything that was genuinely negative, what fraction did I correctly leave alone?*

**Worked out:**
```
Specificity = 940 / (940 + 10) = 940 / 950 = 0.9895  →  98.95%
```

**The symmetry worth internalizing:**

| | Positive class | Negative class |
|---|---|---|
| "Of the real ones, how many did I catch?" | **Recall** = TP/(TP+FN) | **Specificity** = TN/(TN+FP) |

Recall is to positives what specificity is to negatives. They're mirror images.

**False Positive Rate (FPR)** is specificity's complement:
```
FPR = 1 − Specificity = FP / (FP + TN) = 10 / 950 = 0.0105  →  1.05%
```

FPR is the x-axis of the ROC curve. You'll need it in a moment.

**Where specificity is the primary metric:** medical diagnostics. A doctor will ask "what's the sensitivity and specificity of this test?" — never "what's the F1?" In medicine the convention is firmly sensitivity/specificity, so know that vocabulary if you're interviewing in healthtech.

**A worked medical intuition — why high specificity matters more than you'd think.** Suppose a disease affects 1 in 10,000 people and your test has 99% sensitivity and 99% specificity. Test 1,000,000 people:

```
True cases:  100 people.  You catch 99 of them.        (TP = 99, FN = 1)
Healthy:     999,900.     You wrongly flag 1%.          (FP = 9,999, TN = 989,901)

Precision = 99 / (99 + 9,999) = 99 / 10,098 = 0.98%
```

**A test with 99% specificity produces 100 false alarms for every true case.** If you test positive, there's a 99% chance you're fine. This is the base-rate fallacy, it's a famous interview question, and it's why mass screening of rare diseases is so difficult. Specificity of 99% sounds excellent and is nowhere near good enough.

---

### 6.9 ROC curve and AUC

Everything so far required picking a threshold. **What if you want to evaluate the model itself, independent of any threshold?**

### How the ROC curve is built

1. Sort every prediction by its score, highest first.
2. Slide the threshold from 1.0 down to 0.0.
3. At each threshold compute **TPR (= Recall)** and **FPR (= 1 − Specificity)**.
4. Plot **TPR on the y-axis** against **FPR on the x-axis**.

### Reading the plot

```
      1.0 ┤        ╭─────────────  ← great model hugs the top-left
          │     ╭──╯
   T      │   ╭─╯          ╱
   P      │  ╱          ╱
   R      │ ╱        ╱         ← diagonal = random guessing
          │╱      ╱
      0.0 ┼────────────────────
          0.0      FPR      1.0
```

| Point on the plot | Threshold | What's happening |
|---|---|---|
| Bottom-left (0,0) | 1.0 | Flag nothing. No catches, no false alarms. |
| Top-right (1,1) | 0.0 | Flag everything. All catches, all false alarms. |
| **Top-left (0,1)** | — | **The dream.** Catch everything, zero false alarms. |
| **The diagonal y = x** | — | **Random guessing.** Your TPR only rises as fast as your FPR. |

**AUC = Area Under the Curve.** One number summarizing the whole thing.

| AUC | Verdict |
|---|---|
| 1.0 | Perfect separation |
| 0.9 – 1.0 | Excellent |
| 0.8 – 0.9 | Good |
| 0.7 – 0.8 | Fair |
| **0.5** | **Random. The model is worthless.** |
| < 0.5 | Worse than random — you've almost certainly flipped your labels |

### The interpretation to memorize

> **AUC is the probability that a randomly chosen positive example receives a higher score from the model than a randomly chosen negative example.**

AUC = 0.84 means: pick one spam email and one legit email at random; 84% of the time the model gives the spam a higher spam-score. **It's a measure of ranking quality.** It doesn't care what the raw probabilities are, only whether positives are ranked above negatives.

### Computing AUC by hand from our 10 emails

This makes it click. Ranked by score, positives (spam) at positions **1, 2, 4, 5, 7**; negatives at **3, 6, 8, 9, 10**.

We count, for every (positive, negative) pair, whether the positive ranked higher.

```
Total pairs = 5 positives × 5 negatives = 25

Positive at rank 1 → negatives below it: ranks 3,6,8,9,10  = 5 ✓
Positive at rank 2 → negatives below it: ranks 3,6,8,9,10  = 5 ✓
Positive at rank 4 → negatives below it: ranks 6,8,9,10    = 4 ✓ (loses to rank-3 negative)
Positive at rank 5 → negatives below it: ranks 6,8,9,10    = 4 ✓
Positive at rank 7 → negatives below it: ranks 8,9,10      = 3 ✓ (loses to ranks 3 and 6)

Correctly ordered pairs = 5 + 5 + 4 + 4 + 3 = 21

AUC = 21 / 25 = 0.84
```

**That's the whole thing.** AUC is literally counting how often the model got a pairwise comparison right. Three of our 25 comparisons went wrong because the legit email E3 (score 0.85) outranked three real spam emails.

### Strengths of AUC
- **Threshold-free** — compare models without committing to an operating point.
- Beautifully interpretable as a pairwise probability.
- Stable if the class balance shifts between your test set and production (it's computed from TPR and FPR, both of which are within-class rates).

### The critical weakness

**AUC is dangerously optimistic on imbalanced data.** Here's exactly why.

Look at FPR's denominator: `FP / (FP + TN)`. When negatives massively outnumber positives, TN is gigantic — and it **drowns out FP**.

**Concrete scenario — credit card fraud:**
```
1,000,000 transactions.  100 are fraud.  999,900 are legitimate.

Your model flags 10,000 transactions and catches 90 frauds.

TP = 90        FN = 10
FP = 9,910     TN = 989,990

TPR (Recall) = 90 / 100         = 0.900   ← looks fantastic
FPR          = 9,910 / 999,900  = 0.0099  ← looks fantastic
```

Both coordinates sit in the beautiful top-left corner. **AUC would be very high.** Executives would be delighted.

Now compute precision:
```
Precision = 90 / (90 + 9,910) = 90 / 10,000 = 0.009  →  0.9%
```

**99.1% of your fraud alerts are false alarms.** Your investigation team reviews 10,000 cases to find 90 frauds. They would quit. The system is operationally unusable, and ROC-AUC never hinted at it.

The reason: 9,910 false positives is an enormous, business-breaking number, but divided by 999,900 it becomes 0.0099 — invisible.

**This is the single most common "gotcha" in classification interviews.** Which brings us to:

---

### 6.10 Precision–Recall curve and PR-AUC

Same idea, different axes: plot **Precision (y)** against **Recall (x)** as you sweep the threshold. The area underneath is **PR-AUC**, often reported as **Average Precision (AP)**.

**Why it fixes the imbalance problem — look at what's in the formulas:**
```
Precision = TP / (TP + FP)     ← no TN
Recall    = TP / (TP + FN)     ← no TN
```

**True negatives appear nowhere.** The giant pile of correctly-ignored legitimate transactions cannot inflate your score. In the fraud example above, PR-AUC would immediately reveal the 0.9% precision disaster.

### Computing Average Precision from our 10 emails

Walk down the ranked list. **Every time you hit a positive, record the precision at that point.** Then average those recorded values.

| Rank | Email | Score | Truth | Positives so far | Precision here |
|---|---|---|---|---|---|
| 1 | E1 | 0.95 | **SPAM** | 1 | 1/1 = **1.000** ✓ record |
| 2 | E2 | 0.90 | **SPAM** | 2 | 2/2 = **1.000** ✓ record |
| 3 | E3 | 0.85 | legit | 2 | — (skip) |
| 4 | E4 | 0.80 | **SPAM** | 3 | 3/4 = **0.750** ✓ record |
| 5 | E5 | 0.65 | **SPAM** | 4 | 4/5 = **0.800** ✓ record |
| 6 | E6 | 0.55 | legit | 4 | — (skip) |
| 7 | E7 | 0.45 | **SPAM** | 5 | 5/7 = **0.714** ✓ record |

```
AP = (1.000 + 1.000 + 0.750 + 0.800 + 0.714) / 5
   = 4.264 / 5
   = 0.853
```

Notice the structure: **you only record precision at the positions where a positive appears.** Positives near the top of the list get high precision values recorded; positives buried deep get low ones. So AP directly rewards ranking positives high.

### The baseline differs — and this catches people out

| Metric | Random-guessing baseline |
|---|---|
| **ROC-AUC** | **Always 0.5**, regardless of class balance |
| **PR-AUC** | **The positive class rate** |

So if 1% of your data is fraud, a random model scores PR-AUC ≈ 0.01.

**A PR-AUC of 0.40 on a 1%-positive dataset is a 40× improvement over random — genuinely excellent work** — even though 0.40 *sounds* mediocre compared to an AUC of 0.95. Beginners see 0.40 and think they've failed. **Always state the baseline alongside PR-AUC.**

### The decision rule

| Situation | Use |
|---|---|
| Balanced classes, both matter equally | **ROC-AUC** |
| Imbalanced, you care about the rare positive class | **PR-AUC / Average Precision** |
| Explaining to a non-technical audience | Precision & Recall at a chosen threshold |
| Class balance will shift between test and production | ROC-AUC is more stable |

---

### 6.11 Log Loss (Cross-Entropy) — scoring the probabilities themselves

Every metric so far asked only *"was the label right?"* Log loss asks something more demanding: **"how confident were you, and did that confidence deserve to be there?"**

**Formula (binary case):**
```
Log Loss = −(1/n) × Σ [ yᵢ × log(pᵢ)  +  (1 − yᵢ) × log(1 − pᵢ) ]

where pᵢ = the model's predicted probability that example i is positive
      yᵢ = 1 if truly positive, 0 if truly negative
```

**The formula looks scary but it's a switch.** Because `y` is either 0 or 1, exactly one of the two terms survives:

```
If y = 1 (truly positive):   the second term × 0 vanishes  →  loss = −log(p)
If y = 0 (truly negative):   the first term  × 0 vanishes  →  loss = −log(1 − p)
```

**So the whole formula reduces to one instruction:**

> **Take the probability your model assigned to the CORRECT answer. Take its log. Negate it.**

That's it. The `y` and `(1−y)` are just a mathematician's way of writing an if-statement.

### Why the minus sign

A probability is between 0 and 1, so `log(p)` is always **negative** (Section 2, fact 2). Negating makes the loss positive, so lower is better.

### Worked table — the behaviour is the whole point

Suppose an email is genuinely spam (`y = 1`). Different models assign different probabilities:

| Model's `p(spam)` | Correct? | −log(p) | Verdict |
|---|---|---|---|
| 0.99 | ✅ yes | −ln(0.99) = **0.010** | Confident and right. Almost no penalty. |
| 0.90 | ✅ yes | −ln(0.90) = **0.105** | Good. |
| 0.60 | ✅ yes | −ln(0.60) = **0.511** | Right, but hedging. Moderate penalty. |
| 0.50 | ⚖️ coin flip | −ln(0.50) = **0.693** | No information whatsoever. |
| 0.40 | ❌ no | −ln(0.40) = **0.916** | Wrong, but at least unsure. |
| 0.10 | ❌ no | −ln(0.10) = **2.303** | Confidently wrong. Ouch. |
| 0.01 | ❌ no | −ln(0.01) = **4.605** | **Very confidently wrong. Brutal.** |
| 0.001 | ❌ no | −ln(0.001) = **6.908** | Catastrophic. |
| 0.000001 | ❌ no | **13.8** | It keeps going. Forever. |

**The defining property: log loss punishes confident wrong answers without limit.** As the probability assigned to the true answer approaches zero, the loss approaches **infinity**. There is no cap.

Compare this to accuracy, where predicting 0.49 and predicting 0.0001 for a positive example are *identically* wrong. Log loss says one is a near-miss and the other is a catastrophic failure of calibration.

### The baseline

A model that always outputs 0.5 (total ignorance) on balanced data gets:
```
Log Loss = −ln(0.5) = 0.693
```
**0.693 = ln(2) is your "learned nothing" baseline.** If your log loss is above 0.693 on balanced data, your model is worse than a coin flip.

### When you specifically need log loss

**Scenario — a fraud investigation queue.** You have 50 analysts and 200,000 daily transactions. You rank transactions by `P(fraud) × transaction_amount` to decide what to investigate first.

Here the **actual probability value matters**, not just whether it's above a threshold. If the model says 0.9 for cases that are really 0.3 likely, your expected-value calculation is wrong by a factor of three and you're prioritizing the wrong cases. **Log loss (and calibration) is the metric.**

**Scenario — a weather app showing "70% chance of rain."** The number is shown directly to users. It had better mean something.

**Scenario where log loss is the wrong choice:** when you only act on the final label and the probability is never used downstream. Then log loss is measuring precision of something you throw away.

**One practical warning:** log loss is extremely sensitive to a single catastrophic prediction. One example where the model assigned 0.0001 to the true class contributes 9.2 to the sum — potentially more than a thousand well-behaved predictions combined. Always check whether a bad log loss is a systematic problem or one broken row.

---

### 6.12 Brier Score and calibration

**Brier Score** = mean squared error applied to probabilities:
```
Brier = (1/n) × Σ (pᵢ − yᵢ)²
```

**Worked example** — four predictions:

| True (y) | Predicted (p) | (p − y)² |
|---|---|---|
| 1 | 0.90 | 0.01 |
| 1 | 0.60 | 0.16 |
| 0 | 0.20 | 0.04 |
| 0 | 0.70 | 0.49 |

```
Brier = (0.01 + 0.16 + 0.04 + 0.49) / 4 = 0.70 / 4 = 0.175
```

**Range: 0 to 1.** Lower is better. 0.25 is the "always predict 0.5" baseline.

**Brier vs log loss:** same goal, different temperament. Brier is **bounded** (worst case = 1.0) and so is gentler on confident mistakes. Log loss is **unbounded** and savage. Use Brier when you don't want one disastrous prediction to dominate; use log loss when you genuinely want to stamp out overconfidence.

### Calibration — a separate and important idea

> **A model is calibrated if, among all the cases where it said "70% chance," the thing actually happened about 70% of the time.**

**A model can have excellent AUC and terrible calibration.** This surprises people, so here's why:

AUC only cares about **ordering**. Suppose your model outputs 0.91, 0.89, 0.87 for three cases whose true probabilities are 0.30, 0.20, 0.10. The *ordering* is perfect — AUC = 1.0. But every single number is wildly overconfident. If anyone uses those probabilities for a decision, they'll be badly misled.

**How to check: a reliability diagram.**

1. Bucket your predictions: 0–10%, 10–20%, … 90–100%.
2. For each bucket, compute the average predicted probability (x-axis) and the actual observed frequency of positives (y-axis).
3. Plot. A perfectly calibrated model sits on the 45° diagonal.

```
Observed  1.0┤                   ╱
frequency    │                 ╱  ← perfect calibration
             │               ╱
          0.5┤        ╱    ╱
             │    ╱      ╱   ← model curve BELOW the line
             │  ╱      ╱        = OVERCONFIDENT
          0.0┼────────────────
             0.0    0.5    1.0
              Predicted probability
```

A curve **below** the diagonal = overconfident (says 80%, happens 60% of the time). **Above** = underconfident.

**Expected Calibration Error (ECE)** turns this into one number: the weighted average gap between the two across buckets.

**How to fix miscalibration:**
- **Platt scaling** — fit a small logistic regression that maps your model's raw scores to calibrated probabilities. Simple, works well, needs little data.
- **Isotonic regression** — fit a flexible monotonic step function. More powerful, needs more data, can overfit on small validation sets.

Both are fitted on **validation data**, never on training data.

**Why this comes up:** modern neural networks are notoriously overconfident out of the box. A deep model routinely outputs 0.99 for cases it gets wrong. If your probabilities feed a downstream decision, calibration isn't optional.

---

### 6.13 Metrics built for imbalance

#### Balanced Accuracy
```
Balanced Accuracy = (Recall + Specificity) / 2
```

Average the per-class performance, so the majority class can't dominate.

**Our example:**
```
(0.60 + 0.9895) / 2 = 0.795  →  79.5%
```

Compare: plain accuracy said 97%. Balanced accuracy says 79.5%. **The second number is the honest one**, because it gives the 50 spam emails the same weight as the 950 legitimate ones.

And the do-nothing baseline model? `(0 + 1.0)/2 = 0.50` — correctly identified as random-level, where plain accuracy gave it 95%.

#### Matthews Correlation Coefficient (MCC)

```
MCC = (TP×TN − FP×FN) / √[ (TP+FP) × (TP+FN) × (TN+FP) × (TN+FN) ]
```

Ugly, but it's the only common binary metric that uses **all four cells** and stays honest under imbalance.

**Worked out on our example** (TP=30, FP=10, FN=20, TN=940):

```
Numerator:
  (30 × 940) − (10 × 20)
  = 28,200 − 200
  = 28,000

Denominator — four terms, each a row or column total:
  (TP + FP) = 30 + 10  = 40      ← everything flagged positive
  (TP + FN) = 30 + 20  = 50      ← everything truly positive
  (TN + FP) = 940 + 10 = 950     ← everything truly negative
  (TN + FN) = 940 + 20 = 960     ← everything flagged negative

  Product = 40 × 50 × 950 × 960 = 1,824,000,000
  √1,824,000,000 ≈ 42,708

MCC = 28,000 / 42,708 = 0.6556
```

**Range: −1 to +1.**

| MCC | Meaning |
|---|---|
| +1 | Perfect prediction |
| 0 | **No better than random** |
| −1 | Perfectly wrong (every prediction inverted) |

**Sanity-check it against the do-nothing model** (TP=0, FP=0, FN=50, TN=950):
```
Numerator = (0 × 950) − (0 × 50) = 0
MCC = 0
```
**MCC correctly gives the useless model a score of exactly zero**, while accuracy gave it 95%. That's the argument for MCC in one line.

**Why researchers increasingly recommend it as the default:** unlike F1, it's symmetric (swapping which class is "positive" doesn't change it), and unlike accuracy it can't be inflated by a majority class. It only produces a high score when the model does well on **both** classes.

#### Cohen's Kappa
```
Kappa = (observed accuracy − expected accuracy by chance) / (1 − expected accuracy by chance)
```

Measures agreement **above what chance would produce**.

**Worked intuition:** suppose your observed accuracy is 0.85, and given the class distribution, random guessing would produce 0.70 by chance alone.
```
Kappa = (0.85 − 0.70) / (1 − 0.70) = 0.15 / 0.30 = 0.50
```
"You captured half of the improvement that was available above chance."

**Rough interpretation scale:** < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 near-perfect.

**Its other major use:** measuring agreement **between two human annotators**, which tells you whether your labels are trustworthy in the first place. If two doctors only reach κ = 0.4 on whether an X-ray shows pneumonia, **no model can exceed that ceiling**, and you should say so before anyone sets a target of 95% accuracy. This is a very strong point to raise in an interview about data quality.

---

### 6.14 Multi-class classification

Now 3+ mutually exclusive classes: {cat, dog, bird}. The confusion matrix becomes N×N.

**Example 3×3 confusion matrix** (rows = truth, columns = prediction):

|  | Pred: Cat | Pred: Dog | Pred: Bird |
|---|---|---|---|
| **True: Cat** | **850** | 40 | 10 |
| **True: Dog** | 15 | **70** | 5 |
| **True: Bird** | 3 | 5 | **2** |

Read the diagonal for correct predictions, off-diagonal for confusions. Immediately visible: **birds are a disaster** (2 correct out of 10), and the model mostly guesses dog or cat instead.

### One-vs-rest

Precision, recall, and F1 are defined per class using **one-vs-rest**: treat "cat" as positive and {dog, bird} together as negative, compute the metrics; repeat for each class.

For **Cat**:
```
TP = 850
FP = 15 + 3 = 18       (dogs and birds wrongly called cat — column sum minus diagonal)
FN = 40 + 10 = 50      (cats called something else — row sum minus diagonal)

Precision = 850/868 = 0.979
Recall    = 850/900 = 0.944
F1        = 2 × (0.979 × 0.944)/(1.923) = 0.961
```

For **Bird**:
```
TP = 2
FP = 10 + 5 = 15
FN = 3 + 5 = 8

Precision = 2/17 = 0.118
Recall    = 2/10 = 0.200
F1        = 2 × (0.118 × 0.200)/(0.318) = 0.148
```

Now you have three F1 scores and need to combine them.

### The three averaging methods

Using these per-class results:

| Class | Support (count) | F1 |
|---|---|---|
| Cat | 900 | 0.961 |
| Dog | 90 | ~0.700 |
| Bird | 10 | 0.148 |

**Macro-average** — plain unweighted mean:
```
(0.961 + 0.700 + 0.148) / 3 = 1.809 / 3 = 0.603
```
**Every class counts equally regardless of size.** Bird's failure is fully visible and drags the score down hard. **Use when rare classes matter as much as common ones** — rare disease types, minority language dialects, low-frequency but high-value product categories.

**Weighted-average** — weighted by class size:
```
(0.961×900 + 0.700×90 + 0.148×10) / 1000
= (864.9 + 63.0 + 1.5) / 1000
= 929.4 / 1000
= 0.929
```
**Reflects what a typical user experiences**, since most inputs are cats. But look what it did: **0.929 vs 0.603.** Bird's catastrophic performance is almost completely invisible. A model that had *entirely given up on birds* would still score ~0.928 here.

**Micro-average** — pool all TPs, FPs, FNs across classes and compute once:
```
Total TP = 850 + 70 + 2 = 922
Micro-F1 = 922 / 1000 = 0.922
```
In single-label multi-class, **micro-F1 exactly equals accuracy**. Use when every individual prediction matters equally and you don't care about class structure.

### The takeaway

```
Macro  = 0.603    ← exposes the broken class
Micro  = 0.922    ← same as accuracy
Weighted = 0.929  ← hides the broken class
```

**Same model. Three numbers ranging from 0.60 to 0.93.** Whichever one you report shapes the entire conversation, which is why you should report all three plus the per-class table.

> **Interview answer:** "I'd lead with macro-F1 because it's the one that exposes a model quietly ignoring a minority class. I'd show weighted-F1 alongside so stakeholders see the typical-user experience, and I'd always include the per-class breakdown with support counts — an averaged number can hide a class the model has completely given up on, and in this case it does exactly that."

### Top-k accuracy

In a 1,000-class image classifier, **Top-5 accuracy** asks: is the correct answer among my top 5 guesses?

**When it's legitimate:** when a human picks from your suggestions (autocomplete, product search, photo tagging where the user confirms), or when classes are genuinely confusable (distinguishing 120 dog breeds). ImageNet's standard metric is Top-5 for exactly this reason.

**When it's a cop-out:** when the system acts automatically on the top-1 prediction. If nothing downstream sees positions 2–5, top-5 accuracy is measuring something nobody uses.

---

### 6.15 Multi-label classification

Different from multi-class: **each example can have several labels at once.** A news article tagged {politics, economy, Europe}. A photo tagged {beach, sunset, people}.

**Worked example.** True labels: `{politics, economy, Europe}`. Predicted: `{politics, economy, sports}`.

Across 5 possible tags (politics, economy, Europe, sports, health), as binary vectors:
```
True:      [1, 1, 1, 0, 0]
Predicted: [1, 1, 0, 1, 0]
            ✓  ✓  ✗  ✗  ✓
```

| Metric | Computation | Result |
|---|---|---|
| **Subset Accuracy (Exact Match)** | Did the whole set match exactly? | **0** — one tag wrong = total failure |
| **Hamming Loss** | Fraction of individual slots wrong: 2 wrong of 5 | **0.40** (lower is better) |
| **Samples-F1** | Per-example F1: P=2/3, R=2/3 → F1 = **0.667** | 0.667 |

**Subset accuracy is brutally strict** — getting 4 of 5 tags right scores exactly the same as getting 0 right: zero. Only use it when partial credit is genuinely worthless.

**Hamming loss is forgiving and granular** but can be misleading when most labels are absent: predicting "no tags at all" gives excellent Hamming loss if articles average 3 tags out of 500 possible.

| Metric | Best for |
|---|---|
| **Subset Accuracy** | All-or-nothing tasks |
| **Hamming Loss** | Granular view, but watch the all-zeros trap |
| **Micro-F1** | Overall tagging quality; dominated by frequent tags |
| **Macro-F1** | Exposes rare tags |
| **Samples-F1** | "How good was a typical document's tagging" |

---

### 6.16 Scenario-based practice: choosing a classification metric

**Scenario 1 — A hospital screening chest X-rays for tuberculosis in a high-burden region.**
*Think:* Prevalence maybe 2%. A missed TB case means an untreated, contagious patient. A false positive means a confirmatory test — cheap and safe. Radiologists are scarce, so the model is a triage filter, not a final decision.
*Answer:* **Recall (sensitivity) is the primary metric, with a floor** — e.g. "recall ≥ 98%." Report **precision at that recall** to show the workload created. Use **PR-AUC** for model selection (2% prevalence makes ROC-AUC flattering). Add **specificity**, because the medical audience expects sensitivity/specificity language. Absolutely check **per-subgroup recall** by age, sex and imaging device — a model that works on new machines and fails on old ones in rural clinics is worse than useless.

**Scenario 2 — An e-commerce site auto-blocking fraudulent orders at checkout.**
*Think:* Fraud is ~0.3%. A blocked legitimate order means a lost sale *and* an angry customer who may never return — expensive. A missed fraud means a chargeback. There's a middle path: send borderline cases to manual review.
*Answer:* This is a **three-way decision**, not binary: auto-approve / review / auto-block. Use **two thresholds**. Report **precision at the auto-block threshold (target > 95%)** — you must almost never block a real customer. Report **recall at the review threshold** — the review queue should catch most of the rest. Track **review queue volume** as a capacity constraint. Use **PR-AUC** for model comparison. Framing it as a three-way decision instead of forcing a binary is the answer that distinguishes a strong candidate.

**Scenario 3 — A bank's credit default model determining loan interest rates.**
*Think:* The output is not a yes/no. The probability feeds directly into pricing: `interest_rate = f(P(default))`. The number itself must be right.
*Answer:* **Log loss or Brier score as primary, plus an explicit calibration analysis** (reliability diagram + ECE). AUC is useful for feature-selection work but is *not* sufficient — a model can rank perfectly and be systematically overconfident, which would mis-price every loan. Also mandatory here: **fairness metrics across protected groups**, since this is a regulated lending decision.

**Scenario 4 — Classifying support tickets into 40 categories for routing.**
*Think:* Multi-class with a long tail — 5 categories cover 80% of tickets, 20 categories are rare. Misrouting a rare-but-critical category (e.g. "security incident") is much worse than misrouting a common one.
*Answer:* **Macro-F1** as the headline, because the rare categories are the ones that hurt. Report the **full per-class table with support counts**. Consider **per-class recall targets** for the critical categories specifically. And consider **top-3 accuracy** if the routing UI can suggest three options to a human agent.

**Scenario 5 — Content moderation flagging hate speech on a social platform.**
*Think:* Both errors are politically and reputationally costly. Wrongly removing legitimate speech makes headlines. Leaving hate speech up also makes headlines. Volume is enormous so full human review is impossible.
*Answer:* Tiered again: **high-precision auto-removal for the most confident cases, a human-review band in the middle, auto-allow below.** Report **precision in the auto-remove band (target > 99%)** and **recall overall**. Critically, report **per-group performance** — false positive rates often differ dramatically across dialects and languages, and an aggregate F1 completely hides that. Mentioning subgroup analysis unprompted is a strong signal.

---

### 6.17 Classification decision tree

```
STEP 1: What's your class balance?
Compute:  majority_class_count / total_count
This is your accuracy baseline. If it's above ~80%, treat the data as imbalanced.

STEP 2: Balanced data, both errors cost the same?
└── → Accuracy + confusion matrix. Done.

STEP 3: Imbalanced, or errors cost differently ↓

    Which error hurts more?
    ├── False alarms hurt more (banning users, spam folder, recommendations)
    │   → Precision, F0.5, and "Precision at fixed Recall"
    │
    ├── Misses hurt more (disease, fraud, safety, first-stage retrieval)
    │   → Recall, F2, and "Recall at fixed Precision"
    │
    └── Genuinely both
        → F1, MCC, PR-AUC

STEP 4: Do you need the probability value itself downstream?
└── → Log Loss or Brier + a reliability diagram. AUC alone is NOT enough.

STEP 5: Comparing models without committing to a threshold?
├── Balanced   → ROC-AUC
└── Imbalanced → PR-AUC  (and state the positive-rate baseline!)

STEP 6: More than two classes?
└── → Macro-F1 (headline) + weighted-F1 + the full per-class table

ALWAYS: report the confusion matrix, the baseline, and per-subgroup
performance. A single averaged number hides more than it reveals.
```

---

# PART III — SPECIALIZED TASKS

## 7. Ranking, search & embedding retrieval

This covers search engines, recommendation systems, and **vector/embedding retrieval** — which is the retrieval half of every RAG system.

### What makes ranking fundamentally different

In classification, each prediction stands alone. In ranking, **the model returns an ordered list, and position matters enormously.**

Real user behaviour on a search results page:
```
Position 1  →  ~30% of clicks
Position 2  →  ~15%
Position 3  →  ~10%
Position 5  →  ~5%
Position 10 →  ~2%
Position 20 →  ~0.5%
Position 50 →  essentially zero
```

So putting the right document at position 1 rather than position 10 is worth roughly 15× more. **A metric that treats a hit at position 1 the same as a hit at position 30 is measuring the wrong thing.** Everything in this section is about encoding position-awareness.

### Our running example

A user searches for something. The system returns 5 documents. `R` = relevant, `N` = not relevant.

```
Rank:      1    2    3    4    5
Result:    R    N    R    N    R
           ✓    ✗    ✓    ✗    ✓
```

Assume there are exactly **3 relevant documents in the entire collection**, and all three happen to appear in our top 5.

---

### 7.1 Precision@k and Recall@k

**Precision@k** = (relevant items in the top k) ÷ k

*"Of what I showed you, how much was useful?"*

```
P@1 = 1 relevant in top 1  ÷ 1 = 1.000
P@3 = 2 relevant in top 3  ÷ 3 = 0.667
P@5 = 3 relevant in top 5  ÷ 5 = 0.600
```

**Recall@k** = (relevant items in the top k) ÷ (total relevant in the whole collection)

*"Of everything useful that exists, how much did I surface?"*

```
R@1 = 1 ÷ 3 = 0.333
R@3 = 2 ÷ 3 = 0.667
R@5 = 3 ÷ 3 = 1.000
```

**Notice the pattern:** as k grows, **precision tends to fall** (you're adding lower-quality results) while **recall can only rise or stay flat** (more chances to find things). Same tradeoff as Section 6.5, now expressed through k instead of a threshold.

### Choosing k — this matters more than people think

**k should match reality, not be a round number you liked.**

| Context | Sensible k | Why |
|---|---|---|
| Google-style search page | 10 | That's what fits above the fold |
| Mobile app search | 3–5 | Smaller screen, less scrolling |
| Recommendation carousel | 6 | That's how many slots exist |
| **RAG chunk retrieval** | **3–10** | **However many chunks you paste into the prompt** |
| First-stage retrieval before reranking | 100–1000 | You only need to not lose the answer |

> **Recall@k is THE metric for the retrieval stage of a RAG system.** If `Recall@5 = 0.6`, then **40% of the time the answer simply is not in the context you handed to the LLM.** No prompt engineering, no better model, no temperature tweak will fix that. The information isn't there. Always diagnose retrieval before blaming the generator — and saying this unprompted is one of the highest-signal things you can say in a RAG interview.

### Hit Rate @ k

Did *at least one* relevant item appear in the top k? Binary per query (1 or 0), then averaged across all queries.

**Worked example** — 5 queries:
```
Q1: relevant item found at rank 2  → hit  (1)
Q2: relevant item found at rank 7  → miss (0)  [if k=5]
Q3: relevant item found at rank 1  → hit  (1)
Q4: nothing relevant retrieved     → miss (0)
Q5: relevant item found at rank 4  → hit  (1)

Hit Rate@5 = 3/5 = 0.60
```

Simple, intuitive, very common in recommender systems. **Its weakness:** it ignores position entirely within the top k and ignores how many relevant items you found. A hit at rank 1 and a hit at rank 5 score identically.

---

### 7.2 MRR — Mean Reciprocal Rank

**The question it answers:** *How far down the list is the FIRST correct answer?*

**Formula:**
```
MRR = (1/Q) × Σ (1 / rankᵢ)

where rankᵢ = position of the first relevant result for query i
      Q     = number of queries
```

**"Reciprocal" just means 1 divided by it.** The reciprocal of 4 is 1/4.

**Why reciprocal?** It creates a steep decay that matches user patience:

| First relevant result at rank | Reciprocal rank | Credit you get |
|---|---|---|
| 1 | 1/1 = **1.000** | Full |
| 2 | 1/2 = **0.500** | Half |
| 3 | 1/3 = **0.333** | A third |
| 5 | 1/5 = **0.200** | A fifth |
| 10 | 1/10 = **0.100** | A tenth |
| Not found | 0 | Nothing |

Dropping from rank 1 to rank 2 costs you **half your score.** That's aggressive, and deliberately so.

**Worked example across 3 queries:**

| Query | Results | First relevant at | RR |
|---|---|---|---|
| "how to reset password" | N **R** N N N | rank 2 | 1/2 = 0.500 |
| "refund policy" | **R** N R N N | rank 1 | 1/1 = 1.000 |
| "contact support" | N N **R** N N | rank 3 | 1/3 = 0.333 |

```
MRR = (0.500 + 1.000 + 0.333) / 3
    = 1.833 / 3
    = 0.611
```

**Interpretation:** the first good answer sits, on average, around position `1/0.611 ≈ 1.6`.

### When MRR is right — and when it's badly wrong

**MRR ignores everything after the first hit.** Query 2 above had a second relevant result at rank 3, and MRR gave it zero credit.

**Use MRR when there is essentially ONE right answer:**
- Question answering ("what year did the Taj Mahal get built?")
- Navigational search ("facebook login" — you want the login page, not ten of them)
- Looking up a specific customer, order, or document
- Autocomplete

**Don't use MRR when multiple results matter:**
- Shopping ("running shoes" — the user wants to see several options)
- Research ("papers on transformers" — you want many)
- **RAG retrieval where the answer spans multiple chunks** — MRR would say you succeeded after finding one chunk, when you needed three

---

### 7.3 MAP — Mean Average Precision

**The question it answers:** *Did I rank ALL the relevant items high, not just the first one?*

**How to compute Average Precision for one query:** walk down the ranked list. **Every time you land on a relevant document, write down Precision@that position.** At the end, average all the numbers you wrote down, dividing by the total number of relevant documents.

**Worked example** with our `R N R N R` list (3 relevant documents total):

| Rank | Result | Relevant? | Running count | Precision here | Record it? |
|---|---|---|---|---|---|
| 1 | **R** | ✅ | 1 relevant so far | 1/1 = **1.000** | ✓ |
| 2 | N | ❌ | 1 | — | skip |
| 3 | **R** | ✅ | 2 | 2/3 = **0.667** | ✓ |
| 4 | N | ❌ | 2 | — | skip |
| 5 | **R** | ✅ | 3 | 3/5 = **0.600** | ✓ |

```
AP = (1.000 + 0.667 + 0.600) / 3
   = 2.267 / 3
   = 0.756
```

### Seeing how AP rewards better ranking

Put the same three relevant documents at ranks 1, 2, 3 instead:

| Rank | Result | Precision |
|---|---|---|
| 1 | R | 1/1 = 1.000 |
| 2 | R | 2/2 = 1.000 |
| 3 | R | 3/3 = 1.000 |

```
AP = (1.000 + 1.000 + 1.000) / 3 = 1.000    ← perfect
```

Now push them to ranks 3, 4, 5:

| Rank | Result | Precision |
|---|---|---|
| 3 | R | 1/3 = 0.333 |
| 4 | R | 2/4 = 0.500 |
| 5 | R | 3/5 = 0.600 |

```
AP = (0.333 + 0.500 + 0.600) / 3 = 0.478
```

**Same three documents, all three retrieved in every case. AP goes 1.000 → 0.756 → 0.478 purely from ordering.** That's the metric doing its job.

**MAP** = the mean of AP across all your queries.

**Use MAP when:** relevance is **binary** (relevant or not — no degrees) and multiple items are relevant. It's the standard in information retrieval benchmarks and, as **mAP**, in object detection.

---

### 7.4 NDCG — Normalized Discounted Cumulative Gain

The most sophisticated ranking metric and the one production search teams actually use. It handles **graded relevance** — not just relevant/irrelevant, but *how* relevant.

**Why graded relevance matters.** A user searches "python tutorial":
```
Score 3 — the official Python beginner's tutorial      (perfect)
Score 2 — a well-reviewed third-party Python course    (good)
Score 1 — a blog post mentioning Python in passing     (marginal)
Score 0 — an article about snakes                      (irrelevant)
```
MAP would lump 3, 2 and 1 together as simply "relevant" and throw away the distinction. NDCG keeps it.

### Building the name up, one word at a time

**Gain** — the relevance score of a document. A "perfect" match has gain 3.

**Cumulative Gain** — add up the gains of your top k:
```
CG@3 for [3, 2, 1] = 3 + 2 + 1 = 6
CG@3 for [1, 2, 3] = 1 + 2 + 3 = 6     ← identical!
```
**Problem: CG completely ignores order.** Useless on its own. Hence:

**Discounted** — divide each gain by `log₂(rank + 1)`, so deeper positions count for less:

| Rank | log₂(rank+1) | Divisor | Credit multiplier |
|---|---|---|---|
| 1 | log₂(2) = 1.000 | 1.000 | **100%** |
| 2 | log₂(3) = 1.585 | 1.585 | **63.1%** |
| 3 | log₂(4) = 2.000 | 2.000 | **50.0%** |
| 4 | log₂(5) = 2.322 | 2.322 | **43.1%** |
| 5 | log₂(6) = 2.585 | 2.585 | **38.7%** |
| 10 | log₂(11) = 3.459 | 3.459 | **28.9%** |
| 20 | log₂(21) = 4.392 | 4.392 | **22.8%** |

**Why a logarithm and not something simpler?** Look at the decay shape. From rank 1 to rank 2 you lose 37% of your credit — a big drop, because users really do notice the difference between first and second. From rank 10 to rank 20 you lose only 6 percentage points — a small drop, because by then the user has probably stopped looking and position 15 vs 16 is irrelevant. **The log curve is steep at the top and flat at the bottom, which mirrors real attention.**

Contrast with MRR's `1/rank`, which drops 50% between rank 1 and 2 — much harsher. NDCG's gentler decay is more appropriate when users genuinely scan several results.

**Normalized** — divide by the score of the *perfect* ranking (called **IDCG**, Ideal DCG), so the result lands between 0 and 1 and is comparable across queries with different numbers of relevant documents.

### Full worked example

Your system returns 3 documents with relevance scores **3, 2, 3** in that order.

**Step 1 — compute DCG (your actual ranking):**
```
Rank 1: gain 3, divisor log₂(2) = 1.000  →  3 / 1.000 = 3.000
Rank 2: gain 2, divisor log₂(3) = 1.585  →  2 / 1.585 = 1.262
Rank 3: gain 3, divisor log₂(4) = 2.000  →  3 / 2.000 = 1.500

DCG = 3.000 + 1.262 + 1.500 = 5.762
```

**Step 2 — compute IDCG (the best possible ordering of these same documents).** Sort the gains descending: **3, 3, 2**.
```
Rank 1: 3 / 1.000 = 3.000
Rank 2: 3 / 1.585 = 1.893
Rank 3: 2 / 2.000 = 1.000

IDCG = 3.000 + 1.893 + 1.000 = 5.893
```

**Step 3 — normalize:**
```
NDCG = DCG / IDCG = 5.762 / 5.893 = 0.978
```

**Interpretation:** 0.978 is very good. You retrieved exactly the right documents; you just had ranks 2 and 3 swapped, which costs a little.

### Seeing NDCG punish a bad ordering

Same three documents, worst possible order — **2, 3, 3**:
```
DCG = 2/1.000 + 3/1.585 + 3/2.000
    = 2.000 + 1.893 + 1.500
    = 5.393

NDCG = 5.393 / 5.893 = 0.915
```

And if you'd instead retrieved three *irrelevant* documents (gains 0, 0, 0):
```
DCG = 0  →  NDCG = 0
```

**Use NDCG when:** relevance has degrees and full ordering matters. **It's the default for web search, e-commerce search, and feed ranking.** When a company says "we improved search by 3%," they almost always mean NDCG@10 went up by 3%.

**One caveat worth mentioning:** NDCG requires graded relevance labels, which means human annotators rating documents — expensive. Many teams derive proxy grades from click data (clicked = 1, clicked and dwelled long = 2, purchased = 3). That's practical but introduces **position bias**: documents shown at position 1 get clicked more *because* they were at position 1, not necessarily because they're better. Correcting for this (inverse propensity weighting) is a real and interview-worthy topic.

---

### 7.5 Metrics specific to vector / embedding retrieval

Beyond ranking quality, vector databases have engineering concerns that are just as important.

**Why "approximate" nearest neighbour exists.** Finding the true nearest vectors means comparing your query to every single one of, say, 50 million vectors. That's exact but slow — hundreds of milliseconds. **ANN indexes (HNSW, IVF, ScaNN) trade a little accuracy for a lot of speed**, returning results in 2–5 ms. The question is: how much accuracy did you give up?

| Metric | What it measures | Why it matters |
|---|---|---|
| **ANN Recall@k** | Of the *true* k nearest neighbours (found by slow exhaustive search), how many did the fast index actually return? | 0.95 recall means you're **silently** losing 5% of your best matches. Nothing errors out; the results are just quietly worse. |
| **Latency p50 / p95 / p99** | Response time at the median, 95th and 99th percentile | **Always report p95/p99, never the mean.** Mean latency of 20 ms sounds great; if p99 is 2,000 ms, one in every hundred users waits two seconds. |
| **QPS (throughput)** | Queries per second at a given recall level | The real comparison axis: "95% recall at 2,000 QPS on one machine." |
| **Index build time** | How long to construct the index | HNSW can take hours on large datasets. Matters if you re-index frequently. |
| **Memory footprint** | RAM needed to hold the index | Often the actual binding constraint. HNSW is fast but memory-hungry. |

### The tradeoff you must be able to articulate

**ANN search is a curve, not a point.** HNSW has a parameter `efSearch`: raise it and the index explores more of the graph, giving better recall but higher latency.

```
efSearch = 32   →  Recall 0.88,  p95 latency  3 ms
efSearch = 64   →  Recall 0.94,  p95 latency  6 ms
efSearch = 128  →  Recall 0.97,  p95 latency 11 ms
efSearch = 512  →  Recall 0.995, p95 latency 40 ms
```

> **Quoting recall without latency is meaningless**, and vice versa. The correct statement is always "X% recall at Y ms p95." If someone says "our index has 99% recall," the right follow-up is "at what latency and what QPS?"

### Evaluating the embedding model itself (not the index)

- **MTEB (Massive Text Embedding Benchmark)** — the standard public leaderboard, covering retrieval, clustering, classification, reranking and semantic similarity across many datasets. The first place to look when choosing an embedding model.
- **Spearman correlation on STS (Semantic Textual Similarity)** — humans rate how similar sentence pairs are; you check whether the model's cosine similarities **rank** the pairs the same way. Spearman (rank correlation) rather than Pearson, because you care about ordering, not absolute values.
- **Alignment and Uniformity** — research diagnostics. *Alignment*: are semantically similar items actually close? *Uniformity*: is the embedding space well spread out, or has it collapsed into a narrow cone where everything is similar to everything? Collapse is a real failure mode in poorly-trained embedding models.

**An important practical point:** a model that tops MTEB may not be best for *your* data. MTEB is mostly general web text. If you're retrieving over legal contracts or Telugu medical records, **build a small domain evaluation set of 100–300 query/document pairs and test on that.** Saying this shows you understand benchmarks are a starting point, not an answer.

---

### 7.6 Recommendation-system-specific metrics

Relevance isn't everything in recommenders. **A system that recommends only the top-10 bestsellers to everyone will score respectably on NDCG and be commercially useless** — those items would have been found anyway, and 99% of your catalogue is dead inventory.

| Metric | What it measures | Why you'd care |
|---|---|---|
| **Coverage** | What % of the catalogue ever gets recommended to anyone? | Low coverage = the long tail is invisible = wasted inventory |
| **Diversity** | How dissimilar are items *within* one recommendation list? Usually `1 − average pairwise similarity` | Stops "you watched one cooking video, here are ten cooking videos" |
| **Novelty** | Are you recommending things the user likely hasn't already seen? Often measured via inverse item popularity | Recommending the #1 bestseller adds no information |
| **Serendipity** | Relevant **and** unexpected | The actual goal. Genuinely hard to measure automatically. |
| **Popularity bias** | Is the model over-recommending already-popular items relative to their true quality? | Creates a rich-get-richer feedback loop |

**Scenario making this concrete.** A music streaming service optimizes NDCG. NDCG goes up 8%. Six months later, listening hours are flat and artist-discovery complaints have doubled. What happened? The model learned that recommending already-popular songs is the safest way to score well offline — users do click them — so it stopped surfacing new artists. Offline NDCG rose; the product got worse.

> **Interview answer:** "Offline relevance metrics reward recommending what the user would have found anyway, and they're computed on logged data generated by the previous system, so they systematically favour popularity-biased models. I'd pair NDCG with coverage and intra-list diversity, and validate with an online A/B test on long-horizon engagement rather than next-session clicks."

---

### 7.7 Ranking decision tree

```
Is there exactly ONE correct answer per query?
└── → MRR
    (QA, navigational search, lookups, autocomplete)

Multiple correct answers, relevance is binary (yes/no)?
└── → MAP
    (or just Recall@k if you only need "did we find it")

Relevance comes in degrees (perfect / good / fair / bad)?
└── → NDCG@k
    (web search, e-commerce, feed ranking — the production default)

Evaluating the retriever inside a RAG pipeline?
└── → Recall@k (primary — this is your ceiling)
    + MRR (secondary)
    + Context Precision (Section 12)

Evaluating a vector index as an engineering artifact?
└── → ANN Recall@k at a FIXED p95 latency
    + QPS + memory footprint

Building a recommender?
└── → NDCG@k + Coverage + Diversity
    and don't trust any of it until the A/B test
```

---

## 8. Time series forecasting

Time series is where beginners make the most **damaging** mistakes — and the worst ones aren't about metrics at all, they're about the evaluation setup.

### 8.1 The cardinal rule: never shuffle

Standard k-fold cross-validation randomly shuffles your data. On time series this is **catastrophic**.

**Here's exactly what goes wrong.** You have daily sales data for a year. Random shuffling produces a training set containing March, May, July and October, and a test set containing April, June and November.

**Your model is now using July's data to predict April.** It has seen the future. In production it obviously cannot do this.

**Why the damage is so severe:** time series are autocorrelated — today's value is very close to yesterday's and tomorrow's. If the model has seen days 100 and 102 during training, predicting day 101 is nearly trivial interpolation. Your offline RMSE looks superb. Then you deploy, the model has to genuinely extrapolate into unseen future, and performance collapses. **You will not find out until it's in production.**

### The correct approach: rolling-origin (walk-forward) validation

```
Fold 1:  TRAIN [Jan─Mar]        TEST [Apr]
Fold 2:  TRAIN [Jan────Apr]     TEST [May]
Fold 3:  TRAIN [Jan─────May]    TEST [Jun]
Fold 4:  TRAIN [Jan──────Jun]   TEST [Jul]
                                 ↑
         training always ends BEFORE the test period begins
```

Average the scores across folds. This mimics what actually happens in production: you retrain on everything up to today and forecast forward.

**Expanding window** (above) keeps adding data. **Sliding window** drops the oldest period each time, keeping the training window a fixed length — better when old patterns have gone stale (a retailer after a business model change, or post-2020 data where pre-pandemic patterns no longer apply).

### The gap — a detail that impresses

Suppose in production your forecast is generated on Monday and used for the following Monday. There's a 7-day lag between the last data you have and the period you're predicting.

**Your validation must reproduce that gap:**
```
TRAIN [────────────]  [gap: 7 days]  TEST [────]
```
Without the gap, you're validating a model that has access to data it won't actually have. This is a subtle form of leakage and very few beginners think of it.

---

### 8.2 Familiar metrics, applied here

MAE, RMSE, MAPE and WAPE from Section 5 all apply directly. Two important adjustments:

#### Report per forecast horizon, not just averaged

Predicting tomorrow is far easier than predicting 30 days out. A single averaged MAE blends the two and hides the structure.

**Report a table:**

| Horizon | MAE | What it tells you |
|---|---|---|
| h = 1 day | 42 | Short-term accuracy |
| h = 7 days | 89 | Weekly planning accuracy |
| h = 30 days | 210 | Monthly planning accuracy |

**Why this matters commercially:** your supply chain might reorder with a 14-day lead time. Then `MAE@h=14` is the *only* number that affects business outcomes; `MAE@h=1` is decoration. Knowing which horizon the business actually uses, and reporting that one, is a concrete sign of maturity.

#### MAPE is a trap here specifically

Time series data is full of zeros (no sales on Sunday, a discontinued SKU, a new product before launch). As Section 5.4 showed, MAPE divides by zero and explodes. **Use WAPE for demand forecasting.**

---

### 8.3 MASE — Mean Absolute Scaled Error

**The problem MASE solves:** "MAE = 47 units" is meaningless without context. Is 47 good? For a product selling 10,000 units/day it's excellent. For one selling 50 units/day it's catastrophic.

**The idea: compare your model to the dumbest possible forecast.**

Two naive baselines:
- **Naive:** tomorrow = today. `ŷₜ = yₜ₋₁`
- **Seasonal naive:** this Monday = last Monday. `ŷₜ = yₜ₋ₘ` where m is the season length (7 for daily-with-weekly-pattern, 12 for monthly-with-yearly-pattern)

**Formula:**
```
MASE = MAE_of_your_model / MAE_of_the_naive_forecast
```

**Worked example.** Your fancy model gets MAE = 47. You compute the naive forecast ("tomorrow = today") on the same data and it gets MAE = 62.
```
MASE = 47 / 62 = 0.758
```
**Interpretation: your model's error is 76% of the naive model's error — a 24% improvement.** Now the number means something.

### The interpretation scale — this is the whole point

| MASE | Verdict |
|---|---|
| **< 1** | ✅ Your model beats the naive baseline |
| **= 1** | ⚠️ Exactly as good as "tomorrow = today" |
| **> 1** | ❌ **Your deep learning model is worse than one line of code** |

MASE is scale-free, handles zeros (unlike MAPE), is symmetric (unlike MAPE), and is comparable across completely different series. It's the metric used in the M-competitions, the standard academic forecasting benchmarks.

> **The most important sentence you can say in a forecasting interview:** "I always compare against naive and seasonal-naive baselines using MASE. A large fraction of published deep-learning forecasting results fail to beat seasonal naive on the same data — the baselines are much stronger than people expect, especially on short series."

**A scenario that shows why.** A retailer asks you to forecast daily footfall. You build an LSTM. MAE = 340 people/day. Impressive-sounding. Then you compute seasonal naive — "this Tuesday will be like last Tuesday" — and it gets MAE = 295. **MASE = 1.15.** Your neural network is 15% worse than a one-line rule, because footfall is dominated by day-of-week effects that seasonal naive captures perfectly and your model learned imperfectly. Without MASE you'd never have known.

---

### 8.4 Bias — the metric everyone forgets

```
Bias = (1/n) × Σ (ŷᵢ − yᵢ)
```

**Note: no absolute value, no squaring.** Signs are kept deliberately, so over- and under-predictions cancel out. That's the entire point.

**Why it's essential.** Two forecasting models over 100 days:

| Model | Errors | MAE | Bias |
|---|---|---|---|
| **A** | Randomly ±50, sometimes over, sometimes under | 50 | **≈ 0** |
| **B** | Always exactly +50 (always over-forecasts) | 50 | **+50** |

**Identical MAE. Completely different business consequences.**

- Model A's errors cancel out over a month. Your total monthly inventory plan is roughly right; individual days are noisy.
- Model B over-forecasts *every single day*. Over a month you've ordered 1,500 units of surplus stock. Warehouse full, cash tied up, possible spoilage.

**MAE cannot tell these apart because it takes absolute values.** Bias is the only metric that catches systematic drift in one direction, and systematic bias is usually a bigger business problem than random error.

**Always report bias alongside MAE.** It costs one line of code and it catches a whole class of failure that every other regression metric is blind to.

---

### 8.5 Probabilistic forecasting

Increasingly you don't want a single number — you want a **range with a confidence level.** "Demand will be between 400 and 600 units, with 90% confidence."

| Metric | What it measures |
|---|---|
| **Pinball / Quantile Loss** | Scores a specific quantile forecast (Section 5.6). The standard training loss for quantile forecasting. |
| **PICP (Prediction Interval Coverage Probability)** | Did your stated 90% interval actually contain the truth 90% of the time? |
| **MPIW (Mean Prediction Interval Width)** | How wide are the intervals, on average? |
| **CRPS (Continuous Ranked Probability Score)** | Generalizes MAE to whole probability distributions — scores the entire predicted distribution against the single observed value. Gold standard in weather forecasting. |
| **Winkler Score** | Combines interval width with a penalty when the truth falls outside. |

### Why coverage and width must always be reported together

**Scenario.** Your model claims 90% prediction intervals.

**Case 1:** measured coverage = 62%. Your "90% confident" intervals contain the truth less than two-thirds of the time. **The intervals are lying** and anyone planning around them is under-prepared. The model is overconfident.

**Case 2:** measured coverage = 99.8%. Sounds great — until you check the width and find the interval is `[0, 50,000]` on a product selling 500 units/day. **You've achieved coverage by predicting "somewhere between nothing and everything."** Technically correct, completely useless.

```
An interval of [−∞, +∞] has 100% coverage and zero information.
```

**So the rule: report PICP and MPIW together, always.** The goal is the *narrowest* interval that achieves the *stated* coverage. If your 90% interval covers 89.4% of cases and is as tight as possible, that's a well-calibrated probabilistic forecast.

---

### 8.6 Other time series considerations

**Directional accuracy** — did you get the *direction* of change right, regardless of magnitude?
```
Directional Accuracy = % of periods where sign(ŷₜ − yₜ₋₁) = sign(yₜ − yₜ₋₁)
```
Often more valuable than magnitude in finance. A trading strategy that always gets the direction right and the size wrong makes money. One that nails the magnitude but gets the direction backwards loses it.

**Anomaly detection in time series** uses classification metrics, but with an important twist: **"close enough" should count.** If a machine failure begins at 14:00 and your system flags it at 14:03, a strict point-by-point evaluation calls 14:00–14:02 false negatives and 14:03 a false positive — scoring near zero on a detection that was operationally a success. Look up **range-based** or **point-adjusted** precision and recall, which credit a detection that overlaps the true anomaly window.

**Backtesting for trading specifically** adds its own layer: **Sharpe ratio** (return per unit of risk), **maximum drawdown** (worst peak-to-trough loss), and **turnover** (how much trading the strategy generates, which drives transaction costs). A strategy with great RMSE can lose money after costs.

---

### 8.7 Time series decision tree

```
FIRST: is your validation set up correctly?
  ✗ Random k-fold          → STOP. This is leakage.
  ✓ Rolling-origin / walk-forward, with a production-matching gap

Need a stakeholder-friendly number in real units?
└── → MAE  (and ALWAYS report bias alongside it)

Need to know whether your model beats "do nothing"?
└── → MASE   ← include this every single time

Comparing across products/series of different scales?
└── → MASE or WAPE   (never MAPE — zeros will break it)

Are large errors disproportionately costly (capacity, grid, staffing)?
└── → RMSE

Forecasting a range rather than a point?
└── → Pinball loss for training
    + PICP (coverage) AND MPIW (width) together for evaluation
    or CRPS for the full distribution

Financial / trading context?
└── → Directional accuracy + Sharpe + max drawdown

ALWAYS: break results down by forecast horizon (h=1, h=7, h=30),
and lead with the horizon the business actually plans around.
```

---

## 9. Clustering (grouping without labels)

Clustering groups similar items **with no labels at all** — customer segments, topic groups, image clusters, anomaly groups.

**The fundamental difficulty:** there's no ground truth. Nobody can tell you the "correct" number of customer segments. So evaluation splits into two families depending on whether you happen to have labels lying around.

---

### 9.1 Internal metrics (no labels needed)

These measure the **geometry**: are points tightly packed within a cluster and far from other clusters?

### Silhouette Score — the one to know properly

For each individual point, compute two distances:

- **`a`** = the average distance from this point to the **other points in its own cluster**
  → *"how tight is my group?"* (smaller is better)
- **`b`** = the average distance from this point to the points in the **nearest other cluster**
  → *"how far away is the next group?"* (larger is better)

```
silhouette(point) = (b − a) / max(a, b)
```

Then average across all points to get the overall score.

### Making the formula intuitive with numbers

**Case 1 — a well-placed point.** It sits snugly in a tight cluster, far from everything else.
```
a = 2   (close to its own group)
b = 10  (far from the next group)

silhouette = (10 − 2) / max(2, 10) = 8 / 10 = 0.80   ← excellent
```

**Case 2 — a point on a boundary.** Equally close to its own cluster and the neighbouring one.
```
a = 5
b = 5.2

silhouette = (5.2 − 5) / 5.2 = 0.2 / 5.2 = 0.038     ← ambiguous
```

**Case 3 — a misassigned point.** It's actually closer to a *different* cluster than its own.
```
a = 9   (far from its assigned group)
b = 3   (close to a different group!)

silhouette = (3 − 9) / max(9, 3) = −6 / 9 = −0.667   ← wrong cluster
```

**Range: −1 to +1**

| Score | Meaning |
|---|---|
| near **+1** | Point is much closer to its own cluster than any other. Excellent. |
| near **0** | Point sits on a boundary. Ambiguous. |
| **negative** | **The point is probably in the wrong cluster.** |

**Overall interpretation guide:**

| Average silhouette | Verdict |
|---|---|
| > 0.70 | Strong, well-separated structure |
| 0.50 – 0.70 | Reasonable structure |
| 0.25 – 0.50 | Weak — the clusters may be artificial |
| < 0.25 | **No meaningful cluster structure found** |

### The practical workflow: choosing k

Run k-means for k = 2, 3, 4, …, 10 and plot the average silhouette:

```
Silhouette
   0.70 ┤        ●                  ← peak at k=4
   0.60 ┤      ●   ●
   0.50 ┤   ●          ●
   0.40 ┤ ●                ●   ●
        └─┬───┬───┬───┬───┬───┬──
          2   3   4   5   6   7    k
```
Pick the peak. **This is usually more decisive than the elbow method**, because a peak is unambiguous while an "elbow" is often just a gentle bend that three people will read three different ways.

**Also plot the per-point silhouette diagram**, not just the average. It shows the distribution within each cluster — you might find that clusters 1, 2 and 3 are excellent while cluster 4 is a garbage bin of leftover points with negative scores. The average alone would hide that.

### The other internal metrics

| Metric | Direction | The idea |
|---|---|---|
| **Davies-Bouldin Index** | **Lower is better** (0 = ideal) | For each cluster, find its most similar neighbour and measure the ratio of within-cluster spread to between-cluster distance; average the worst cases. |
| **Calinski-Harabasz Index** (Variance Ratio) | **Higher is better** | Between-cluster spread ÷ within-cluster spread. Very fast to compute. Tends to favour larger k, so don't use it alone to choose k. |
| **Inertia / WCSS** | Lower, but see below | Sum of squared distances from each point to its cluster centre. |
| **Dunn Index** | Higher is better | Smallest between-cluster distance ÷ largest within-cluster diameter. Very sensitive to a single outlier. |

### Why inertia alone can't choose k — and what the elbow method is

**Inertia always decreases as k increases.** If k = n (one cluster per point), every point *is* its own centre and inertia = 0. So "minimize inertia" would always tell you to use as many clusters as you have data points. Useless.

**The elbow method** works around this by looking at the *rate* of decrease:

```
Inertia
  high ┤●
       │ ●
       │  ●
       │   ●___          ← the "elbow" — adding clusters stops helping much
       │       ●___
       │           ●___●___●
   low └─┬──┬──┬──┬──┬──┬──┬──
         1  2  3  4  5  6  7   k
```

Before the elbow, each extra cluster buys a big reduction in inertia (you're splitting genuinely different groups). After it, you're just chopping up coherent groups for marginal gain.

**Its weakness: the elbow is often not sharp.** On real data the curve is frequently a smooth bend where reasonable people pick k = 3, 4 or 5. Use it alongside silhouette rather than instead of it.

### An honest caveat you should raise in interviews

**All internal metrics assume clusters should be compact and roughly spherical**, because they're built on distances to centres or averages.

This means they **systematically penalize DBSCAN and other density-based methods**, which are designed to find elongated, crescent-shaped or irregular clusters. If your data looks like two interlocking crescents, DBSCAN will find the *correct* clusters and silhouette will score it worse than k-means, which finds the *wrong* ones but makes them nicely round.

> **Don't use silhouette to compare k-means against DBSCAN.** Use it to tune k *within* k-means. Saying this unprompted signals you've actually used these tools rather than read about them.

---

### 9.2 External metrics (you do have labels)

Sometimes you have ground-truth groupings and want to check whether clustering recovers them.

**Why you can't just use accuracy.** Cluster labels are **arbitrary names**. Your algorithm calling a group "cluster 3" when the truth calls it "class 1" is not an error — the grouping is identical, only the name differs.

```
Truth:      [A, A, B, B, C, C]
Clustering: [2, 2, 0, 0, 1, 1]

Accuracy would say 0% correct. But the grouping is PERFECT.
```

So external metrics are **invariant to label permutation.** They ask *"did the same points end up together?"* not *"did the names match?"*

| Metric | Range | Notes |
|---|---|---|
| **Adjusted Rand Index (ARI)** | −1 to 1 (0 = random) | Counts **pairs** of points: for each pair, are they together in both partitions, apart in both, or disagreeing? "Adjusted" = corrected for chance. **The standard default.** |
| **Normalized Mutual Information (NMI)** | 0 to 1 | Information-theoretic: how much does knowing a point's cluster tell you about its true label? Slightly biased toward larger k. |
| **Adjusted Mutual Information (AMI)** | ~0 to 1 | NMI with chance correction. **Prefer AMI over NMI.** |
| **Homogeneity** | 0 to 1 | Does each cluster contain only members of a single true class? (Like precision.) |
| **Completeness** | 0 to 1 | Are all members of a true class in the same cluster? (Like recall.) |
| **V-measure** | 0 to 1 | Harmonic mean of homogeneity and completeness. (Like F1.) |
| **Fowlkes-Mallows** | 0 to 1 | Geometric mean of pairwise precision and recall. |
| **Purity** | 0 to 1 | For each cluster take its most common true label, sum those counts, divide by n. **Easily gamed.** |

### Homogeneity vs completeness — made concrete

These two are mirror images, and the examples make it obvious:

**Case A — split every true class in half.** True classes {Cats, Dogs} → you produce 4 clusters: {Cats-1}, {Cats-2}, {Dogs-1}, {Dogs-2}.
```
Homogeneity  = 1.00   ← every cluster contains only one true class. Pure!
Completeness = LOW    ← each true class is scattered across 2 clusters
```

**Case B — dump everything into one cluster.**
```
Completeness = 1.00   ← every member of each class is together. Complete!
Homogeneity  ≈ 0      ← the single cluster is a total mixture
```

Each is trivially maximizable on its own, which is exactly why you need **V-measure** (their harmonic mean) — the same logic as F1 in Section 6.6.

**Why purity is dangerous:** make n clusters of 1 point each. Every cluster is perfectly pure. Purity = 1.000. You have learned nothing. Purity has no penalty for the number of clusters, so never report it alone.

---

### 9.3 The most important clustering advice

Metrics can tell you whether clusters are geometrically clean. **They cannot tell you whether the clusters are useful**, and usefulness is the entire point of clustering.

**Scenario.** You cluster 2 million retail customers and get silhouette = 0.68. Solid number. You take it to the marketing team. They ask: "so what are the segments?" You look at the clusters and find:

- Cluster 1: customers who joined in 2019
- Cluster 2: customers who joined in 2020
- Cluster 3: customers who joined in 2021

Geometrically excellent. Commercially worthless — the model just found signup date, which they already knew and which doesn't suggest any different marketing action.

> **What to say:** "Internal metrics tell me whether the clusters are geometrically sound, but not whether they're useful. I'd use silhouette to sanity-check k, then **profile each cluster** on business-meaningful dimensions — average spend, tenure, category mix, churn rate — and put those profiles in front of the marketing team. If they can't name the segments and act differently toward each one, the clustering has failed no matter what silhouette says."

**Also always test stability.** Re-run the clustering on bootstrap samples of the data, or with different random seeds. If cluster assignments swing wildly between runs, the structure isn't real — you've found noise, and next quarter's data will produce completely different segments. Measure this with **ARI between runs** (same data, different seeds): ARI > 0.75 across runs suggests genuine structure; ARI < 0.4 suggests you're fitting noise.

---

# PART IV — LANGUAGE & MODERN AI

## 10. NLP text generation metrics

Here's where evaluation gets genuinely hard.

When the output is a number, right and wrong are obvious. When the output is a **category**, right and wrong are obvious. When the output is **text**, there are a thousand correct answers.

```
Reference:  "The cat sat on the mat."
Output A:   "A feline was resting upon the rug."     ← same meaning, ~0 shared words
Output B:   "The cat sat on the hat."                ← 5 of 6 words match, wrong meaning
```

**Any metric based on word overlap scores B higher than A.** That's backwards. Every metric in this section is an imperfect attempt to deal with this, and understanding *which* imperfection each one has is the interview skill.

### The three generations of approach

| Generation | Method | Speed | Quality | Examples |
|---|---|---|---|---|
| **1. N-gram overlap** | Count matching word sequences | Very fast, free | Blind to meaning | BLEU, ROUGE, METEOR |
| **2. Embedding-based** | Compare meaning via vectors | Moderate | Understands paraphrase, not facts | BERTScore, BLEURT, COMET |
| **3. LLM-as-judge** | Ask a strong model to grade | Slow, costs money | Best human correlation | G-Eval, Ragas, MT-Bench |

The field has moved decisively toward generation 3 for anything open-ended, but generations 1 and 2 remain standard for benchmarking and are still asked about constantly.

### First: what is an n-gram?

**An n-gram is a sequence of n consecutive words.** For the sentence "the cat sat on the mat":

| n | Name | The n-grams |
|---|---|---|
| 1 | unigrams | the, cat, sat, on, the, mat — **6 of them** |
| 2 | bigrams | "the cat", "cat sat", "sat on", "on the", "the mat" — **5** |
| 3 | trigrams | "the cat sat", "cat sat on", "sat on the", "on the mat" — **4** |
| 4 | 4-grams | "the cat sat on", "cat sat on the", "sat on the mat" — **3** |

**Rule:** a sentence of `L` words has `L − n + 1` n-grams. 6 words → 6 unigrams, 5 bigrams, 4 trigrams.

**Why longer n-grams matter:** unigram overlap measures whether you used the right *words*. 4-gram overlap measures whether you used them in the right *order* — which is a proxy for fluency and grammar.

### Our running example for this section

```
Reference (human-written):  "the cat sat on the mat"
Candidate (model output):   "the cat is on the mat"
```
One word changed: "sat" → "is".

---

### 10.1 BLEU — the machine translation standard

BLEU (BiLingual Evaluation Understudy) is **precision-oriented**: *of the n-grams I generated, how many appear in the reference?*

### The three components

**Component 1 — modified n-gram precision.**

Plain precision would be: `(matching n-grams) / (total n-grams in candidate)`.

**"Modified" prevents an obvious cheat.** Without it, a model could output "the the the the the" against a reference containing "the" twice, and score 5/5 = 100% unigram precision. The modification: **you cannot get credit for an n-gram more times than it appears in the reference.** "the" appears twice in the reference, so a maximum of 2 of your 5 "the"s count → 2/5 = 40%.

**Component 2 — geometric mean across n = 1, 2, 3, 4.**

Why geometric rather than arithmetic? Because a geometric mean goes to **zero if any single component is zero**. A candidate with decent word choice but no correct 4-word sequences gets BLEU = 0, correctly flagging it as disfluent. An arithmetic mean would let good unigram precision cover for terrible grammar.

**Component 3 — Brevity Penalty.**

Pure precision rewards being short. Output one word you're certain about → precision 1.0. The brevity penalty multiplies your score down when the candidate is shorter than the reference:
```
BP = 1                      if candidate_length > reference_length
BP = e^(1 − ref/cand)       otherwise
```

**Worked example of BP.** Reference has 20 words; your translation has 10.
```
BP = e^(1 − 20/10) = e^(1 − 2) = e^(−1) = 0.368
```
**Your score gets multiplied by 0.368 — cut by nearly two-thirds** for being half the required length. That's the intended deterrent.

### Our example computed

**Unigram precision:**
```
Candidate unigrams: the, cat, is, on, the, mat   (6 total)
Reference has:      the(×2), cat, sat, on, mat

Matches: the(×2) ✓, cat ✓, on ✓, mat ✓, is ✗
       = 5 of 6  →  0.833
```

**Bigram precision:**
```
Candidate bigrams: "the cat", "cat is", "is on", "on the", "the mat"   (5)
Reference bigrams: "the cat", "cat sat", "sat on", "on the", "the mat"

Matches: "the cat" ✓, "on the" ✓, "the mat" ✓
       = 3 of 5  →  0.600
```

**Trigram precision:**
```
Candidate trigrams: "the cat is", "cat is on", "is on the", "on the mat"   (4)
Reference trigrams: "the cat sat", "cat sat on", "sat on the", "on the mat"

Matches: "on the mat" ✓
       = 1 of 4  →  0.250
```

**4-gram precision:**
```
Candidate 4-grams: "the cat is on", "cat is on the", "is on the mat"   (3)
Reference 4-grams: "the cat sat on", "cat sat on the", "sat on the mat"

Matches: none  →  0.000
```

**Notice what happened:** as n increases, precision collapses — 0.833 → 0.600 → 0.250 → 0.000. **One wrong word destroys every n-gram that contains it**, and longer n-grams are more likely to contain it. With a zero in the geometric mean, standard BLEU-4 would be **0** for this sentence, despite it being a nearly perfect translation.

(In practice BLEU is computed over whole corpora, not single sentences, so isolated zeros get diluted. Sentence-level BLEU is known to be unreliable for exactly this reason — another good thing to know.)

**Lengths are equal (6 vs 6), so BP = 1.**

### BLEU's known problems

1. **It ignores meaning entirely.** Synonyms score zero. "big" vs "large" → no match.
2. **It correlates poorly with human judgment at the sentence level** (as we just saw).
3. **It's tokenization-sensitive** — different tokenizers give different BLEU on identical text, which made published numbers incomparable for years. **Always use SacreBLEU**, which standardizes tokenization. Interviewers who work in MT will notice if you say this.
4. **It's unfair to morphologically rich languages.** In Tamil, Telugu, Turkish or Finnish, a single concept can be expressed as one long inflected word. Word-level matching is too brittle.

**Use BLEU for:** machine translation benchmarking, with multiple references, comparing systems on a standard test set. **Don't use it for:** open-ended generation, chatbots, summarization, or anything where multiple valid phrasings exist.

### The alternatives

**chrF / chrF++** — the same idea at the **character** n-gram level. "running" and "runs" share the characters "run", so partial credit is possible. **Much better for morphologically rich languages** and increasingly preferred over BLEU generally.

**TER (Translation Edit Rate)** — the minimum number of edits (insert, delete, substitute, shift) to turn the candidate into the reference, divided by reference length. **Lower is better.** Its virtue is direct interpretability: TER = 0.25 means a human post-editor would need to change about a quarter of the words. That's an actual business number — it translates to cost.

---

### 10.2 ROUGE — the summarization standard

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is BLEU's mirror image: **recall-oriented**. *Of the n-grams in the reference, how many did I produce?*

**Why that orientation is right for summarization:** the question isn't "did you avoid saying anything extra?" — it's **"did you capture the important content?"** Recall measures coverage. (BLEU's precision orientation is right for translation, where extra content is a genuine error.)

### ROUGE-N — n-gram recall

**ROUGE-1 (unigrams), worked out:**
```
Reference unigrams: the, cat, sat, on, the, mat   → 6 total
Candidate unigrams: the, cat, is, on, the, mat    → 6 total

Overlapping: the(×2), cat, on, mat = 5

ROUGE-1 Recall    = 5 / 6 (reference length) = 0.833
ROUGE-1 Precision = 5 / 6 (candidate length) = 0.833
ROUGE-1 F1        = 0.833
```

**ROUGE-2 (bigrams):**
```
Reference bigrams: "the cat", "cat sat", "sat on", "on the", "the mat"   → 5
Candidate bigrams: "the cat", "cat is", "is on", "on the", "the mat"

Overlapping: "the cat", "on the", "the mat" = 3

ROUGE-2 Recall = 3 / 5 = 0.600
```

ROUGE-2 is stricter than ROUGE-1 and is usually the headline number, because it partially captures fluency — you can only match bigrams if your word order is right.

### ROUGE-L — longest common subsequence

**What's a subsequence?** A sequence of words that appear in the same *order* but **need not be consecutive.**

```
Reference:  the  cat  sat  on  the  mat
Candidate:  the  cat  is   on  the  mat
            ✓    ✓         ✓   ✓    ✓

LCS = "the cat on the mat"  →  length 5
```

We skipped "sat"/"is" but everything else lines up in order.

```
ROUGE-L Recall    = 5 / 6 = 0.833
ROUGE-L Precision = 5 / 6 = 0.833
ROUGE-L F1        = 0.833
```

**Why LCS is useful:** it captures sentence-level structure without demanding contiguous matches. A summary that preserves the reference's information flow but inserts a connecting word still scores well, where ROUGE-2 would be penalized.

**ROUGE-Lsum** is the multi-sentence variant, computing LCS per sentence and aggregating — use this for summaries longer than one sentence.

### What to report

The convention is **ROUGE-1, ROUGE-2 and ROUGE-L together**:
- **ROUGE-1** → content coverage (did you mention the right things?)
- **ROUGE-2** → fluency (did you phrase them naturally?)
- **ROUGE-L** → structure (did you preserve the flow?)

### ROUGE's fatal weakness — and the reason Section 11 exists

**ROUGE cannot detect hallucination or factual error.**

```
Source:    "Revenue rose 20% to ₹450 crore in Q3."
Reference: "Revenue increased 20% to ₹450 crore."
Summary A: "Revenue increased 20% to ₹450 crore."     ← correct
Summary B: "Revenue decreased 20% to ₹450 crore."     ← FACTUALLY REVERSED
```

Summary B shares 5 of 6 words with the reference. **Its ROUGE score is nearly identical to the correct summary's.** The one word that inverted the meaning is worth exactly as much as any other word.

This is not a corner case. Summarization models hallucinate numbers, flip attributions, and invent quotes routinely — and ROUGE sees none of it. It's the single reason the field moved to faithfulness metrics.

---

### 10.3 METEOR

An improvement on BLEU that explicitly handles the synonym problem, matching in stages:

1. **Exact match** — identical words
2. **Stem match** — "running" matches "run", "runs", "ran"
3. **Synonym match** — via WordNet, "big" matches "large"
4. **Paraphrase match** — via a paraphrase table

Then it weights **recall 9× more than precision** (unlike BLEU) and adds a **fragmentation penalty** for matches that appear out of order.

**Correlates notably better with human judgment than BLEU.** Its limitation is practical: it needs WordNet and paraphrase tables, which exist well for English and poorly for most other languages.

---

### 10.4 BERTScore — comparing meanings, not words

Instead of matching words, match **meanings**.

### How it works

1. Run the candidate and reference through a transformer (BERT, RoBERTa, DeBERTa) to get a **contextual embedding** for every token. "Contextual" matters — the word "bank" gets a different vector in "river bank" than in "savings bank."
2. For each **candidate** token, find the reference token with the highest cosine similarity → average these → **precision**.
3. For each **reference** token, find the best-matching candidate token → average → **recall**.
4. Combine into **F1**.
5. Optionally **rescale** against a baseline.

### Why it fixes the paraphrase problem

```
Reference: "The film was excellent."
Candidate: "The movie was great."

ROUGE-1: only "The" and "was" match → 2/4 = 0.50
         (and those are the two meaningless words!)

BERTScore: "film"≈"movie" (cos ≈ 0.85), "excellent"≈"great" (cos ≈ 0.88)
         → F1 ≈ 0.95
```

BERTScore gets the right answer; ROUGE gets it badly wrong.

### Why rescaling exists

Raw BERTScores are squashed into a narrow band — almost everything lands between 0.82 and 0.96, because any two English sentences share *some* semantic structure. That makes differences hard to read. **Rescaling** subtracts a baseline (the expected score for random sentence pairs) and stretches the range so scores spread across 0 to 1 usefully. Always state whether you're reporting rescaled or raw.

### Limitations

- **Slower** — needs a forward pass through a transformer for every evaluation.
- **Still can't verify facts.** "Revenue rose 20%" and "Revenue fell 20%" are *semantically very similar* in embedding space — same topic, same entities, same structure. BERTScore would rate them as near-identical. **Semantic similarity is not factual correctness.**
- **Model-dependent.** BERTScore computed with RoBERTa-large differs from BERTScore with DistilBERT. **Always state which model you used** or the number isn't reproducible.

### The related family

- **BLEURT** — a BERT model **fine-tuned on human quality ratings** to directly predict what a human would score. Better correlation than BERTScore, but tied to the domains and languages it was trained on.
- **MoverScore** — uses Earth Mover's Distance between embedding distributions rather than greedy token matching.
- **COMET** — **the current state of the art for machine translation evaluation.** Trained on human judgments, and crucially it can use the **source sentence** as input, not just the reference. That lets it catch errors a reference-only metric can't. **If an interviewer asks about modern MT evaluation, COMET is the answer — not BLEU.**

---

### 10.5 Perplexity — for language models

**The question it answers:** *How surprised is the model by the text it's reading?* Lower = less surprised = better model.

**Formula:**
```
Perplexity = exp( average negative log-likelihood )
           = 2^(cross-entropy)
```

That's just Section 6.11's log loss, exponentiated to undo the logarithm and return to a readable scale.

### The intuition that makes it click

> **Perplexity is roughly "how many words was the model effectively choosing between at each step?"**

| Perplexity | Meaning |
|---|---|
| **1** | The model knew the next word with total certainty |
| **10** | About as uncertain as picking from 10 equally likely options |
| **100** | Quite uncertain |
| **50,000** (= vocab size) | **Learned nothing at all** — uniform guessing |

**Worked intuition.** Consider the sentence fragment "The capital of France is ___".
- A good model puts ~0.95 probability on "Paris" → very low perplexity at that position.
- Now "My favourite colour is ___". Even a perfect model can't know — it might spread probability across blue, green, red, black. Perplexity ≈ 5–10, and **that's correct behaviour**, not a failure. Some text is genuinely unpredictable.

Perplexity is measured over a whole corpus, averaging the easy positions and the hard ones.

### Two critical caveats (both common interview probes)

**1. Perplexity is only comparable across models with the SAME tokenizer AND the same test set.**

Why: perplexity is per-token. If Model A splits a word into 1 token and Model B splits the same word into 3 tokens, they're being scored on different units. Model B has three easier predictions instead of one hard one. **The numbers are not comparable.** Anyone quoting "our model has perplexity 12 vs their 15" across different tokenizers is saying nothing.

**2. Low perplexity ≠ useful model.**

Perplexity measures how well a model **predicts text**. It says nothing about whether the model follows instructions, is truthful, is safe, or is helpful. A model trained purely to minimize perplexity on web text will be excellent at continuing web text and terrible as an assistant.

**This is precisely why nobody evaluates chat models with perplexity.** Instruction tuning and RLHF often *increase* perplexity while making the model dramatically more useful. Knowing this distinction is a strong signal in an LLM interview.

---

### 10.6 Task-specific NLP metrics

#### Extractive Question Answering (SQuAD-style)

The model highlights a span of text in a passage.

**Exact Match (EM)** — did the predicted span exactly equal a gold answer, after normalizing case, articles ("a", "an", "the") and punctuation? Binary.

**Token-level F1** — treat prediction and gold answer as bags of tokens, compute precision and recall over them.

**Worked example:**
```
Gold answer:  "Albert Einstein"
Prediction:   "Einstein"

Exact Match = 0  (not identical)

Token F1:
  Precision = 1 matched / 1 predicted token  = 1.000
  Recall    = 1 matched / 2 gold tokens      = 0.500
  F1        = 2 × (1.0 × 0.5) / 1.5          = 0.667
```

**Always report both.** EM alone is harshly pessimistic — it gives zero credit for an answer a human would accept. F1 alone is too generous — a prediction of "Albert Einstein was a physicist born in Germany" gets decent F1 while being a bad span. Together they bracket the truth.

#### Named Entity Recognition (NER)

**Evaluate at the ENTITY level, not the token level.**

```
Text:  "She flew to New York City yesterday."
Gold:  [New York City] = LOCATION
Pred:  [New York] = LOCATION
```

**Token-level F1** would say 2 of 3 tokens correct → looks like 67% success.
**Entity-level strict F1** says the boundary is wrong → this is **one FP and one FN**, scoring zero for this entity.

Strict matching is usually right, because a downstream system looking up "New York" gets a different place than "New York City." But it can be too harsh for some applications, so the standard evaluation schemes (implemented in `seqeval`) offer variants:

- **Strict** — boundary and type must both match exactly
- **Exact** — boundary matches, type ignored
- **Partial** — credit for overlapping boundaries
- **Type** — correct type with any overlap

**Always say which scheme you used**, and report per-entity-type breakdown — models typically do well on PERSON and badly on ORG, and an aggregate F1 hides that.

#### Speech Recognition (ASR)

**WER (Word Error Rate):**
```
WER = (Substitutions + Insertions + Deletions) / (words in reference)
```

**Worked example:**
```
Reference:  "the quick brown fox jumps over the lazy dog"   (9 words)
Hypothesis: "the quick brown fox jumped over lazy dog"

  "jumps" → "jumped"     = 1 substitution
  "the" (before lazy) missing = 1 deletion

WER = (1 + 0 + 1) / 9 = 2/9 = 0.222  →  22.2%
```

**Two facts worth knowing:**
1. **WER can exceed 100%.** If the model hallucinates 20 extra words against a 5-word reference, insertions alone push WER above 4.0. The denominator is the reference length, not the hypothesis length.
2. **WER treats all words equally.** Getting "not" wrong inverts the meaning of a sentence; getting "a" wrong changes nothing. WER scores them identically. For meaning-critical applications (medical dictation, legal transcription), supplement with entity-level accuracy on the terms that matter.

**CER (Character Error Rate)** — same formula at character level. Use for languages without clear word boundaries (Chinese, Japanese, Thai) and for OCR.

#### Code generation

**pass@k** — generate k candidate programs; did **any** of them pass the unit tests?

```
pass@1  = the model's first attempt works        ← what matters in production
pass@10 = at least one of 10 attempts works      ← relevant if a human picks
```

**Never evaluate code with BLEU or any text-similarity metric.** Two programs can be character-for-character different and functionally identical (different variable names, a `for` loop vs a list comprehension). Conversely, two nearly identical programs can differ by one character — `<` vs `<=` — and one works while the other has an off-by-one bug that text similarity can't see.

**Execution is the only meaningful test.** Supplement pass@k with **test coverage** (do the tests actually exercise the edge cases?) and, for agentic coding benchmarks like SWE-bench, **resolution rate** on real GitHub issues.

---

## 11. Summarization

Summarization gets its own section because the historical standard metric — ROUGE — measures the wrong thing, and understanding why is a genuinely valuable interview topic.

### 11.1 The four dimensions that actually matter

| Dimension | The question | How to measure it |
|---|---|---|
| **Relevance** | Did it capture the important content? | ROUGE-1/2/L, BERTScore, LLM judge |
| **Faithfulness** (factual consistency) | Is every claim actually supported by the source? | **SummaC, QAFactEval, AlignScore, LLM judge** |
| **Coherence** | Does it read as organized, logical prose? | LLM judge, human rating |
| **Fluency** | Is the grammar and wording clean? | LLM judge, human rating |

**Only the first row is what ROUGE measures.** The second row is where summarization models actually fail.

### 11.2 Why faithfulness is the hard problem

Modern abstractive summarizers hallucinate in predictable ways:

| Hallucination type | Example |
|---|---|
| **Number distortion** | Source says 23%; summary says 32% |
| **Direction flip** | Source says "declined"; summary says "grew" |
| **Attribution error** | Source quotes the CFO; summary attributes it to the CEO |
| **Unsupported inference** | Source lists three risks; summary concludes "the company is in trouble" |
| **Entity substitution** | Source says Infosys; summary says TCS |

**ROUGE cannot detect a single one of these**, because in every case the hallucinated summary shares nearly all its vocabulary with the correct one. The word that's wrong is worth exactly as much as every other word.

Worse: **ROUGE and faithfulness scores are close to uncorrelated in published studies.** So optimizing for ROUGE doesn't improve faithfulness, and can make it worse — a model rewarded for word overlap learns to produce summaries that *sound* like the reference without checking whether they're true.

### 11.3 How faithfulness is actually measured

#### NLI-based methods (SummaC, AlignScore)

**NLI = Natural Language Inference.** An NLI model takes two sentences and classifies the relationship:
- **Entailment** — the second follows from the first ✅
- **Contradiction** — the second contradicts the first ❌
- **Neutral** — unrelated, can't tell ❌

**The procedure:**
1. Split the summary into individual sentences.
2. For each summary sentence, run NLI against the source document.
3. Anything not entailed is a potential hallucination.
4. Score = fraction of summary sentences entailed by the source.

**Worked example.**
```
Source: "Infosys reported Q3 revenue of ₹38,821 crore, up 1.3% year-on-year.
         The company added 6,400 employees during the quarter."

Summary sentence 1: "Infosys revenue grew 1.3% in Q3."        → ENTAILED   ✅
Summary sentence 2: "Infosys hired 6,400 people."             → ENTAILED   ✅
Summary sentence 3: "The company expects growth to continue." → NEUTRAL    ❌
                     (nothing in the source says this)

Faithfulness = 2/3 = 0.667
```

Sentence 3 is the hallucination. It sounds plausible, it's the kind of thing a business article says, and ROUGE would have loved it. NLI catches it.

#### QA-based methods (QAGS, QAFactEval)

A genuinely elegant approach:

1. **Generate questions from the summary.** ("By how much did revenue grow?")
2. **Answer them using the summary.** → "1.3%"
3. **Answer the same questions using the source document.** → "1.3%"
4. **Compare.** Agreement = faithful. Disagreement = hallucination.

```
Question generated from summary: "How many employees were added?"
  Answer from summary: "6,400"
  Answer from source:  "6,400"        → MATCH ✅

Question: "What is the growth outlook?"
  Answer from summary: "expected to continue"
  Answer from source:  [unanswerable]  → HALLUCINATION ❌
```

**Why this works so well:** it converts a vague question ("is this faithful?") into a precise one ("do these two answers match?") that existing QA models are good at.

#### LLM-as-judge

The most common production approach today. Prompt a strong model:

> "Here is a source document and a summary. List every factual claim in the summary that is not directly supported by the source. Output JSON."

This is the mechanism underneath Ragas's faithfulness metric (Section 12). It's flexible, requires no specialized model, and correlates well with humans — at the cost of money, latency, and non-determinism.

### 11.4 Baselines you must include

**Lead-3** — just take the first three sentences of the source. No model at all.

**This baseline beats a surprising number of published neural summarization models on ROUGE**, because news articles are written with the most important information first (the "inverted pyramid"). If your model can't beat Lead-3, it isn't summarizing — and reporting this comparison honestly is a strong signal.

**Also measure:**
- **Compression ratio** = summary length / source length. Are you actually condensing?
- **Abstractiveness** = % of summary n-grams that do *not* appear verbatim in the source. A model that copies sentences wholesale gets great ROUGE and great faithfulness while performing no summarization. Low abstractiveness + high ROUGE = you built an extractive system by accident.

### 11.5 What to actually say

> "I'd report ROUGE-1/2/L for continuity with published baselines, but treat **faithfulness as the primary metric** — either an NLI-based score like AlignScore or an LLM judge using claim decomposition. ROUGE and faithfulness are nearly uncorrelated, so optimizing ROUGE alone can increase hallucination. I'd also include the Lead-3 baseline and report abstractiveness, because a model can score well on ROUGE by simply copying. And before trusting any LLM judge, I'd validate it against 100–200 human-labelled examples and report the agreement."

---

## 12. RAG evaluation & Ragas

**RAG = Retrieval-Augmented Generation.** The user asks a question; you search a knowledge base for relevant chunks; you paste those chunks into the LLM's prompt; the LLM answers using them.

It's the dominant architecture for enterprise AI applications, and evaluating it is one of the most common interview topics right now.

### 12.1 The key insight: evaluate the two halves separately

```
Question → [ RETRIEVER ] → chunks → [ GENERATOR (LLM) ] → Answer
               ↑                          ↑
    Did we FIND the right info?   Did we USE it correctly?
```

**These fail in completely different ways and need completely different fixes.**

| Failure | Symptom | Fix |
|---|---|---|
| **Retrieval failed** | The needed information was never in the context | Chunking, embeddings, hybrid search, k, reranking |
| **Generation failed** | Information was right there and the model ignored it or made something up | Prompting, temperature, model choice, citation requirements |

**If retrieval failed, no amount of prompt engineering will help.** The information isn't in the prompt. People waste weeks tuning prompts on problems that are 100% retrieval failures.

> **Always report retrieval metrics and generation metrics separately.** This is the single highest-signal thing to say in a RAG interview.

---

### 12.2 Retrieval-side metrics

Everything from Section 7 applies. The specifically RAG-flavoured ones:

| Metric | What it catches | Why it matters |
|---|---|---|
| **Context Recall** | Did we retrieve everything needed? | **The hardest ceiling.** If information is missing, answer quality is capped no matter what. |
| **Context Precision** | Are the relevant chunks ranked at the top? | Irrelevant chunks waste context window, cost tokens, and actively distract the LLM |
| **Recall@k / MRR / NDCG@k** | Standard ranking quality | Model-agnostic comparison of retrievers |

---

### 12.3 The Ragas metrics, explained properly

[Ragas](https://docs.ragas.io) is the most widely used open-source RAG evaluation framework. **Most of its metrics are computed by an LLM judge, not by formula** — which is exactly what lets them measure things like faithfulness that no formula could.

**The four inputs a Ragas evaluation uses:**

| Input | What it is |
|---|---|
| **question** | The user's query |
| **contexts** | The chunks your retriever returned |
| **answer** | What your LLM generated |
| **ground_truth** | A reference answer (only some metrics need this) |

---

#### Faithfulness — the hallucination detector

**The question:** *Does the answer stick to the retrieved context?*

**How it works:**
1. An LLM breaks the generated answer into individual **atomic claims**.
2. For each claim, an LLM checks: can this be inferred from the retrieved contexts?
3. `Faithfulness = supported claims / total claims`

**Worked example:**
```
Question: "What is the return policy?"

Retrieved context: "Customers may return unused items within 30 days
                    of purchase with the original receipt."

Generated answer: "You can return items within 30 days with a receipt.
                   Returns are free and we also offer exchanges."

Claim decomposition:
  1. "Items can be returned within 30 days"     → supported by context  ✅
  2. "A receipt is required"                    → supported by context  ✅
  3. "Returns are free"                         → NOT in the context    ❌
  4. "Exchanges are offered"                    → NOT in the context    ❌

Faithfulness = 2/4 = 0.50
```

Claims 3 and 4 are hallucinations — plausible, helpful-sounding, and completely invented. In a real product this would generate customer complaints and possibly legal exposure.

**Inputs needed:** question, answer, contexts. **No ground truth.** Range 0–1.

> **Critical nuance to state in an interview:** a faithfulness of 1.0 means the answer never went beyond its sources. It does **not** mean the answer is *correct*. If the retrieved context was itself wrong or outdated, a perfectly faithful answer is a perfectly wrong answer. Faithfulness measures *groundedness*, not *truth*.

---

#### Answer Relevancy — is the answer on-topic?

**The question:** *Does the answer actually address what was asked?*

**How it works — this mechanism is clever and worth explaining:**
1. An LLM reads the generated answer and **reverse-engineers what question it seems to be answering** — generating several candidate questions.
2. Embed those candidate questions and the original question.
3. Compute cosine similarity between each candidate and the original.
4. Average them. That's the score.

**Worked example:**
```
Original question: "What is the return window for electronics?"

Answer A: "Electronics can be returned within 15 days of purchase."
  Reverse-engineered questions:
    "How long do I have to return electronics?"    → cos sim 0.94
    "What is the electronics return period?"       → cos sim 0.96
  Answer Relevancy ≈ 0.95  ✅

Answer B: "Our return policy is designed to be customer-friendly and
           we pride ourselves on service."
  Reverse-engineered questions:
    "What is your company's attitude to customers?"  → cos sim 0.42
    "Do you have good customer service?"             → cos sim 0.38
  Answer Relevancy ≈ 0.40  ❌
```

Answer B is fluent, positive and completely evasive. It never states the return window. **Answer relevancy catches that specific failure mode** — the corporate non-answer — which faithfulness would happily pass (it didn't hallucinate; it just didn't answer).

**Inputs needed:** question, answer. Range 0–1.

**Nuance:** this measures *relevance*, not *correctness*. A confidently wrong but directly on-topic answer scores high. Always pair it with faithfulness and correctness.

---

#### Context Precision — are the good chunks at the top?

**The question:** *Of the chunks I retrieved, are the useful ones ranked highest?*

**How it works:** for each retrieved chunk, an LLM judges whether it was relevant to answering the question. Then a **rank-weighted** calculation (mathematically this is Average Precision from Section 7.3) rewards relevant chunks near position 1 more than relevant chunks at position 8.

**Worked example** — retrieved 5 chunks, relevance as judged:
```
Rank 1: relevant    ✅  → precision so far 1/1 = 1.000
Rank 2: irrelevant  ❌
Rank 3: relevant    ✅  → precision so far 2/3 = 0.667
Rank 4: irrelevant  ❌
Rank 5: irrelevant  ❌

Context Precision = (1.000 + 0.667) / 2 relevant chunks = 0.833
```

**What low context precision tells you:** your reranker is weak, or your `k` is too high and you're padding the prompt with noise. Both cost money (tokens) and quality (LLMs genuinely get distracted by irrelevant context — this is the documented "lost in the middle" problem where models attend poorly to information buried in long contexts).

---

#### Context Recall — did we retrieve everything needed?

**The question:** *Is all the information required to answer present in the retrieved chunks?*

**How it works:** take the **ground truth answer**, break it into claims, and check each claim: is it supported by the retrieved contexts?

```
Context Recall = (ground-truth claims found in contexts) / (total ground-truth claims)
```

**Worked example:**
```
Question: "What are the eligibility criteria for the loan?"

Ground truth answer: "You must be 21 or older, have an annual income
                      above ₹3 lakh, and a credit score above 700."

Claims in the ground truth:
  1. "Minimum age 21"              → found in retrieved chunk 2  ✅
  2. "Income above ₹3 lakh"        → found in retrieved chunk 2  ✅
  3. "Credit score above 700"      → NOT in any retrieved chunk  ❌

Context Recall = 2/3 = 0.667
```

**One third of the required information was never retrieved.** The LLM literally cannot produce a complete answer. Whatever it says about credit scores will be either an omission or a hallucination.

> **This is the metric to fix first.** Low context recall sets a hard ceiling on the entire system. Common causes: chunk size too small (splitting a table across chunks), chunk size too large (diluting the embedding), weak embedding model for your domain, `k` too small, or a question requiring multi-hop reasoning across documents.

**Needs ground truth**, so this is an offline metric on a curated evaluation set.

---

#### The remaining Ragas metrics

**Context Entities Recall** — of the named entities (people, dates, products, amounts) in the ground truth, how many appear in the retrieved contexts? A sharper version of context recall for fact-dense domains like finance or legal.

**Answer Correctness** — compares the generated answer to the ground truth, blending **factual overlap** (claim-level TP/FP/FN → an F1) with **semantic similarity** (embeddings). The single headline number if you need one.

**Answer Semantic Similarity** — cosine similarity between the answer's embedding and the ground truth's. Cheap, fast, no LLM call. Useful as a smoke test but too coarse to rely on.

**Noise Sensitivity** — how often does the model produce incorrect claims **when irrelevant chunks are present**? Directly measures robustness to retrieval noise. **Lower is better.** A model with low noise sensitivity can tolerate a sloppy retriever; one with high noise sensitivity needs a very clean context.

---

### 12.4 Ragas quick reference

| Metric | Measures | Ground truth? | Half of pipeline |
|---|---|---|---|
| **Faithfulness** | Hallucination / groundedness | **No** | Generation |
| **Answer Relevancy** | On-topic-ness | **No** | Generation |
| **Context Precision** | Chunk ranking quality | No* | Retrieval |
| **Context Recall** | Retrieval completeness | **Yes** | Retrieval |
| **Context Entities Recall** | Entity coverage | **Yes** | Retrieval |
| **Answer Correctness** | Overall correctness | **Yes** | End-to-end |
| **Answer Semantic Similarity** | Meaning match | **Yes** | End-to-end |
| **Noise Sensitivity** | Robustness to bad chunks | **Yes** | End-to-end |

\* a reference-free variant exists

### The production trick worth mentioning

**The "No ground truth" metrics — faithfulness and answer relevancy — can run on live production traffic**, because they only need the question, the retrieved contexts and the generated answer. All three are things you already log.

So the practical setup is:
- **Offline**, on a curated eval set: all metrics, run on every model or prompt change
- **Online**, on sampled live traffic: faithfulness + answer relevancy, as continuous monitoring with alerting

Saying this shows you've thought about **operating** a system, not just benchmarking one — which is a meaningfully different level of experience.

---

### 12.5 A practical RAG debugging workflow

```
The answer is bad.
│
├── STEP 1: Check CONTEXT RECALL
│   │
│   └── LOW → Retrieval failed. The info was never there. Fix:
│             • Revisit chunk size and overlap (tables and lists
│               often get split mid-way — check this first)
│             • Try a stronger/domain-appropriate embedding model
│             • Add hybrid search (BM25 keyword + dense vector)
│               — catches exact product codes, names, acronyms
│               that embeddings often miss
│             • Increase k, then add a reranker to restore precision
│             • Query rewriting or HyDE for vague/short questions
│             • For multi-hop questions, consider iterative retrieval
│
└── Context recall is HIGH (the info WAS retrieved) ↓
    │
    ├── STEP 2: Check FAITHFULNESS
    │   │
    │   └── LOW → The LLM is hallucinating despite good context. Fix:
    │             • Strengthen the prompt: "Answer ONLY using the
    │               provided context. If the answer isn't there,
    │               say you don't know."
    │             • Lower temperature (often to 0)
    │             • Require inline citations to specific chunks —
    │               this alone reduces hallucination substantially
    │             • Try a stronger model
    │
    └── Faithfulness is HIGH ↓
        │
        └── STEP 3: Check ANSWER RELEVANCY and CONTEXT PRECISION
            │
            ├── LOW relevancy  → the model is dodging or rambling.
            │                    Tighten the prompt, add output format
            │                    requirements, add few-shot examples.
            │
            └── LOW precision  → too much noise in the context.
                                 Add a reranker, reduce k, improve
                                 chunking. Check "lost in the middle".
```

### 12.6 Operational RAG metrics

Never forget these in a system-design interview:

| Metric | Why |
|---|---|
| **End-to-end latency (p95)** | Users abandon slow interfaces. Break it down: retrieval vs generation. |
| **Cost per query** | Embedding + retrieval + generation tokens. Multiply by expected QPS and you have a monthly bill. Reranking and large k both increase it. |
| **Token usage** | Directly drives cost and latency. A k of 20 instead of 5 quadruples your prompt tokens. |
| **"I don't know" rate** | A system that appropriately abstains is usually **better** than one that always answers. Track it and make sure it's neither 0% (never abstains = hallucinating) nor too high (uselessly cautious). |

---

## 13. LLM evaluation

### 13.1 LLM-as-a-Judge

The dominant approach for evaluating open-ended generation, because no formula can assess "is this a good explanation?"

### Three formats

| Format | How it works | Reliability |
|---|---|---|
| **Pointwise (direct scoring)** | "Rate this answer 1–5 for helpfulness." | Simplest. **Least reliable** — scores drift over time, across prompts, and across judge model versions. |
| **Pairwise comparison** | "Which answer is better, A or B?" | **More reliable.** Relative judgments are much easier than absolute ones — humans are the same way. Produces a **win rate**. |
| **Reference-based** | "Here's the gold answer. Does the candidate match?" | Most reliable when you have references. |

**Why pairwise beats pointwise:** ask a human "rate this essay out of 10" and you'll get inconsistent answers from the same person on different days. Ask "which of these two essays is better?" and you'll get stable, reproducible answers. LLM judges behave identically.

**G-Eval** is a well-known refinement: have the judge first generate chain-of-thought evaluation *steps*, then score, and use the **token probabilities** of the score tokens to produce a smooth weighted average rather than a jumpy integer. (If the model puts 0.6 probability on "4" and 0.4 on "5", the score is 4.4 rather than 4.)

### The biases you must be able to name

This is a favourite interview question. Know all five.

| Bias | What happens | Mitigation |
|---|---|---|
| **Position bias** | Judges systematically favour whichever answer appears first (or sometimes last) | **Run both orderings and average.** Non-negotiable for pairwise evaluation. |
| **Verbosity bias** | Longer answers score higher regardless of quality | Control for length; instruct the judge explicitly to ignore length; report length alongside scores |
| **Self-enhancement bias** | A model rates its own outputs higher than others' | **Use a different model family as judge** |
| **Formatting bias** | Bullet points, headers and bold text score higher | Normalize formatting before judging, or instruct the judge to ignore it |
| **Sycophancy** | The judge agrees with whatever the prompt implies is correct | Neutral prompt wording; don't reveal which answer is "yours" |

### How to make a judge trustworthy

1. **Write a specific rubric with concrete criteria** — not "rate quality 1–10" but "score 1 if any factual claim is unsupported; 2 if all claims are supported but the answer is incomplete; …". Vague rubrics produce vague, unstable scores.
2. **Give few-shot examples** of scored outputs, especially borderline ones.
3. **Set temperature = 0** for reproducibility.
4. **Validate against human labels.** Have humans score 100–200 examples, then measure agreement with **Cohen's Kappa** (Section 6.13) or **Spearman correlation**. If κ < 0.4, your judge isn't measuring what you think it is, and every number you've produced with it is suspect.
5. **Version the judge prompt AND the judge model.** Upgrading the judge silently changes all your historical numbers and destroys your ability to compare across time. Pin the version and note it in every report.

> **Interview answer:** "I'd use pairwise comparison rather than absolute scoring, run both orderings to cancel position bias, use a judge from a different model family than the one being evaluated, and validate against a few hundred human labels before trusting any of it. I'd also pin the judge model version, because an upgrade silently invalidates every prior measurement."

---

### 13.2 Benchmarks and leaderboards

| Benchmark | What it measures |
|---|---|
| **MMLU / MMLU-Pro** | Broad multi-subject knowledge (57 subjects), multiple choice |
| **GSM8K, MATH** | Mathematical reasoning — GSM8K is grade-school word problems, MATH is competition level |
| **HumanEval, MBPP** | Code generation from docstrings |
| **SWE-bench** | Resolving **real GitHub issues** — much harder and much more realistic than HumanEval |
| **HellaSwag, ARC, WinoGrande** | Commonsense reasoning |
| **TruthfulQA** | Resistance to repeating common human misconceptions |
| **MT-Bench** | Multi-turn conversation, LLM-judged |
| **Chatbot Arena (LMSYS)** | Human pairwise votes → **Elo rating** |
| **MTEB** | Text embedding quality |
| **BBH (Big-Bench Hard)** | Hard reasoning tasks where models historically underperformed |

### The caveat every interviewer wants to hear: contamination

**Benchmark test sets leak into training data.** These benchmarks are published on the web, discussed on GitHub, quoted in papers — and models are trained on scraped web text. A model can score well on MMLU partly by **memorization** rather than reasoning.

**How the field detects and mitigates it:**
- **Private held-out evaluations** that are never published
- **Freshly generated test sets** created after the model's training cutoff
- **Rephrased benchmarks** — take MMLU questions, reword them while preserving meaning. **A large score drop on rephrased questions reveals memorization**, since genuine understanding should survive rewording.
- **Canary strings** embedded in benchmark files so dataset creators can detect inclusion

### Elo rating (Chatbot Arena)

Models are compared pairwise by anonymous human voters, and ratings update like chess ratings. 

**Why it's the most trusted LLM leaderboard despite having no ground truth at all:**
- It measures **human preference directly**, which is what you actually care about
- It's robust to adding new models (no need to re-run anything)
- It's very hard to contaminate — you can't memorize an Elo score
- Large vote volumes make it statistically stable

**Its limitation:** it measures *preference*, which correlates with but isn't identical to *correctness*. Voters may prefer confident, well-formatted answers over hedged, accurate ones. And it skews toward the kinds of questions people ask an arena, which aren't representative of enterprise workloads.

---

### 13.3 Safety, fairness and behaviour metrics

| Metric | What it measures |
|---|---|
| **Refusal rate** | How often the model declines. Measure on **two** sets: harmful prompts (should refuse) and benign prompts (**over-refusal rate** — should not refuse). Reporting only the first is misleading. |
| **Toxicity** | Scored with classifiers like Perspective API or Detoxify |
| **Jailbreak / attack success rate** | % of adversarial prompts that bypass safeguards |
| **Consistency** | Ask the same question three different ways — does the answer change? |
| **Bias / fairness** | Performance disparities across demographic groups |

### The fairness metrics, and why they conflict

| Metric | Definition |
|---|---|
| **Demographic Parity** | Equal positive prediction rates across groups |
| **Equalized Odds** | Equal TPR **and** equal FPR across groups |
| **Equal Opportunity** | Equal TPR across groups (the weaker version) |
| **Calibration within groups** | A predicted 70% means 70% for every group |

**The crucial fact: these are mathematically incompatible.** It has been formally proven that you cannot simultaneously satisfy demographic parity, equalized odds and calibration across groups — except in degenerate cases where base rates are identical across groups or the classifier is perfect.

**What this means practically:** fairness is not a box you tick. **You must choose which definition matters for your context and defend that choice.** In lending, calibration within groups might be the legal requirement. In hiring, demographic parity might be the policy goal. They pull against each other, and pretending otherwise is a mistake.

**Mentioning this impossibility result unprompted is one of the strongest signals you can give in an ML ethics discussion** — it shows you've engaged with the actual literature rather than absorbed a slogan.

---

# PART V — EVERYTHING ELSE & INTERVIEW PREP

## 14. Other tasks

### 14.1 Object detection

The model must draw boxes around objects **and** label them. Two independent ways to be wrong: wrong label, or right label but a badly-placed box.

#### IoU — Intersection over Union

The foundation of all detection and segmentation metrics.

```
IoU = (area where the predicted box and true box OVERLAP)
      ───────────────────────────────────────────────────
      (total area covered by EITHER box)
```

**Worked example.** True box is 100×100 pixels = 10,000 px². Your predicted box is also 100×100 but shifted 20 px right and 20 px down.

```
Overlap region = 80 × 80 = 6,400 px²

Union = area(true) + area(pred) − overlap
      = 10,000 + 10,000 − 6,400
      = 13,600 px²

IoU = 6,400 / 13,600 = 0.47
```

(We subtract the overlap in the union because otherwise we'd count it twice.)

**The IoU scale:**

| IoU | Meaning |
|---|---|
| 1.0 | Perfect overlap |
| **0.5** | The conventional threshold for "counts as a detection" |
| 0.75 | Tight, precise localization |
| 0.0 | No overlap at all |

A prediction counts as a **True Positive** if its IoU with a true box exceeds the threshold **and** the class label is right. Otherwise it's a False Positive. Any true box with no matching prediction is a False Negative. Once you have TP/FP/FN, precision and recall work exactly as in Section 6.

#### mAP — mean Average Precision

Compute **Average Precision** (Section 7.3) per object class, then average across classes.

**mAP@0.5** — the classic PASCAL VOC metric, using a 0.5 IoU threshold. Generous about localization.

**mAP@[0.5:0.95]** — the modern COCO primary metric. Compute mAP at ten IoU thresholds (0.50, 0.55, 0.60, …, 0.95) and average all ten.

**Why the stricter version matters.** A model that reliably gets IoU ≈ 0.55 scores identically to one that gets IoU ≈ 0.90 under mAP@0.5 — both pass the threshold. But for a robot arm picking up an object, or a self-driving car judging distance, the difference between a sloppy box and a tight one is everything. mAP@[0.5:0.95] rewards precise localization instead of just "roughly there."

---

### 14.2 Image segmentation

Per-pixel classification — every pixel gets a class label.

| Metric | Formula / idea | Notes |
|---|---|---|
| **IoU / Jaccard** | Same as above, applied to pixel masks. **mIoU** = averaged across classes | The standard for semantic segmentation |
| **Dice Coefficient** | `2 × \|A ∩ B\| / (\|A\| + \|B\|)` | **The default in medical imaging** |
| **Pixel Accuracy** | % of correctly classified pixels | **Almost always misleading** |
| **Hausdorff Distance** | The worst-case distance between predicted and true boundaries | Surgical planning, where the edge is the point |

**Dice vs IoU.** They're mathematically related: `Dice = 2·IoU / (1 + IoU)`. Dice is always ≥ IoU and weights overlap more generously.

**Why medical imaging prefers Dice:** the region of interest (a tumour) is often a tiny fraction of the image. IoU is harsh on small regions — being a few pixels off around the boundary of a 200-pixel tumour costs a lot of IoU. Dice is more forgiving there, which better reflects clinical usefulness.

**Why pixel accuracy is a trap** — the same accuracy paradox from Section 6.2, in a new costume:
```
Brain MRI slice: 512 × 512 = 262,144 pixels
Tumour region:   ~2,000 pixels  (0.76% of the image)

A model that predicts "no tumour anywhere" achieves:
Pixel accuracy = 260,144 / 262,144 = 99.24%
```
**99.24% accuracy, zero clinical value.** Always use Dice or IoU on the region of interest.

---

### 14.3 Image generation

| Metric | Measures | Direction |
|---|---|---|
| **FID (Fréchet Inception Distance)** | Statistical distance between the feature distributions of real and generated images | **Lower is better.** The standard. |
| **Inception Score (IS)** | Are images recognizable *and* diverse? | Higher is better. Largely superseded. |
| **CLIPScore** | Similarity between the generated image and the text prompt | Higher is better |
| **LPIPS** | Perceptual similarity between two specific images | Lower = more similar |
| **Human preference** | Still the gold standard | — |

**Why FID replaced Inception Score:** IS never looks at the real data at all — it only checks whether generated images are confidently classifiable and varied. A model can game it. FID compares the actual *distribution* of generated images against the distribution of real ones, which catches mode collapse and unrealistic artifacts.

**Why you need CLIPScore alongside FID:** FID measures whether your images look like real images. **It says nothing about whether they match the prompt.** A model that ignores "a cat wearing a hat" and generates a beautiful photorealistic landscape every time gets an excellent FID. CLIPScore catches that by measuring image–text alignment. **Report both.**

---

### 14.4 Anomaly / outlier detection

Extreme class imbalance — anomalies might be 0.01% of the data. So:

- **PR-AUC, not ROC-AUC** (Section 6.10 explains exactly why).
- **Precision@k where k = your alert budget.** This is the metric that matters operationally. If your team can investigate 100 alerts a day, then precision in the top 100 is the *only* thing that affects their working life. Everything below rank 100 is invisible.
- **Recall at a fixed alert budget** — the complementary framing: "of all real anomalies, what fraction do we catch in our top 100 alerts?"
- For time series: **point-adjusted** or **range-based** precision/recall, so a detection 3 minutes late still counts (Section 8.6).

**Scenario.** A datacenter monitoring system flags hardware failures. 50,000 servers, maybe 5 genuine failures a week. The ops team can investigate 20 alerts a day.

The right metric is **Precision@20 per day** and **weekly recall**. A model with 0.98 ROC-AUC that puts the real failures at rank 400 is worthless here.

---

### 14.5 Survival analysis / time-to-event

Predicting **when** something will happen — customer churn, equipment failure, patient mortality — where some subjects hadn't experienced the event when the study ended. That's called **censoring**, and it's why ordinary regression doesn't work: you know the customer hadn't churned by day 400, but you don't know when they will.

- **Concordance Index (C-index)** — the survival equivalent of AUC. Of all *comparable* pairs of subjects, in what fraction did the model correctly rank who experiences the event first? **0.5 = random, 1.0 = perfect.** Same interpretation as AUC.
- **Time-dependent Brier Score** / **Integrated Brier Score** — calibration of the predicted survival probability at each time point.

---

### 14.6 Topic modelling

- **Topic Coherence (C_v, UMass, NPMI)** — do the top words of a topic tend to co-occur in real documents? The best automatic proxy for "is this topic interpretable by a human?"
- **Topic Diversity** — % of unique words across all topics' top-N word lists. Catches the failure where five topics are all basically the same.
- **Perplexity** — available, but there's a famous result you should know: **perplexity is negatively correlated with human interpretability** in the classic LDA literature. Models with better perplexity produced topics humans found *less* meaningful. It's one of the cleanest real examples of a metric actively misleading a field, and it's worth citing.

---

### 14.7 Online / production metrics

**Offline metrics predict. Online metrics decide.**

| Concept | What you need to know |
|---|---|
| **A/B test** | Randomly split users, measure the business metric. Needs a properly powered sample size computed *in advance*. |
| **Statistical significance** | p-values and confidence intervals. Report the effect size and CI, not just "p < 0.05" — a statistically significant 0.02% lift may not be worth shipping. |
| **Multiple comparisons problem** | Test 20 metrics at p < 0.05 and **one will look significant by pure chance**. Apply Bonferroni or FDR correction, or pre-register a single primary metric. |
| **Novelty effect** | New features get engagement just for being new. Run the test long enough (2–4 weeks) for it to wash out. |
| **Guardrail metrics** | Things that must **not** degrade even if the primary metric improves — latency, error rate, unsubscribe rate, complaint rate. **Define these before launching**, not after. |
| **Interleaving** | For search/ranking: mix results from two rankers into one list and see which system's results get clicked. **Far more sensitive than A/B testing** and needs much less traffic, because each user compares both systems directly. |

### Why offline and online disagree — the answer that impresses

> "Offline metrics are a cheap filter, not the decision. They're computed on **logged data generated by the old model**, so I only observe outcomes for items the previous system chose to show — that's selection bias, and it systematically favours models that behave similarly to the incumbent. The offline metric may also be a poor proxy: NDCG can improve while revenue doesn't, because the model learned to rank popular items higher, which users click but would have found anyway. I'd use offline metrics to decide **what's worth A/B testing**, and let the online business metric make the actual call."

---

### 14.8 Reinforcement learning

| Metric | What it measures |
|---|---|
| **Cumulative / episodic reward** | The primary signal |
| **Sample efficiency** | Reward achieved per environment step. Often the real constraint — a policy that needs 10 million simulation steps may be impractical. |
| **Regret** | Total reward lost versus the optimal policy. The standard metric for multi-armed bandits. |
| **Stability across seeds** | How much results vary between random initializations |

**On seeds — this matters more in RL than anywhere else.** RL results vary *enormously* between random seeds. The same algorithm on the same environment can produce a working policy with seed 1 and complete failure with seed 2.

**Always report mean ± std over at least 5 seeds** (10 is better). Papers reporting a single seed are not trustworthy, and noting this in an interview shows genuine familiarity with the field's reproducibility problems.

---

## 15. The master cheat sheet

| Task | Primary metric | Also report | Watch out for |
|---|---|---|---|
| **Regression, general** | RMSE | MAE, R², bias | Outliers inflating RMSE — check the RMSE/MAE ratio |
| **Regression, outlier-heavy** | MAE / Median AE | RMSE for contrast | — |
| **Regression, orders of magnitude** | RMSLE | MAE per segment | — |
| **Regression, need a %** | WAPE | MAPE | MAPE breaks on zeros, biases low |
| **Regression, need a range** | Pinball loss | Interval coverage **and** width | Coverage without width is meaningless |
| **Binary classification, balanced** | F1 or Accuracy | ROC-AUC, confusion matrix | — |
| **Binary classification, imbalanced** | PR-AUC | F1, MCC, Recall@fixed Precision | ROC-AUC and accuracy both lie |
| **Fraud / disease detection** | Recall at acceptable precision | F2, PR-AUC, alert volume | Cost of a missed case |
| **Spam / moderation / auto-ban** | Precision | F0.5, review queue size | Cost of a false flag |
| **Probability used downstream** | Log Loss | Brier, reliability diagram, ECE | Great AUC ≠ calibrated |
| **Multi-class** | Macro-F1 | Weighted-F1, per-class table, top-k | Averaging hides dead classes |
| **Multi-label** | Micro-F1 + Macro-F1 | Hamming loss, subset accuracy | All-zeros gets good Hamming loss |
| **Search / ranking** | NDCG@k | MRR, MAP, Recall@k | Position bias in click-derived labels |
| **Recommender** | NDCG@k | Coverage, diversity, novelty | Popularity bias; offline ≠ online |
| **RAG retrieval** | Recall@k | MRR, Context Precision, p95 latency | **Fix retrieval before touching prompts** |
| **ANN vector index** | Recall@k **at fixed p95 latency** | QPS, memory, build time | Quoting recall without latency |
| **Time series** | MASE | MAE, **bias**, per-horizon breakdown | Shuffled splits = leakage |
| **Demand forecasting** | WAPE + MASE | Bias, pinball loss | MAPE breaks on zeros |
| **Probabilistic forecast** | CRPS or pinball | PICP **and** MPIW together | Wide intervals fake good coverage |
| **Clustering, no labels** | Silhouette | Davies-Bouldin, stability (ARI across seeds), qualitative profiles | Penalizes non-spherical clusters |
| **Clustering, labels exist** | ARI | AMI, V-measure | Purity is trivially gameable |
| **Machine translation** | COMET | chrF, BLEU (comparability), TER | BLEU alone is outdated |
| **Summarization** | **Faithfulness** (NLI or LLM judge) | ROUGE-1/2/L, BERTScore, abstractiveness | ROUGE can't see hallucination |
| **Extractive QA** | Token F1 | Exact Match | EM alone is too harsh |
| **NER** | Entity-level strict F1 | Per-type breakdown, scheme used | Token-level F1 overstates |
| **Speech recognition** | WER | CER, entity accuracy | WER can exceed 100%; treats "not" like "a" |
| **Code generation** | pass@1 | pass@10, test coverage | **Never use BLEU on code** |
| **RAG end-to-end** | Faithfulness + Context Recall | Answer Relevancy, Context Precision, cost, p95 latency, IDK rate | Evaluate the two halves separately |
| **Open-ended LLM output** | Pairwise win rate (LLM judge) | Human eval on a sample, Elo | Position, verbosity, self-enhancement bias |
| **Object detection** | mAP@[0.5:0.95] | mAP@0.5, per-class AP | — |
| **Segmentation** | mIoU / Dice | Boundary metrics | Pixel accuracy is a trap |
| **Image generation** | FID | **CLIPScore**, human preference | FID ignores the prompt entirely |
| **Anomaly detection** | PR-AUC | **Precision@alert-budget** | ROC-AUC looks fake-good |
| **Survival analysis** | C-index | Time-dependent Brier | — |
| **Topic modelling** | Coherence (C_v) | Topic diversity | Perplexity anti-correlates with quality |
| **Production** | Business KPI via A/B test | Guardrails, p95 latency, CI + effect size | Novelty effect, multiple comparisons |
| **RL** | Episodic reward | Regret, sample efficiency, **seed variance** | Single-seed results are anecdotes |

---

## 16. Scenario drills (test yourself)

Read each scenario, decide your answer, then check. These are close to what you'll actually be asked.

---

**Drill 1.** *A hospital wants a model to predict which discharged patients will be readmitted within 30 days, so a nurse can call them. They have capacity to call 50 patients a week out of 2,000 discharges. Readmission rate is 12%. What's your metric?*

<details><summary>Answer</summary>

**Precision@50** is the primary metric — because the intervention capacity *is* 50, nothing below rank 50 exists operationally. Also report **Recall@50** ("of the 240 patients who will be readmitted, how many are in our top 50?"). Use **PR-AUC** for comparing candidate models since 12% positives makes ROC-AUC flattering.

**The strong extra point:** the metric that ultimately matters is whether the calls *reduce* readmissions, which requires a randomized trial, not a prediction metric. A model can rank perfectly and the intervention can still not work.
</details>

---

**Drill 2.** *Your colleague reports a churn model with ROC-AUC 0.91 and says it's ready to ship. Churn rate is 3%. What do you ask?*

<details><summary>Answer</summary>

Ask for **PR-AUC and precision at the operating point they intend to use.** At 3% positives, ROC-AUC 0.91 is compatible with precision under 15% — meaning most flagged customers won't churn, and if you send them retention discounts you're burning margin on people who were staying anyway.

Also ask: what's the **baseline PR-AUC** (0.03)? What **threshold** will actually be used? What does a false positive **cost** (a discount given away) versus a false negative (a lost customer)?
</details>

---

**Drill 3.** *You're forecasting daily demand for 4,000 SKUs. Many have zero sales on most days. Which metric?*

<details><summary>Answer</summary>

**Not MAPE** — the zeros will produce divide-by-zero errors and infinite values.

Use **WAPE** (aggregates before dividing, so zeros are harmless and errors are weighted by business volume) and **MASE** against a seasonal-naive baseline (scale-free, so you can compare across SKUs with wildly different volumes).

**Also report bias**, because systematic over- or under-forecasting across 4,000 SKUs compounds into a serious inventory problem that MAE would never reveal.

**And segment the results** — a single number across 4,000 SKUs hides everything. Break it down by volume tier.
</details>

---

**Drill 4.** *Your RAG chatbot gives confident, fluent, wrong answers. Where do you look first?*

<details><summary>Answer</summary>

**Context Recall first.** Check whether the information was ever retrieved. If it wasn't, the model is filling the gap by inventing — and no prompt change will fix that. Look at chunking (are tables being split?), embedding model fit for the domain, hybrid search for exact terms, and k.

**Only if context recall is high** does this become a generation problem — then check **Faithfulness**, and fix with a stricter prompt ("answer only from the context; say you don't know"), temperature 0, and mandatory citations.

Saying "I'd check retrieval before touching the prompt" is the answer.
</details>

---

**Drill 5.** *A recruiter asks: your image classifier has 99.2% accuracy across 50 classes. Good?*

<details><summary>Answer</summary>

**Ask for the per-class breakdown and the class distribution.** With 50 classes, if 2 classes hold 95% of the data, 99.2% accuracy is compatible with the model completely failing on 30 rare classes.

Report **macro-F1** — it will be dramatically lower than accuracy if rare classes are broken. And ask which classes *matter*: if one of the failing rare classes is "defective product" or "malignant," aggregate accuracy is irrelevant.
</details>

---

**Drill 6.** *You improved NDCG@10 by 4% offline. The A/B test shows no change in revenue. What are three possible explanations?*

<details><summary>Answer</summary>

1. **Selection bias in the offline data** — it was logged by the old ranker, so you only see outcomes for items it chose to show. Your gain may be on items users would never have reached.
2. **NDCG is a poor proxy for revenue here** — the relevance labels may be click-derived, and clicks don't equal purchases. You might be ranking clickbait higher.
3. **The test may be underpowered or too short** — a 4% ranking improvement may produce a revenue effect too small to detect with the traffic and duration used, or novelty effects may be masking it.

Bonus: **a guardrail may have degraded** — if the new ranker is slower, the latency cost could be cancelling the relevance gain.
</details>

---

**Drill 7.** *Your summarization model scores ROUGE-2 = 0.28, beating the previous model's 0.24. A domain expert reviews the outputs and says they're worse. Who's right?*

<details><summary>Answer</summary>

**Probably the expert.** ROUGE measures n-gram overlap and is blind to factual errors. The new model may have learned to match the reference's *phrasing* more closely while hallucinating numbers, flipping directions, or misattributing quotes — all of which ROUGE cannot see and a domain expert can.

Measure **faithfulness** (NLI-based like AlignScore, or an LLM judge with claim decomposition). Also check **abstractiveness** — if the new model is just copying more source text, ROUGE rises while the summarization value falls.

And check the **Lead-3 baseline**; it might beat both.
</details>

---

**Drill 8.** *A credit model has AUC 0.88 and is used to set interest rates. Anything missing?*

<details><summary>Answer</summary>

**Calibration.** AUC only measures *ranking*. The model can order borrowers perfectly while being systematically overconfident, which means every interest rate is mis-priced.

Report **log loss or Brier score**, plot a **reliability diagram**, compute **ECE**, and apply **Platt scaling or isotonic regression** on validation data if it's miscalibrated.

Also: since this is a regulated lending decision, report **fairness metrics across protected groups** — and be ready to explain that demographic parity, equalized odds and calibration-within-groups are mathematically incompatible, so you must choose and defend one.
</details>

---

## 17. Interview questions & model answers

**Q: Your model has 99% accuracy. Are you happy?**
> Not yet — I'd first ask about class balance. If 99% of examples belong to one class, a model that always predicts the majority gets exactly 99%, so accuracy tells me nothing. I'd compute the majority-class baseline, look at the confusion matrix, and report precision and recall for the minority class plus PR-AUC or MCC. I'd also want to know what a false positive and a false negative each cost, because that determines which metric should lead.

**Q: Explain precision and recall to a non-technical stakeholder.**
> Precision is "when the system raises an alarm, how often is it right?" Recall is "of all the real cases out there, how many did the system catch?" You can always trade one for the other by making the system more or less trigger-happy — flag everything and you catch all of them but most alarms are wrong; flag only the obvious cases and every alarm is right but you miss a lot. The question for you is which mistake costs us more.

**Q: When would you use ROC-AUC vs PR-AUC?**
> ROC-AUC when classes are reasonably balanced and both matter. PR-AUC when the data is imbalanced and I care about the rare positive class. The reason is structural: ROC's false positive rate has true negatives in its denominator, so when negatives vastly outnumber positives, even thousands of false positives barely move FPR and ROC-AUC stays flattering while precision is terrible. PR-AUC contains no true negatives so it can't be inflated that way. I'd also note PR-AUC's random baseline is the positive class rate, not 0.5 — so a PR-AUC of 0.4 on 1% positives is excellent, not mediocre.

**Q: Why is F1 a harmonic mean rather than a plain average?**
> Because the harmonic mean is dragged toward the smaller value. A model with precision 1.0 and recall 0.01 gets an arithmetic mean of 0.505, which flatters a useless model; F1 gives about 0.02, which is honest. The harmonic mean refuses to let a perfect score cover for a collapsed one, so you can only get a high F1 if both components are genuinely high.

**Q: Can R² be negative? What does that mean?**
> Yes. R² compares your model's squared error against the error of always predicting the mean. Negative R² means your squared error is *larger* than the mean baseline's — your model is worse than doing nothing. It usually signals severe overfitting, a pipeline bug, or evaluating on data from a different distribution than you trained on. It's most often seen on a test set for a model that looked fine in training.

**Q: Why can't you use normal k-fold cross-validation on time series?**
> Because random shuffling lets the model train on future data to predict the past. Time series are autocorrelated, so if the model has seen days 100 and 102, predicting day 101 is near-trivial interpolation — your offline scores look excellent and collapse in production where genuine extrapolation is required. I'd use rolling-origin validation where every training window ends before its test window begins, and I'd insert a gap matching the real forecast lag, since in production there's always a delay between the last available data and the period being predicted.

**Q: What's MASE and why do you like it?**
> It's your model's MAE divided by the MAE of a naive forecast — "tomorrow equals today," or seasonal naive. Below 1 means you beat the baseline; above 1 means your model is worse than one line of code. I like it because it's scale-free, handles zeros, and forces the baseline comparison. A lot of published deep-learning forecasting results fail to beat seasonal naive, and without MASE you wouldn't know.

**Q: Your RAG system gives bad answers. How do you debug it?**
> I'd split the diagnosis between retrieval and generation, because they need completely different fixes. First, context recall — was the needed information even in the retrieved chunks? If not, it's retrieval: chunking strategy, embedding model, hybrid search for exact terms, k, reranking. Only if the context was good do I look at faithfulness — is the model hallucinating despite having the right material? That's prompting, temperature, citations, or model capability. Then context precision and answer relevancy for noise and evasiveness. Tuning prompts when retrieval is broken is wasted effort, and that's the most common mistake I see.

**Q: How does Ragas measure faithfulness without a ground truth answer?**
> It uses an LLM to decompose the generated answer into atomic claims, then checks each claim against the retrieved contexts for entailment. The score is supported claims over total claims. Because it needs only the question, answer and contexts — all of which you already log — it can run on live production traffic as continuous monitoring. That's a real operational advantage over metrics that need labelled references. One caveat: faithfulness measures groundedness, not truth. If the retrieved context was wrong, a perfectly faithful answer is still wrong.

**Q: What's wrong with ROUGE for summarization?**
> It measures n-gram overlap, so it's blind to factual correctness. A summary that reverses a number or attributes a quote to the wrong person shares nearly all its words with the reference and scores well. ROUGE and faithfulness scores are close to uncorrelated in published studies, so optimizing ROUGE can actually increase hallucination. I'd keep ROUGE for comparability with published baselines but treat an NLI-based or LLM-judged faithfulness score as primary — and I'd include Lead-3 as a baseline, since simply taking the first three sentences beats a lot of neural models on ROUGE.

**Q: Is a low-perplexity model a good model?**
> Not necessarily. Perplexity measures how well a model predicts the next token in a corpus — it says nothing about instruction-following, factuality, or safety. It's also only comparable between models sharing a tokenizer and test set, since different tokenizations mean different units of prediction. And instruction tuning often *increases* perplexity while making a model dramatically more useful, which is why chat models aren't evaluated this way.

**Q: How do you pick the number of clusters?**
> No single metric decides it. I'd plot silhouette across a range of k and look for a peak, cross-check against the elbow on inertia and Davies-Bouldin, and test stability by re-running on bootstrap samples and measuring ARI between runs — if assignments swing wildly, the structure isn't real. But the deciding factor is usually qualitative: profile each cluster on business-meaningful features and see whether stakeholders can name the segments and act differently toward each. Geometrically clean clusters that nobody can use are a failed clustering. I'd also note that silhouette assumes roughly spherical clusters, so it's not a fair way to compare k-means against DBSCAN.

**Q: How do you know your LLM judge is any good?**
> I'd validate it against humans — have annotators score two hundred outputs, then measure agreement using Cohen's Kappa or Spearman correlation. I'd control for the known biases: run pairwise comparisons in both orderings to cancel position bias, check whether the judge simply prefers longer answers, and avoid having a model judge its own family. I'd use pairwise comparison rather than absolute scoring since relative judgments are far more stable. And I'd pin the judge model and prompt version, because upgrading the judge silently invalidates every historical number.

**Q: Offline metrics improved but the A/B test was flat. What happened?**
> Several plausible causes. Offline evaluation uses logged data generated by the old system, so I only observe outcomes for items the previous model chose to show — that's selection bias, and it favours models similar to the incumbent. The offline metric may also be a weak proxy; NDCG can improve while revenue doesn't if the model learned to rank popular items higher, which users click but would have found anyway. It's also worth checking whether the test was adequately powered, whether novelty effects distorted the window, and whether a guardrail like latency degraded enough to cancel the gain.

**Q: Precision is 0.9 and recall is 0.3. What do you do?**
> First I'd establish which error costs more. If misses are expensive — fraud, disease, safety — recall of 0.3 is unacceptable and I'd lower the threshold, trading precision away, choosing the point using an explicit cost estimate for each error type. But before assuming the model is weak, I'd check PR-AUC: if it's high, the model ranks well and this is purely a threshold choice, solvable in one line. If PR-AUC is also low, threshold tuning won't save it and I need better features, more positive examples, or a different model class. Distinguishing "bad threshold" from "bad model" is the first diagnostic.

**Q: What's calibration and why does it matter?**
> A model is calibrated if, among the cases where it said 70%, the thing actually happened about 70% of the time. It matters whenever the probability is used downstream rather than just thresholded — pricing a loan, ranking an investigation queue by expected value, showing a number to a user. A model can have excellent AUC and terrible calibration, because AUC only measures ordering. I'd check with a reliability diagram and ECE, and fix with Platt scaling or isotonic regression on validation data. Modern neural networks are notoriously overconfident out of the box.

**Q: What's data leakage? Give an example.**
> It's when information unavailable at prediction time sneaks into training, making offline scores optimistic and production performance a disaster. The most common version is preprocessing leakage — fitting a scaler on the full dataset before splitting, so training statistics are contaminated by test data. Target leakage is more dramatic: a churn model using a "cancellation reason" field that only gets populated after the customer has already churned. The heuristic I use is to ask of every feature: would I actually have this value at the moment I need to predict? And if accuracy looks suspiciously high, assume leakage until proven otherwise.

---

## 18. Mistakes that get you rejected

1. **Reporting accuracy on imbalanced data** without the majority-class baseline. The single most common junior mistake.
2. **Evaluating on training data** — or worse, tuning hyperparameters against the test set and then presenting the test score as an unbiased estimate.
3. **Preprocessing leakage.** Fitting a scaler, imputer, or encoder on the full dataset before splitting.
4. **Shuffling time series data.** Instant disqualification for a forecasting role.
5. **No baseline at all.** Majority class for classification, mean for regression, naive/seasonal-naive for forecasting, BM25 for retrieval, Lead-3 for summarization, most-popular for recommendation. Every result needs one.
6. **Reporting a single number.** Always give the confusion matrix, per-class breakdown, or distribution. Averages hide broken subgroups — and the broken subgroup is often the one that matters.
7. **Ignoring variance.** One split or one seed is an anecdote. Report mean ± std across folds or seeds.
8. **Optimizing a metric that isn't the goal.** Maximizing engagement can maximize outrage. Maximizing ROUGE can maximize hallucination. Maximizing NDCG can kill catalogue diversity. (Goodhart's Law: when a measure becomes a target, it stops being a good measure.)
9. **Confusing offline improvement with real improvement.** Offline is a filter; production is the verdict.
10. **Not knowing the cost of errors.** If you can't say roughly what a false positive and a false negative cost the business, you can't defend your metric choice — and that's the actual question being asked.
11. **Quoting ANN recall without latency**, or reporting mean latency instead of p95/p99.
12. **Trusting an LLM judge that was never validated against humans.**
13. **Reporting a metric without its baseline range.** PR-AUC 0.40 is excellent on 1% positives and terrible on 40% positives. The number alone means nothing.
14. **Forgetting bias in regression and forecasting.** Absolute-value metrics are structurally blind to systematic over- or under-prediction, which is usually a bigger business problem than random error.

---

## Final thought

The formulas are the easy part. You can look them up, and honestly, so can the interviewer.

What separates a strong candidate is the sentence that comes **after** naming the metric:

> **"I'd use *X*, because in this problem the cost of *this specific mistake* is higher — and here's the baseline I'd compare against."**

Metric. Reason. Baseline. Every single time.

If you internalize nothing else from this document, internalize that pattern and the four numbers in the confusion matrix. Everything else follows from those.

Good luck with the interviews.
