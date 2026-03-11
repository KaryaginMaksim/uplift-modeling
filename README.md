# 🚀 Uplift Modeling for Marketing Campaign Optimization

> End-to-end uplift modeling project for product marketing communication: identify which customers should actually receive a campaign in order to maximize incremental impact and reduce wasted marketing spend.

---

## ✨ Overview

This repository is designed for **uplift modeling in product marketing campaigns** where the business goal is not simply to predict conversion, but to estimate:

- who will convert **because of** communication
- who will convert **without** communication
- who may react negatively to communication
- where marketing budget is being wasted

The project compares three increasingly advanced approaches:

- **Two-Model Approach**
- **Single-Model Class Transformation**
- **Uplift Random Forest** inspired by the methodology popularized by **Leo Guelman**

The result is a practical framework for **customer-level treatment targeting** and **marketing cost optimization**.

---

## 🧠 Problem This Solves

In many product businesses, campaigns are still sent using:

- propensity models
- business rules
- broad segmentation
- static CRM heuristics

That usually leads to expensive mistakes:

- messaging customers who would convert anyway
- wasting discount or incentive budget
- contacting customers who are unlikely to respond
- harming users who react negatively to outreach

Uplift modeling solves this by estimating the **incremental causal effect of treatment** for each client.

Instead of asking:

> “Who is likely to buy?”

we ask:

> “Who is more likely to buy if we contact them?”

That is the key difference between **response modeling** and **uplift modeling**.

---

## ⚙️ Project Goals

✔ Optimize marketing costs  
✔ Improve campaign ROI  
✔ Rank customers by incremental treatment effect  
✔ Compare classical and specialized uplift approaches  
✔ Provide interpretable, business-friendly evaluation artifacts  
✔ Build a reusable experimentation template for CRM / lifecycle / retention use cases

---

## 🏗 Methodology

This repository is structured as a progression from baseline to specialized causal modeling.

### 1 — Two-Model Approach

Train two separate models:

- one on the **treated** population
- one on the **control** population

Then estimate uplift as:

```text
uplift(x) = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)
```

**Why use it**

- simple and intuitive
- easy to explain to stakeholders
- strong baseline for many tabular problems

**Key limitation**

Two independent models may become unstable if treatment and control populations differ materially or if sample sizes are imbalanced.

---

### 2 — Single-Model Approach with Class Transformation

Transform the target into an uplift-aware class label and train a single model to learn the treatment effect signal directly.

Typical intuition:

- reward cases where treatment helped
- penalize cases where treatment hurt
- let one model learn the differential effect pattern

**Why use it**

- one unified model
- often simpler deployment logic
- can be efficient for ranking treatment impact

**Key limitation**

Interpretation is less straightforward than in the two-model setup, and implementation details matter for correctness.

---

### 3 — Uplift Random Forest

The final and most specialized stage uses **Uplift Random Forest**, a tree-based method explicitly designed for treatment effect heterogeneity.

This approach is associated with the practical uplift modeling line of work described by **Leo Guelman** and widely used in direct marketing and retention problems.

**Why use it**

- directly optimizes splits for uplift separation
- captures nonlinear treatment-response interactions
- better aligned with the business objective than standard classifiers

**Key limitation**

- more specialized tooling
- more careful validation required
- harder to productionize than baseline models if the organization is early in causal ML maturity

---

## 📈 Business Use Cases

This repository is intended for product and CRM scenarios such as:

- push / email / SMS targeting
- retention and churn prevention campaigns
- discount allocation
- onboarding nudges
- reactivation campaigns
- upsell / cross-sell communication

Example decision rule:

```text
Contact only users with positive predicted uplift above a business threshold.
```

This reduces spend on:

- sure things
- lost causes
- sleeping dogs

and concentrates budget on the segment where communication creates real incremental value.

---

## 🧪 Evaluation Strategy

This project focuses on uplift-specific evaluation rather than plain classification quality.

### Core metrics

- **Qini curve**
- **Qini coefficient**
- **Uplift curve**
- **AUUC** (Area Under the Uplift Curve)
- incremental gain by treatment decile / percentile

### Business metrics

- incremental conversions
- incremental revenue
- contact cost savings
- ROI under targeting policy

### Why AUC is not enough

A standard classifier can have strong ROC-AUC and still be useless for campaign optimization, because it predicts **conversion probability**, not **incremental treatment effect**.

---

## 🧱 Expected Repository Structure

```text
uplift-modeling/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── src/
    └── .gitkeep
```

---

## 📦 Recommended Workflow

### 1 — Prepare data

Input dataset should contain:

- customer features `X`
- treatment flag `T` (`1` = contacted, `0` = control)
- outcome flag `Y`

Optional but useful:

- campaign cost
- revenue / margin
- channel metadata
- customer segment labels
- experiment timestamp

---

### 2 — Build baselines

Start with:

- two-model uplift baseline
- class transformation model

This gives interpretable first results before moving to specialized tree-based uplift algorithms.

---

### 3 — Train specialized uplift model

Use uplift random forest as the final benchmark and compare ranking quality and business value against simpler methods.

---

### 4 — Evaluate targeting policy

Do not stop at scores. Simulate decisions such as:

- top 5% treated by uplift
- top 10% treated by uplift
- treatment only above zero uplift
- constrained budget policy

This is where the real marketing value appears.

---

## 🚀 Quick Start

### 1 — Enter the repository

```bash
cd uplift-modeling
```

### 2 — Create environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Core Approaches Covered

| Approach | Role in project | Strength |
|-----|---------------------|----------------------------|
| **Two-Model** | Baseline uplift estimation | Simple, intuitive, business-friendly |
| **Class Transformation** | Compact single-model alternative | Unified scoring logic |
| **Uplift Random Forest** | Final specialized model | Best fit for heterogeneous treatment effects |

---

## 🛡 Practical Considerations

- Use only randomized or well-designed treatment/control data for credible uplift estimation
- Monitor treatment imbalance across important segments
- Avoid leakage from post-treatment features
- Evaluate both statistical quality and campaign economics
- Recalibrate targeting thresholds based on contact cost and margin

---

## 🔐 Modeling Discipline

To make uplift modeling useful in production, the repository should enforce:

- strict train / validation / test separation
- experiment tracking
- business-threshold simulation
- reproducible preprocessing
- transparent feature documentation

Without this, uplift models easily become interesting offline exercises with weak deployment value.

---

## 📈 What Success Looks Like

This project succeeds if it helps answer questions like:

- Which customers should we contact at all?
- Which customers should explicitly not be contacted?
- How much budget can be saved with uplift targeting?
- Which modeling approach gives the best incremental business impact?

The final outcome is not “best accuracy”.

The final outcome is:

> better campaign decisions with lower cost and higher incremental return.

