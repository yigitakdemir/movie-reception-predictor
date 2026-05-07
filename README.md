# What Makes a "Good" Movie?
### A Genre, Theme and Feeling-Based Machine Learning Approach

**Course:** MPCS 53120 Applied Data Analysis — University of Chicago  
**Author:** Yigit Akdemir

---

## Overview

This project investigates what content elements — genres, themes, and feelings — make a movie well-received by audiences and critics. Rather than predicting box office revenue (which depends heavily on marketing and timing), this project focuses on a movie's intrinsic content and asks: **given a film's genre, thematic elements, and the feelings it evokes in viewers, can we predict whether it will be considered "good"?**

The dataset covers **1,500 critically and culturally significant films** spanning 1920–2024, with ratings drawn from IMDb (general audience), Rotten Tomatoes Tomatometer, and Metacritic. The response variable is the average of these three rating sources, normalized to a 0–100 scale.

The project's most novel contribution is a **custom-built feelings dataset** — three independent feeling measurements per film, generated through review-based NLP, transformer models, and GPT-4o — described in detail below. The feelings dataset is published as open-source on Kaggle: [Movie Feelings: Emotion Features for 1,500 Films](https://www.kaggle.com/datasets/yakdemir/movie-feelings-emotion-features-for-1500-films)

---

## Repository Structure

```
├── Yigit_Akdemir_ADA_Final_Project_Data_Gathering.ipynb   # Data pipeline (1,412 lines)
├── Yigit_Akdemir_ADA_Final_Project_Analysis.ipynb         # Modeling and analysis (1,426 lines)
├── max_ds.xlsx                                             # Main dataset (1,500 movies, 2,157 predictors)
└── README.md
```

### Supporting data files (required to run notebooks)
| File | Description |
|------|-------------|
| `clean_rt_reviews.xlsx` | Rotten Tomatoes reviews linked by IMDb ID |
| `clean_top_emotions_with_plots.xlsx` | F2 feelings (transformer-based, from movie plots) |
| `feeling_counts_normalized.xlsx` | F1 feelings (NLP review-based, normalized 0–1) |
| `filtered_reviews_350.xlsx` | Stanford Movie Review Database subset |
| `gptf_df_clean.xlsx` | F3 feelings (GPT-4o generated, cleaned) |
| `omdb_with_rt.xlsx` | OMDB data with Tomatometer scores |
| `updated_ewg.xlsx` | 50 feelings × 313 related words (F1 synonym list) |

---

## Data Sources

| Source | Usage |
|--------|-------|
| **TMDB** (via Kaggle, 930k movies) | Primary theme/keyword data (T1: 1,915 keywords) + genres |
| **IMDb** (non-commercial datasets) | General audience ratings + genres + IMDb IDs for linkage |
| **OMDB API** | Metacritic scores, Tomatometer, genres, movie plots |
| **Rotten Tomatoes** (via Kaggle) | Tomatometer scores + 880k+ critic/audience reviews |
| **Stanford Large Movie Review Dataset** | Additional 880k reviews for feeling tallies |
| **MPST Dataset** (Kar et al., 2018) | 69 thematic tags (T2) |
| **Kaggle IMDb Reviews** | Additional review corpus for F1 feeling generation |

---

## The Feelings Dataset — Novel Contribution

No existing dataset classified films across a broad spectrum of emotional responses. This project created **three independent feeling measurements** for each of 50 emotions (e.g., awe, triumph, frustration, compassion, tension, regret):

### F1 — Review-Based NLP Feelings
- Built a custom **313-word synonym list** across 50 feelings through multiple GPT-4o iterations and manual review
- Scanned **~880,000 movie reviews** (IMDb + Rotten Tomatoes + Stanford) for feeling-related word matches
- Each feeling scored 0–1 per film (proportion of reviews mentioning that feeling)

### F2 — Transformer-Based Plot Feelings
- Applied **Facebook's BART (zero-shot, MNLI fine-tuned)** to classify each movie's OMDB plot synopsis into a shortlist of 10 top feelings
- Re-ranked using a **RoBERTa cross-encoder** to identify the top 3 most semantically relevant feelings per film
- Hybrid approach chosen to balance accuracy and computation time

### F3 — GPT-4o Generated Feelings
- Used **OpenAI GPT-4o** to independently generate the top 3 feelings per film from the same 50-feeling list
- Required multiple prompt iterations to minimize out-of-list outputs
- Provided a third independent signal for cross-validation of feeling measurements

---

## Predictor Set

The maximal dataset contained **2,157 predictors** across 6 feature types:

| Type | Description | Count |
|------|-------------|-------|
| G | Genres (combined from TMDB, IMDb, OMDB) | 23 |
| T1 | TMDB keywords mapped to 3+ films | 1,915 |
| T2 | MPST thematic tags | 69 |
| F1 | Review-based feelings | 50 |
| F2 | Transformer plot feelings | 50 |
| F3 | GPT-4o feelings | 50 |

For regression models, a **base 118-predictor set** was used (23G + 45 most common T1 + 50 F1) to satisfy the 10 observations-per-predictor rule of thumb at 80/20 train-test split.

---

## Models and Results

Both continuous (average rating) and binary (above/below median = 78.5) response variables were evaluated across 13 model classes with 10-fold cross-validation.

### Continuous Response — Test RMSE Comparison
| Model | Test RMSE |
|-------|-----------|
| **Boosting** (lasso-selected 150 predictors, shrinkage=0.01) | **7.429** ✅ best |
| Bagging | 7.576 |
| Lasso (α≈0.38, 150 predictors) | 7.645 |
| PLS (1 component) | 7.768 |
| Ridge | 7.768 |
| Backward Selection (98 predictors) | 7.697 |
| Linear Regression (24 sig. predictors) | 7.806 |
| Forward Selection (125 predictors) | 8.469 |
| PCA (30 components) | 8.629 |

### Binary Response — Accuracy Comparison
| Model | Accuracy |
|-------|----------|
| **SVM** (polynomial, d=1, C=100, 118-predictor base set) | **73.6%** ✅ best |
| Bagging | 73.3% |
| Logistic Regression (15 sig. predictors) | 73.2% |
| Boosting | 72.6% |
| SVC | 72.6% |
| Neural Network (relu, adam, lr=0.01, structure 100-50-25) | 68.2% |
| Forward Selection | 68.2% |
| Ridge | 67.2% |
| PCA | 62.4% |

### Key Findings
- **`f1_awe`** (review-based awe feeling) was the single most important predictor in the boosting model by a large margin — followed by `f1_triumph`, `f1_frustration`, and `f1_compassion`
- F1 feelings (review-based NLP) consistently outperformed F2 (transformer) and F3 (GPT-4o) as predictors
- Animation genre significantly increases expected rating (~+5%); horror and sport significantly decrease it (~-4 to -6%)
- Black-and-white films are notably better-received (~+6 rating points), likely due to selection effects in the curated film set
- Models with as few as 15 statistically significant predictors achieved accuracy comparable to models with 150 predictors

---

## Notable Challenges

- **Neural network convergence failure:** Initial logistic activation + 0.1 learning rate caused all predictions to collapse to 1. Resolved by switching to relu activation, adam solver, 0.01 learning rate, and 5× epochs — improving accuracy from 50% to 68.2%
- **F1 feelings synonym construction:** WordNet/NLTK synonyms were too generic for movie-specific sentiment. Built a custom 313-word list through iterative GPT-4o generations and manual trial runs on the full review corpus
- **F2 speed vs. accuracy tradeoff:** Running RoBERTa cross-encoder on all 50 feelings per movie was computationally prohibitive. Used BART to first shortlist to 10, then applied cross-encoder — reducing computation while maintaining precision
- **Predictor dimensionality:** 2,157 predictors for 1,500 observations required careful selection strategies; only 12 predictor pairs had >80% correlation

---

## Limitations

- **`f1_awe` and `f1_regret` may be partially confounded:** Words like "wonderful" and "fascinating" (mapped to awe) are also used as generic positive expressions in reviews, partially making `f1_awe` a proxy for overall positivity. Similarly, "disappoint" and "embarrass" are generic negative signals captured under `f1_regret`.
- **After-credits scene predictor:** Statistically significant as a positive predictor in logistic regression, likely due to confounding with Marvel films rather than a direct causal effect.
- **Overall accuracy ceiling (~74%):** Suggests that a film's genre, themes, and emotional content — while predictive — do not fully determine audience reception. Cinematography, cast, soundtrack, marketing, and release timing likely account for a meaningful portion of the remaining variance.

---

## Software and Libraries

**Data Gathering:**
`pandas`, `numpy`, `kagglehub`, `requests`, `BeautifulSoup`, `transformers` (BART, RoBERTa), `sentence-transformers`, `torch`, `sklearn` (cosine similarity), `NLTK`, `OpenAI API`

**Analysis:**
`statsmodels` (linear/logistic regression), `sklearn` (forward/backward selection, ridge, lasso, PCA, PLS, neural nets, decision trees, SVC/SVM, train-test split, CV, confusion matrix), `matplotlib`, `seaborn`

**Environment:** Google Colab

---

## Kaggle Dataset

The novel movie feelings dataset generated in this project is publicly available on Kaggle:

**[Movie Feelings: Emotion Features for 1,500 Films](https://www.kaggle.com/datasets/yakdemir/movie-feelings-emotion-features-for-1500-films)**

The published dataset includes identity columns, ratings, plot synopses, and all 150 feeling features (F1/F2/F3) across 1,500 films.

---

## Notes

- API keys (OMDB, OpenAI) are not included in this repository. Set your own keys in the relevant notebook cells before running.
- The data gathering notebook must be run before the analysis notebook to reproduce the full pipeline.
- Throughout development, ChatGPT and Gemini were used for code debugging assistance and suggestions on specific implementation approaches.
