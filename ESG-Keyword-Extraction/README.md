# ESG Keyword Extraction & Disclosure Mapping

An NLP project that adapts classic keyword-extraction techniques to corporate ESG disclosures. The pipeline extracts ESG-relevant keywords/keyphrases, maps them to Environmental, Social, and Governance categories, and supports reporting-framework mapping.

## What this project does

- Accepts paragraph/sentence-level ESG disclosure text.
- Extracts keywords/keyphrases using TF-IDF, RAKE, and TextRank.
- Applies ESG-domain vocabulary weighting.
- Classifies extracted terms into Environmental, Social, and Governance categories.
- Maps concepts to selected ESG reporting frameworks.
- Produces a disclosure profile and method comparison.
- Provides a reproducible evaluation framework using human-labelled ESG examples.

## Project structure

```text
ESG-Keyword-Extraction/
├── data/
│   ├── esg_documents.csv
│   └── labeled_sentences.csv
├── 01_eda_esg.ipynb
├── 02_esg_extraction.ipynb
├── 03_esg_evaluation.ipynb
├── esg_dictionary.py
├── utils.py
├── requirements.txt
└── README.md
```

## Methods

### TF-IDF
Ranks terms that are important to a document while reducing the influence of terms that occur broadly across the corpus. The implementation supports unigrams, bigrams, and trigrams so concepts such as `scope 1 emissions` can be retained.

### RAKE
Extracts candidate multi-word phrases and scores them from word co-occurrence and phrase structure.

### TextRank
Builds a word co-occurrence graph and applies a PageRank-style algorithm to identify central terms.

### Hybrid ESG ranker
Combines normalized TF-IDF, RAKE, TextRank, and ESG-domain scores. The coefficients are configurable and should be tuned against a validation set for production use.

## ESG taxonomy

The included starter vocabulary covers three top-level categories:

- **Environmental:** emissions, energy, climate, water, waste, biodiversity, circular economy.
- **Social:** health and safety, human rights, workforce, diversity, training, community, customers, supply chain.
- **Governance:** board structure, ethics, anti-corruption, whistleblowing, risk, compliance, remuneration, related parties, privacy.

The dictionary is deliberately editable so it can be extended for sector-specific and framework-specific terminology.

## Dataset schema

### `esg_documents.csv`

| Column | Description |
|---|---|
| company | Company name |
| year | Reporting year |
| document_type | Annual Report, BRSR, Sustainability Report, etc. |
| section | Disclosure section |
| text | Paragraph or sentence used for analysis |

### `labeled_sentences.csv`

| Column | Description |
|---|---|
| sentence | Sentence being evaluated |
| keyword | ESG concept(s) expected in the sentence, separated by `\|` |
| is_esg | 1 if the concept is ESG-relevant, otherwise 0 |
| category | E, S, G, or blank |

The included labels are starter examples for testing. For a portfolio-grade benchmark, expand the labelled set with manually reviewed ESG disclosure text.

## Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order:

```text
01_eda_esg.ipynb
02_esg_extraction.ipynb
03_esg_evaluation.ipynb
```

## Example

Input:

> The company reduced Scope 1 greenhouse gas emissions through energy-efficiency projects and increased renewable electricity procurement.

Possible extracted concepts:

```text
scope 1 greenhouse gas emissions
renewable electricity
energy efficiency
greenhouse gas emissions
```

Typical classification:

```text
Environmental
```

## Important limitation

Keyword extraction is not the same as materiality assessment, assurance, or an investment recommendation. Results should be reviewed in context, particularly where terminology is ambiguous or where the same concept appears in targets, policies, historical discussion, and performance data.
