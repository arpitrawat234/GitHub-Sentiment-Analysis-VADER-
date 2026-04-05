# GitHub Sentiment Analyser

Pulls issues, PR comments and commits from any public GitHub repo and runs them through VADER to figure out how developers are feeling about the project.

Built this for our DA-304T stats project at MAIT.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste a repo like `microsoft/vscode` into the sidebar, pick your data sources, hit Run.

---

## GitHub Token

Not required, but without one you're limited to 60 API calls/hour which runs out fast on big repos.

Get one at: `github.com → Settings → Developer Settings → Personal Access Tokens`  
Only needs `public_repo` scope.

---

## What it shows

- Sentiment split (positive / negative / neutral) across all fetched items
- How sentiment changes month by month
- Which authors tend to write more positive or negative comments
- The most extreme items in each category
- Full CSV export

---

## How VADER labels text

| Score | Label |
|---|---|
| ≥ 0.05 | Positive |
| ≤ -0.05 | Negative |
| in between | Neutral |

---

## Stack

Python · GitHub REST API · vaderSentiment · pandas · matplotlib · seaborn · Streamlit

--
