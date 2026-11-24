# Amazon India AI Business Intelligence Dashboard

This version is designed around your existing AI knowledge:
- LLM integration
- Retrieval-Augmented Generation (RAG)
- Role-based multi-agent orchestration
- Prompt engineering

It intentionally does **not** use traditional ML algorithms such as Random Forest,
K-Means, TensorFlow, PyTorch or Scikit-learn. Pandas is used for data processing
and calculations, not as an ML model.

## Project files
- `app.py` — complete Streamlit dashboard and AI copilot.
- `requirements.txt` — required Python packages.
- `Amazon Sales Data India.xlsx` — put your supplied workbook beside `app.py`,
  or upload it from the dashboard sidebar.

## Setup on Windows
1. Install Python 3.10+.
2. Extract this ZIP into a folder.
3. Copy `Amazon Sales Data India.xlsx` into the same folder as `app.py`.
4. Open the folder in VS Code.
5. Open Terminal and run:

   `python -m venv .venv`

   `.venv\Scripts\activate`

   `pip install -r requirements.txt`

   `streamlit run app.py`

## AI copilot setup
The dashboard charts work without an API key. To use the AI copilot:
- Add your API key in the sidebar, or set the `OPENAI_API_KEY` environment variable.
- The default model field is `gpt-4o-mini`; change it if your provider uses another
  model name.
- The optional Base URL field supports providers that are compatible with the
  OpenAI Python SDK.

## How the AI workflow works
1. Intent router selects a specialist role (sales, product, orders, or channels).
2. RAG prepares business evidence from the currently filtered dataset.
3. Specialist analyst LLM answers using only retrieved evidence.
4. Executive recommendation agent turns the analysis into practical next steps.

The retrieval in this starter version is transparent keyword-based retrieval over
aggregated workbook facts, not embedding/vector retrieval. This keeps setup simple
and lets you understand the RAG flow first. Embeddings and a vector database can be
added as a later upgrade.

## Important data interpretation
The source workbook contains recorded sales for different order statuses. The app
labels this as "recorded sales" and does not automatically treat all of it as
realized revenue. The dataset does not include customer IDs, return reasons,
inventory or advertising spend, so the app does not invent those insights.
