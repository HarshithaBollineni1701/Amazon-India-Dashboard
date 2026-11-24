# Amazon India AI Business Intelligence Dashboard

This version is designed around your existing AI knowledge:
- LLM integration
- Retrieval-Augmented Generation (RAG)
- Role-based multi-agent orchestration
- Prompt engineering

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
