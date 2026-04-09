# Due Diligence News Agent

An agentic pipeline that extracts structured business events from trade news articles using Claude. Built for due diligence research workflows to ensure data extraction is optimised, accurate and consistent.

## What it does

The agent takes a list of article URLs, fetches each article, and uses Claude to extract key business events - company actions, market developments, and regulatory updates — in a structured, validated format. Results are written to a consolidated JSON report.

Each extraction is validated against a schema before being accepted. If validation fails, the agent retries with a refined prompt. This ensures output quality without manual review of every article.

The pipeline runs autonomously via GitHub Actions, making it suitable for overnight or scheduled research runs.

## Architecture
urls.json → agent.py → tools.py (fetch article) → Claude API (extract events) → schema.py (validate output) → report.json

- **agent.py** - core agent loop: iterates over URLs, manages tool calls, handles retries, writes output
- **tools.py** — fetch article tool: retrieves and cleans article text for Claude to process
- **schema.py** — defines the event schema and validates Claude's structured output before it is accepted

## Setup

```bash
git clone https://github.com/your-username/due-diligence-agent.git
cd due-diligence-agent
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python src/agent.py
```

To run via GitHub Actions, add ANTHROPIC_API_KEY as a repository secret and trigger the run_agent workflow manually or on push.

## Sample output

Each extracted event follows this structure:

```json
{
  "url": "https://example.com/article",
  "events": [
    {
      "topic": "NEW_BUILDS",
      "company": "Oatly",
      "description": "Oatly is building a new plant in Fort Worth to process approximately 40 million gallons of oat milk per year.",
      "location": "Fort Worth, United States",
      "scale": "40 million gallons per year capacity, 275,000 square foot facility"
    }
  ],
  "validated": true,
  "retries": 0
}
```

## Why this exists
Built to accelerate due diligence research at a finance firm where analysts want to evaluate hundreds of trade news articles for topics, trends and sentiment across time. Manual extraction took hours. This agent processes a batch of 50 articles in under 10 minutes with consistent structured output.
