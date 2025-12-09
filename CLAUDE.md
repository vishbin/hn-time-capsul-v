# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HN Time Capsule pulls the Hacker News frontpage from exactly 10 years ago, fetches article content and comments, runs LLM analysis to evaluate prescience with hindsight, and generates HTML reports.

## Commands

```bash
# Run the full pipeline (fetch → prompt → analyze → parse → render)
uv run python pipeline.py all

# Run with article limit for testing
uv run python pipeline.py all --limit 5

# Run individual stages
uv run python pipeline.py fetch
uv run python pipeline.py prompt
uv run python pipeline.py analyze --model gpt-5-mini  # cheaper for testing
uv run python pipeline.py parse
uv run python pipeline.py render

# Specific date
uv run python pipeline.py fetch --date 2015-06-15
```

## Architecture

### Main Files
- `pipeline.py` - The main pipeline with 5 stages. This is the primary entry point.
- `parse_grades.py` - Standalone grade parser (can test with `uv run python parse_grades.py output.txt`)

### Pipeline Stages (in pipeline.py)
1. **fetch** (`stage_fetch`) - Fetches HN frontpage, article content, and comments. Caches everything.
2. **prompt** (`stage_prompt`) - Generates LLM prompts from fetched data.
3. **analyze** (`stage_analyze`) - Calls OpenAI API to analyze each article. Uses `client.responses.create()` API.
4. **parse** (`stage_parse`) - Extracts grades from LLM responses using regex.
5. **render** (`stage_render`) - Generates summary.html with all results.

### Data Structure
```
data/{date}/
  frontpage.json         # Article list from HN
  all_grades.json        # Aggregated grades
  summary.html           # Final HTML output
  {item_id}/
    meta.json            # Article metadata
    article.txt          # Fetched article (or article_error.txt)
    comments.json        # Comment tree from Algolia API
    prompt.md            # Full LLM prompt
    response.md          # LLM response
    grades.json          # Parsed grades
```

### Key Implementation Details

**Fetching:**
- HN frontpage: HTML scraping (no API for historical frontpages)
- Comments: Algolia API `https://hn.algolia.com/api/v1/items/{id}` - single request gets full tree
- Articles: Direct fetch with fallback handling for 404s, paywalls, non-HTML

**OpenAI API (important!):**
```python
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-5.1",  # or gpt-5-mini for testing
    input=prompt,
    reasoning={"effort": "medium"},
    text={"verbosity": "medium"},
)
result = response.output_text
```

**Grade Parsing:**
- Looks for "Final grades" section header
- Parses lines like `- username: A+`
- Converts to GPA (A+=4.3, A=4.0, B+=3.3, etc.)

### Environment
- Python 3.10+ with uv
- Dependencies: openai, python-dotenv
- API key in `.env` file: `OPENAI_API_KEY=...`

## Current State / TODO

**Working:**
- Full pipeline runs end-to-end
- Caching at each stage (won't re-fetch or re-analyze existing data)
- HTML rendering with expandable analysis sections
- Grade aggregation across articles

**Potential improvements:**
- Aggregate grades across multiple days
- Better HTML styling
- Handle very long comment threads (truncation?)
- Archive.org fallback for dead article links
- Rate limiting / retry logic for OpenAI API
- Progress bars for long runs
