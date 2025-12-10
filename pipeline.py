"""
HN Time Capsule Pipeline

Stages:
1. fetch    - Fetch frontpage and articles for a given date
2. prompt   - Generate LLM prompts for each article
3. analyze  - Run LLM analysis on prompts
4. parse    - Parse grades from LLM responses
5. render   - Generate HTML summary
"""

import json
import os
import re
import html
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Article:
    rank: int
    title: str
    url: str
    hn_url: str
    points: int
    author: str
    comment_count: int
    item_id: str


@dataclass
class Comment:
    id: str
    author: str
    text: str
    children: list = field(default_factory=list)

    def to_dict(self):
        return {
            'id': self.id,
            'author': self.author,
            'text': self.text,
            'children': [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d['id'],
            author=d['author'],
            text=d['text'],
            children=[cls.from_dict(c) for c in d.get('children', [])]
        )


# -----------------------------------------------------------------------------
# HTML Parsing
# -----------------------------------------------------------------------------

class HNFrontpageParser(HTMLParser):
    """Parse HN frontpage HTML to extract article listings."""

    def __init__(self):
        super().__init__()
        self.articles = []
        self.current_article = {}
        self.in_titleline = False
        self.in_title_link = False
        self.in_subline = False
        self.in_score = False
        self.in_user = False
        self.in_subline_links = False
        self.current_rank = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "span" and attrs_dict.get("class") == "rank":
            self.in_titleline = True
        if tag == "span" and attrs_dict.get("class") == "titleline":
            self.in_titleline = True
        if self.in_titleline and tag == "a" and "href" in attrs_dict:
            if not self.current_article.get("title"):
                self.current_article["url"] = attrs_dict["href"]
                self.in_title_link = True
        if tag == "span" and attrs_dict.get("class") == "subline":
            self.in_subline = True
        if self.in_subline:
            if tag == "span" and attrs_dict.get("class") == "score":
                self.in_score = True
            if tag == "a" and attrs_dict.get("class") == "hnuser":
                self.in_user = True
            if tag == "a" and "href" in attrs_dict and "item?id=" in attrs_dict["href"]:
                href = attrs_dict["href"]
                item_id = href.split("item?id=")[-1]
                self.current_article["item_id"] = item_id
                self.current_article["hn_url"] = f"https://news.ycombinator.com/{href}"
                self.in_subline_links = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self.in_title_link:
            self.current_article["title"] = data
        if self.in_score:
            try:
                self.current_article["points"] = int(data.split()[0])
            except (ValueError, IndexError):
                self.current_article["points"] = 0
        if self.in_user:
            self.current_article["author"] = data
        if self.in_subline_links:
            if "comment" in data.lower():
                try:
                    self.current_article["comment_count"] = int(data.split()[0])
                except (ValueError, IndexError):
                    self.current_article["comment_count"] = 0
            elif data.lower() == "discuss":
                self.current_article["comment_count"] = 0
        if data.endswith(".") and data[:-1].isdigit():
            self.current_rank = int(data[:-1])
            self.current_article["rank"] = self.current_rank

    def handle_endtag(self, tag):
        if tag == "a":
            self.in_title_link = False
            self.in_user = False
            self.in_subline_links = False
        if tag == "span":
            self.in_score = False
            if self.in_titleline:
                self.in_titleline = False
        if tag == "tr" and self.in_subline:
            self.in_subline = False
            if self.current_article.get("title") and self.current_article.get("item_id"):
                self.articles.append(Article(
                    rank=self.current_article.get("rank", 0),
                    title=self.current_article.get("title", ""),
                    url=self.current_article.get("url", ""),
                    hn_url=self.current_article.get("hn_url", ""),
                    points=self.current_article.get("points", 0),
                    author=self.current_article.get("author", ""),
                    comment_count=self.current_article.get("comment_count", 0),
                    item_id=self.current_article.get("item_id", ""),
                ))
            self.current_article = {}


class ArticleTextParser(HTMLParser):
    """Extract main text content from article HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe'}
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
        if tag == 'br':
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.skip_depth > 0:
            self.skip_depth -= 1
        if tag in ('p', 'div', 'article', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text_parts.append('\n\n')

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = data.strip()
        if text:
            self.text_parts.append(text + ' ')

    def get_text(self) -> str:
        text = ''.join(self.text_parts)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +\n', '\n', text)
        return text.strip()


# -----------------------------------------------------------------------------
# Fetching functions
# -----------------------------------------------------------------------------

def fetch_url(url: str, retries: int = 5, timeout: int = 15) -> str:
    """Fetch URL content with retry logic. Uses requests library to avoid TLS fingerprint blocking."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # 2, 4, 8, 16 seconds
                print(f"  Retry {attempt}/{retries-1} after {wait_time}s...")
                time.sleep(wait_time)
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403 and attempt < retries - 1:
                print(f"  Got 403, will retry...")
                continue
            raise
    raise Exception(f"Failed to fetch {url} after {retries} retries")


def fetch_frontpage(day: str) -> list[Article]:
    """Fetch HN frontpage for a specific day (YYYY-MM-DD format)."""
    url = f"https://news.ycombinator.com/front?day={day}"
    print(f"Fetching frontpage: {url}")
    page_html = fetch_url(url)
    parser = HNFrontpageParser()
    parser.feed(page_html)
    return parser.articles


def fetch_comments(item_id: str) -> list[Comment]:
    """Fetch all comments for an HN item using Algolia API."""
    url = f"https://hn.algolia.com/api/v1/items/{item_id}"
    print(f"  Fetching comments: {item_id}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    def parse_children(children) -> list[Comment]:
        comments = []
        for child in children:
            if child.get("type") != "comment" or child.get("text") is None:
                continue
            comment = Comment(
                id=str(child.get("id", "")),
                author=child.get("author") or "[deleted]",
                text=clean_html_to_text(child.get("text", "")),
                children=parse_children(child.get("children", [])),
            )
            comments.append(comment)
        return comments

    return parse_children(data.get("children", []))


MAX_ARTICLE_CHARS = 15000


def fetch_article_content(url: str) -> tuple[str, str | None]:
    """Fetch and extract text from article URL. Returns (text, error)."""
    if not url.startswith(('http://', 'https://')):
        return "", "Not a web URL"
    if any(x in url for x in ['.pdf', 'youtube.com', 'youtu.be', 'twitter.com', 'x.com']):
        return "", f"Skipped URL type"

    print(f"  Fetching article: {url[:60]}...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return "", f"Not HTML: {content_type}"
            data = response.read(5 * 1024 * 1024)
            try:
                page_html = data.decode('utf-8')
            except UnicodeDecodeError:
                page_html = data.decode('latin-1', errors='replace')

        page_html = html.unescape(page_html)
        parser = ArticleTextParser()
        parser.feed(page_html)
        text = parser.get_text()

        if len(text) < 100:
            return "", "Article too short or failed to extract"

        if len(text) > MAX_ARTICLE_CHARS:
            truncate_at = text.rfind('. ', MAX_ARTICLE_CHARS - 500, MAX_ARTICLE_CHARS)
            if truncate_at == -1:
                truncate_at = MAX_ARTICLE_CHARS
            text = text[:truncate_at + 1] + "\n\n[TRUNCATED]"

        return text, None

    except urllib.error.HTTPError as e:
        return "", f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "", f"URL error: {e.reason}"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def clean_html_to_text(text: str) -> str:
    """Convert HN comment HTML to clean text."""
    text = html.unescape(text)
    text = re.sub(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', r'[\2](\1)', text)
    text = re.sub(r'<i>([^<]+)</i>', r'*\1*', text)
    text = re.sub(r'<b>([^<]+)</b>', r'**\1**', text)
    text = re.sub(r'<code>([^<]+)</code>', r'`\1`', text)
    text = text.replace("<p>", "\n\n").replace("</p>", "")
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# -----------------------------------------------------------------------------
# Prompt generation
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """The following is an article that appeared on Hacker News 10 years ago, and the discussion thread.

Let's use our benefit of hindsight now in 6 sections:

1. Give a brief summary of the article and the discussion thread.
2. What ended up happening to this topic? (research the topic briefly and write a summary)
3. Give out awards for "Most prescient" and "Most wrong" comments, considering what happened.
4. Mention any other fun or notable aspects of the article or discussion.
5. Give out grades to specific people for their comments, considering what happened.
6. At the end, give a final score (from 0-10) for how interesting this article and its retrospect analysis was.

As for the format of Section 5, use the header "Final grades" and follow it with simply an unordered list of people and their grades in the format of "name: grade (optional comment)". Here is an example:

Final grades
- speckx: A+ (excellent predictions on ...)
- tosh: A (correctly predicted this or that ...)
- keepamovin: A
- bgwalter: D
- fsflover: F (completely wrong on ...)

Your list may contain more people of course than just this toy example. Please follow the format exactly because I will be parsing it programmatically. The idea is that I will accumulate the grades for each account to identify the accounts that were over long periods of time the most prescient or the most wrong.

As for the format of Section 6, use the prefix "Article hindsight analysis interestingness score:" and then the score (0-10) as a number. Give high scores to articles/discussions that are prominent, notable, or interesting in retrospect. Give low scores in cases where few predictions are made, or the topic is very niche or obscure, or the discussion is not very interesting in retrospect.

Here is an example:
Article hindsight analysis interestingness score: 8

---

"""


def comments_to_markdown(comments: list[Comment], indent: int = 0) -> str:
    """Convert comment tree to markdown format."""
    lines = []
    for comment in comments:
        prefix = "  " * indent
        lines.append(f"{prefix}- **{comment.author}**: {comment.text}")
        if comment.children:
            lines.append(comments_to_markdown(comment.children, indent + 1))
    return "\n\n".join(lines)


def generate_prompt(article: Article, article_text: str, article_error: str | None,
                    comments: list[Comment]) -> str:
    """Generate full LLM prompt for an article."""
    lines = [
        PROMPT_TEMPLATE,
        f"# {article.title}",
        "",
        "## Article Info",
        "",
        f"- **Original URL**: {article.url}",
        f"- **HN Discussion**: {article.hn_url}",
        f"- **Points**: {article.points}",
        f"- **Submitted by**: {article.author}",
        f"- **Comments**: {article.comment_count}",
        "",
        "## Article Content",
        "",
    ]

    if article_error:
        lines.append(f"*Could not fetch article: {article_error}*")
    else:
        lines.append(article_text)

    lines.extend([
        "",
        "## HN Discussion",
        "",
        comments_to_markdown(comments),
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Grade parsing
# -----------------------------------------------------------------------------

def parse_grades(text: str) -> dict[str, dict]:
    """Parse the Final grades section from LLM output.

    Returns dict of username -> {"grade": "A", "rationale": "explanation..."}
    """
    grades = {}
    # Match "Final grades" with optional leading section number, #, or other prefixes
    pattern = r'(?:^|\n)(?:\d+[\.\)]\s*)?(?:#+ *)?Final grades\s*\n'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return grades

    grades_section = text[match.end():]
    # Pattern: - username: GRADE (rationale text)
    # Also handle: - username (qualifier): GRADE (rationale)
    # Note: handle both ASCII +/- and Unicode minus (−)
    line_pattern = r'^[\-\*]\s*([^:]+):\s*([A-F][+\-−]?)(?:\s*\(([^)]+)\))?'

    for line in grades_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#') or line.startswith('['):
            break
        m = re.match(line_pattern, line)
        if m:
            username = m.group(1).strip()
            grade = m.group(2).strip()
            rationale = m.group(3).strip() if m.group(3) else ""
            grades[username] = {"grade": grade, "rationale": rationale}

    return grades


def grade_to_numeric(grade: str) -> float:
    """Convert letter grade to GPA."""
    base = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    if not grade:
        return 0.0
    value = base.get(grade[0].upper(), 0.0)
    if len(grade) > 1:
        if grade[1] == '+':
            value += 0.3
        elif grade[1] in '-−':  # ASCII minus and Unicode minus
            value -= 0.3
    return value


def parse_interestingness_score(text: str) -> int | None:
    """Parse the interestingness score (0-10) from LLM output."""
    pattern = r'Article hindsight analysis interestingness score:\s*(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(10, score))  # Clamp to 0-10
    return None


# -----------------------------------------------------------------------------
# Pipeline stages
# -----------------------------------------------------------------------------

def get_data_dir(target_date: str) -> Path:
    """Get the data directory for a given date."""
    return Path("data") / target_date


def get_output_dir(target_date: str | None = None) -> Path:
    """Get the output directory, optionally for a specific date."""
    base = Path("output")
    if target_date:
        return base / target_date
    return base


def stage_fetch(target_date: str, limit: int | None = None):
    """Stage 1: Fetch frontpage and all article data."""
    data_dir = get_data_dir(target_date)
    data_dir.mkdir(parents=True, exist_ok=True)

    frontpage_file = data_dir / "frontpage.json"

    # Fetch frontpage if not cached
    if frontpage_file.exists():
        print(f"Loading cached frontpage from {frontpage_file}")
        with open(frontpage_file) as f:
            articles = [Article(**a) for a in json.load(f)]
    else:
        articles = fetch_frontpage(target_date)
        with open(frontpage_file, 'w') as f:
            json.dump([asdict(a) for a in articles], f, indent=2)
        print(f"Saved frontpage to {frontpage_file}")

    if limit:
        articles = articles[:limit]

    print(f"\nFetching data for {len(articles)} articles...")

    for article in articles:
        article_dir = data_dir / article.item_id
        article_dir.mkdir(exist_ok=True)

        # Save article metadata
        meta_file = article_dir / "meta.json"
        if not meta_file.exists():
            with open(meta_file, 'w') as f:
                json.dump(asdict(article), f, indent=2)

        # Fetch article content
        article_file = article_dir / "article.txt"
        error_file = article_dir / "article_error.txt"
        if not article_file.exists() and not error_file.exists():
            text, error = fetch_article_content(article.url)
            if error:
                with open(error_file, 'w') as f:
                    f.write(error)
            else:
                with open(article_file, 'w') as f:
                    f.write(text)
            time.sleep(0.5)  # Be nice

        # Fetch comments
        comments_file = article_dir / "comments.json"
        if not comments_file.exists():
            comments = fetch_comments(article.item_id)
            with open(comments_file, 'w') as f:
                json.dump([c.to_dict() for c in comments], f, indent=2)
            time.sleep(0.2)  # Be nice

    print(f"\nFetch complete. Data saved to {data_dir}")


def stage_prompt(target_date: str):
    """Stage 2: Generate prompts for all articles."""
    data_dir = get_data_dir(target_date)

    for article_dir in sorted(data_dir.iterdir()):
        if not article_dir.is_dir():
            continue

        prompt_file = article_dir / "prompt.md"
        if prompt_file.exists():
            continue

        meta_file = article_dir / "meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            article = Article(**json.load(f))

        # Load article content
        article_file = article_dir / "article.txt"
        error_file = article_dir / "article_error.txt"
        if article_file.exists():
            article_text = article_file.read_text()
            article_error = None
        elif error_file.exists():
            article_text = ""
            article_error = error_file.read_text()
        else:
            article_text = ""
            article_error = "Not fetched"

        # Load comments
        comments_file = article_dir / "comments.json"
        if comments_file.exists():
            with open(comments_file) as f:
                comments = [Comment.from_dict(c) for c in json.load(f)]
        else:
            comments = []

        # Generate prompt
        prompt = generate_prompt(article, article_text, article_error, comments)
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        print(f"Generated prompt for {article.item_id}: {article.title[:50]}...")

    print(f"\nPrompts generated in {data_dir}")


def stage_analyze(target_date: str, model: str = "gpt-5.1", max_workers: int = 5):
    """Stage 3: Run LLM analysis on all prompts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    data_dir = get_data_dir(target_date)

    # Collect articles to analyze
    to_analyze = []
    for article_dir in sorted(data_dir.iterdir()):
        if not article_dir.is_dir():
            continue

        prompt_file = article_dir / "prompt.md"
        response_file = article_dir / "response.md"

        if not prompt_file.exists() or response_file.exists():
            continue

        meta_file = article_dir / "meta.json"
        with open(meta_file) as f:
            article = Article(**json.load(f))

        to_analyze.append((article_dir, article, prompt_file.read_text()))

    if not to_analyze:
        print("No articles to analyze.")
        return

    print(f"Analyzing {len(to_analyze)} articles with {max_workers} workers...")

    def analyze_one(item):
        article_dir, article, prompt = item
        response_file = article_dir / "response.md"
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
            )
            result = response.output_text
            with open(response_file, 'w') as f:
                f.write(result)
            return (article.item_id, article.title[:50], len(result), None)
        except Exception as e:
            return (article.item_id, article.title[:50], 0, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_one, item): item for item in to_analyze}
        for future in as_completed(futures):
            item_id, title, chars, error = future.result()
            if error:
                print(f"  {item_id}: {title}... Error: {error}")
            else:
                print(f"  {item_id}: {title}... Done ({chars} chars)")

    print(f"\nAnalysis complete. Results in {data_dir}")


def stage_parse(target_date: str):
    """Stage 4: Parse grades from all responses."""
    data_dir = get_data_dir(target_date)
    all_grades = {}  # username -> list of {grade, rationale, article}

    for article_dir in sorted(data_dir.iterdir()):
        if not article_dir.is_dir():
            continue

        response_file = article_dir / "response.md"
        grades_file = article_dir / "grades.json"
        score_file = article_dir / "score.json"

        if not response_file.exists():
            continue

        response = response_file.read_text()
        grades = parse_grades(response)  # Now returns {username: {grade, rationale}}
        score = parse_interestingness_score(response)

        with open(grades_file, 'w') as f:
            json.dump(grades, f, indent=2)

        with open(score_file, 'w') as f:
            json.dump({"interestingness": score}, f, indent=2)

        item_id = article_dir.name
        for username, grade_info in grades.items():
            if username not in all_grades:
                all_grades[username] = []
            all_grades[username].append({
                "grade": grade_info["grade"],
                "rationale": grade_info["rationale"],
                "article": item_id
            })

        score_str = f", score={score}" if score is not None else ""
        print(f"Parsed {len(grades)} grades from {item_id}{score_str}")

    # Save aggregated grades
    agg_file = data_dir / "all_grades.json"
    with open(agg_file, 'w') as f:
        json.dump(all_grades, f, indent=2)

    print(f"\nParsed grades saved to {agg_file}")

    # Print summary
    if all_grades:
        print("\n--- Grade Summary ---")
        user_gpas = []
        for username, grades_list in all_grades.items():
            gpas = [grade_to_numeric(g["grade"]) for g in grades_list]
            avg_gpa = sum(gpas) / len(gpas)
            user_gpas.append((username, avg_gpa, len(grades_list)))

        user_gpas.sort(key=lambda x: x[1], reverse=True)
        for username, gpa, count in user_gpas[:10]:
            print(f"  {username}: {gpa:.2f} ({count} articles)")


def stage_clean(target_date: str, stage: str | None = None, article_id: str | None = None):
    """Clean cached data for a date, optionally filtered by stage or article."""
    data_dir = get_data_dir(target_date)

    if not data_dir.exists():
        print(f"No data directory for {target_date}")
        return

    # Define what files each stage produces
    stage_files = {
        "fetch": ["meta.json", "article.txt", "article_error.txt", "comments.json"],
        "prompt": ["prompt.md"],
        "analyze": ["response.md"],
        "parse": ["grades.json", "score.json"],
    }

    # Stages and their downstream dependencies
    stage_order = ["fetch", "prompt", "analyze", "parse", "render"]

    # Determine which stages to clean
    if stage:
        if stage not in stage_order:
            print(f"Unknown stage: {stage}")
            return
        # Clean this stage and all downstream stages
        stage_idx = stage_order.index(stage)
        stages_to_clean = stage_order[stage_idx:-1]  # exclude render (it's just summary.html)
    else:
        stages_to_clean = stage_order[:-1]  # all except render

    files_to_delete = set()
    for s in stages_to_clean:
        files_to_delete.update(stage_files.get(s, []))

    # Get article directories to clean
    if article_id:
        article_dirs = [data_dir / article_id]
        if not article_dirs[0].exists():
            print(f"Article {article_id} not found")
            return
    else:
        article_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    # Clean files
    deleted_count = 0
    for article_dir in article_dirs:
        for filename in files_to_delete:
            filepath = article_dir / filename
            if filepath.exists():
                filepath.unlink()
                deleted_count += 1

        # If cleaning fetch, remove the entire article directory if empty
        if "fetch" in stages_to_clean:
            if article_dir.exists() and not any(article_dir.iterdir()):
                article_dir.rmdir()

    # Clean top-level files
    if not article_id:
        if "fetch" in stages_to_clean:
            frontpage = data_dir / "frontpage.json"
            if frontpage.exists():
                frontpage.unlink()
                deleted_count += 1

        if "parse" in stages_to_clean:
            all_grades = data_dir / "all_grades.json"
            if all_grades.exists():
                all_grades.unlink()
                deleted_count += 1

        # Always clean summary.html if cleaning any stage
        summary = data_dir / "summary.html"
        if summary.exists():
            summary.unlink()
            deleted_count += 1

    stage_desc = f"stage '{stage}' and downstream" if stage else "all stages"
    article_desc = f" for article {article_id}" if article_id else ""
    print(f"Cleaned {stage_desc}{article_desc}: {deleted_count} files deleted")


def get_all_output_dates() -> list[str]:
    """Get all dates that have been rendered to output directory."""
    output_base = get_output_dir()
    if not output_base.exists():
        return []
    dates = []
    for d in output_base.iterdir():
        if d.is_dir() and (d / "index.html").exists():
            dates.append(d.name)
    return sorted(dates)


def stage_render(target_date: str, update_index: bool = True):
    """Stage 5: Render HTML summary to output directory."""
    data_dir = get_data_dir(target_date)
    output_dir = get_output_dir(target_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frontpage
    frontpage_file = data_dir / "frontpage.json"
    with open(frontpage_file) as f:
        articles = [Article(**a) for a in json.load(f)]

    # Collect all article data
    articles_data = []
    for article in articles:
        article_dir = data_dir / article.item_id

        # Load response if available
        response_file = article_dir / "response.md"
        response = response_file.read_text() if response_file.exists() else ""

        # Load prompt if available
        prompt_file = article_dir / "prompt.md"
        prompt = prompt_file.read_text() if prompt_file.exists() else ""

        # Load interestingness score if available
        score_file = article_dir / "score.json"
        score = None
        if score_file.exists():
            with open(score_file) as f:
                score_data = json.load(f)
                score = score_data.get("interestingness")

        # Load grades if available
        grades_file = article_dir / "grades.json"
        grades = {}
        if grades_file.exists():
            with open(grades_file) as f:
                grades = json.load(f)

        articles_data.append({
            "article": article,
            "response": response,
            "prompt": prompt,
            "score": score,
            "grades": grades,
        })

    # Get all dates for navigation
    all_dates = get_all_output_dates()
    # Add current date if not yet in list (we're about to render it)
    if target_date not in all_dates:
        all_dates = sorted(all_dates + [target_date])
    current_idx = all_dates.index(target_date)
    prev_date = all_dates[current_idx - 1] if current_idx > 0 else None
    next_date = all_dates[current_idx + 1] if current_idx < len(all_dates) - 1 else None

    # Build HTML
    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HN Time Capsule - {target_date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 0; line-height: 1.6; height: 100vh; overflow: hidden; }}
        .container {{ display: flex; height: 100vh; }}

        /* Sidebar */
        .sidebar {{ width: 350px; min-width: 350px; background: #f5f5f5; border-right: 1px solid #ddd;
                   overflow-y: auto; padding: 15px; }}
        .sidebar h1 {{ color: #ff6600; font-size: 1.3em; margin: 0 0 5px 0; }}
        .sidebar h1 a {{ color: #ff6600; text-decoration: none; }}
        .sidebar h1 a:hover {{ text-decoration: underline; }}
        .sidebar h2 {{ font-size: 0.95em; color: #666; margin: 0 0 8px 0; font-weight: normal; }}
        .nav {{ display: flex; gap: 10px; margin-bottom: 15px; font-size: 0.85em; }}
        .nav a {{ color: #0066cc; text-decoration: none; }}
        .nav a:hover {{ text-decoration: underline; }}
        .nav .disabled {{ color: #ccc; }}
        .article-item {{ padding: 10px; margin-bottom: 8px; background: #fff; border-radius: 6px;
                        cursor: pointer; border: 2px solid transparent; transition: all 0.15s;
                        display: flex; align-items: flex-start; gap: 10px; }}
        .article-item:hover {{ border-color: #ff6600; }}
        .article-item.selected {{ border-color: #ff6600; background: #fff5f0; }}
        .article-item .score-box {{ width: 36px; height: 36px; border-radius: 6px; display: flex;
                                   align-items: center; justify-content: center; font-weight: bold;
                                   font-size: 0.85em; flex-shrink: 0; }}
        .article-item .score-box.score-10 {{ background: #c2410c; color: white; }}
        .article-item .score-box.score-9 {{ background: #ea580c; color: white; }}
        .article-item .score-box.score-8 {{ background: #f97316; color: white; }}
        .article-item .score-box.score-7 {{ background: #fb923c; color: #333; }}
        .article-item .score-box.score-6 {{ background: #fdba74; color: #333; }}
        .article-item .score-box.score-5 {{ background: #fed7aa; color: #333; }}
        .article-item .score-box.score-4 {{ background: #e5e7eb; color: #666; }}
        .article-item .score-box.score-3 {{ background: #d1d5db; color: #666; }}
        .article-item .score-box.score-2 {{ background: #9ca3af; color: white; }}
        .article-item .score-box.score-1 {{ background: #6b7280; color: white; }}
        .article-item .score-box.score-0 {{ background: #4b5563; color: white; }}
        .article-item .score-box.score-none {{ background: #eee; color: #999; font-size: 0.7em; }}
        .article-item .content {{ flex: 1; min-width: 0; }}
        .article-item .title {{ font-size: 0.9em; font-weight: 500; margin-bottom: 4px; color: #333; }}
        .article-item .meta {{ font-size: 0.75em; color: #888; }}
        .score {{ display: inline-block; padding: 2px 6px; border-radius: 10px; font-weight: bold;
                 font-size: 0.7em; margin-left: 6px; vertical-align: middle; }}
        .score.score-10 {{ background: #c2410c; color: white; }}
        .score.score-9 {{ background: #ea580c; color: white; }}
        .score.score-8 {{ background: #f97316; color: white; }}
        .score.score-7 {{ background: #fb923c; color: #333; }}
        .score.score-6 {{ background: #fdba74; color: #333; }}
        .score.score-5 {{ background: #fed7aa; color: #333; }}
        .score.score-4 {{ background: #e5e7eb; color: #666; }}
        .score.score-3 {{ background: #d1d5db; color: #666; }}
        .score.score-2 {{ background: #9ca3af; color: white; }}
        .score.score-1 {{ background: #6b7280; color: white; }}
        .score.score-0 {{ background: #4b5563; color: white; }}

        /* Main content */
        .main {{ flex: 1; overflow-y: auto; padding: 30px 40px; background: #fff; }}
        .main-inner {{ max-width: 800px; }}
        .main h1 {{ margin-top: 0; font-size: 1.5em; color: #333; }}
        .main .article-meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; padding-bottom: 15px;
                              border-bottom: 1px solid #eee; }}
        .main .article-meta a {{ color: #0066cc; }}
        .analysis {{ font-size: 0.95em; line-height: 1.5; }}
        .grades-section {{ background: #f9f9f9; padding: 15px; border-radius: 6px; margin-top: 20px; }}
        .grade {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 3px; font-size: 0.8em; }}
        .grade-a {{ background: #c6efce; color: #006100; }}
        .grade-b {{ background: #ffeb9c; color: #9c5700; }}
        .grade-c {{ background: #ffc7ce; color: #9c0006; }}
        .grade-d, .grade-f {{ background: #f4cccc; color: #990000; }}
        .prompt-section {{ margin-top: 20px; }}
        .prompt-section summary {{ cursor: pointer; color: #0066cc; font-size: 0.9em; }}
        .prompt-content {{ white-space: pre-wrap; font-size: 0.85em; background: #f5f5f5;
                          padding: 15px; border-radius: 4px; margin-top: 10px; max-height: 400px; overflow-y: auto; }}
        .placeholder {{ color: #999; text-align: center; margin-top: 100px; }}

        /* Markdown content styling */
        .analysis h1, .analysis h2, .analysis h3 {{ margin-top: 1.2em; margin-bottom: 0.4em; color: #333; }}
        .analysis h1 {{ font-size: 1.3em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        .analysis h2 {{ font-size: 1.15em; }}
        .analysis h3 {{ font-size: 1.05em; }}
        .analysis p {{ margin: 0.6em 0; }}
        .analysis ul, .analysis ol {{ margin: 0.6em 0; padding-left: 1.5em; }}
        .analysis li {{ margin: 0.25em 0; }}
        .analysis strong {{ color: #333; }}
        .analysis blockquote {{ border-left: 3px solid #ff6600; margin: 0.8em 0; padding-left: 1em; color: #555; }}
        .analysis code {{ background: #f5f5f5; padding: 0.15em 0.4em; border-radius: 3px; font-size: 0.9em; }}
        .analysis pre {{ background: #f5f5f5; padding: 0.8em; border-radius: 4px; overflow-x: auto; margin: 0.6em 0; }}
        .analysis pre code {{ background: none; padding: 0; }}
        .analysis hr {{ border: none; border-top: 1px solid #eee; margin: 1em 0; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1><a href="../index.html">HN Time Capsule</a></h1>
            <h2>{target_date} (10 years ago)</h2>
            <div class="nav">
                {f'<a href="../{prev_date}/index.html">&larr; {prev_date}</a>' if prev_date else '<span class="disabled">&larr; prev</span>'}
                <span>|</span>
                {f'<a href="../{next_date}/index.html">{next_date} &rarr;</a>' if next_date else '<span class="disabled">next &rarr;</span>'}
            </div>
"""]

    # Sidebar items
    for i, data in enumerate(articles_data):
        article = data["article"]
        score = data["score"]
        if score is not None:
            score_class = f"score-{score}"
            score_box = f'<div class="score-box {score_class}">{score}</div>'
        else:
            score_box = '<div class="score-box score-none">--</div>'

        selected = "selected" if i == 0 else ""
        html_parts.append(f"""
            <div class="article-item {selected}" id="article-{article.item_id}" onclick="selectArticle({i})">
                {score_box}
                <div class="content">
                    <div class="title">{article.rank}. {html.escape(article.title)}</div>
                    <div class="meta">{article.points} pts &middot; {article.comment_count} comments</div>
                </div>
            </div>""")

    html_parts.append("""
        </div>
        <div class="main">
            <div class="main-inner" id="main-content">
                <div class="placeholder">Select an article from the sidebar</div>
            </div>
        </div>
    </div>

    <script>
    // Configure marked for tighter output
    marked.setOptions({ breaks: false, gfm: true });

    // Preprocess markdown to remove excessive whitespace
    function cleanMarkdown(md) {
        return md
            .replace(/  +$/gm, '')
            .replace(/\\n\\n\\n+/g, '\\n\\n');
    }

    const articles = [""")

    # JavaScript data
    for i, data in enumerate(articles_data):
        article = data["article"]
        response = data["response"]
        prompt = data["prompt"]
        grades = data["grades"]
        score = data["score"]

        # Build grades HTML
        grade_html = ""
        if grades:
            # Handle both old format (username: "grade") and new format (username: {grade, rationale})
            def get_grade(item):
                val = item[1]
                if isinstance(val, dict):
                    return val.get("grade", "")
                return val
            sorted_grades = sorted(grades.items(), key=lambda x: -grade_to_numeric(get_grade(x)))
            for username, grade_info in sorted_grades[:20]:
                grade = get_grade((username, grade_info))
                if not grade:
                    continue
                grade_class = f"grade-{grade[0].lower()}"
                grade_html += f'<span class="grade {grade_class}">{html.escape(username)}: {grade}</span> '

        # Escape for JS
        title_js = json.dumps(article.title)
        url_js = json.dumps(article.url)
        hn_url_js = json.dumps(article.hn_url)
        response_js = json.dumps(html.escape(response))
        prompt_js = json.dumps(html.escape(prompt))
        grade_html_js = json.dumps(grade_html)

        html_parts.append(f"""
        {{
            id: "{article.item_id}",
            title: {title_js},
            url: {url_js},
            hn_url: {hn_url_js},
            points: {article.points},
            comments: {article.comment_count},
            score: {json.dumps(score)},
            response: {response_js},
            prompt: {prompt_js},
            grades: {grade_html_js}
        }},""")

    html_parts.append("""
    ];

    function selectArticle(idx, updateHash = true) {
        // Update sidebar selection
        document.querySelectorAll('.article-item').forEach((el, i) => {
            el.classList.toggle('selected', i === idx);
        });

        const a = articles[idx];
        const scoreHtml = a.score !== null ?
            `<span class="score score-${a.score}">${a.score}/10</span>` : '';

        document.getElementById('main-content').innerHTML = `
            <h1>${a.title}${scoreHtml}</h1>
            <div class="article-meta">
                ${a.points} points &middot; ${a.comments} comments &middot;
                <a href="${a.url}" target="_blank">Original Article</a> &middot;
                <a href="${a.hn_url}" target="_blank">HN Discussion</a>
            </div>
            ${a.grades ? `<div class="grades-section"><strong>Grades:</strong> ${a.grades}</div>` : ''}
            <div class="analysis">${a.response ? marked.parse(cleanMarkdown(a.response)) : '<em>No analysis available</em>'}</div>
            ${a.prompt ? `<details class="prompt-section"><summary>View LLM prompt</summary><div class="prompt-content">${a.prompt}</div></details>` : ''}
        `;

        // Update URL hash without scrolling
        if (updateHash) {
            history.replaceState(null, '', '#article-' + a.id);
        }
    }

    function selectArticleById(id) {
        const idx = articles.findIndex(a => a.id === id);
        if (idx >= 0) {
            selectArticle(idx, false);
            // Scroll sidebar item into view
            document.getElementById('article-' + id)?.scrollIntoView({block: 'nearest'});
        }
    }

    // Handle initial load and hash changes
    function handleHash() {
        const hash = window.location.hash;
        if (hash.startsWith('#article-')) {
            const id = hash.substring(9);  // Remove '#article-'
            selectArticleById(id);
            return true;
        }
        return false;
    }

    // On load: check for hash, otherwise select first
    if (!handleHash() && articles.length > 0) {
        selectArticle(0);
    }

    // Handle hash changes (e.g., back/forward navigation)
    window.addEventListener('hashchange', handleHash);
    </script>
</body>
</html>""")

    output_file = output_dir / "index.html"
    with open(output_file, 'w') as f:
        f.write("\n".join(html_parts))

    print(f"Rendered HTML to {output_file}")

    if update_index:
        stage_render_index()


def stage_render_index():
    """Render the main index page and re-render all day pages to update navigation."""
    output_base = get_output_dir()
    output_base.mkdir(parents=True, exist_ok=True)

    # Find all dates that have data (not just output)
    data_base = Path("data")
    all_dates = []
    if data_base.exists():
        for d in data_base.iterdir():
            if d.is_dir() and (d / "frontpage.json").exists():
                all_dates.append(d.name)
    all_dates = sorted(all_dates)

    if not all_dates:
        print("No dates to index.")
        return

    # Re-render all day pages to update prev/next navigation
    print(f"Re-rendering {len(all_dates)} day pages...")
    for d in all_dates:
        stage_render(d, update_index=False)

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HN Time Capsule</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; }
        h1 { color: #ff6600; }
        .intro { color: #666; margin-bottom: 30px; }
        .hall-of-fame-link { display: inline-block; margin-bottom: 30px; padding: 12px 24px;
                            background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white;
                            text-decoration: none; border-radius: 8px; font-weight: 600;
                            transition: transform 0.15s, box-shadow 0.15s; }
        .hall-of-fame-link:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(251, 191, 36, 0.4); }
        h2 { color: #333; margin-top: 40px; margin-bottom: 20px; font-size: 1.2em; }
        .date-list { list-style: none; padding: 0; }
        .date-list li { margin-bottom: 10px; }
        .date-list a { display: block; padding: 15px 20px; background: #f5f5f5; border-radius: 6px;
                      text-decoration: none; color: #333; transition: all 0.15s; }
        .date-list a:hover { background: #ff6600; color: white; }
        .date-list .date { font-weight: 500; }
        .date-list .desc { font-size: 0.85em; color: #888; margin-top: 3px; }
        .date-list a:hover .desc { color: rgba(255,255,255,0.8); }
    </style>
</head>
<body>
    <h1>HN Time Capsule</h1>
    <p class="intro">
        Revisiting Hacker News frontpages from 10 years ago, with the benefit of hindsight.
        Each day's articles are analyzed by an LLM to see what predictions came true and which commenters were most prescient.
    </p>
    <a href="hall-of-fame.html" class="hall-of-fame-link">Hall of Fame</a>
    <h2>Browse by Date</h2>
    <ul class="date-list">
"""

    for d in reversed(all_dates):  # Most recent first
        html += f"""        <li>
            <a href="{d}/index.html">
                <div class="date">{d}</div>
                <div class="desc">10 years ago today</div>
            </a>
        </li>
"""

    html += """    </ul>
</body>
</html>"""

    output_file = output_base / "index.html"
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Rendered index to {output_file} ({len(all_dates)} dates)")

    # Also render the Hall of Fame
    stage_render_hall_of_fame()


def stage_render_hall_of_fame():
    """Render the Hall of Fame page aggregating all user grades across all dates."""
    output_base = get_output_dir()
    data_base = Path("data")

    # Collect all grades from all dates
    all_user_grades = {}  # username -> list of {grade, rationale, article, date, article_title, hn_url}

    if not data_base.exists():
        print("No data directory found.")
        return

    for date_dir in sorted(data_base.iterdir()):
        if not date_dir.is_dir():
            continue

        frontpage_file = date_dir / "frontpage.json"
        all_grades_file = date_dir / "all_grades.json"

        if not all_grades_file.exists():
            continue

        # Load frontpage for article titles
        article_info = {}
        if frontpage_file.exists():
            with open(frontpage_file) as f:
                for article in json.load(f):
                    article_info[article["item_id"]] = {
                        "title": article["title"],
                        "hn_url": article["hn_url"]
                    }

        # Load grades
        with open(all_grades_file) as f:
            grades_data = json.load(f)

        target_date = date_dir.name
        for username, grade_list in grades_data.items():
            if username not in all_user_grades:
                all_user_grades[username] = []

            for g in grade_list:
                article_id = g["article"]
                info = article_info.get(article_id, {})
                all_user_grades[username].append({
                    "grade": g["grade"],
                    "rationale": g.get("rationale", ""),
                    "article_id": article_id,
                    "date": target_date,
                    "article_title": info.get("title", f"Article {article_id}"),
                    "hn_url": info.get("hn_url", f"https://news.ycombinator.com/item?id={article_id}")
                })

    if not all_user_grades:
        print("No grades found to render.")
        return

    # Calculate stats for each user
    user_stats = []
    for username, grades in all_user_grades.items():
        gpas = [grade_to_numeric(g["grade"]) for g in grades]
        avg_gpa = sum(gpas) / len(gpas)
        user_stats.append({
            "username": username,
            "avg_gpa": avg_gpa,
            "num_grades": len(grades),
            "grades": grades
        })

    # Filter to users with at least 2 grades
    user_stats = [u for u in user_stats if u["num_grades"] >= 2]

    # Sort by average GPA (highest first), then by number of grades
    user_stats.sort(key=lambda x: (-x["avg_gpa"], -x["num_grades"]))

    # Generate HTML
    def grade_color(grade: str) -> str:
        """Return background color for a grade with +/- variations."""
        # Color mapping: each grade has base, plus, minus variants
        colors = {
            'A+': '#15803d',  # dark green
            'A':  '#22c55e',  # green
            'A-': '#4ade80',  # light green
            'A−': '#4ade80',  # light green (unicode minus)
            'B+': '#1d4ed8',  # dark blue
            'B':  '#3b82f6',  # blue
            'B-': '#60a5fa',  # light blue
            'B−': '#60a5fa',  # light blue (unicode minus)
            'C+': '#d97706',  # dark amber
            'C':  '#f59e0b',  # amber
            'C-': '#fbbf24',  # light amber
            'C−': '#fbbf24',  # light amber (unicode minus)
            'D+': '#ea580c',  # dark orange
            'D':  '#f97316',  # orange
            'D-': '#fb923c',  # light orange
            'D−': '#fb923c',  # light orange (unicode minus)
            'F':  '#ef4444',  # red
        }
        return colors.get(grade, '#6b7280')  # gray fallback

    page_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hall of Fame - HN Time Capsule</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1000px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; }
        h1 { color: #ff6600; }
        .intro { color: #666; margin-bottom: 30px; }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #ff6600; text-decoration: none; }
        .back-link a:hover { text-decoration: underline; }

        .user-card { background: #f9fafb; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                    border: 1px solid #e5e7eb; }
        .user-header { display: flex; align-items: center; gap: 15px; margin-bottom: 15px; }
        .user-name { font-size: 1.25em; font-weight: 600; }
        .user-name a { color: #333; text-decoration: none; }
        .user-name a:hover { color: #ff6600; }
        .user-stats { color: #666; font-size: 0.9em; }
        .avg-gpa { background: #ff6600; color: white; padding: 4px 10px; border-radius: 4px;
                  font-weight: 600; font-size: 0.9em; }

        .grades-list { display: flex; flex-direction: column; gap: 10px; }
        .grade-item { display: flex; align-items: flex-start; gap: 12px; padding: 10px;
                     background: white; border-radius: 6px; border: 1px solid #e5e7eb; }
        .grade-badge { min-width: 36px; height: 36px; display: flex; align-items: center;
                      justify-content: center; border-radius: 6px; color: white;
                      font-weight: 700; font-size: 0.85em; }
        .grade-content { flex: 1; }
        .grade-article { font-weight: 500; margin-bottom: 4px; }
        .grade-article a { color: #333; text-decoration: none; }
        .grade-article a:hover { color: #ff6600; }
        .grade-rationale { color: #666; font-size: 0.9em; font-style: italic; }
        .grade-meta { font-size: 0.8em; color: #999; margin-top: 4px; }
        .grade-meta a { color: #ff6600; text-decoration: none; }
        .grade-meta a:hover { text-decoration: underline; }

        .rank { font-size: 1.5em; font-weight: 700; color: #d1d5db; min-width: 40px; }
        .rank.gold { color: #fbbf24; }
        .rank.silver { color: #9ca3af; }
        .rank.bronze { color: #d97706; }
    </style>
</head>
<body>
    <div class="back-link"><a href="index.html">&larr; Back to dates</a></div>
    <h1>Hall of Fame</h1>
    <p class="intro">
        The most prescient Hacker News commenters, ranked by their average grade across all analyzed threads.
        Grades are assigned by an LLM evaluating how well each comment predicted the future with 10 years of hindsight.
    </p>
"""

    for i, user in enumerate(user_stats):
        rank = i + 1
        rank_class = ""
        if rank == 1:
            rank_class = "gold"
        elif rank == 2:
            rank_class = "silver"
        elif rank == 3:
            rank_class = "bronze"

        # Format GPA as letter grade equivalent
        gpa = user["avg_gpa"]
        if gpa >= 3.85:
            gpa_letter = "A"
        elif gpa >= 3.5:
            gpa_letter = "A-"
        elif gpa >= 3.15:
            gpa_letter = "B+"
        elif gpa >= 2.85:
            gpa_letter = "B"
        elif gpa >= 2.5:
            gpa_letter = "B-"
        elif gpa >= 2.15:
            gpa_letter = "C+"
        elif gpa >= 1.85:
            gpa_letter = "C"
        elif gpa >= 1.5:
            gpa_letter = "C-"
        elif gpa >= 1.15:
            gpa_letter = "D+"
        elif gpa >= 0.85:
            gpa_letter = "D"
        else:
            gpa_letter = "F"

        hn_user_url = f"https://news.ycombinator.com/user?id={html.escape(user['username'])}"

        page_html += f"""
    <div class="user-card">
        <div class="user-header">
            <div class="rank {rank_class}">#{rank}</div>
            <div class="user-name"><a href="{hn_user_url}" target="_blank">{html.escape(user['username'])}</a></div>
            <div class="avg-gpa">{gpa_letter} ({gpa:.2f})</div>
            <div class="user-stats">{user['num_grades']} grade{"s" if user['num_grades'] > 1 else ""}</div>
        </div>
        <div class="grades-list">
"""

        # Sort grades by grade (best first)
        sorted_grades = sorted(user["grades"], key=lambda g: -grade_to_numeric(g["grade"]))

        for g in sorted_grades:
            color = grade_color(g["grade"])
            rationale_part = f'<div class="grade-rationale">"{html.escape(g["rationale"])}"</div>' if g["rationale"] else ""
            analysis_url = f"{g['date']}/index.html#article-{g['article_id']}"

            page_html += f"""            <div class="grade-item">
                <div class="grade-badge" style="background: {color}">{g['grade']}</div>
                <div class="grade-content">
                    <div class="grade-article"><a href="{analysis_url}">{html.escape(g['article_title'])}</a></div>
                    {rationale_part}
                    <div class="grade-meta">
                        <a href="{analysis_url}">View analysis</a> &middot;
                        <a href="{g['hn_url']}" target="_blank">HN thread</a> &middot; {g['date']}
                    </div>
                </div>
            </div>
"""

        page_html += """        </div>
    </div>
"""

    page_html += """</body>
</html>"""

    output_file = output_base / "hall-of-fame.html"
    with open(output_file, 'w') as f:
        f.write(page_html)

    print(f"Rendered Hall of Fame to {output_file} ({len(user_stats)} users)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="HN Time Capsule Pipeline")
    parser.add_argument("stage", choices=["fetch", "prompt", "analyze", "parse", "render", "render-index", "all", "clean"],
                        help="Pipeline stage to run")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD), defaults to 10 years ago")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles (for testing)")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model for analysis")
    parser.add_argument("--workers", type=int, default=15, help="Number of parallel workers for analysis")
    parser.add_argument("--clean-stage", choices=["fetch", "prompt", "analyze", "parse"],
                        help="For clean: only clean this stage and downstream (default: all)")
    parser.add_argument("--article", help="For clean: only clean specific article by item_id")

    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        today = date.today()
        target_date = today.replace(year=today.year - 10).isoformat()

    print(f"Target date: {target_date}\n")

    if args.stage == "clean":
        stage_clean(target_date, args.clean_stage, args.article)
    elif args.stage == "render-index":
        stage_render_index()
    else:
        if args.stage == "fetch" or args.stage == "all":
            stage_fetch(target_date, args.limit)
        if args.stage == "prompt" or args.stage == "all":
            stage_prompt(target_date)
        if args.stage == "analyze" or args.stage == "all":
            stage_analyze(target_date, args.model, args.workers)
        if args.stage == "parse" or args.stage == "all":
            stage_parse(target_date)
        if args.stage == "render" or args.stage == "all":
            stage_render(target_date)  # This also calls stage_render_index()


if __name__ == "__main__":
    main()
