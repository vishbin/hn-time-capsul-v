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

def fetch_url(url: str, retries: int = 3, timeout: int = 15) -> str:
    """Fetch URL content with retry logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 403 and attempt < retries - 1:
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

Let's use our benefit of hindsight now:

1. What ended up happening to this topic? (research the topic briefly and write a summary)
2. Give out awards for "Most prescient" and "Most wrong" comments, considering what happened.
3. Mention any other fun or notable aspects of the article or discussion.
4. Give out grades to specific people for their comments, considering what happened.
5. At the end, give a final score (from 0-10) for how interesting this article and its retrospect analysis was.

As for the format of (4), use the header "Final grades" and follow it with simply an unordered list of people and their grades in the format of "name: grade (optional comment)". Here is an example:

Final grades
- speckx: A+ (excellent predictions on ...)
- tosh: A (correctly predicted this or that ...)
- keepamovin: A
- bgwalter: D
- fsflover: F (completely wrong on ...)

Your list may contain more people of course than just this toy example. Please follow the format exactly because I will be parsing it programmatically. The idea is that I will accumulate the grades for each account to identify the accounts that were over long periods of time the most prescient or the most wrong.

As for the format of (5), use the prefix "Article hindsight analysis interestingness score:" and then the score (0-10) as a number. Here is an example:
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

def parse_grades(text: str) -> dict[str, str]:
    """Parse the Final grades section from LLM output."""
    grades = {}
    pattern = r'(?:^|\n)(?:#+ *)?Final grades\s*\n'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return grades

    grades_section = text[match.end():]
    line_pattern = r'^[\-\*]\s*([^:]+):\s*([A-F][+-]?)'

    for line in grades_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#') or line.startswith('['):
            break
        m = re.match(line_pattern, line)
        if m:
            grades[m.group(1).strip()] = m.group(2).strip()

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
        elif grade[1] == '-':
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
    all_grades = {}  # username -> list of (grade, article_id)

    for article_dir in sorted(data_dir.iterdir()):
        if not article_dir.is_dir():
            continue

        response_file = article_dir / "response.md"
        grades_file = article_dir / "grades.json"
        score_file = article_dir / "score.json"

        if not response_file.exists():
            continue

        response = response_file.read_text()
        grades = parse_grades(response)
        score = parse_interestingness_score(response)

        with open(grades_file, 'w') as f:
            json.dump(grades, f, indent=2)

        with open(score_file, 'w') as f:
            json.dump({"interestingness": score}, f, indent=2)

        item_id = article_dir.name
        for username, grade in grades.items():
            if username not in all_grades:
                all_grades[username] = []
            all_grades[username].append({"grade": grade, "article": item_id})

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


def stage_render(target_date: str):
    """Stage 5: Render HTML summary."""
    data_dir = get_data_dir(target_date)

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
        .sidebar h2 {{ font-size: 0.95em; color: #666; margin: 0 0 15px 0; font-weight: normal; }}
        .article-item {{ padding: 10px; margin-bottom: 8px; background: #fff; border-radius: 6px;
                        cursor: pointer; border: 2px solid transparent; transition: all 0.15s;
                        display: flex; align-items: flex-start; gap: 10px; }}
        .article-item:hover {{ border-color: #ff6600; }}
        .article-item.selected {{ border-color: #ff6600; background: #fff5f0; }}
        .article-item .score-box {{ width: 36px; height: 36px; border-radius: 6px; display: flex;
                                   align-items: center; justify-content: center; font-weight: bold;
                                   font-size: 0.85em; flex-shrink: 0; }}
        .article-item .score-box.score-high {{ background: #ff6600; color: white; }}
        .article-item .score-box.score-medium {{ background: #ffcc00; color: #333; }}
        .article-item .score-box.score-low {{ background: #ddd; color: #666; }}
        .article-item .score-box.score-none {{ background: #eee; color: #999; font-size: 0.7em; }}
        .article-item .content {{ flex: 1; min-width: 0; }}
        .article-item .title {{ font-size: 0.9em; font-weight: 500; margin-bottom: 4px; color: #333; }}
        .article-item .meta {{ font-size: 0.75em; color: #888; }}
        .score {{ display: inline-block; padding: 2px 6px; border-radius: 10px; font-weight: bold;
                 font-size: 0.7em; margin-left: 6px; vertical-align: middle; }}
        .score-high {{ background: #ff6600; color: white; }}
        .score-medium {{ background: #ffcc00; color: #333; }}
        .score-low {{ background: #ddd; color: #666; }}

        /* Main content */
        .main {{ flex: 1; overflow-y: auto; padding: 30px 40px; background: #fff; }}
        .main-inner {{ max-width: 800px; }}
        .main h1 {{ margin-top: 0; font-size: 1.5em; color: #333; }}
        .main .article-meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; padding-bottom: 15px;
                              border-bottom: 1px solid #eee; }}
        .main .article-meta a {{ color: #0066cc; }}
        .analysis {{ font-size: 0.95em; white-space: pre-wrap; line-height: 1.7; }}
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
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>HN Time Capsule</h1>
            <h2>{target_date} (10 years ago)</h2>
"""]

    # Sidebar items
    for i, data in enumerate(articles_data):
        article = data["article"]
        score = data["score"]
        if score is not None:
            score_class = "score-high" if score >= 7 else "score-medium" if score >= 4 else "score-low"
            score_box = f'<div class="score-box {score_class}">{score}</div>'
        else:
            score_box = '<div class="score-box score-none">--</div>'

        selected = "selected" if i == 0 else ""
        html_parts.append(f"""
            <div class="article-item {selected}" onclick="selectArticle({i})">
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
            sorted_grades = sorted(grades.items(), key=lambda x: -grade_to_numeric(x[1]))
            for username, grade in sorted_grades[:20]:
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

    function selectArticle(idx) {
        // Update sidebar selection
        document.querySelectorAll('.article-item').forEach((el, i) => {
            el.classList.toggle('selected', i === idx);
        });

        const a = articles[idx];
        const scoreHtml = a.score !== null ?
            `<span class="score score-${a.score >= 7 ? 'high' : a.score >= 4 ? 'medium' : 'low'}">${a.score}/10</span>` : '';

        document.getElementById('main-content').innerHTML = `
            <h1>${a.title}${scoreHtml}</h1>
            <div class="article-meta">
                ${a.points} points &middot; ${a.comments} comments &middot;
                <a href="${a.url}" target="_blank">Original Article</a> &middot;
                <a href="${a.hn_url}" target="_blank">HN Discussion</a>
            </div>
            ${a.grades ? `<div class="grades-section"><strong>Grades:</strong> ${a.grades}</div>` : ''}
            <div class="analysis">${a.response || '<em>No analysis available</em>'}</div>
            ${a.prompt ? `<details class="prompt-section"><summary>View LLM prompt</summary><div class="prompt-content">${a.prompt}</div></details>` : ''}
        `;
    }

    // Select first article on load
    if (articles.length > 0) selectArticle(0);
    </script>
</body>
</html>""")

    output_file = data_dir / "summary.html"
    with open(output_file, 'w') as f:
        f.write("\n".join(html_parts))

    print(f"Rendered HTML to {output_file}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="HN Time Capsule Pipeline")
    parser.add_argument("stage", choices=["fetch", "prompt", "analyze", "parse", "render", "all", "clean"],
                        help="Pipeline stage to run")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD), defaults to 10 years ago")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles (for testing)")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model for analysis")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers for analysis")
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
    elif args.stage == "fetch" or args.stage == "all":
        stage_fetch(target_date, args.limit)
    if args.stage == "prompt" or args.stage == "all":
        stage_prompt(target_date)
    if args.stage == "analyze" or args.stage == "all":
        stage_analyze(target_date, args.model, args.workers)
    if args.stage == "parse" or args.stage == "all":
        stage_parse(target_date)
    if args.stage == "render" or args.stage == "all":
        stage_render(target_date)


if __name__ == "__main__":
    main()
