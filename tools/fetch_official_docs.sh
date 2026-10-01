#!/bin/bash
# ----------------------------------------------------------------------------
# Save the official RoboSynChallenge documentation for offline reading.
#   1) mirror of the Sphinx docs  https://edem-ai.github.io/RoboSynChallenge/html/
#      (pages + css/js/images, links rewritten to local files; the plain-text
#       page sources under docs/_sources/ (.md/.rst) are the easiest to read/grep)
#   2) the FAQ and home pages of https://robosyn-bench.net (#/faq, #/home) -- a
#      JavaScript app, so they are rendered with headless Chrome and saved as
#      HTML, PDF and plain text (the home page carries the timeline)
#
# Usage: bash tools/fetch_official_docs.sh [out_dir]
#   default out_dir: official_docs/ in the repository root
# Re-running refreshes everything (wget only re-downloads changed files).
# ----------------------------------------------------------------------------
set -euo pipefail
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/official_docs}"
DOCS_URL="https://edem-ai.github.io/RoboSynChallenge/html/"
FAQ_URL="https://robosyn-bench.net/#/faq"
HOME_URL="https://robosyn-bench.net/#/home"
mkdir -p "$OUT"
cd "$OUT"

echo "[1/2] mirroring $DOCS_URL -> $OUT/docs"
# --cut-dirs=2 drops "RoboSynChallenge/html" so the site root becomes docs/
wget --mirror --no-parent --page-requisites --convert-links --adjust-extension \
     --no-host-directories --cut-dirs=2 -e robots=off --wait=0.2 --tries=3 \
     --directory-prefix=docs "$DOCS_URL" 2>&1 | tail -3 || true

echo "[2/2] rendering $FAQ_URL -> $OUT/faq.{html,pdf,txt}"
CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
PROFILE="$(mktemp -d)"   # throw-away profile, never touches the user's browser profile
trap 'rm -rf "$PROFILE"' EXIT
CHROME_ARGS=(--headless=new --disable-gpu --no-first-run --user-data-dir="$PROFILE"
             --virtual-time-budget=20000 --run-all-compositor-stages-before-draw)
"$CHROME" "${CHROME_ARGS[@]}" --dump-dom "$FAQ_URL" > faq.html 2>/dev/null
"$CHROME" "${CHROME_ARGS[@]}" --no-pdf-header-footer --print-to-pdf="$OUT/faq.pdf" "$FAQ_URL" 2>/dev/null
# the home page carries the timeline (deadlines, finalist announcement)
"$CHROME" "${CHROME_ARGS[@]}" --dump-dom "$HOME_URL" > home.html 2>/dev/null
"$CHROME" "${CHROME_ARGS[@]}" --no-pdf-header-footer --print-to-pdf="$OUT/home.pdf" "$HOME_URL" 2>/dev/null
for page in faq home; do python3 - "$page.html" "$page.txt" <<'PYEOF'
import sys, re, html
from html.parser import HTMLParser

class Text(HTMLParser):
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr", "br", "section", "summary", "details", "dt", "dd"}
    def __init__(self):
        super().__init__(); self.out = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"): self.skip += 1
        if tag in self.BLOCK: self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"): self.out.append("#" * int(tag[1]) + " ")
        if tag == "li": self.out.append("- ")
    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"): self.skip -= 1
        if tag in self.BLOCK: self.out.append("\n")
    def handle_data(self, data):
        if not self.skip: self.out.append(data)

p = Text(); p.feed(open(sys.argv[1], encoding="utf-8").read())
text = html.unescape("".join(p.out))
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip() + "\n"
open(sys.argv[2], "w", encoding="utf-8").write(text)
PYEOF
done

{
  echo "Fetched $(date '+%Y-%m-%d %H:%M %Z') by tools/fetch_official_docs.sh"
  echo "  docs/index.html          <- $DOCS_URL"
  echo "  docs/_sources/**/*.md|.rst  plain-text (Markdown/RST) source of every docs page"
  echo "  faq.html / faq.pdf / faq.txt <- $FAQ_URL (rendered with headless Chrome)"
  echo "  home.html / home.pdf / home.txt <- $HOME_URL (timeline and key dates)"
} > README.txt
echo "done: $(find docs -name '*.html' | wc -l) html pages, $(find docs/_sources -type f 2>/dev/null | wc -l) page sources, faq.txt $(wc -l < faq.txt) lines"
