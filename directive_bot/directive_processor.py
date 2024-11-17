from pathlib import Path

import regex as re
from bs4 import BeautifulSoup, Tag

KNOWLEDGEBASE_DIR = "knowledgebase"
DIRECTIVE_FILE = "directive.html"


class DirectiveProcessor:
    def __init__(self) -> None:
        self._directive_filepath = Path(__file__).parent / KNOWLEDGEBASE_DIR / DIRECTIVE_FILE
        self._ensure_directive()

    def process_table(self, table: Tag) -> str:
        result = []

        for tr in table.find_all("tr"):
            row_cells = []
            for cell in tr.find_all(["td", "th"]):
                cell_text = " ".join(cell.get_text(strip=False).split())
                row_cells.append(cell_text)

            if row_cells:
                result.append(" ".join(row_cells))

        if result:
            return "\n".join(result) + "\n"

        return ""

    def process_element(self, element: Tag) -> str:
        if element.name == "table":
            return self.process_table(element)

        if element.name == "p":
            text = " ".join(element.get_text(strip=False).split())

            return f"\n{text.strip()}\n"

        text_parts = []
        for child in element.children:
            if isinstance(child, Tag):
                processed = self.process_element(child)
                if processed:
                    text_parts.append(processed)

            elif isinstance(child, str) and child.strip():
                text_parts.append(child.strip())

        return " ".join(text_parts)

    def apply_processed_transforms(self, text: str) -> str:
        text = self._remove_special_chars(text)
        text = self._apply_enumerate_spacing(text)

        return text.strip()

    def clean_text(self) -> str:
        with open(self._directive_filepath, "r") as f:
            soup = BeautifulSoup(f, "html.parser")

        for element in soup(["script", "style"]):
            element.decompose()

        main_contents = soup.find_all("div", class_="eli-container")

        processed_text = ""
        for main_content in main_contents:
            processed_text += "\n" + self.apply_processed_transforms(self.process_element(main_content))

        lines = []
        for line in processed_text.split("\n"):
            if not (line := line.strip()):
                continue

            lines.append(self._apply_headings_spacing(line))

        text = "\n".join(lines)

        return text

    def _remove_special_chars(self, text: str) -> str:
        text = re.sub(r"[‘’]", '"', text)
        return re.sub(r"\t", " ", text)

    def _apply_enumerate_spacing(self, text: str) -> str:
        text = re.sub(r"^(\(\w+\))(\w)", r"\1 \2", text, flags=re.MULTILINE)
        return re.sub(r"^([\d\.]+)\n\s\n?(\w)", r"\1 \2", text, flags=re.MULTILINE)

    def _apply_headings_spacing(self, text: str) -> str:
        if "Article" in text:
            prefix = ""
        elif "CHAPTER" in text:
            prefix = "### "
        elif "TITLE" in text:
            prefix = "## "
        elif "PART" in text:
            prefix = "# "
        elif "ANNEX" in text:
            prefix = "# "
        else:
            # No heading
            return text

        return re.sub(r"^(Article|CHAPTER|TITLE|PART|ANNEX) (\w+)$", rf"\n{prefix}\1 \2", text)

    def _ensure_directive(self) -> None:
        if self._directive_filepath.exists():
            return

        raise FileNotFoundError(f"Directive file not found: {self._directive_filepath}!")
