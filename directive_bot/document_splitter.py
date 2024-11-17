from dataclasses import dataclass

import regex as re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Article:
    """Represents an article with its content."""

    identifier: str
    description: str
    content: str


class DocumentSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200) -> None:
        self._patterns = {
            "part": re.compile(r"^# (PART \w+)\n(.*)$", re.MULTILINE),
            "title": re.compile(r"^## (TITLE \w+)\n(.*)$", re.MULTILINE),
            "chapter": re.compile(r"^### (CHAPTER \w+)\n(.*)$", re.MULTILINE),
            "article": re.compile(r"^(Article \w+)\n(.*)$", re.MULTILINE),
            "annex": re.compile(r"^# (ANNEX \w+)\n(.*)$", re.MULTILINE),
        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _process_section(
        self,
        ind: int,
        curr_match: re.Match,
        curr_matches: list[re.Match],
        parent_text: str,
        match_section: str,
        futher_matches: list[re.Match] | None = None,  # Matches for Annexes - only needed for the last PART
    ) -> tuple[list[re.Match], str]:
        """Process a section and get its subsections."""
        next_match_start = None
        if ind + 1 != len(curr_matches):
            next_match_start = curr_matches[ind + 1].start()
        elif futher_matches is not None:
            # Last PART - get the start of the next section
            next_match_start = futher_matches[0].start()

        current_text = parent_text[curr_match.start() : next_match_start]
        subsection_matches = list(self._patterns[match_section].finditer(current_text))

        return subsection_matches, current_text

    def _split_article_content(self, article: Article, metadata: dict) -> list[Document]:
        if len(article.content) <= self.text_splitter._chunk_size:
            return [
                Document(
                    page_content=article.content,
                    metadata={
                        **metadata,
                        "article": article.identifier,
                        "article_description": article.description,
                        "chunk_type": "article_full",
                    },
                )
            ]

        chunks = self.text_splitter.create_documents(
            texts=[article.content],
            metadatas=[
                {
                    **metadata,
                    "article": article.identifier,
                    "article_description": article.description,
                    "chunk_type": "article_split",
                }
            ],
        )

        # Add article identifier to chunk content for context
        for i, chunk in enumerate(chunks):
            if i != 0:
                prefix = f"{article.identifier}\n{article.description}\n"
                chunk.page_content = prefix + chunk.page_content

            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _process_articles(self, matches: list[re.Match], text: str, metadata: dict) -> list[Document]:
        """Process articles within a section and split into smaller chunks."""
        chunks = []
        for a_i, curr_article in enumerate(matches):
            next_article_start = None
            if a_i + 1 != len(matches):
                next_article_start = matches[a_i + 1].start()

            article_text = text[curr_article.start() : next_article_start]
            article = Article(
                identifier=curr_article.group(1),
                description=curr_article.group(2) or "UNDEFINED",
                content=article_text,
            )

            article_chunks = self._split_article_content(article, metadata)
            chunks.extend(article_chunks)

        return chunks

    def create_chunks(self, text: str) -> list[Document]:
        """Create chunks based on document structure and size limits."""
        chunks = []
        part_matches = list(self._patterns["part"].finditer(text))
        annex_matches = list(self._patterns["annex"].finditer(text))

        # Process parts
        for p_i, curr_part in enumerate(part_matches):
            part_metadata = {
                "type": "part",
                "part": curr_part.group(1),
                "part_description": curr_part.group(2) or "UNDEFINED",
            }

            title_matches, part_text = self._process_section(p_i, curr_part, part_matches, text, "title")

            if title_matches:
                for t_i, curr_title in enumerate(title_matches):
                    title_metadata = {
                        **part_metadata,
                        "title": curr_title.group(1),
                        "title_description": curr_title.group(2) or "UNDEFINED",
                    }

                    chapter_matches, title_text = self._process_section(t_i, curr_title, title_matches, part_text, "chapter")

                    if chapter_matches:
                        for c_i, curr_chapter in enumerate(chapter_matches):
                            chapter_metadata = {
                                **title_metadata,
                                "type": "chapter",
                                "chapter": curr_chapter.group(1),
                                "chapter_description": curr_chapter.group(2) or "UNDEFINED",
                            }

                            article_matches, chapter_text = self._process_section(
                                c_i, curr_chapter, chapter_matches, title_text, "article"
                            )
                            chunks.extend(self._process_articles(article_matches, chapter_text, chapter_metadata))
                    else:
                        article_matches, title_text = self._process_section(
                            t_i, curr_title, title_matches, part_text, "article"
                        )
                        chunks.extend(self._process_articles(article_matches, title_text, {**title_metadata, "type": "title"}))
            else:
                article_matches, part_text = self._process_section(
                    p_i,
                    curr_part,
                    part_matches,
                    text,
                    "article",
                    futher_matches=annex_matches,
                )
                chunks.extend(self._process_articles(article_matches, part_text, part_metadata))

        for a_i, curr_annex in enumerate(annex_matches):
            if a_i + 1 < len(annex_matches):
                next_annex_start = annex_matches[a_i + 1].start()

                annex_text = text[curr_annex.start() : next_annex_start]

                annex_chunks = self.text_splitter.create_documents(
                    texts=[annex_text],
                    metadatas=[
                        {
                            "type": "annex",
                            "annex": curr_annex.group(1),
                            "annex_description": curr_annex.group(2) or "UNDEFINED",
                        }
                    ],
                )

                for i, chunk in enumerate(annex_chunks):
                    if i != 0:
                        prefix = f"{curr_annex.group(1)}\n{curr_annex.group(2) or 'UNDEFINED'}\n"
                        chunk.page_content = prefix + chunk.page_content

                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["total_chunks"] = len(annex_chunks)

                chunks.extend(annex_chunks)

            else:
                # Last annex - correlation table that needs to be removed
                annex_text = "Correlation table - not included"
                chunks.append(
                    Document(
                        page_content=annex_text,
                        metadata={
                            "type": "annex",
                            "annex": curr_annex.group(1),
                            "annex_description": curr_annex.group(2) or "UNDEFINED",
                        },
                    )
                )

        return chunks
