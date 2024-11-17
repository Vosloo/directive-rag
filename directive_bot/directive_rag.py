import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()


FAISS_DIR = Path("directive_faiss")


class DirectiveRAG:
    """
    RAG system for querying the EU directive using FAISS and Groq.
    No history is stored - the system is stateless.
    """

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        embeddings_model: str = "BAAI/bge-large-en-v1.5",
        k_documents: int = 6,
    ) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._vector_store = None
        self._k_documents = k_documents

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API key must be provided either through constructor or GROQ_API_KEY environment variable!")

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.1,
        )  # type: ignore

        self._system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specialized in understanding and explaining EU directives.
Your task is to answer questions about the provided directive sections accurately and precisely.
Base your answers solely on the provided context. If the context doesn't contain enough information to answer the question fully, acknowledge this limitation.

When referencing specific parts of the directive, cite the relevant Article / Chapter / Title / Part / Annex based on the Source location provided in the context, e.g. for text starting with:
```
[Source location: PART I > CHAPTER II]

Article 3
...
```
you should reference it as Article 3 of Chapter II of Part I - do NOT provide it in a `raw` format, e.g. `PART I > CHAPTER II > Article 3`.

Rules:
- Provide accurate, precise (preferably concise) answers based on the context.
- Beside answering user's question, do not provide additional information.
- Start directly with the answer.
- Do NOT deviate from the context provided nor from the eu directive in general - you can only talk about topics related to the directive.
- Always give citations to the relevant parts (provided in the context as `Source location`) of the directive when answering questions.

Previous conversation:
{chat_history}

Context:
{context}""",  # noqa: E501
                ),
                ("human", "{question}"),
            ]
        )

        self._chain = (
            {
                "context": lambda x: self.retrieve(x["question"]),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", "No previous conversation."),
            }
            | self._system_prompt
            | self.llm
            | StrOutputParser()
        )

    def load_store(self, documents: list[Document]) -> None:
        if self._vector_store is None:
            if FAISS_DIR.exists():
                self._vector_store = FAISS.load_local(
                    str(FAISS_DIR),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                self._vector_store = FAISS.from_documents(documents, self._embeddings)
        else:
            print("Loading vector store ...")
            self._vector_store.add_documents(documents)

        print("Done.")
        self._vector_store.save_local("directive_faiss")

    def retrieve(self, query: str) -> str:
        if not self._vector_store:
            raise ValueError("No documents have been added to the vector store")

        docs = self._vector_store.similarity_search_with_score(query, k=self._k_documents)

        context_parts = []
        for doc, score in docs:
            metadata = doc.metadata
            if metadata["type"] == "chapter":
                section_info = f"{metadata['part']} > {metadata['title']} > {metadata['chapter']}"
            elif metadata["type"] == "title":
                section_info = f"{metadata['part']} > {metadata['title']}"
            elif metadata["type"] == "part":
                section_info = metadata["part"]
            elif metadata["type"] == "annex":
                section_info = metadata["annex"]
            else:
                section_info = "General Section"

            context_parts.append(f"[Source location: {section_info}]\n{doc.page_content}\n")

        return "\n\n".join(context_parts)

    def format_history(self, history: list[dict]) -> str:
        messages = []
        for msg in history:
            role = "Human" if msg["role"] == "user" else "System"
            messages.append(f"{role}: {msg['content']}")

        return "\n".join(messages)

    async def query(self, question: str, history: list[dict]) -> str:
        chat_history = self.format_history(history)

        chain_input = {
            "question": question,
            "chat_history": chat_history,
        }

        return await self._chain.ainvoke(chain_input)
