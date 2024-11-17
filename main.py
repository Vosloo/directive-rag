import gradio as gr

from directive_bot import DirectiveProcessor, DirectiveRAG, DocumentSplitter


def main():
    processor = DirectiveProcessor()
    chunker = DocumentSplitter()
    rag = DirectiveRAG()

    text = processor.clean_text()
    chunks = chunker.create_chunks(text)

    rag.load_store(chunks)

    iface = gr.ChatInterface(rag.query, type="messages")
    iface.launch(show_error=True)


if __name__ == "__main__":
    main()
