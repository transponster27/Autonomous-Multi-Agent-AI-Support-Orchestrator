# src/generation/context_builder.py

class ContextBuilder:

    @staticmethod
    def build(retrieved_docs):
        context = ""
        
        for i, doc in enumerate(retrieved_docs, 1):
            doc_name = doc.get("document_name", "Unknown Document")
            page = doc.get("page_number", "N/A")
            text = doc.get("chunk_text", "")
            
            # ENHANCEMENT: Give the LLM clear boundaries and source info
            # This helps the LLM understand where the text came from!
            context += f"--- Source {i}: {doc_name} (Page {page}) ---\n{text}\n\n"
            
        return context