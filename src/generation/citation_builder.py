# src/generation/citation_builder.py

class CitationBuilder:

    @staticmethod
    def build(retrieved_docs):
        citations = []
        seen = set() # Prevent duplicate citations for the same page
        
        for doc in retrieved_docs:
            doc_name = doc.get("document_name", "Unknown")
            
            # Note: MetadataExtractor uses "page_number", so we use that key here.
            page = doc.get("page_number", "N/A") 
            
            # Create a unique key to avoid duplicates
            key = f"{doc_name}_{page}"
            if key not in seen:
                citations.append({
                    "document": doc_name,
                    "page": page 
                })
                seen.add(key)
                
        return citations