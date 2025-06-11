import os
import json
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle

# Constants
KNOWLEDGE_BASE_DIR = "../data/knowledge_base"
INDEX_PATH = "../data/vector_store.faiss"
DOC_CHUNKS_PATH = "../data/doc_chunks.pkl"

def load_config():
    """Loads environment variables and configures API keys."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)

def get_embedding(text, model="models/embedding-001"):
    """Generates embedding for a given text using Gemini.
    Retries on failure.
    """
    try:
        result = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:50]}...': {e}")
        # Simple retry or could implement exponential backoff
        try:
            print("Retrying embedding generation...")
            result = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
            return result['embedding']
        except Exception as e_retry:
            print(f"Retry failed for embedding generation: {e_retry}")
            return None

def load_markdown_documents(directory):
    """Loads all markdown documents from a specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({'name': filename, 'content': content})
    return documents

def split_documents(documents):
    """Splits documents into manageable chunks."""
    all_chunks = []
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Fallback for text that isn't split by headers or is too long
    chunk_size = 1000 # characters
    chunk_overlap = 150 # characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for doc in documents:
        md_header_splits = markdown_splitter.split_text(doc['content'])
        
        for split in md_header_splits:
            # If a header split is still too large, use the recursive character splitter
            if len(split.page_content) > chunk_size * 1.5: # Heuristic for further splitting
                sub_chunks = text_splitter.split_text(split.page_content)
                for i, sub_chunk_text in enumerate(sub_chunks):
                    chunk_metadata = split.metadata.copy()
                    chunk_metadata['doc_name'] = doc['name']
                    # Create a unique ID for the chunk
                    chunk_id = hashlib.md5(f"{doc['name']}_{chunk_metadata.get('Header 1', '')}_{chunk_metadata.get('Header 2', '')}_{chunk_metadata.get('Header 3', '')}_{i}".encode()).hexdigest()
                    all_chunks.append({
                        'id': chunk_id,
                        'text': sub_chunk_text,
                        'metadata': chunk_metadata
                    })
            else:
                chunk_metadata = split.metadata.copy()
                chunk_metadata['doc_name'] = doc['name']
                chunk_id = hashlib.md5(f"{doc['name']}_{chunk_metadata.get('Header 1', '')}_{chunk_metadata.get('Header 2', '')}_{chunk_metadata.get('Header 3', '')}".encode()).hexdigest()
                all_chunks.append({
                    'id': chunk_id,
                    'text': split.page_content,
                    'metadata': chunk_metadata
                })
        
        # Handle documents that might not have headers or very short content
        if not md_header_splits:
            sub_chunks = text_splitter.split_text(doc['content'])
            for i, sub_chunk_text in enumerate(sub_chunks):
                chunk_metadata = {'doc_name': doc['name']}
                chunk_id = hashlib.md5(f"{doc['name']}_part_{i}".encode()).hexdigest()
                all_chunks.append({
                    'id': chunk_id,
                    'text': sub_chunk_text,
                    'metadata': chunk_metadata
                })

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
    return all_chunks

def build_vector_store(chunks_with_embeddings):
    """Builds a FAISS vector store from chunks and their embeddings."""
    if not chunks_with_embeddings:
        print("No embeddings to build vector store.")
        return None

    embeddings = np.array([item['embedding'] for item in chunks_with_embeddings if item['embedding'] is not None])
    if embeddings.size == 0:
        print("No valid embeddings found to build vector store.")
        return None
        
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(embeddings)
    return index

def main():
    load_config()
    print("Loading documents...")
    documents = load_markdown_documents(KNOWLEDGE_BASE_DIR)
    
    print("Splitting documents into chunks...")
    doc_chunks = split_documents(documents)
    
    chunks_with_embeddings = []
    print(f"Generating embeddings for {len(doc_chunks)} chunks...")
    for i, chunk in enumerate(doc_chunks):
        print(f"Processing chunk {i+1}/{len(doc_chunks)}: {chunk['metadata'].get('doc_name', '')} - {chunk['text'][:30]}...")
        embedding = get_embedding(chunk['text'])
        if embedding:
            chunks_with_embeddings.append({
                'id': chunk['id'],
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'embedding': embedding
            })
        else:
            print(f"Skipping chunk due to embedding failure: {chunk['id']}")

    if not chunks_with_embeddings:
        print("No embeddings were generated. Exiting.")
        return

    print("Building FAISS vector store...")
    vector_store = build_vector_store(chunks_with_embeddings)
    
    if vector_store:
        print(f"Saving FAISS index to {INDEX_PATH}...")
        faiss.write_index(vector_store, INDEX_PATH)
        
        # Save the text chunks and their metadata (without embeddings for this file, as FAISS stores the vectors)
        # We need a mapping from FAISS index ID back to our chunk information
        # For IndexFlatL2, the order is preserved, so the FAISS index i corresponds to the i-th embedding added.
        # We'll save the chunks that successfully got embeddings in the same order.
        successful_chunks = [item for item in chunks_with_embeddings if item['embedding'] is not None]
        print(f"Saving document chunks and metadata to {DOC_CHUNKS_PATH}...")
        with open(DOC_CHUNKS_PATH, 'wb') as f:
            pickle.dump(successful_chunks, f)
        print("Knowledge base indexing complete.")
    else:
        print("Failed to build vector store. Index and chunks not saved.")

if __name__ == "__main__":
    # Adjust KNOWLEDGE_BASE_DIR, INDEX_PATH, DOC_CHUNKS_PATH if running script from a different CWD
    # This script assumes it's run from the 'scripts' directory or paths are relative to project root
    # For simplicity, we assume paths are relative to the project root if this script is in `scripts/`
    # and data is in `data/`.
    
    # Make paths relative to the script's location if needed, or use absolute paths.
    # This current setup assumes the script is run from the project root, or paths are adjusted.
    # If running from `scripts/` directory: 
    # KNOWLEDGE_BASE_DIR = "../data/knowledge_base"
    # INDEX_PATH = "../data/vector_store.faiss"
    # DOC_CHUNKS_PATH = "../data/doc_chunks.pkl"
    
    # If running from project root (e.g., python scripts/build_knowledge_index.py):
    KNOWLEDGE_BASE_DIR = "data/knowledge_base"
    INDEX_PATH = "data/vector_store.faiss"
    DOC_CHUNKS_PATH = "data/doc_chunks.pkl"

    main()
