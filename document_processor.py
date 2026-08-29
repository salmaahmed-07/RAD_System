# document_processor.py
import os
import tempfile
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

# Document loaders from langchain-community
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
# Text splitter from langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Document schema from langchain-core
from langchain_core.documents import Document

# For embeddings
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor with chunking parameters.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        # Initialize embedding model (use the same one as your RAG system)
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")
    
    def process_file(self, file) -> List[Dict[str, Any]]:
        """
        Process an uploaded file and return chunks with embeddings.
        
        Args:
            file: Uploaded file object (from Streamlit/Gradio)
            
        Returns:
            List of chunks with metadata and embeddings
        """
        file_extension = file.name.split('.')[-1].lower()
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(file.getbuffer())
            temp_path = tmp_file.name
        
        try:
            # Load based on file type
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(temp_path)
            elif file_extension == 'txt':
                loader = TextLoader(temp_path, encoding='utf-8')
            elif file_extension == 'html':
                loader = UnstructuredHTMLLoader(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load documents
            documents = loader.load()
            
            # Process and chunk
            chunks = self._process_documents(documents, file.name, file_extension)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            raise
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _process_documents(self, documents: List[Document], source_name: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Process loaded documents into chunks with embeddings.
        """
        all_chunks = []
        
        for doc in documents:
            # Split text into chunks
            chunks_text = self.text_splitter.split_text(doc.page_content)
            
            # Add metadata to each chunk
            for i, chunk_text in enumerate(chunks_text):
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                # Generate embedding
                embedding = self.model.encode(
                    "passage: " + chunk_text,
                    normalize_embeddings=True
                )
                
                chunk_data = {
                    "title": source_name,
                    "text": chunk_text,
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "source": source_name,
                        "source_type": "user_upload",
                        "file_type": file_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks_text),
                        "character_count": len(chunk_text)
                    }
                }
                
                all_chunks.append(chunk_data)
        
        return all_chunks
    
    def process_multiple_files(self, files) -> List[Dict[str, Any]]:
        """
        Process multiple uploaded files.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        
        for file in files:
            try:
                chunks = self.process_file(file)
                all_chunks.extend(chunks)
                print(f"✅ Processed: {file.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"❌ Failed to process {file.name}: {str(e)}")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str = "embeddings.json"):
        """
        Save chunks to a JSON file.
        """
        # Load existing chunks if file exists
        existing_chunks = []
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        
        # Add new chunks
        existing_chunks.extend(chunks)
        
        # Save combined chunks
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(chunks)} new chunks to {output_file}")
        print(f"📊 Total chunks in database: {len(existing_chunks)}")
    
    def get_file_info(self, file) -> Dict[str, Any]:
        """
        Get information about a file without processing it.
        """
        file_extension = file.name.split('.')[-1].lower()
        return {
            "name": file.name,
            "size": len(file.getbuffer()),
            "extension": file_extension,
            "type": self._get_file_type_description(file_extension)
        }
    
    def _get_file_type_description(self, extension: str) -> str:
        """Get a human-readable description of the file type."""
        descriptions = {
            'pdf': 'PDF Document',
            'docx': 'Word Document',
            'txt': 'Text File',
            'html': 'HTML Page',
            'png': 'Image (text extraction may be limited)',
            'jpg': 'Image (text extraction may be limited)',
            'jpeg': 'Image (text extraction may be limited)'
        }
        return descriptions.get(extension, 'Unknown')