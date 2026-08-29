# test_document_processor.py
from document_processor import DocumentProcessor
import os

def test_processor():
    # Initialize processor
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    print("="*60)
    print("📄 DOCUMENT PROCESSOR TEST")
    print("="*60)
    
    # Check for test files
    test_files = []
    
    # Look for sample files in current directory
    sample_files = ['sample.txt', 'sample.pdf', 'sample.docx', 'sample.html']
    
    for filename in sample_files:
        if os.path.exists(filename):
            test_files.append(filename)
    
    if not test_files:
        print("\n⚠️ No sample files found. Create a test file:")
        print("   - Create sample.txt with some text")
        print("   - Or upload a file through Streamlit later")
        return
    
    print(f"\n📁 Found test files: {test_files}")
    
    # Process each file
    for filename in test_files:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print("="*60)
        
        # Open file as file-like object
        with open(filename, 'rb') as f:
            # Create a mock file object
            class MockFile:
                def __init__(self, name, content):
                    self.name = name
                    self._content = content
                
                def getbuffer(self):
                    return self._content
            
            mock_file = MockFile(filename, f.read())
            
            # Get file info
            info = processor.get_file_info(mock_file)
            print(f"📊 File Info:")
            print(f"   Name: {info['name']}")
            print(f"   Size: {info['size']} bytes")
            print(f"   Type: {info['type']}")
            
            # Process file
            try:
                chunks = processor.process_file(mock_file)
                print(f"\n✅ Processed {len(chunks)} chunks")
                
                if chunks:
                    print(f"\n📝 First chunk preview:")
                    print(f"   Title: {chunks[0]['title']}")
                    print(f"   Text length: {len(chunks[0]['text'])} chars")
                    print(f"   Preview: {chunks[0]['text'][:200]}...")
                    print(f"   Embedding size: {len(chunks[0]['embedding'])}")
                    
                    # Show metadata
                    print(f"\n📋 Metadata:")
                    for key, value in chunks[0]['metadata'].items():
                        print(f"   {key}: {value}")
                
                # Save chunks
                processor.save_chunks(chunks, "embeddings.json")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_processor()