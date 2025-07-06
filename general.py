import os
from preprocessing import create_chunks, hybrid_search, init_chroma, load_docs




## General----------------------------------------------------------------
def general_query():
    """
    Main function to demonstrate usage of preprocessing functions with RAG chain.
    """
    # Your PDF path
    pdf = r"D:\1. INDIUM\Project\Agentic_AI_project\PDFs\TESLA.pdf"
    
    # Extract filename without extension for the create_chunks function
    pdf_filename = os.path.splitext(os.path.basename(pdf))[0]
    
    print("🚀 Starting PDF processing and RAG setup...")
    
    # Step 1: Create chunks from your PDF
    create_chunks([pdf_filename])
    
    # Step 2: Initialize ChromaDB
    chroma_store = init_chroma()
    
    # Step 3: Load processed documents
    docs = load_docs()
    print(f"📚 Loaded {len(docs)} document chunks")
    
    # Step 4: Perform hybrid search with RAG chain and get the answer
    sample_query = "What is the main topic of this document?"
    print(f"🔍 Processing query: '{sample_query}'")
    
    # Get the RAG chain answer (string response)
    answer = hybrid_search(sample_query)
    
    print(f"\n🤖 RAG Chain Answer:")
    print("=" * 50)
    print(answer)
    print("=" * 50)
    
    # Optional: Ask more questions
    interactive_mode = input("\n❓ Do you want to ask more questions? (y/n): ").lower().strip()
    
    if interactive_mode == 'y':
        print("\n🔍 Interactive Q&A Mode (type 'quit' to exit):")
        while True:
            try:
                user_query = input("\n💬 Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_query:
                    print("⚠️ Please enter a question.")
                    continue
                
                print(f"\n🔍 Processing: {user_query}")
                response = hybrid_search(user_query)
                
                print(f"\n🤖 Answer:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    return answer

if __name__ == "__main__":
    # Replace with your actual PDF path
    pdf_path = r"D:\1. INDIUM\Project\Agentic_AI_project\PDFs\TESLA.pdf"
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable with the correct path to your PDF file.")
    else:
        general_query()

# ------------------------------------------------------------------------------

