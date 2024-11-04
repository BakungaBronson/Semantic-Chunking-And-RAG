import os
import openai
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Initialize OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OpenAI API key is not set in the environment variables.")

    # Ensure database directory exists
    os.makedirs("db", exist_ok=True)

    # Initialize Chroma client with persistent storage
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db/"  # Directory to store the database
    ))

    # Create or get a collection in Chroma
    collection = client.create_collection(name="markdown_documents")

    def load_markdown_files(directory):
        """Load and read all Markdown files in the specified directory."""
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        return documents

    def semantic_chunking(text, chunk_size=1000, chunk_overlap=200):
        """Split text into semantically meaningful chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def get_embeddings(texts):
        """Generate embeddings for a list of texts using OpenAI's API."""
        embeddings = []
        for text in texts:
            try:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response['data'][0]['embedding'])
            except Exception as e:
                print(f"Error generating embedding: {e}")
        return embeddings

    # Directory containing your Markdown files
    markdown_directory = 'path_to_markdown_files'
    if not os.path.exists(markdown_directory):
        raise ValueError("The specified path to markdown files does not exist.")

    # Load and process Markdown files
    documents = load_markdown_files(markdown_directory)

    # Process and add documents to Chroma
    for doc in documents:
        chunks = semantic_chunking(doc)
        embeddings = get_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            collection.add(
                documents=[chunk],
                embeddings=[embedding]
            )

    def retrieve_documents(query, collection, k=5):
        """Retrieve relevant documents from Chroma based on the query."""
        query_embedding = get_embeddings([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []

    def generate_response(query, collection):
        """Generate a response from GPT-4o using retrieved documents as context."""
        retrieved_docs = retrieve_documents(query, collection)
        if not retrieved_docs:
            return "No relevant documents found for your query."

        context = "\n\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating response from GPT-4o: {e}"

    # Example usage
    query = "What is the time of the day?"
    response = generate_response(query, collection)
    print("Response:", response)

if __name__ == "__main__":
    main()
