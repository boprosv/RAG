# RAG
AI-Powered Data Science and Machine Learning Knowledge Assistant

In this project, I built a Streamlit application that serves as an AI copilot for Data Science and Machine Learning (ML) tasks. It leverages LangChain, OpenAI's GPT models, and Chroma for vector storage and retrieval. While pre-configured for Data Science and ML tasks, it is highly customizable to tackle a wide range of problems across various domains. Utilizing a Retrieval-Augmented Generation (RAG) framework, this assistant can be tailored to specific industries, use cases, and different databases, making it a powerful tool for AI-driven insights and automation.

To build a knowledge database, I developed an AI-powered application that processes multiple files (including PDFs and other formats), extracts meaningful insights using OpenAI’s embedding model, and stores the data in ChromaDB for fast and intelligent retrieval. I used 10 different books, making this a highly versatile and efficient assistant for creating customized knowledge databases.

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.environ["OPENAI_API_KEY"])

# Create (or load existing) ChromaDB instance
vectordb = Chroma(persist_directory="Data_Science.db", embedding_function=embeddings)

# Add newly processed document chunks to the database
vectordb.add_documents(chunks)
vectordb.persist()  # Save embeddings for future use

st.success(f"📦 Successfully stored all embeddings in 'Data_Science.db'.")

![Alt image}()

Once the "Data_Science.db" were created I 
