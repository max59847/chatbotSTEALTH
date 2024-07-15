import os
import pickle
import traceback
import logging
from flask import Flask, request, jsonify, send_from_directory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../static')

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Set file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, 'faiss_index.pickle')
metadata_path = os.path.join(current_dir, 'metadata.pkl')
retriever_path = os.path.join(current_dir, 'retriever_components.pkl')

# Initialize global variables
qa = None

def load_qa_system():
    global qa
    
    if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(retriever_path):
        logger.info("Loading index, metadata, and retriever components...")
        try:
            # Load FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(index_path, embeddings)

            # Load metadata and retriever components
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            with open(retriever_path, 'rb') as f:
                retriever_components = pickle.load(f)

            # Reconstruct the retrievers
            bm25_retriever = BM25Retriever.from_documents(retriever_components['bm25_docs'])
            bm25_retriever.k = 20

            vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vectorstore_retriever],
                weights=[0.5, 0.5]
            )

            # Recreate the multi-query retriever
            llm = ChatOpenAI(temperature=0)
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=llm
            )

            # Create the QA system
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(),
                chain_type="stuff",
                retriever=multi_query_retriever,
                return_source_documents=True
            )
            logger.info("QA system created successfully.")
        except Exception as e:
            logger.error(f"Error initializing QA system: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.error(f"Error: Index, metadata, or retriever component file not found.")

# Load the QA system when the app starts
load_qa_system()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if qa is None:
        return jsonify({"error": "QA system not loaded. Check if all required files exist."}), 500
    
    try:
        question = request.json['question']
        logger.info(f"Received question: {question}")
        
        result = qa({"query": question})
        logger.info("QA system processed the question successfully")
        
        answer = result["result"]
        sources = [doc.metadata.get('course_id', 'Unknown') for doc in result.get('source_documents', [])]
        
        logger.info(f"Answer: {answer}")
        logger.info(f"Sources: {sources}")
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "debug_info": {
                "question": question,
                "similar_docs": [doc.page_content[:100] + "..." for doc in result.get('source_documents', [])]
            }
        })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)