from flask import Flask, request, jsonify, session
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pickle  # For saving objects in session

# Load environment variables (for OpenAI API key)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management
CORS(app)  # Allow cross-origin requests from frontend

# Setup OpenAI API Key (ensure this is properly loaded from .env)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure OpenAI API Key is available
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing. Make sure it's properly set in your environment.")

# Process PDF and create conversation chain
def process_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vectorstore using OpenAI's embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Use your OpenAI API key here
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create a conversational retrieval chain using OpenAI for answering questions
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key)  # Use OpenAI for chat model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )
    return conversation_chain

# Route to upload and process PDF
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    pdf_file = request.files['file']
    if pdf_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process PDF
        text_chunks = process_pdf(pdf_file)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        # Store conversation chain in session (as a serialized object)
        session['conversation_chain'] = pickle.dumps(conversation_chain)
        print("Conversation chain stored successfully.")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    return jsonify({"message": "PDF processed successfully", "text_chunks": text_chunks})

# Route to ask questions based on processed PDF
@app.route('/ask-question', methods=['POST'])
def ask_question():
    data = request.get_json()  # Ensure you are parsing the JSON body
    user_question = data.get('question')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Retrieve the conversation chain from session
        if 'conversation_chain' not in session:
            return jsonify({"error": "No conversation chain found. Please process a PDF first."}), 400

        conversation_chain = pickle.loads(session['conversation_chain'])
        print("Conversation chain retrieved successfully.")
        response = conversation_chain({'question': user_question})
    except Exception as e:
        print(f"Error during question processing: {str(e)}")
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

    return jsonify({"response": response['answer']})

if __name__ == "__main__":
    app.run(debug=True)
