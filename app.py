# # Just runs .complete to make sure the LLM is listening
# from llama_index.llms.ollama import Ollama

# llm = Ollama(model="tinydolphin")
# response = llm.complete("Who is Laurie Voss?")
# print(response)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# read documents
documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="tinydolphin", request_timeout=30.0)

ll_index = VectorStoreIndex.from_documents(
documents,
)

# initialize flask
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def process_question(question):
    query_engine = ll_index.as_query_engine()
    response = query_engine.query(question+"?")
    return str(response)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            response = process_question(question)
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'No question provided'}), 400
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            response = process_question(question)
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'No question provided'}), 400
    else:
        return jsonify({'error': 'Method not allowed'}), 405