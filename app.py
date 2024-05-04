import configparser

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS, cross_origin

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
flask_port = config.get('flask', 'port')
ollama_model = config.get('ollama', 'ollama_model')
data_folder = config.get('ollama', 'data_folder')

# read documents
documents = SimpleDirectoryReader(data_folder).load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model=ollama_model, request_timeout=30.0)

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
    # print(response)
    return str(response)

process_question("who are you?")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            response = process_question(question)
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'No question provided'}), 400
    with open('./templates/index.html', 'r') as f:
        html_content = f.read()
    # return render_template('index.html')
    return html_content

@app.route('/public/<path:path>')
def serve_image(path):
    return send_from_directory('static', path)

@app.route('/ask', methods=['POST'])
def ask_question():
        question = request.form.get('question')
        if question:
            response = process_question(question)
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'No question provided'}), 400

@app.route('/random-tip', methods=['GET'])
def random_question():
            response = process_question("give me a single point from any of the safety policies")
            return jsonify({'response': response})

if __name__ == '__main__':
    # Run Flask app with the configured port
    app.run(port=int(flask_port))
