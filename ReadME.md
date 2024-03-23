# Flask server serving requests to a Retrieval-augmented generation model from Ollama

## Requirements
- Ollama

## Run the project

### create a virtual environment
python -m venv env
.\env\Scripts\activate

### install dependencies
pip install -r requirements.txt

### update the config file
[flask]
port = 5000

[ollama]
ollama_model = tinydolphin
data_folder = ./data

### run service
flask --app app run
