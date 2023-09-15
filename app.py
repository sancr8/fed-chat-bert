from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertForMaskedLM, BertTokenizer

app = Flask(__name__)

# Load your pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define a function to generate responses
def generate_response(input_text):
    # Process the user's input (you can replace this with your specific processing logic)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(debug=True)
