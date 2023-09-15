from transformers import BertForMaskedLM, BertTokenizer
import torch

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # You can choose a specific variant
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Freeze the weights of the pre-trained model
for param in model.parameters():
    param.requires_grad = False
    
# Save the model's state_dict to a checkpoint file
checkpoint_path = "bert_checkpoint.pth"
torch.save(model.state_dict(), checkpoint_path)

# Initialize the model with random weights
num_classes = 1000
model.classifier = torch.nn.Linear(768, num_classes)
model.init_weights()

print(f"Pre-trained BERT model '{model_name}' checkpoint saved to {checkpoint_path}")
