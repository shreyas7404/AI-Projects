#Task1: Text Generation with GPT- 2 

'''Problem Statement: Train a model to generate coherent and contextually relevant text based on a given prompt.
Starting with GPT-2, transformer model developed by Open AI, you will learn how to fine tune the model on a custom
dataset to create text that mimics the style and structure of your training data This is the name and problem statement 
of my project Can you please explain me the project.'''


# Step 1: Install required libraries
!pip install transformers datasets accelerate

# Step 2: Upload your dataset (if using Google Colab)
from google.colab import files
uploaded = files.upload()

# Step 3: Load your dataset
from datasets import load_dataset

# Load the dataset and take a small subset for testing
dataset = load_dataset('text', data_files={'train': '1342-0.txt'})
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(100))  # Select 100 samples for testing

# Step 4: Tokenize the dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    # Ensure the labels are the same as the input_ids
    tokenized_output['labels'] = tokenized_output['input_ids'].copy()
    return tokenized_output

tokenized_datasets = small_train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 5: Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Update the model's config to recognize the new padding token
model.resize_token_embeddings(len(tokenizer))

# Ensure the model uses the GPU if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 6: Fine-tuning the model
from transformers import Trainer, TrainingArguments

# Define training arguments with gradient accumulation and mixed precision
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Reduce number of epochs for testing
    per_device_train_batch_size=8,  # Increase batch size if memory allows
    gradient_accumulation_steps=4,  # Accumulate gradients to effectively increase batch size
    fp16=True,  # Use mixed precision training
    save_steps=50,  # Reduce the save steps for quicker testing
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Fine-tune the model
trainer.train()

# Step 7: Save the model
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')

# Step 8: Generate text with the fine-tuned model
from transformers import pipeline

# Load the fine-tuned model and tokenizer
fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2')
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2')

# Create a text generation pipeline
text_generator = pipeline('text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Generate text based on a prompt
prompt = "I love the"
generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)

print(generated_text[0]['generated_text'])
