import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

def train_local_model():
    # 1. Configuration
    model_id = "meta-llama/Llama-2-7b-hf" # Or any Hugging Face model
    dataset_path = "campus_dataset.json"   # Your local training data
    output_dir = "./campus_model_output"
    
    print(f"--- Starting Local Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with GPU optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, # Use half-precision for GPU speed
        device_map="auto"          # Automatically use available GPU
    )

    # 3. Load Dataset
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please provide a JSON dataset.")
        return

    dataset = load_dataset("json", data_files=dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments (Optimized for local GPU)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,                  # Required for modern GPUs
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        push_to_hub=False,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # 6. Run Training
    print("--- Training in Progress... ---")
    trainer.train()

    # 7. Save Final Model
    trainer.save_model(output_dir)
    print(f"--- Training Complete! Model saved to {output_dir} ---")

if __name__ == "__main__":
    train_local_model()
