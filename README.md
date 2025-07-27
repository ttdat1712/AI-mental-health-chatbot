# Chatbot for Answering Mental Health Problems

This project builds an intelligent chatbot system to support mental health consultation, utilizing Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) techniques.

## 1. Environment Setup

### 1.1. System Requirements
- Operating System: Windows / Linux / MacOS
- RAM: Minimum 16GB (32GB recommended)
- GPU: NVIDIA GPU with at least 8GB VRAM
- Disk Space: Minimum 20GB free

### 1.2. Software Requirements
- Python >= 3.9
- CUDA >= 11.8 (for GPU support)
- MongoDB >= 5.0

### 1.3. Required Python Libraries
```bash
# Core dependencies
streamlit>=1.24.0
transformers>=4.30.0
torch>=2.0.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
trl>=0.7.0

# Database and data processing
pymongo>=4.3.3
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.2.2

# Embeddings and model utilities
sentence-transformers>=2.2.2
chromadb>=0.3.26
rank-bm25>=0.2.2

# Evaluation metrics
nltk>=3.8.1
sacrebleu>=2.3.1
rouge>=1.0.1

# Environment and utilities
python-dotenv>=1.0.0
tensorboard>=2.13.0
matplotlib>=3.7.1
```

## 2. How to Run

### 2.1. Environment Setup
1. Clone repository về máy:
```bash
git clone <repository-url>
cd mental-health-chatbot
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a .env file and set environment variables:
```
HUGGINGFACE_TOKEN=your_token_here
MONGO_URI=your_mongodb_uri
MONGO_DB_NAME=chatbot
MONGO_COLLECTION=chat_history
```

### 2.2. Run the Application
1. Start MongoDB:
```bash
# Ensure MongoDB is installed and running
mongod
```

2. Launch the Streamlit web application:
```bash
streamlit run app/app.py
```

3. Access the app at: http://localhost:8501

## 3. Fine-tuning the Model

### 3.1. Training Data
1. Data structure:
- Format: CSV with 2 columns (Question, Answer)
- Language: Vietnamese
- Content: Question–answer pairs about mental health
- Number of samples: approximately 20,000 Q&A pairs

2. Data example:
```
Question: "Có loại thực phẩm nào giúp giảm trầm cảm không?"
Answer: "Một số thực phẩm giàu omega-3 (cá hồi, hạt chia), vitamin B (chuối, trứng) và tryptophan (sữa, hạnh nhân) có thể giúp cải thiện tâm trạng"

Question: "Làm thế nào để kiểm soát lo âu?"
Answer: "Bạn có thể thử các phương pháp như hít thở sâu, thiền định, tập thể dục nhẹ nhàng. Việc chia sẻ với người thân hoặc chuyên gia cũng rất hữu ích."
```

3. Data characteristics:
   - Question scope:
     + Symptoms and signs of psychological issues
     + Treatment and management methods
     + Healthy habits and lifestyle
     + Nutrition and diet
   - Answer characteristics:
     + Accurate, science-based information
     + Clear and friendly language
     + Provides specific solutions
     + Encourages seeking professional support when needed

4. Storage location:
   - Raw data: `data_finetune/dataset2.csv`
   - Processed data: `data/`

### 3.2. Reproducing the Training Process (Experiment 2)

1. Environment Setup:
   ```bash
   # Install required packages for training
   pip install -r requirements.txt
   pip install accelerate bitsandbytes transformers trl
   ```

2. Data Preparation:
   - Place your training data in `data_finetune/dataset2.csv`
   - Format: CSV with columns 'Question' and 'Answer'
   - Ensure text is in UTF-8 encoding

3. Choose the Model to Train:
   - Open one of the notebooks in `Finetuning models/`:
     + `finetune_gemma.ipynb`: For Gemma 2B
     + `finetune_llama.ipynb`: For Llama 3.2B
     + `Finetune_Qwen.ipynb`: For Qwen 2.5B

4. Configure Training Parameters:
   ```python
   training_args = TrainingArguments(
       output_dir="output_directory",
       max_steps=1000,
       per_device_train_batch_size=1,
       gradient_accumulation_steps=8,
       learning_rate=2e-4,
       weight_decay=0.01,
       fp16=True,
       logging_steps=10,
       save_steps=200
   )
   
   # QLoRA parameters
   peft_config = LoraConfig(
       r=64,
       lora_alpha=16,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
       bias="none",
       task_type="CAUSAL_LM"
   )
   ```

5. Execute Training:
   - Run all cells in the notebook sequentially
   - Monitor training progress through loss values and evaluation metrics
   - Training takes approximately 4-6 hours on T4/V100 GPU

6. Save and Export Model:
   - The model will be saved in the specified output directory
   - Convert to GGUF format if needed:
   ```python
   from transformers import AutoModelForCausalLM
   model.save_pretrained("final_model", safe_serialization=True)
   ```

7. Model Evaluation:
   ```python
   # Calculate metrics
   results = trainer.evaluate()
   print(f"Loss: {results['eval_loss']}")
   print(f"Perplexity: {math.exp(results['eval_loss'])}")
   ```

## 4. Demo Usage

### 4.1. Launching the application
```bash
streamlit run app/app.py
```

### 4.2. Key features
1. Chat interface::
   - Chat window for interaction
   - Suggested question panel
   - Conversation history

2. Example interaction:
```
User: "Triệu chứng của trầm cảm là gì?"
Bot: [Detailed answer about depression symptoms]

User: "Làm thế nào để kiểm soát lo âu?"
Bot: [Guidance on managing anxiety]
```

### 4.3. Directory structure
```
app/
├── answer_generator.py    # Answer generation logic
├── app.py                # Streamlit interface
├── configuration.py      # System configuration
├── data_processor.py     # Data processing
├── embedding_utils.py    # Embedding generation
├── main.py              # CLI interface
├── model_loader.py      # Model loading
├── mongo_manager.py     # MongoDB management
├── question_suggester.py # Question suggestion
├── search_engine.py     # Search engine
└── utils.py             # Utilities

data/                    # data
data_finetune             # data to finetune models
models/                  # Trained models
Finetuning models/      # Fine-tuning notebooks
Experiment 1/           # Method evaluation experiment
```

### 4.4. Reproducing Context Retrieval Experiments (Experiment 1)

To reproduce the experiments comparing different context retrieval methods:

1. Environment Setup:
   ```bash
   pip install -r requirements.txt
   pip install chromadb sentence-transformers rank-bm25 nltk matplotlib seaborn scikit-learn
   ```

2. Data Preparation:
   - Ensure `data/data.txt` contains your document chunks
   - Place ground truth data in `ground_truth.xlsx` with columns:
     + query: Test questions
     + relevant_chunk: Semicolon-separated chunk IDs that are relevant

3. Running the Experiment:
   - Open `Experiement 1/experiment_1.ipynb`
   - Execute cells sequentially to:
     1. Load and preprocess data
     2. Initialize retrieval methods (TF-IDF, BM25, BGE-M3 embeddings)
     3. Set up reranking with BGE-Reranker-v2
     4. Run evaluation with test queries
     5. Generate performance visualizations

4. Methods Evaluated:
   - TF-IDF (with/without reranker)
   - BM25 (with/without reranker)
   - Embedding using BGE-M3 (with/without reranker)
   - Hybrid: BM25 + Embedding (with/without reranker)

5. Evaluation Metrics:
   - Precision
   - Recall
   - Mean Reciprocal Rank (MRR)
   - Mean Average Precision (MAP)

6. Visualizations Generated:
   - Bar charts for each metric
   - Radar chart comparing methods
   - Heatmap of performance metrics
   - Improvement analysis with reranker

The notebook will automatically generate comprehensive visualizations and statistical analyses of the results. Results are displayed in both tabular and graphical formats for easy comparison.
