# LRP-Merge: Layer-wise Relevance Propagation for LLM Merging

LRP-Merge is a custom model merging method built on top of [Mergekit](https://github.com/arcee-ai/mergekit). 

Standard merging techniques (like Linear or SLERP) often treat all model weights equally or prune them based purely on weight magnitude. **LRP-Merge** introduces a hybrid pruning-averaging approach that uses **Layer-wise Relevance Propagation (LRP)** scores to identify and preserve the weights that are functionally critical to a model's learned capabilities, while discarding the noise.

---

## 🧠 The Core Concept

### The Problem with Standard Merging
When merging fine-tuned models, a simple weighted average (`θ_merged = 0.7 × θ_model_A + 0.3 × θ_model_B`) dilutes the specific knowledge learned by each model. In reality, only a small fraction of a model's weights drive its new capabilities; the rest are effectively noise.

### The LRP Solution
LRP-Merge applies an XAI (Explainable AI) technique to score each weight matrix: *"How much did this specific weight contribute to the correct prediction?"*

Our merging algorithm executes in three steps for every tensor:

1. **Compute the Task Vector (Delta):**
   ```python
   δ = θ_model - θ_base
   ```
   We isolate the fine-tuned knowledge by subtracting the base model weights. The base model's foundational knowledge is preserved intact.

2. **Functional Trimming (Sparsification):**
   ```python
   mask = (relevance_scores >= threshold) # Keep top k% 
   δ_sparse = δ * mask
   ```
   Instead of pruning by the magnitude of the weight itself, we prune based on its LRP relevance score. (If LRP scores are unavailable for a specific tensor, the method gracefully falls back to magnitude pruning: `r = |δ|`).
   *Note: 1D tensors (biases, LayerNorm scales) are passed through untouched to prevent breaking the model architecture.*

3. **Weighted Parameter Averaging:**
   ```python
   θ_merged = θ_base + Σ (λᵢ / Σλ) × δ_sparseᵢ
   ```
   The sparsified task vectors are normalized and added back to the base model.

---

## 🚀 Installation

Because LRP-Merge is a custom extension, it requires a local installation of `mergekit` with our custom script injected.

1. Clone this repository (which includes the base `mergekit` and the custom LRP logic):
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. Install the modified `mergekit` in editable mode:
   ```bash
   cd mergekit_repo
   pip install -e .
   cd ..
   ```

---

## 🛠️ Usage

### 1. Configure your Merge (YAML)
Create a `lrp_config.yaml` file. Note the `merge_method: lrp` and the `density` parameter (which controls what percentage of weights to keep per layer).

```yaml
merge_method: lrp
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
parameters:
  density: 0.7  # Keep the top 70% most relevant weights
models:
  - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    parameters:
      weight: 0.7
  - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    parameters:
      weight: 0.3
```

### 2. Execute the Merge
Run the merge engine. We recommend setting `--out-shard-size` to prevent Out-Of-Memory (OOM) errors during serialization.

```bash
# Optional: Clear previous outputs if re-running
rm -rf ./merged-model-directory/*

# Run Mergekit
mergekit-yaml lrp_config.yaml ./merged-model-directory \
    --copy-tokenizer \
    --lazy-unpickle \
    --allow-crimes \
    --out-shard-size 300M
```

### 3. Test the Merged Model
You can test the output using standard HuggingFace `transformers`. Be sure to load the model in `float16` and use `device_map="auto"` to prevent memory spikes.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./merged-model-directory"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.float16
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the secret to time travel?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```

### Example Output
Running the script on an LRP-merged **Qwen2.5-1.5B** (Base + Instruct) produces coherent, instruction-following text:
```text
User: What is photosynthesis?
Assistant: Photosynthesis is a process used by plants, algae, and some bacteria to convert light energy into chemical energy. This chemical energy is stored in glucose, a simple sugar that the plant can use for energy. Photosynthesis occurs in the leaves of the plant...
```

---

## 🔮 Next Steps: Injecting Real LRP Scores

Currently, the engine is designed to look for pre-calculated LRP relevance maps (saved as `.pt` tensors) in the pipeline. If it does not find them, it defaults to magnitude pruning.

To fully utilize LRP-Merge:
1. Run an LRP attribution pass on your fine-tuned models using an evaluation dataset.
2. Save the per-weight relevance score tensors.
3. Pass these score tensors into the `mergekit` execution graph (currently configured to look for `tensors.get(f"{model_ref}_lrp")`).
