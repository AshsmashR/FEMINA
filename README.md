FEMINA: A Lightweight HAI-DEF Clinical Reasoning Framework for Women’s Health
FEMINA is a modular MedGemma-based HAI-DEF system that uses lightweight LoRA adapters and rule-grounded RAGS
Problem statement
Women’s endocrine and reproductive disorders (PCOS, thyroid dysfunction, endometriosis, gestational diabetes, preeclampsia, menstrual cycle disruption) are common, multi-factorial, and often assessed through fragmented signals: labs, anthropometrics (BMI/WHR), symptoms, and history. In practice, interpretation quality varies across settings; patients frequently receive binary labels without transparent reasoning, while clinicians face time pressure and heterogeneous data. This creates an unmet need for a tool that can (i) standardize structured interpretation, (ii) communicate risk patterns clearly, and (iii) remain non-diagnostic and clinically cautious. If effective, FEMINA improves clinical feasibility by accelerating triage and follow-up planning (e.g., identifying “moderate metabolic risk pattern” vs “low endocrine disruption pattern”), and improves patient understanding by producing consistent, explainable reasoning without claiming a diagnosis.

Overall solution
Workflow of the project
The system follows a modular instruction-tuning and retrieval-augmented reasoning architecture built on MedGemma-4B-IT. The complete workflow aligns directly with the implemented code structure.

1. Data Structuring and Instruction Formatting
Raw structured clinical datasets (multi-disorder profiles including hormonal panels, metabolic markers, menstrual characteristics, and categorical disease indicators) are converted into an instruction-tuning format.

Each dataset entry is mapped into a single "text" field using a deterministic template:

If input exists:

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
If input is empty:

### Instruction:
{instruction}

### Response:
{output}
This formatting is applied using a map() transformation function:

Ensures uniform training distribution.
Enforces structured reasoning layout.
Preserves deterministic response boundaries.
This template is reused identically during evaluation and inference, ensuring alignment between training and deployment behavior.

2. Base Model Loading (Lightweight Configuration)
The healthcare-aligned HAI-DEF model google/medgemma-4b-it is loaded in 4-bit quantized mode using BitsAndBytes:

load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
device_map="auto"
attn_implementation="eager"
Quantization reduces GPU memory usage while maintaining reasoning stability.
The base model is loaded once and reused across adapters.

Tokenizer adjustments:

padding_side = "right"
pad_token = eos_token (if absent)
Caching is disabled during training (use_cache=False) to support gradient flow.

3. LoRA Adapter Fine-Tuning
Two independent LoRA adapters are trained on top of the frozen MedGemma base:

Adapter A: Structured endocrine + reproductive reasoning.
Adapter B: Structured reasoning for Preeclampsia, gestational diabetes.
Training characteristics:

Parameter-efficient fine-tuning (LoRA).
Base weights remain frozen.
Training runtime ≈ 4 hours total (~2 hours per adapter).
Conducted on Google Colab Pro GPU.
Lightweight optimization without full model retraining.
This design allows:

Specialization without duplication of model weights.
Fast experimentation.
Modular deployment.
4. Response-Only Loss Masking (Evaluation Strategy)
Evaluation strictly measures generation quality on the Response section only.

Implementation steps:

Tokenize full "text" example.
Locate the index of:
   RESPONSE_TAG = "### Response:\n"
Tokenize the prompt portion (Instruction + Input).
Assign label -100 to prompt tokens.
Compute cross-entropy loss only on response tokens.
This prevents loss inflation from prompt memorization and ensures evaluation reflects reasoning performance.

Collation:

Dynamic batch padding.
-100 label padding for masked tokens.
Attention mask aligned to padded tokens.
5. Validation Results (Reproducible)
Adapter B Evaluation

Val samples: 800
Val loss: 1.0953
Val perplexity: 2.99
Perplexity < 3 indicates strong token-level confidence and stable structured generation.

Sample output:

Cycle Pattern: Cycle within typical range
Lifestyle Assessment: Moderate lifestyle-related cycle disruption risk
Interpretation:
Lifestyle factors such as BMI, stress level, sleep duration, and exercise frequency...
Response Status: Low Risk
This assessment does not constitute a medical diagnosis...
Adapter A Evaluation

Val samples: 4000
Val loss: 1.2608
Val perplexity: 3.528
Example (PCOS-endometriosis reasoning):

Predicted probability: 31.5%
Gold reference: 32.5%

Demonstrates near-aligned probabilistic reasoning under structured template constraints.

6. Controlled Decoding Strategy
Inference uses deterministic and constrained decoding:

do_sample=False
max_new_tokens bounded
repetition_penalty ≈ 1.1–1.15
no_repeat_ngram_size=4
Explicit eos_token_id and pad_token_id
This reduces:

Over-generation
Repetition loops
Textbook-style drift
Hallucinated expansions
Post-generation processing extracts text after "### Response:" to remove prompt artifacts.

7. Clinical Rulebook RAG Layer
A structured rulebook text file encodes:

PCOS reasoning patterns
Thyroid functional states
Endometriosis probability phrasing
Preeclampsia risk categories
Gestational diabetes metabolic patterns
Menstrual cycle classification rules
Mandatory disclaimer language
Implementation:

Load rulebook text.
Split into sections.
Build TF-IDF vectorizer.
Retrieve top-k relevant rule chunks using cosine similarity.
Inject retrieved rules into prompt under:
   ### Retrieved Clinical Rules:
This ensures:

Style consistency.
Domain constraint enforcement.
Reduced hallucination.
Structured phrasing adherence.
No retraining required for rule enforcement.

8. Multi-Adapter Runtime Switching
Adapters are loaded using:

PeftModel.from_pretrained(...)
model.load_adapter(...)
Runtime switching:

model.set_adapter("Adapter_A")
model.set_adapter("Adapter_B")
This allows:

Instant specialization change.
Shared base model.
Memory-efficient deployment.
9. Gradio Interface (User-Facing Stack)
The deployment stack includes:

Dropdown for adapter selection.
Checkbox for RAG activation.
Instruction textbox.
Clinical input textbox.
Max token slider.
Output display panel.
Workflow:

User input → Adapter selection → Rule retrieval → Prompt assembly → Controlled generation → Structured output display.

This mirrors real-world clinical triage reasoning rather than chatbot conversation.

10. Computational Feasibility
4-bit quantized base model.
Lightweight LoRA adapters.
≈4 hours total training time.
Single GPU runtime feasible.
No full fine-tuning required.
Modular architecture.
This demonstrates realistic deployability in constrained environments.

11. Planned Extension (2–3 Days)
MedGemma’s multimodal capabilities will be integrated to process:

Ovarian ultrasound imagery.
Thyroid ultrasound.
Additional imaging biomarkers.
This extends the architecture from text-only reasoning to structured image-text multimodal reasoning.

System Summary
FEMINA follows a reproducible workflow:

Structured dataset → Instruction mapping → Quantized MedGemma → LoRA specialization → Response-masked evaluation → Controlled decoding → Rule-grounded RAG → Multi-adapter UI deployment.

The architecture prioritizes:

Clinical feasibility
Lightweight adaptation
Structured reasoning stability
Safe non-diagnostic output
Reproducible evaluation
