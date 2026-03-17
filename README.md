<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/30f5b738-c033-4237-ba57-8c82ea13aa4e" />

🌸 FEMINA 🌸

A Lightweight HAI-DEF Clinical Reasoning Framework for Women’s Health
FEMINA is a modular Human-AI Decision Framework (HAI-DEF) built on MedGemma-4B-IT, designed to generate structured, explainable, and clinically cautious reasoning for women’s health conditions. Instead of producing binary labels, the system focuses on interpretable risk patterns, enabling more consistent clinical understanding and patient communication.

The framework integrates parameter-efficient LoRA fine-tuning, rule-grounded retrieval (RAG), and controlled decoding to produce stable, non-diagnostic outputs aligned with real-world clinical reasoning.

Problem Context: Women’s endocrine and reproductive disorders—including PCOS, thyroid dysfunction, endometriosis, gestational diabetes, preeclampsia, and menstrual irregularities—are inherently multi-factorial. Clinical interpretation depends on fragmented inputs such as laboratory values, anthropometric measures (BMI, WHR), symptoms, and patient history.

In practice, this leads to inconsistent interpretations across settings. Patients are often given binary labels without explanation, while clinicians must operate under time constraints with heterogeneous data. The result is a gap between data availability and interpretable reasoning.

FEMINA addresses this by structuring clinical interpretation into a standardized, explainable reasoning pipeline that communicates risk without making diagnostic claims.

System Workflow

FEMINA follows a modular pipeline:

Structured Clinical Data  
→ Instruction Formatting  
→ Quantized MedGemma (4-bit)  
→ LoRA Adapter Specialization  
→ Response-Masked Training  
→ Controlled Decoding  
→ Rule-Grounded RAG Injection  
→ Structured Output

The workflow is designed to mirror how clinicians synthesize information—progressively refining interpretation rather than jumping to conclusions.

Core Methodology

The system begins by transforming structured clinical datasets into a unified instruction-tuning format. Each data point—containing hormonal panels, metabolic markers, menstrual characteristics, and categorical indicators—is mapped into a deterministic template:

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}

This format is consistently used across training, validation, and inference. It ensures that the model learns not just patterns, but also the structure of reasoning itself, preventing drift between training and deployment behavior.

The base model, MedGemma-4B-IT, is loaded in 4-bit quantized mode using BitsAndBytes. This significantly reduces memory usage while maintaining reasoning stability, enabling deployment on a single GPU. The base model remains frozen throughout training.

On top of this, FEMINA introduces two LoRA adapters, each specializing in a subset of clinical reasoning tasks. One focuses on endocrine and reproductive conditions, while the other targets preeclampsia and gestational diabetes. This modular design allows specialization without duplicating model weights, keeping the system lightweight and flexible.

Training is further refined using response-only loss masking, where the model is evaluated strictly on the generated reasoning portion. Prompt tokens are masked out, ensuring that performance reflects genuine reasoning ability rather than memorization of input structure.

During inference, FEMINA uses a controlled decoding strategy. Sampling is disabled, repetition is constrained, and token generation is bounded. This reduces hallucinations, prevents over-generation, and maintains concise, clinically aligned outputs.

Rule-Grounded Reasoning (RAG Layer)

A key component of FEMINA is its rule-based retrieval layer. A curated clinical rulebook encodes domain knowledge such as PCOS reasoning patterns, thyroid functional states, risk categorization for preeclampsia, and standardized disclaimer language.

At runtime, relevant rule segments are retrieved using TF-IDF and cosine similarity, and injected into the prompt under a dedicated section:

### Retrieved Clinical Rules:
{top_k_rules}

This ensures that outputs remain stylistically consistent, clinically grounded, and constrained, without requiring additional model training.

Example Input and Output

Example 1: Hormonal + Symptom Profile

Input

FSH: 6.8 IU/L  
Testosterone: 45 ng/dL  
Cycle: Irregular  
Symptoms: Acne, mild weight gain

Output

Cycle Pattern: Irregular cycle observed  
Lifestyle Assessment: Moderate endocrine disruption pattern  

Interpretation:
Elevated testosterone levels combined with irregular cycles and dermatological symptoms suggest a moderate likelihood of hormonal imbalance. The pattern aligns with early-stage endocrine disruption and may indicate a PCOS-like profile.  

Risk Status: Moderate Risk  

Note: This assessment does not constitute a medical diagnosis. Clinical consultation is advised.

Example 2: Pregnancy + Metabolic Indicators

Input

BP: 150/95 mmHg  
Glucose: 145 mg/dL  
Gestational age: 32 weeks

✅UI OF FEMINA ✅
<img width="1401" height="719" alt="image" src="https://github.com/user-attachments/assets/3a1c75fe-c0d7-4154-81e3-2f66ad52e105" />

Output

Clinical Pattern: Elevated blood pressure with impaired glucose regulation  

Interpretation:
The combination of hypertension and elevated glucose levels during late pregnancy indicates a high-risk metabolic profile. This may align with preeclampsia-related risk patterns or gestational metabolic stress.  

Risk Status: High Risk  

Note: This assessment is non-diagnostic and intended for clinical support only.
Performance Summary

The system demonstrates stable and confident structured generation:
Adapter B: Perplexity ~2.99
Adapter A: Perplexity ~3.52

These values indicate strong token-level confidence and consistent adherence to structured output formats. Additionally, probabilistic outputs closely align with ground truth values, demonstrating reliable reasoning calibration.

Deployment and Efficiency

FEMINA is designed for practical deployment:

4-bit quantized base model
Lightweight LoRA adapters
~4 hours total training time
Single GPU compatibility
Modular architecture

This makes it feasible for use in resource-constrained clinical or research environments.

Future Direction
The next extension introduces multimodal reasoning, integrating imaging inputs such as ovarian and thyroid ultrasound data. This will enable combined image + structured clinical reasoning, extending FEMINA beyond text-based interpretation.
The Adapter testing and response fine tuning of the model is ongoing as to prevent hallucinations.

Disclaimer
FEMINA does not provide medical diagnoses.
It is intended strictly for:
Clinical decision support
Risk interpretation
Educational use

