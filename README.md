# LLM & Generative kAI Q&A Reference Guide

## Language Models Fundamentals

### What is a Language Model (LM)?

A language model is a probabilistic model of text that predicts the next word in a sequence based on probability distributions of previous words.

### What is an Encoder?

A model that converts a sequence of words into embeddings (vector representations).

**Examples:** MiniLM, Embed-Light, BERT, RoBERTa, DistilBERT, SBERT

**Example Output:**

- Input: "They sent me a"
- Output: [33242, 234234, 234, 234234] (embedding vector)

### What is a Decoder?

A language model that takes a sequence of words and outputs the next word.

**Examples:** GPT-4, Llama, BLOOM, Falcon

**Example Output:**

- Input: "They sent me a"
- Output: "note"

### What is Decoding?

The process of generating text with a Large Language Model (LLM).

### What is Temperature in Decoding?

A parameter that modulates the probability distribution over the vocabulary, flattening it across all words.

- **Higher temperature** = more random/creative results
- **Lower temperature** = more deterministic/focused results

### Types of Decoding

#### Greedy Decoding

- Always picks the highest probability word at each step
- Temperature: 0

#### Non-Deterministic Decoding

- Randomly picks among high-probability candidates
- Temperature: 0 < t < 1

### What is an Encoder-Decoder?

Encodes a sequence of words and uses the encoding to output the next word.

**Examples:** T5, UL2, BART

---

## When to Use Each Model Type

|Task|Encoder|Decoder|Encoder-Decoder|
|---|---|---|---|
|Embedding Text|✅|❌|❌|
|Abstractive QA|❌|✅|✅|
|Extractive QA|✅|⚙️|✅|
|Translation|❌|⚙️|✅|
|Creative Writing|❌|✅|❌|
|Abstractive Summarization|❌|✅|✅|
|Extractive Summarization|✅|⚙️|✅|
|Chat|❌|✅|❌|
|Forecasting|❌|❌|❌|
|Code|❌|✅|✅|

**Legend:** ✅ = Yes, ❌ = No, ⚙️ = Maybe/Possible

---

## Prompting Techniques

### What is Prompting?

The simplest way to affect the probability distribution of vocabulary output.

### What is Prompt Engineering?

The process of iteratively refining a prompt to elicit a particular style of response.

### Types of Prompting

#### In-Context Learning

Conditioning the LLM with instructions or demonstrations of tasks.

#### K-Shot Prompting

Explicitly providing K examples of the intended task in the prompt.

**Example:**

```
5 & 1 = 5
5 & 2 = 10
5 & 3 = 15
5 & 4 = ?
```

#### Chain of Thoughts

Breaking a complex problem into sequential steps.

#### Least to Most

Providing easier problems first, then gradually increasing complexity.

#### Step-Back

Prompting to identify high-level concepts related to the task.

---

## Grounding and Issues

### What is Grounded?

Text that has been generated from a true source with proper citations. Groundedness means the information is written somewhere in a document.

### What is Ungrounded?

Text that has no source and no proper citations.

### Issues with Prompting

#### Prompt Injection

Giving instructions to ignore other instructions, cause harm, or behave contrary to deployment expectations.

#### Memorization

Leaking information using clever prompts.

#### Hallucination

Generating text that is non-factual or ungrounded.

---

## Domain Adaptation

### What is Domain Adaptation?

Training to enhance performance outside the domain/subject the model was originally trained on. This involves continuously updating model parameters until expected answers are generated.

### Training Styles

|Training Style|Modifies|Data|Summary|OCI Example|
|---|---|---|---|---|
|Fine-Tuning (FT)|All Parameters|Labeled, task-specific|Expensive but comprehensive|Vanilla|
|Parameter Efficient FT|Few, new parameters|Labeled, task-specific|Adds a small set of additional parameters|T-Few|
|Soft Prompting|Few, new parameters|Labeled, task-specific|Learnable prompt parameters|-|
|Continued Pre-Training|All parameters|Unlabeled|Same as LLM pre-training|-|

---

## Applications of LLM

### RAG (Retrieval Augmented Generation)

Model retrieves supporting documents before answering the prompt.

### Code Models

Trained on programming code instead of natural language.

**Examples:** Co-pilot, Codex, Code Llama

### Multi-Modal

Models trained on multiple modalities: language, images, audio.

**Examples:** DALL-E, Stable Diffusion

### Language Agents

Using external tools with LLM to perform actions.

---

## Generative AI Parameters

### Preamble Override (`SYSTEM`)

Initial context or guiding message for a chat model that sets the model's overall chat behavior and conversation style.

### Max Output Tokens

Maximum number of tokens generated per response.

### Temperature

Controls the randomness of LLM output. Use 0 to generate the same output for a prompt every time.

### Top-K

Ensures only the top K most likely tokens are considered in generated output. 0 means disabled.

- Picks the next token from the top K tokens in the distribution.

### Top-P (Nucleus Sampling)

Ensures only the most likely tokens with cumulative probability P are considered for generation at each step.

- Picks tokens where: `t1_weight + t2_weight + ... = P`

### Frequency Penalty

Assigns penalty when a token appears frequently.

### Presence Penalty

Assigns penalty when a token appears even once.

---

## OCI Generative AI Features

### Core Features

- Fully managed service
- Choice of models
- Flexible fine-tuning
- Dedicated AI Clusters

### Available Models

#### command-r-plus

- Entry-level use case, affordable
- 128k prompt tokens → 4k response tokens
- Highly performant

#### command-r-16k

- Smaller and faster
- 16k prompt tokens → 4k response tokens

#### llama 3.1-70b/405b instruct

- Largest publicly available model in the market

### Embedding Models

- embed-english-v3
- embed-multilingual-v3
- embed-english-light-v3
- embed-multilingual-light-v3
- embed-english-light-v2

### Fine-Tuning Approaches

- **T-Few:** Enables fast and efficient customization (PEFT)
- **LoRA:** Parameter-Efficient Fine-Tuning (PEFT)

### Dedicated AI Clusters Features

- **Security:** GPUs allocated for a customer's generative AI tasks are isolated from other GPUs
- Uses RDMA (Remote Direct Memory Access) Cluster network for connecting GPUs

---

## Embeddings

### What are Embeddings?

Numerical representations of text converted to number sequences by an Embedding Model.

- Groups similar words together in a multidimensional space
- Semantically similar items have closer values

### Embedding Distance Metrics

1. Dot product
2. Cosine similarity

---

## Model Customization

### Steps for Model Customization

1. Start with simple prompt
2. Add few-shot prompting
3. Add simple retrieval using RAG
4. Fine-tune the model
5. Optimize the retrieval on fine-tuned model

### What is Fine-Tuning?

Optimizing a pretrained foundation model on a smaller domain-specific dataset to:

- Improve model performance on specific tasks
- Improve model efficiency

**Use when:** A pretrained model doesn't perform your task well or you want to teach it something new.

---

## Retrieval Augmented Generation (RAG)

### What is RAG?

A method for generating text using additional information fetched from an external data source.

- Language model queries enterprise knowledge bases
- Overcomes model limitations
- Mitigates bias in training data
- Effective when supplementing with specific information not present in training data

### RAG Workflow

**Ingestion:**

1. Document → Chunks → Embedding → Database

**Retrieval:** 2. Query → Index → Top K Results → Generation

### RAG Ingestion Steps

1. **Load the document**
    
    - Formats: PDF, CSV, JSON, Web Pages, Markdown
    - Uses loader classes
2. **Break documents into chunks**
    
    - Separators: ["\n\n", "\n", ""]
3. **Convert chunks to embeddings**
    
4. **Index embeddings into database**
    

---

## Model Customization Comparison

|Method|Description|When to Use|Pros|Cons|
|---|---|---|---|---|
|K-Shot Prompting|Provide few examples in the prompt before the actual request|LLM already understands topics necessary for text generation|Very simple, no cost|Adds latency to each model request|
|Fine-Tuning|Adapt a pretrained LLM to perform a specific task on private data|LLM doesn't work well on task; data is too large; latency is too high|Increases model performance on specific tasks; no impact on latency|Requires labeled dataset (expensive and time-consuming)|
|RAG|Optimize LLM output with targeted information without modifying model|Data changes rapidly; want to mitigate hallucinations|Accesses latest data; grounds results; no fine-tuning required|More complex to set up; requires compatible data source|

---

## OCI Infrastructure Components

### What is a Model Endpoint?

A designated point on a dedicated AI Cluster where an LLM can accept user requests and send back responses (e.g., generated text).

### Cluster Types Available in OCI

- **Fine-tuning:** Used for training a pretrained foundational model
- **Hosting:** Used for hosting a custom model endpoint for inference

### Dedicated AI Cluster Units

1. **Large Cohere Dedicated:** Fine-tuning and hosting
2. **Small Cohere Dedicated:** Fine-tuning and hosting
3. **Embed Cohere Dedicated:** Embedding tasks
4. **Large Meta Dedicated:** Meta Llama models

---

## Fine-Tuning Parameters

### T-Few Fine-Tuning Training Parameters

1. **Total Training Epochs:** Number of iterations through entire training dataset
2. **Training Batch Size:** Number of samples processed before parameter update
3. **Learning Rate:** Rate at which model parameters are updated after each batch
4. **Early Stopping Threshold:** Minimum improvement in loss required to prevent premature termination
5. **Early Stopping Patience:** Tolerance of stagnation in loss before stopping training
6. **Log Model Metrics Intervals:** Frequency of logging model metrics

### Fine-Tuning Output Parameters

1. **Accuracy:** Ratio of correct tokens (max = 1)
2. **Loss:** Ratio of incorrect responses (lower is better)

### Training Data Format

```json
[
   {"prompt": "question or input", "completion": "expected output"},
   {"prompt": "another question", "completion": "another output"}
]
```

---

## LangChain

### What is LangChain?

A wrapper/framework for communication with generative AI systems, simplifying integration and interaction.

---

## AI Agents

### Knowledge Base for Agents

1. Vector-based storage
2. Stores ingested data

### What is a Session in AI Agents?

A series of exchanges where the user sends queries or prompts and the agent responds with relevant information.

### What are Endpoints in AI Agents?

Specific points of access in a network or system that agents use to interact with other systems or services.

### What is Trace in AI Agents?

A feature to track and display conversation history (both original prompts and generated responses) during a chat conversation.

### What is Citation in AI Agents?

The source of information for the agent's response, providing transparency and verifiability.

---

## Content Moderation

### What is Content Moderation?

A feature designed to detect or filter out toxic, violent, abusive, hateful, threatening, insulting, and harassing phrases from:

- Generated responses
- User prompts in large language models

---

## Data Storage Guidelines

### Requirements

1. Data must be uploaded as files to Object Storage bucket
2. Only one bucket per data source
3. Maximum 100MB file size
4. Maximum 8MB multimedia content in PDF
5. Supported formats: PDF, Text

---

## Steps to Create AI Agents

### Process

1. **Create a Knowledge Base**
2. **Create Agent**
3. **Create Agent Endpoint**
    - Maximum: 3 endpoints per agent

---

## Quick Reference Summary

### Key Concepts to Remember

- **Encoders** = Text to vectors (embeddings)
- **Decoders** = Vectors to text (generation)
- **Temperature** = Randomness control (0 = deterministic, higher = creative)
- **RAG** = Retrieval + Generation (grounds responses in data)
- **Fine-Tuning** = Customize model on your data
- **Grounding** = Based on real sources with citations
- **Hallucination** = Ungrounded, incorrect output

### Decision Framework

1. Need embeddings? → Use **Encoder**
2. Need text generation? → Use **Decoder**
3. Need translation/summarization? → Use **Encoder-Decoder**
4. Data changes frequently? → Use **RAG**
5. Need task-specific performance? → Use **Fine-Tuning**
6. Simple customization? → Use **Few-Shot Prompting**

### More Resources
[Tricky and important questions](./Tricky%20and%20Important%20questions.md)
