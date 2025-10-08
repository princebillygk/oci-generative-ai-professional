# Q/A
## What is Language Model (LM)?
A language model is a probabilistic model of text. 
	It guesses the next word based on the given sequence based on their probability.
## What is encoder?
Model that convert a sequence of word to embedding (vector representation)
Example: Examples: MiniLM, Embed-Light, Bert, RoBERTA, DistillBert, SBERT
	Embedding AI
		They sent me a -> 33242,234234,234,234234
## What is is decoder?
A language model that takes a sequence of word and output the next word.
Example:  GPT-4, Llama, BLOOM, Falcon
		They sent me a -> note
## What is decoding? 
The process of generating text with LLM
## What is temperature in decoding?
It is used to modulates the distribution over vocabulary as it flattened over all words
	More temperature means more random result
## What are the types of decoding
### Greedy Decoding
Always pick the highest probability word of each step
	Temperature: 0
### Non-Deterministic Decoding
Pick randomly among high probability candidates
	 Temperature: 0 < t < 1
## What is encoder-decoder?
Encodes a sequence of words and use the encoding + to output a next word
Example: T5, Ul2, BART
	Encoder + Decoder
## When to use encoder, decoder and encoder-decoder?

| Task                      | Encoders | Decoders | Encoder-Decoder |
| ------------------------- | -------- | -------- | --------------- |
| Embedding Text            | Yes      | No       | No              |
| Abstractive QA            | No       | Yes      | Yes             |
| Extractive QA             | Yes      | Maybe    | Yes             |
| Translation               | No       | Maybe    | Yes             |
| Creative Writting         | No       | Yes      | No              |
| Abstractive Summarization | No       | Yes      | Yes             |
| Extractive Summarization  | Yes      | Maybe    | Yes             |
| Chat                      | No       | Yes      | No              |
| Forecasting               | No       | No       | No              |
| Code                      | No       | Yes      | Yes             |
## What is prompting?
Simplest way to affect the distribution of vocabulary.
## What is prompt engineering?
The process of iteratively refining a prompt for the purpose of eliciting a particular style of response
## What are some types of prompt
### In-context learning
Conditioning the LLM with instruction or demonstration of tasks
### K-shot prompting
Explicitly providing k examples of the intended task in the prompt
	 5 & 1 = 5
	 5 & 2 = 10
	 5 & 3 = 15
	 5 & 4 = ?
### Chain of thoughts
Breaking a complex problem into steps
### Least to most
Give easier problem first then slowly increase complexity
### Step-back
Prompt to identify the high-level concepts related to the task
## Whats is grounded?
Text that has been generated from true source and have proper citation
	Groundedness means written somewhere in document
## What is ungrounded?
Text that has no source and no proper citation
## What are the issues with prompting?
### Prompt Injection
Give instruction to ignore other instruction, cause harm, or behave contrary to deployment exceptions
### Memorization
Leaking information using clever prompts
## Hallucination
Generating text that is non factual or ungrounded
## What is Domain Adaption
Training to enhance performance outside the domain/subject it was trained on
	 Keep updating the parameters of model until I get expected answers from generative AI

| Training Style      | Modifies            | Data                   | Summary                                                   | OCI Example |
| ------------------- | ------------------- | ---------------------- | --------------------------------------------------------- | ----------- |
| Fine Tuning (FT)    | All Parameters      | Labeled, task-specific | Expensive                                                 | Vanila      |
| Param. Efficient FT | Few, new parameters | Labeled, task-specific | A small set of parameters and adding aditional parameters | T-Few       |
| Soft Prompting      | Few, new parameters | Labeled, task-specific | Learnable prompt params                                   |             |
| Cont. Pre Training  | All parameters      | unlabled               | Same as LLM pre-training                                  |             |
# What are the Applications of LLM
### RAG (Retrieval Augmented Generation)
Model retrieve support documents before answering the prompt
### Code Models
Training on Code (programming) instead of language
Example: Co-pilot, Codex, Code Llama
### Multi-modal
Models trained on multiple modalities language, images, audio
Example: DALL-E, Stable Diffusion
### Language Agents
Using other tools with LLM to make it do actions

## What is Fine Tuning?
* Optimizing a pretrained foundation model on a smaller domain specific dataset
	* Improve Model Performance on specific tasks
	* Improve Model Efficiency
* Use when a pretrained model doesn't perform you task well or you want to teach it something new

## What are the features of OCI Generative AI
* Fully managed service
* Choice of models
* Flexible fine tuning
* Dedicated AI Clusters
## What models are available in OCI Generative AI
### command-r-plus
Entry level usecase, afford able
- 128k tokens prompt -> 4k response tokens
- Highly performant,
### command-r-16k
* Smaller and faster
* 16k prompt token -> 4k response tokens
### llama 3.1-70b/40b instruct
* largest publicly available model in the market
## What are the Fine tuning approach available in OCI
- **T-Few**: enables fast and efficient customization
## What are the features of OCI Dedicated AI Clusters
- **Security**: The GPU allocated for a customer's generative AI tasks are isolated from other GPU's
- Uses RDMA (Remote direct memory access) Cluster network for connecting GPU