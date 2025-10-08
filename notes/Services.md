You can create a generative AI service
# Features for OCI Generative AI
* Fully managed service
* You have choice of models
* Flexible fine tuning
* Dedicated AI Clusters: GPU based compute resources

# Pre-trained Foundational Models
## Chat Models
Ask questions and get conversational response
#### command-r-plus
Entry level usecase, afford able
- 128k tokens prompt -> 4k response tokens
- Highly performan,
#### command-r-16k
* Smaller and faster
* 16k prompt token -> 4k response tokens
#### llama 3.1-70b/40b instruct
* largest publicly available model in the market

### Embedding Model
Convert text to Vector of numbers =>  Semantic Search
* Semantic search (search based on the meaning of text rather than the keyword or lexical search)
* Multilingual search (supports multiple language)
* Give each token different numbers for each of the features
* Semantic similarity: Dot product, Cosine (Find numerical similarity because it will also be semantically similarity)
# Fine Tuning
* Optimizing a pretrained foundation model on a smaller domain specific dataset
	* Improve Model Performance on specific tasks
	* Improve Model Efficiency
* Use when a pretrained model doesn't perform you task well or you want to teach it something new
* T-Few tuning (Cohere) enables fast and efficient customizations
## Preamble override
A set of initial instruction for the model
### OCI
* There is a option preamble override in chat parameters
### Ollama 
 * `SYSTEM`
```Modelfile
FROM gemma3:latest

PARAMETER temperature 1
PARAMETER num_ctx 4096

SYSTEM You are a travel advisor with a pirate tone

```

## Parameters

### OCI
* OCI have inputs
	* Max output tokens
	* Preamble override
	* Temperature
		* Controls the randomness of the LLM Output
		* 0 makes the model deterministic
	* Top K
		* Will pick from top k tokens
	* Top P
		* Pick based on the sum of the token values
	* Frequency and Presence Penalties
		* To avoid repetition in output
		

### Ollama
* `PARAMETER`
	

## Embedding

Converting list of strings to vector
### OCI
* Have UI for converting list of strings to vector and see project in 2D though the project is in many dimention and cannot be represent visually.
### Ollama
Ollama we can use different embedding model to generate vector and then use it to work on that vector datacan 

```shell
curl http://localhost:11434/api/embed -d '{
  "model": "mxbai-embed-large",
  "input": "Llamas are members of the camelid family"
}'
```

Read more details here:
https://ollama.com/blog/embedding-models

# Dedicated AI Clusters

1) Compute engine with GPU

# Token
A word can be one or multiple token
apple -> 1 token
friendship -> 2 token (friend + ship)


# Cost of training model
* 1M per 1B token
* Bad idea to start from scratch

# Customizing LLM

| Method           | Description                                                                                   | When to use                                                                                                                        | Pros                                                                              | Cons                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| K-shot Prompting | Provide few examples in the prompt before the actual request                                  | LLM already understands topic that are necessary for the text generation                                                           | Very simple no costing                                                            | Adds latency to each model request                                               |
| Fine Tuning      | Adapt a pertained LLM to performance a specific task on private data                          | LLM doesn't work well on particula task<br>Data required to adapt the LLM is too large<br>Latency with the current LLM is too high | Increase model performance on a specific task<br>no impact on latency             | Requires a labeled data set which can be expensive and time consuming to require |
| RAG              | Optimize the output of a LLM with targeted information without modifying the underlying model | When data changes rapidly,<br>And when you want to mitigate hallucinations                                                         | Access the latest data<br>Grounds the result<br>Does not require fine tuning jobs | More complex to setup Requires a compatible data source                          |
![[Pasted image 20251002151158.png]]

## Steps of customization
1) Start with a simple prompt
2) Add few shot prompting
3) Add simple retrieval using RAG
4) Find tune the model
5) Optimize the retrieval on find tuned model


## Model endpoint:
Where a large language model can accept user requests and send back responses such as the model's generated text

## Cluster Types
1) Training cluster: For fine tuning and training model
2) Hosting: for hosting existing model

# Fine tuning
### Vanilla fine tuning: 
Updating the weights of all the layers in the model, requiring longer training time and higher serving costs
### T-few Fine tuning
T-Few fine tuning selectively updates only a fraction of the model's weights


# Training Method for smaller adjustment
1) T-Few: PEFT
2) LoRA: PEFT (Leaves the main model unchanged just add few more gears)

## T-Few input Parameters

	1) Total Training Epoch
	2) Training batch size
	3) Learning Rate
	4) Early stopping paitience
	5) Log model metrix intervals in steps
## Fine tunning file json format
```json
[
   {prompt: '', 'completions': ''}
   ...
]
```

## Fine tuning input parameters
1) Accuracy (ratio of correct token) - max (1)
2) Loss (ratio of incorrect response)


