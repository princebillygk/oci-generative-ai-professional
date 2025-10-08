# What is LM
LM: Is probabilistic model of text

## LLM Architecture

## Prompting and Training

## Encoder
Models that convert sequence of words to an  embedding (vectors representation)

The sent me a text -> Bert -> Vector representation

Examples: MiniLM, Embed-Light, Bert, RoBERTA, DistillBert, SBERT

# Decoder
Models take a sequence of words and output next word

Examples: GPT-4, Llama, BLOOM, Falcon

Used: 
Answer question, Genertic Dialogs

# Encoder - Decoder
encodes a sequence of words and use the encoding + to output a next word

### Usage of each

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
# Prompting

### Prompt engineering

This process of itteratively refining a prompt to get the expected result
### K-shot Prompting
1) Giving K Examples and then tell to do the next
```
Add 3 + 4 = 7
Add 6 + 5 = 11
Add 1 + 8 = ?
```

## Prompting Strategies

#### Chain of thoughts
Give a problem in small chunks rather than giving the whole problem together

#### Least to most
Easy First and the give harder

#### Step - Back  (Chemistry - Physics)
Questions: Higher level concept


## Issues with prompting
#### Prompt Injection (Issue Bad Prompt)
Tell LLM to ignore previous instruction and do whatever you want
#### Leaked Prompt
Leaking private information



# Training
### Domain Adaptation
Alter the parameters of model until you get exceptable answer


| Training Style      | Modifies            | Data                   | Summary                                                   |
| ------------------- | ------------------- | ---------------------- | --------------------------------------------------------- |
| Fine Tuning (FT)    | All Parameters      | Labeled, task-specific | Expensive                                                 |
| Param. Efficient FT | Few, new parameters | Labeled, task-specific | A small set of parameters and adding aditional parameters |
| Soft Prompting      | Few, new parameters | Labeled, task-specific | Learnable prompt params                                   |
| Cont. Pre Training  | All parameters      | unlabled               | Same as LLM pre-training                                  |
# Decoding

### Greedy Decoding
Pick the highest probability word at each step
(EOS) = End of sentence

### Non-Deterministic Decoding
Pick randomly among high probability candidates at each step
##### Temperature
When temperature is decreased, the distribution is more peaked around the most likely word


# Hallucination
Generated text that is non factual or ungrounded 
(Grounded means it is written somewhere in the document has source)

Decreasing techniques:
1. (K - shot helps a little bit)
2. RAG

# RAG
Input -> Transform question into query -> retrieve doc -> LLM

### Code Model
Copilot, Codex

# Multimodel
Trained on language image, audio
DALL-E 
Stable Diffusion



## Language Agents

#### ReAct
Leveraging LLMS for language agents
#### Toolformer
Pre-training technique where strings are replaced with calls to tools that yeild result. Use tools ad api for few tasks
