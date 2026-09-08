# RAG Evaluation — Notes

## Setup

```bash
uv add langchain-openai langchain-chroma langchain-text-splitters langchain-core chromadb openai sentence-transformers deepeval pytest python-dotenv langchain-cohere
```

build the retriever 1st

```
query->embediing model->vector->vector db->chunks->find->come and give->embed->text->return
```

## Failures during retrivation [failure metric]

1. retriver brother getting different context which are not needed, like need 1,2 getting 9, 10-waste
   --->to evaluate this, wee need a matrix called RECALL
   RECALL=out of all relevant context how many from vector db our retriever can bring,,,example 3/5

2. retriver brother geeting some irrelevant and some relevant context, irrelevent ones are called noise--noisey context,,like 1,2 needed, getting 1,4,5---wasted
   --->to evaluate this, wee need a matrix called PRECISION
   PRECISION = out of brought context how many of them are correct?

# to increase recall we need to increase value of K ex: 5->10, but then precision reduces,,thats a trade off between 2
# as we need the golden dataset to check this, so this 2 metrices are REFERENCE BASED EVALS

![alt text](image-3.png)

as we are chaning chunking size k
value to get good values of recall preciosn, its making change in golden dataset, it need to rebuilded agin for this precess, which is not good thing, each time golden dataset creation is not worth it.

so we gonnma make the golden dataset like this->
![alt text](image.png)
![alt text](image-1.png)
this technique is used by DEEPEVAL->this kinda eval is called CONTEXTUAL EVAL

now, as per this methond lets see how precisiion is calculated...
![alt text](image-4.png)
deepeval precison is called->contextual precison->calculated like this
![alt text](image-2.png)

we nned to check the the rank of the retrived chunk, better rank better precison, better eval

## ways to create golden dataset
1. hand authorisd
2. llm assisted->thn review by human
3. deep eval synthesizer and human review
4. production log mining

llm test case = one row of golden dataset
metric = something that measure its relevance and threshold makes the model fail or pass
evaluate = send all test cases and get final reslut with the metric

so in summary in deep eval 3 things ->
   1. llm test case
   2. metric
   3. evaluate

now to imporve precison score we will implement reranker

### RAG Evaluation Optimization

1. **Baseline** – measure current Recall & Precision.
2. **Chunk Size** – test 500, 750, 1000, 1500; choose best.
3. **Chunk Overlap** – test 0, 50, 100, 150; balance context and redundancy.
4. **Embedding Model** – compare models; re-index after changing.
5. **Initial Top-K** – test 3, 5, 10, 20; higher K usually improves Recall.
6. **Add Reranker** – compare Precision/Recall before vs after.
7. **Reranker Candidate Pool** – test 10, 20, 30 candidates.
8. **Final Top-N** – test 3, 5, 8, 10.
9. **Hybrid Retrieval** – combine vector + BM25.
10. **Metadata Filtering** – remove irrelevant documents.
11. **Query Rewriting/Multi-query** – improve Recall.
12. **Analyze failures** – inspect low Recall/Precision and fix the actual bottleneck.

the result without reranker->
![alt text](image-5.png)

the rsult with reranker->
![alt text](image-3.png)

this happened because, the reranker externally reranked some chunks, what instead of increasing the precisson reduced it

what we are building->
![alt text](image-6.png)

Retriver = finnds relevant chunks from vectordb from query and gives
Generator = take the output of retriver and query and create the answer and gives to user

where generator fails->
1. unfaithful response = other than context llm includes its own knowledge
   Faithulness - as a matrix we will use to resolve
2. retriever gets relevant chunk, but generator making not relevant anser
   Relevence - as a metrix to check the relevence of a generator

n eval_generator.py, the interaction between the generator and the evaluator (using an LLM-as-a-Judge) follows a structured testing pipeline. The script utilizes a golden dataset (stored as a JSON file, typically exported from the ChromaDB vector store) to provide ground truth for evaluation (32:45, 36:06).

How they work together:

Input Preparation: For each entry in the golden dataset, the script extracts the question and the ideal (golden) context (36:36, 37:19).
Generator Execution: The question and the golden context are passed to the generate function from generator.py (36:47). The generator processes these inputs using a Large Language Model (like GPT-4o mini) to produce an answer (04:17, 36:55).

Evaluation Loop: The evaluator, powered by the DeepEval framework, takes the generated answer and the golden context (37:16). It acts as an LLM-as-a-Judge to perform two primary checks:
    Faithfulness: It breaks the answer down into atomic claims and verifies if each claim is directly supported by the provided context, detecting hallucinations (15:33, 37:23).

    Answer Relevancy: It analyzes the generated answer against the original query to ensure the response remains on-topic and helpful (23:55, 37:25).

    Optimization: By aggregating these scores, the script provides a clear performance baseline, allowing for the iterative refinement of the generator's system prompt to minimize errors (37:41, 42:13).

    generator result using golden dataset
![alt text](image-7.png)

Improving Faithfulness
Faithfulness measures whether the generated answer is strictly grounded in the provided context, without hallucinations or external knowledge (08:07, 15:33). To improve this:

Refine System Instructions: Add explicit constraints to the system prompt, such as "Use only information present in the context" and "Do not add outside knowledge" (44:35).
Control Claims: Instruct the generator not to overstate or strengthen claims, and to maintain the specific nuance provided in the source material rather than synthesizing generalized or separate methods (44:51).
Improving Answer Relevancy
Answer relevancy measures if the output directly addresses the user's query rather than just being factually faithful but off-topic (12:07, 14:48). To improve this:

Analyze Failures: Identify specific test cases where the answer was faithful but irrelevant (e.g., ignoring the user's specific question) and use those to add clarifying rules to the system prompt (44:13).
Refine Prompt Logic: Update the system prompt to force the model to prioritize synthesizing and rephrasing information specifically to answer the user's question, rather than just restating the context (45:07).
Iterative Refinement: Run multiple evaluation cycles using tools like DeepEval to measure improvements, adjusting the prompt rules after observing why previous test cases failed

![alt text](image-8.png)

rag pipeline level
![alt text](image-9.png)

summary :
# RAG Eval Metrics — Definitions

## Precision
Of what you retrieved, how much was actually relevant?
```
Precision = (relevant items retrieved) / (total items retrieved)
```
*Requires ground-truth relevance labels.*

## Recall
Of what was actually relevant, how much did you retrieve?
```
Recall = (relevant items retrieved) / (total relevant items that exist)
```
*Requires ground-truth relevance labels.*

## Contextual Relevancy
Of everything retrieved, what fraction is actually relevant to the query? (LLM-judged precision on retrieved context — no ground truth needed.)
1. Break each retrieved chunk into statements.
2. Judge each statement: relevant to query? yes/no.
3. Score = relevant statements / total statements.

## Faithfulness
Of everything the generated answer claims, how much is actually backed by the retrieved context? (Catches hallucination.)
1. Break the answer into atomic claims.
2. Judge each claim: supported by context? yes/no/idk.
3. Score = supported claims / total claims (idk excluded).

## Answer Relevancy
Does the answer actually address the original question? (Catches off-topic or rambling answers, not hallucination.)
1. Generate hypothetical questions the answer would be a good response to.
2. Judge each generated question: aligns with the original query? yes/no.
3. Score = aligned questions / total generated questions.

llm as a judge does not work because wactimme, the answer generated by the llm, varies, so its not a areliable for judging

to resolve this problem we have G Eval
![alt text](image-10.png)
judge prommpt->
![alt text](image-11.png)
just 2 thing in GEVAL what creates the difference->>>>
innovation 1 = 
   it converts a high level  criteria into evaluation step through COT
   so we dont give llm toomuch scope  to shinnk, we just increase steps of instructions, so it work determenisticly

innvoation 2 =
   we just dont use llm's output to judge  the product, instead we go on syep further by checking the log probability
   from llm final layer 10k tokens -> pull TOP 5 tokens->get weighted average-> then score them
![alt text](image-12.png)
   summ of probabily of top 5 tokens-> divide all tokes by sum ->got each results->then multiply each probabs, with their result ->now sum this new reslts =  = weighted average

   

this workflow helps to kepp the evaluatuation not varying in each time

## RAG Application Metrics

- **Correctness** → Is the answer **factually correct**?
  - Compares Actual vs Expected Answer.
  - **Truth / Accuracy**

- **Completeness** → Did the answer cover the **important points**?
  - Checks coverage of Expected Answer.
  - **Coverage**

- **Style** → Is the answer presented in the **desired way**?
  - Checks tone, clarity, explanation, etc.
  - **Presentation**

### How DeepEval Calculates

`Actual + Expected Answer`
→ **GEval**
→ **LLM Judge (Cohere)**
→ Judge follows **evaluation steps + rubric**
→ gives **0–10 score**
→ DeepEval **normalizes to 0–1**

`Score ≥ 0.70 → PASS`  
`Score < 0.70 → FAIL`

**In short:**  
Correctness = **Truth** | Completeness = **Coverage** | Style = **Presentation**
here is the test result :=
 ![alt text](image-13.png)

 metric in deep eval called toxicity
 The video outlines several strategies to improve the toxicity score of an LLM-based application if the initial results are unsatisfactory (58:48 - 1:01:07):

Switch to a better model: Upgrade to a more advanced, state-of-the-art model that is already well-aligned and inherently less prone to producing toxic content.
System Prompt Refinement: Explicitly update the system prompt to define the assistant's desired tone, specifically instructing it to avoid insults, taunts, or inappropriate analogies.
Implement Guardrails: Use input guardrails to analyze and filter queries before they reach the model, or apply output guardrails to screen and filter responses before they are displayed to the user.
Systematic Fine-tuning: As a final resort, if other methods fail, you can fine-tune the model on a specialized dataset to systematically remove toxic behaviors from its output.

results after toxicity eval
![alt text](image-14.png)

![alt text](image-15.png)
The video suggests several approaches to mitigate data leakage and improve security scores in an LLM application (1:11:49 - 1:15:10, 1:15:10 - 1:18:00):

System Prompt Hardening: Incorporate explicit instructions into the system prompt that forbid the model from revealing sensitive information, such as system prompts, internal configuration, or personal user data (1:15:10 - 1:16:15).

Structuring Evaluation Pipelines: Use multi-evaluator setups to specifically audit for different types of leakage, including prompt leakage, corpus/knowledge base leakage, and Personally Identifiable Information (PII) leakage (1:07:20 - 1:11:49).

Implementing Guardrails: Employ dedicated guardrails to filter output or retrieved context before it reaches the end user, ensuring no sensitive data is surfaced in the response (1:11:49 - 1:12:20).
![alt text](image-16.png)
its the result during leakage
![alt text](image-17.png)
![alt text](image-18.png)
To improve the scope adherence of an LLM-based application, the video outlines a structured approach involving defining clear boundaries, building a test dataset, and refining the system instructions. Here are the key steps:

Define a Strict Safety Policy: Explicitly state what the assistant is allowed to answer and what it must refuse. For example, if your bot is a course assistant, define its scope to only answer questions related to your specific learning material (1:23:34 - 1:24:49).
Create a Diverse Evaluation Dataset: Develop a 'golden' dataset containing a mix of benign (valid) questions, adversarial (out-of-scope) questions, and mixed queries (which combine a valid question with an out-of-scope request). This helps identify if the model properly distinguishes between what it should and should not handle (1:24:52 - 1:25:56).
Implement Rigorous Evaluation: Use custom metrics, such as a G-Eval framework, to evaluate the model's responses against your defined scope policy. The metric should check if the model answers the valid portion of a query while refusing the irrelevant or out-of-scope portions (1:26:00 - 1:28:13).
Refine System Prompts: Use the insights from your failures to harden your system prompt. By adding clear instructions and constraints based on real-world edge cases discovered during evaluation, you can guide the LLM to better maintain its operational boundaries (1:31:19 - 1:32:00).
Iterate with CI/CD: Instead of making ad-hoc changes, treat these improvements as part of a regression testing cycle where you run the full evaluation suite to ensure that changes intended to fix scope drift don't negatively impact other performance metrics (1:34:00 - 1:35:14).
result:
![alt text](image-19.png)

Operations evals
![alt text](image-20.png)


![alt text](image-21.png)
p95 = 95% of requests are done in this time period
![alt text](image-22.png)

Improving latency in a RAG system involves optimizing different stages of your pipeline. Based on the session, here are the key strategies to improve system speed:

Model Selection: Choosing a more efficient, smaller model can significantly reduce generation time compared to large, complex models (1:04:20).
Context Optimization: Reducing the context size passed to the model by using smaller chunks or applying contextual compression helps the model process information faster (1:04:31).
Prompt Engineering: Making your system prompt more efficient and concise ensures the model spends less time processing unnecessary instructions (1:04:54).
Caching: Implementing prompt caching allows you to reuse results for recurring queries, which drastically cuts down on redundant processing (1:05:24).
Streaming: Utilizing streaming architectures (as discussed for TTFT - Time to First Token) allows users to see the beginning of the answer immediately, improving the perceived latency even if the total time remains similar (23:00).
Infrastructure Tuning: For production, ensuring your system handles cold starts efficiently and managing your load through horizontal scaling or optimizing your hosting environment are essential (24:45).

To reduce costs in a RAG system, you should primarily focus on optimizing your LLM token usage, as this is typically the most significant expense. Here are the key strategies discussed in the video (1:04:20):

Reduce Context Size: Minimize the amount of information sent to the model by using smaller chunks or contextual compression (1:04:31). Sending fewer tokens directly lowers input costs.
Optimize Prompts: Carefully refine your system prompts to be more efficient and concise, ensuring you aren't paying for unnecessary instructions (1:04:54).
Implement Caching: Use prompt caching to reuse results for recurring queries, which effectively avoids redundant API costs (1:05:24).
Model Selection: Switch to a cheaper, smaller model where high-end capabilities are not strictly required, or use model routing to match query complexity with the appropriate model (1:05:21).
Enforce Output Limits: Instruct the model to provide concise answers or set a hard cap on the output word count to prevent runaway token generation (1:05:11).

![alt text](image-23.png)
![alt text](image-24.png)

To improve the **reliability** of a *RAG* application, the focus must be on minimizing errors, handling failures gracefully, and testing at scale. Key strategies discussed in the video include:

* **Categorize Failures:** Instead of a generic error rate, break down failures by their source (e.g., *LLM API issues*, *retriever failures*, *rate limits*, or *timeouts*). This granular tracking allows for targeted fixes (1:10:10).
* **Use Robust Error Handling:** Implement comprehensive `try-except` blocks throughout your pipeline to catch and categorize errors at each stage—from retrieval and re-ranking to generation (1:11:36).
* **Stress Testing:** A pipeline that works for a single user may fail under heavy load. You must measure reliability under **concurrent user requests** to identify performance bottlenecks that only appear at scale (1:11:54).
* **Increase Sampling:** Use a sufficiently large and diverse set of queries (simple, complex, and long-context) to ensure your reliability metrics are representative of real-world usage rather than just ideal, small-scale test runs (1:12:38, 1:13:04).
* **Implement Retries:** Configure smart retry logic for transient errors, while keeping track of the *retry rate* to understand how often your system requires multiple attempts to successfully serve a request (1:09:42).