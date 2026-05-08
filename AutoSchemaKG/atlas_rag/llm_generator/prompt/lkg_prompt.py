import json
import jsonschema
from typing import Any

ner_prompt =[
{"role": "system",
     "content": """
You are a domain analysis engine. You must provide keywords for searching relevant documents.
When given any academic question, follow these steps:  
1. **Identify Tested Skills:** Determine the *abstract knowledge/skills* required to solve the problem (e.g., "translating universal statements into predicate logic"), not the concrete entities in the question (e.g., "children," "school").  
2. **Extract Domain Specific Term:** Extract domain-specific technical terms (e.g., "school" in *educational policy*), exclude common nouns/verbs describing the question's *scenario* (e.g., "child," "school," "goes to").  
3. **Prioritize Formal Structures:** For logic/math problems, focus on notation rules (e.g., quantifier order, implication vs. conjunction), not scenario labels.  
4.  **Capture Rare Technical Terms:** Include uncommon domain-specific terms critical to the question (e.g., "epigenetics" in biology, "monad" in computer science), even if they appear infrequently in general language.
Your input fields are:
1. question (str): Query for keyword extraction
Your output fields are:
1. keywords (array): Extracted keywords in JSON format
All interactions will be structured as:
[[ ## question ## ]]
{question}
[[ ## keywords ## ]]
{keywords}
The output must be parseable according to JSON schema:
{"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"]}
"""},
     {
        "role": "user",
        "content": "[[ ## question ## ]]\nSolve \(x^2 - 5x + 6 = 0\)."
    },
    {
        "role": "assistant",
        "content": "{\"keywords\": [\"quadratic equation\", \"factoring\", \"roots\", \"algebraic manipulation\"]}"
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nExplain the socio-economic causes of the French Revolution."
    },
    {
        "role": "assistant",
        "content": "{\"keywords\": [\"historical causation\", \"class struggle\", \"economic inequality\", \"Enlightenment philosophy\"]}"
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nProve that the square root of 2 is irrational."
    },
    {
        "role": "assistant",
        "content": "{\"keywords\": [\"proof by contradiction\", \"irrational numbers\", \"number theory\", \"rational/irrational distinction\"]}"
    },
    {
        "role": "user",
        "content": "[[ ## question ## ]]\nExplain the implications of Heisenberg's uncertainty principle on quantum measurements."
    },
    {
        "role": "assistant",
        "content": "{\"keywords\": [\"Heisenberg uncertainty principle\", \"quantum mechanics\", \"measurement theory\", \"quantum state collapse\", \"observable quantities\"]}"
    }
]

simple_ner_prompt = [
    {"role": "system", "content": """You are an advanced AI assistant that always output in valid JSON format."""}
]
keyword_filtering_prompt = [
    {
        "role": "system",
        "content": """You are a precision-focused component of a knowledge retrieval system used by researchers and educators. 
Your task is to filter keywords based on their relevance to the **core domain knowledge** required to answer a given question. 
You must critically evaluate whether each item represents:
1. Academic concepts/methodologies (entities)
2. Critical processes or relationships (events)
The output should be in JSON format, e.g., {"keywords": ["concept1", "concept2"]}. If no keywords are relevant, return {"keywords": []}. 
Accuracy is crucial, as these keywords drive document retrieval for critical research. Do not generate any new keywords or explanations.
**Do not change the content of each object in the list. You must only use text from the candidate list and cannot generate new text.**
**Include all characters of the selected keywords.**
**Do not include any duplicate keywords.**
**The keywords can be a sentence describing a event as long as it is helpful for searching.**
---

**Input Fields:**
1. question (str): Query requiring knowledge analysis
2. keywords_before_filter (list): Candidate keywords to evaluate

**Output Field:**
1. keywords_after_filter (list): Filtered keywords in JSON format

---

**Interaction Structure:**
[[ ## question ## ]]
{question}
[[ ## keywords_before_filter ## ]]
{keywords_before_filter}
[[ ## keywords_after_filter ## ]]
{keywords_after_filter}

**JSON Schema Validation:**
{"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"]}
"""},
        {
        "role": "user",
        "content": """[[ ## question ## ]]
Explain the causes of the French Revolution.
[[ ## candidates ## ]]
["French Revolution", "social inequality", "Enlightenment ideas spreading", "economic crisis", "Bastille storming event"]"""
    },
    {
        "role": "assistant",
        "content": """{"keywords": ["social inequality", "Enlightenment ideas spreading", "economic crisis"]}"""
    },
    {
        "role": "user",
        "content": """[[ ## question ## ]]
Translate 'All children go to some school' into predicate logic.
[[ ## keywords_before_filter ## ]]
["predicate logic", "children", "school", "universal quantifier (∀)", "attendance"]"""
    },
    {
        "role": "assistant",
        "content": """{"keywords": ["predicate logic", "universal quantifier (∀)"]}"""
    },

    {
        "role": "user",
        "content": """[[ ## question ## ]]
Solve \(x^2 - 5x + 6 = 0\).
[[ ## keywords_before_filter ## ]]
["quadratic equation", "polynomial", "algebra", "roots", "classroom teaching"]"""
    },
    {
        "role": "assistant",
        "content": """{"keywords": ["quadratic equation", "polynomial", "roots"]}"""
    }
]


test_messages = [
    [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: What is the phenotype of a congenital disorder impairing the secretion of leptin?
            A. Normal energy intake, normal body weight and hyperthyroidism
            B. Obesity, excess energy intake, normal growth and hypoinsulinaemia
            C. Obesity, abnormal growth, hypothyroidism, hyperinsulinaemia
            D. Underweight, abnormal growth, hypothyroidism, hyperinsulinaemia"""
        }
    ],
    [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and five candidate answers (A, B, C and D), choose the answer.
            Question: Which of the following is notan advantage of stratified random sampling over simple random sampling?
            A. When done correctly, a stratified random sample is less biased than a simple random sample.
            B. When done correctly, a stratified random sampling process has less variability from sample to sample than a simple random sample.
            C. When done correctly, a stratified random sample can provide, with a smaller sample size, an estimate that is just as reliable as that of a simple random sample with a larger sample size.
            D. A stratified random sample provides information about each stratum in the population as well as an estimate for the population as a whole, and a simple random sample does not."""
        }
    ],  
    [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
            A. 0
            B. 4
            C. 2
            D. 6"""
        }
    ],  [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
            A. paralysis of the facial muscles.
            B. paralysis of the facial muscles and loss of taste.
            C. paralysis of the facial muscles, loss of taste and lacrimation.
            D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."""
        }
    ],  [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: What is true for a type-Ia ("type one-a") supernova?
            A. This type occurs in binary systems.
            B. This type occurs in young galaxies.
            C. This type produces gamma-ray bursts.
            D. This type produces high amounts of X-rays."""
        }
    ],   [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: _______ such as bitcoin are becoming increasingly mainstream and have a whole host of associated ethical implications, for example, they are______ and more ______. However, they have also been used to engage in _______.
            A. Cryptocurrencies, Expensive, Secure, Financial Crime
            B. Traditional currency, Cheap, Unsecure, Charitable giving
            C. Cryptocurrencies, Cheap, Secure, Financial crime
            D. Traditional currency, Expensive, Unsecure, Charitable giving"""
        }
    ],  [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: The access matrix approach to protection has the difficulty that
            A. the matrix, if stored directly, is large and can be clumsy to manage
            B. it is not capable of expressing complex protection requirements
            C. deciding whether a process has access to a resource is undecidable
            D. there is no way to express who has rights to change the access matrix itself"""
        }
    ],   [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: In the NoNicks operating system, the time required by a single file-read operation has four nonoverlapping components: disk seek time-25 msec disk latency time-8 msec disk transfer time- 1 msec per 1,000 bytes operating system overhead-1 msec per 1,000 bytes + 10 msec In version 1 of the system, the file read retrieved blocks of 1,000 bytes. In version 2, the file read (along with the underlying layout on disk) was modified to retrieve blocks of 4,000 bytes. The ratio of-the time required to read a large file under version 2 to the time required to read the same large file under version 1 is approximately
            A. 1:4
            B. 1:3.5
            C. 1:1
            D. 1.1:1"""
        }
    ],  [
        {
            "role":"system",
            "content":"You are a helpful asssistant. You will answer the question based on the context provided."
        },
        {
            "role": "user",
            "content":"""Query: Given the following question and four candidate answers (A, B, C and D), choose the answer.
            Question: Which of the following propositions is an immediate (one-step) consequence in PL of the given premises? ~E ⊃ ~F G ⊃ F H ∨ ~E H ⊃ I ~I
            A. E ⊃ F
            B. F ⊃ G
            C. H ⊃ ~E
            D. ~H"""
        }
    ],      
]