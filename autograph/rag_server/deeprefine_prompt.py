REAFINER_FILTERING_SYSTEM_PROMPT = """
As an advanced filtering assistant, your task is to filter the given triples based on the given query.

Output your filtering result in the following format:
<filtering>[{"subject": "...", "relation": "...", "object": "..."}, {"subject": "...", "relation": "...", "object": "..."}, ...]</filtering>

**Important Filtering Rules:**
1. **Be liberal in keeping triples**: When in doubt, KEEP the triple. It's better to keep potentially relevant triples than to remove them.
2. **Entity matching rule**: If ANY entity (subject, object) in a triple appears in the query (exact match or semantically similar), you MUST keep that triple.
3. **Semantic relevance**: Keep triples that are semantically related to the query, even if the connection is indirect.
4. **Only remove clearly irrelevant triples**: Only filter out triples that are completely unrelated to the query topic and contain no entities or concepts mentioned in the query.
5. You must think carefully about the query and the triples before making your filtering. And output your filtering result directly in the specified format. AND YOUR OUTPUT TRIPLES CAN NOT BE EMPTY.
"""

REAFINER_FILTERING_USER_PROMPT = """
Filter the triples based on the given query. **Be very conservative in filtering - only remove triples that are clearly and completely irrelevant to the query.**

**Key guidelines:**
- Keep ALL triples where the subject or object entity appears in the query
- Keep triples that are semantically related to the query topic
- When uncertain about relevance, KEEP the triple
- Only remove triples that are completely unrelated to the query

Query: {query}
Triples: {triples_string}
Filtered Triples:
"""

REAFINER_JUDGEMENT_SYSTEM_PROMPT = """
As an advanced judgement assistant, your task is to judge whether the given question is answerable based on the provided KG context.

Evaluate whether the given question is answerable based on the provided KG context. Output your judgment in the following format:
<judge>Yes</judge> or <judge>No</judge>

**Important:** You must think carefully about the question and the KG context before making your judgment. And output your judgment result directly in the specified format.
"""

REAFINER_JUDGEMENT_USER_PROMPT = """
Question: {question}
Knowledge Graph (KG) context: {triples_string}
"""

REAFINER_EXPANDED_JUDGEMENT_USER_PROMPT = """Additional Knowledge Graph (KG) context:
{triples_string}

Using the original question and all KG context in this conversation, judge answerability again.
Output only <judge>Yes</judge> or <judge>No</judge>."""

REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT = """
As an advanced error abduction assistant, your task is to analyze the error reasons based on the given interaction history.

Analyze the reasons of the unanswerable questions based on the given interaction history from the incompleteness, incorrectness, and redundancy perspectives. Output your analysis in the following format:
<abduction>...</abduction>

**Important:** You must think carefully about the interaction history before making your analysis. And output your analysis result directly in the specified format.
"""

REAFINER_ERROR_ABDUCTION_USER_PROMPT = """
Interaction history: {interaction_history}
"""

REAFINER_ABDUCTION_USER_PROMPT = """The KG expansion stage is complete.
Using the original question, all KG context, and judgement results already present in this conversation,
analyze why the question was initially unanswerable."""

REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT = """
As an advanced knowledge graph refinement assistant, your task is to generate a series of actions (**within 10 actions**) to refine the given KG to make it more suitable for answering the given question.

Based on the given KG and the analysed error reasons, refine the given KG to make it more easily for retrieval and answering the given question. You have the following three types of actions to conduct:

- insert_edge(subject, relation, object): Insert a new edge into the KG to complete the missing information.
- delete_edge(subject, relation, object): Delete an edge from the KG to remove the redundant information or conflicting information.
- replace_node(old_entity, new_entity): Replace an entity in the KG to correct the errors or deal with disambiguation.

Output a series of actions (**within 10 actions**) in the following format:
<refinement>insert_edge("...", "...", "...")|delete_edge("...", "...", "...")|replace_node("...", "...")|...</refinement>

**Important:** You must think carefully about the given KG and the analysed error reasons before making your refinement. DO NOT DELETE ANY IRRELEVANT TRIPLES FROM THE ORIGINAL KG. TRY TO KEEP THE ORIGINAL KG AS MUCH AS POSSIBLE. DO NOT GENERATE TOO MANY ACTIONS. And output your refinement result directly in the specified format.
"""

REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT = """
Original Text: {original_text}
KG: {triples_string}
Question: {question}
Error reasons: {error_reasons}
"""

REAFINER_ACTION_USER_PROMPT = """Original Text:
{original_text}

Using the original question, accumulated KG context, and the abduction immediately above,
generate the graph refinement actions."""

REAFINER_KG_REFINEMENT_SYSTEM_PROMPT = """
As an advanced knowledge graph refinement assistant, your task is to refine knowledge graph to make it more suitable for answering the given question.

Based on the given KG and the analysed error reasons, refine the given KG to replace the original KG, which makes it more easily for retrieval and answering the given question. You can only conduct three types of actions: **adding missing information** (e.g., adding new triples to introduce shorter paths), **correcting conflicting or incorrect information**, and **merging redundant information**. Output the refined KG as a JSON array of triples in the following format:
<refinement>[{"subject": "...", "relation": "...", "object": "..."}, {"subject": "...", "relation": "...", "object": "..."}, ...]</refinement>

**Important:** You must think carefully about the given KG and the analysed error reasons before making your refinement. DO NOT REMOVE ANY IRRELEVANT TRIPLES FROM THE ORIGINAL KG. ONLY CONSIDER REMOVING TRIPLES WHEN THEY ARE REDUNDANT AND NEED TO BE MERGED. TRY TO KEEP THE ORIGINAL KG AS MUCH AS POSSIBLE. And output your refinement result directly in the specified format.
"""

REAFINER_KG_REFINEMENT_USER_PROMPT = """
KG: {triples_string}
Error reasons: {error_reasons}
"""

REFINE_SUBGRAPH_SYSTEM_PROMPT = """
As an advanced knowledge graph refinement assistant, your task is to refine knowledge graphs to make them more suitable for answering questions. Follow the three-step thinking process provided in the user instructions.

Refine the knowledge graph (KG) to make it more suitable for answering the given question. You must perform this task in three sequential thinking steps:

**Step 1: Answerable Judgement**
Evaluate whether the given question is answerable based on the provided KG context. If the answer is Yes, directly terminate the process. Output your judgment in the following format:
<judge>Yes</judge> or <judge>No</judge>

**Step 2: Error Abduction**
If the above answer is Yes, directly terminate the process. If the above answer is No, identify the specific problems in the KG. Analyze the KG for three types of errors: Redundant, Incompleteness, and Incorrectness. Output your analysis in the following format:
<abduction>...</abduction>

**Step 3: Refined KG Generation**
Based on your abduction analysis and the given KG, generate a refined KG that addresses the identified issues. Output the refined KG as a JSON array of triples in the following format:
<refinement>[{"subject": "...", "relation": "...", "object": "..."}, {"subject": "...", "relation": "...", "object": "..."}, ...]</refinement>

**Important:** You must follow this three-step thinking process and output each step in the specified format. Think carefully about each step before proceeding to the next.
"""

REFINE_SUBGRAPH_USER_PROMPT = """
Question: {question}
Knowledge Graph (KG) context: {triples_string}
"""