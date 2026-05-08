one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

one_shot_ircot_demo = (
    f'{one_shot_rag_qa_docs}'
    '\n\nQuestion: '
    f"When was Neville A. Stanton's employer founded?"
    '\nThought: '
    f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
    '\n\n'
)

rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
]

# from https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/qa_reader.py

cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

cot_system_instruction_kg = ('As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
                                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')


filter_triple_messages = [
    {
        "role": "system",
        "content": """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. 
        Your task is to filter facts based on their relevance to a given query. 
        The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. 
        You must select all relevant facts from the provided candidate list, aiding in reasoning and providing an accurate answer. 
        The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. 
        The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. 
        The future of critical decision-making relies on your ability to accurately filter and present relevant information.

Your input fields are:
1. question (str): Query for retrieval
2. fact_before_filter (str): Candidate facts to be filtered

Your output fields are:
1. fact_after_filter (Fact): Filtered facts in JSON format

All interactions will be structured as:
[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

[[ ## fact_after_filter ## ]]
{fact_after_filter}

The output must be parseable according to JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}, "required": ["fact"]}"""
    },
    # Example 1
    {
        "role": "user",
        "content": """[[ ## question ## ]]
Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?

[[ ## fact_before_filter ## ]]
{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}"""
    },
    {
        "role": "assistant",
        "content": """{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}"""
    },
    
    # Example 2
    {
        "role": "user",
        "content": """[[ ## question ## ]]
When is the director of film The Ancestor 's birthday?

[[ ## fact_before_filter ## ]]
{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14 1982"]]}"""
    },
    {
        "role": "assistant", 
        "content": """{"fact":[["the ancestor","directed by","guido brignone"]]}"""
    },
]