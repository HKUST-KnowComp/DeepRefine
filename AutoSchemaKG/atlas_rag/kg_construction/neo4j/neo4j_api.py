import time
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from logging import Logger
from atlas_rag.retriever.lkg_retriever.base import BaseLargeKGRetriever, BaseLargeKGEdgeRetriever
from atlas_rag.kg_construction.neo4j.utils import start_up_large_kg_index_graph
from atlas_rag.llm_generator import LLMGenerator
from neo4j import Driver
from dataclasses import dataclass
import traceback

@dataclass
class LargeKGConfig:
    largekg_retriever: BaseLargeKGRetriever | BaseLargeKGEdgeRetriever = None
    reader_llm_generator : LLMGenerator = None
    driver: Driver = None
    logger: Logger = None
    is_felm: bool = False
    is_mmlu: bool = False
    
    rag_exemption_list = [
    """I will show you a question and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to assess whether each text segment contains errors or not. \nPlease generate using the following format:\nAnswer: List the ids of the segments with errors (separated by commas). Please only output the ids, no more details. If all the segments are correct, output \"ALL_CORRECT\".\n\nHere is one example:\nQuestion: 8923164*7236571?\nSegments: \n1. The product of 8923164 and 7236571 is: 6,461,216,222,844\n2. So, 8923164 multiplied by 7236571 is equal to 6,461,216,222,844.\n\nBelow are your outputs:\nAnswer: 1,2\nIt means segment 1,2 contain errors.""",
    """I will show you a question and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to determine whether each text segment contains factual errors or not. \nPlease generate using the following format:\nAnswer: List the ids of the segments with errors (separated by commas). Please only output the ids, no more details. If all the segments are correct, output \"ALL_CORRECT\".\n\nHere is one example:\nQuestion: A company offers a 10% discount on all purchases over $100. A customer purchases three items, each costing $80. Does the customer qualify for the discount?\nSegments: \n1. To solve this problem, we need to use deductive reasoning. We know that the company offers a 10% discount on purchases over $100, so we need to calculate the total cost of the customer's purchase.\n2. The customer purchased three items, each costing $80, so the total cost of the purchase is: 3 x $80 = $200.\n3. Since the total cost of the purchase is greater than $100, the customer qualifies for the discount. \n4. To calculate the discounted price, we can multiply the total cost by 0.1 (which represents the 10% discount): $200 x 0.1 = $20.\n5. So the customer is eligible for a discount of $20, and the final cost of the purchase would be: $200 - $20 = $180.\n6. Therefore, the customer would pay a total of $216 for the three items with the discount applied.\n\nBelow are your outputs:\nAnswer: 2,3,4,5,6\nIt means segment 2,3,4,5,6 contains errors.""",
    ]

    mmlu_check_list = [
        """Given the following question and four candidate answers (A, B, C and D), choose the answer."""
    ]

app = FastAPI()

@app.on_event("startup")
async def startup():
    global large_kg_config
    start_up_large_kg_index_graph(large_kg_config.driver)

@app.on_event("shutdown")
async def shutdown():
    global large_kg_config
    print("Shutting down the model...")
    del large_kg_config

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "test"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str = None
    name: Optional[str] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1
    retriever_config: Optional[dict] = {
        "topN": 5,
        "number_of_source_nodes_per_ner": 10,
        "sampling_area": 250,
        "Dmax": 2,
        "Wmax": 3
    }
    class Config:
        extra = "allow"
    

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]

class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]
    index: int

class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global large_kg_config
    try:
        if len(request.messages) < 1 or request.messages[-1].role == "assistant":
            print(request)
            # raise HTTPException(status_code=400, detail="Invalid request")
        if large_kg_config.logger is not None:
            large_kg_config.logger.info(f"Request: {request}")
        gen_params = dict(
            messages=request.messages,
            temperature=0.8,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=False,
            repetition_penalty=request.repetition_penalty,
            tools=request.tools,
        )

        last_message = request.messages[-1]
        system_prompt = 'You are a helpful assistant.'
        question = last_message.content if last_message.role == 'user' else request.messages[-2].content

        is_exemption = any(exemption in question for exemption in LargeKGConfig.rag_exemption_list)
        is_mmlu = any(exemption in question for exemption in LargeKGConfig.mmlu_check_list)
        print(f"Is exemption: {is_exemption}, Is MMLU: {is_mmlu}")
        if is_mmlu:
            rag_text = question 
        else:
            parts = question.rsplit("Question:", 1)
            rag_text = parts[-1] if len(parts) > 1 else None
        print(f"RAG text: {rag_text}")
        if not is_exemption:
            passages, passages_score = large_kg_config.largekg_retriever.retrieve_passages(rag_text)
            context = "No retrieved context, Please answer the question with your own knowledge." if not passages else "\n".join([f"Passage {i+1}: {text}" for i, text in enumerate(reversed(passages))])
            
        if is_mmlu:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""Here is the context: {context} \n\n
                    If the context is not useful, you can answer the question with your own knowledge. \n {question}\nThink step by step. Your response should end with 'The answer is ([the_answer_letter])' where the [the_answer_letter] is one of A, B, C and D."""
                }
            ]
        elif not is_exemption:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""{question} Reference doc: {context}"""
                }
            ]
        else:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""{question} """
                }
            ]
        if large_kg_config.logger is not None:
            large_kg_config.logger.info(rag_chat_content)

        response = large_kg_config.reader_llm_generator.generate_response(
            batch_messages=rag_chat_content,
            max_new_tokens=gen_params["max_tokens"],
            temperature=gen_params["temperature"],
            frequency_penalty = 1.1
        )
        message = ChatMessage(
            role="assistant",
            content=response.strip()
        )
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            object="chat.completion",
            choices=[choice_data]
        )
    except Exception as e:
        print("ERROR: ", e)
        print("Catched error")
        traceback.print_exc()
        system_prompt = 'You are a helpful assistant.'
        gen_params = dict(
            messages=request.messages,
            temperature=0.8,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=False,
            repetition_penalty=request.repetition_penalty,
            tools=request.tools,
        )
        last_message = request.messages[-1]
        system_prompt = 'You are a helpful assistant.'
        question = last_message.content if last_message.role == 'user' else request.messages[-2].content
        rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""{question} """
                }
            ]
        response = large_kg_config.reader_llm_generator.generate_response(
            batch_messages=rag_chat_content,
            max_new_tokens=gen_params["max_tokens"],
            temperature=gen_params["temperature"],
            frequency_penalty = 1.1
        )
        message = ChatMessage(
            role="assistant",
            content=response.strip()
        )
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            object="chat.completion",
            choices=[choice_data]
        )


def start_app(user_config:LargeKGConfig, host="0.0.0.0", port=10090, reload=False):
    """Function to start the FastAPI application."""
    global large_kg_config
    large_kg_config = user_config  # Use the passed context if provided

    uvicorn.run(f"atlas_rag.kg_construction.neo4j.neo4j_api:app", host=host, port=port, reload=reload)
    