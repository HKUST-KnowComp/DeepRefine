from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from typing import Union
from logging import Logger

class ReAct():
    def __init__(self, llm:LLMGenerator):
        self.llm = llm
    
    def generate_with_rag_react(self, question: str, retriever: Union['BaseEdgeRetriever', 'BasePassageRetriever'], max_iterations: int = 5, max_new_tokens: int = 1024, logger: Logger = None):
        """
        Generate a response using RAG with ReAct framework, starting with an initial search using the original query.
        
        Args:
            question (str): The question to answer
            retriever: The retriever instance to use for searching
            max_iterations (int): Maximum number of ReAct iterations
            max_new_tokens (int): Maximum number of tokens to generate per iteration
            
        Returns:
            tuple: (final_answer, search_history)
                - final_answer: The final answer generated
                - search_history: List of (thought, action, observation) tuples
        """
        search_history = []
        
        # Perform initial search with the original query
        if isinstance(retriever, BaseEdgeRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = ". ".join(initial_context)
        elif isinstance(retriever, BasePassageRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = "\n".join(initial_context)
        
        # Start ReAct process with the initial context
        for iteration in range(max_iterations):
            # First, analyze if we can answer with current context
            analysis_response = self.llm.generate_with_react(
                question=question,
                context=current_context,
                max_new_tokens=max_new_tokens,
                search_history=search_history,
                logger = logger
            )
            
            if logger:
                logger.info(f"Analysis response: {analysis_response}")
                
            try:
                # Parse the analysis response
                thought = analysis_response.split("Thought:")[1].split("\n")[0]
                if logger:
                    logger.info(f"Thought: {thought}")
                action = analysis_response.split("Action:")[1].split("\n")[0]
                answer = analysis_response.split("Answer:")[1].strip()
                
                # If the answer indicates we can answer with current context
                if answer.lower() != "need more information":
                    search_history.append((thought, action, "Using current context"))
                    return answer, search_history
                
                # If we need more information, perform the search
                if "search" in action.lower():
                    # Extract search query from the action
                    search_query = action.split("search for")[-1].strip()
                    
                    # Perform the search
                    if isinstance(retriever, BaseEdgeRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split(". ")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = ". ".join(new_context)
                    elif isinstance(retriever, BasePassageRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split("\n")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = "\n".join(new_context)
                    
                    # Store the search results as observation
                    if new_context:
                        observation = f"Found information: {new_context}"
                    else:
                        observation = "No new information found. Consider searching for related entities or events."
                    search_history.append((thought, action, observation))
                    
                    # Update context with new search results
                    if new_context:
                        current_context = f"{current_context}\n{new_context}"
                        if logger:
                            logger.info(f"New search results: {new_context}")
                    else:
                        if logger:
                            logger.info("No new information found, suggesting to try related entities")

                else:
                    # If no search is needed but we can't answer, something went wrong
                    search_history.append((thought, action, "No action taken but answer not found"))
                    return "Unable to find answer", search_history
                
            except Exception as e:
                if logger:
                    logger.error(f"Error parsing ReAct response: {e}")
                return analysis_response, search_history
        
        # If we've reached max iterations, return the last answer
        return answer, search_history