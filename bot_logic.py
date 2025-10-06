# bot_logic.py

# --- 1. Imports ---
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# --- 2. The SocraticTutor Class ---
class SocraticTutor:
    def __init__(self, api_key: str, collection_name: str = "socratic_collection"):
        """
        Initializes the Socratic Tutor, setting up connections to the LLM and vector database.
        """
        # --- LLM Configuration ---
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-latest')
        self.collection_name = collection_name
        
        # --- Vector DB and Embedding Model Configuration ---
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Connect to the running Qdrant Docker container
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # --- The Core Socratic Prompt ---
        # This is our final, most effective prompt for beginner-friendly, guided responses.
        self.socratic_prompt_template = """
        You are a Socratic Tutor for beginners. Your main goal is to make learning easy and clear. You are patient, friendly, and you always simplify concepts.

        Follow these rules:
        1.  NEVER give the direct answer.
        2.  Your questions must be very simple and direct. Point the student to the exact location in the text if possible (e.g., a section number).
        3.  Always start with a friendly, encouraging phrase.
        4.  **Guideline on Relevance:** Carefully check if the CONTEXT can answer the student's question, even if the words are different. For example, the student might ask about "speeding," and the text might only mention "exceeding the speed limit" or refer to a specific section number. You must try to connect these ideas. Only if the context is completely unrelated (e.g., about vehicle registration when the question is about fines) should you say you can't find the information.

        ---
        GOOD vs. BAD Response Example:
        Student asks: "What is the fine for speeding?"
        Context contains: "Section 183 of the Act states the penalty for exceeding the speed limit is a fine of Rs. 1000 for a light motor vehicle."

        BAD response: "The text mentions penalties for certain infractions. Which section discusses the monetary consequence for driving too fast?"
        GOOD response: "That's a great question! The text discusses penalties for that offense in Section 183. According to that section, what is the fine for a light motor vehicle?"
        ---

        CONTEXT FROM COURSE MATERIAL:
        {context}
        ---
        CONVERSATION HISTORY:
        {chat_history}
        ---
        STUDENT'S LATEST MESSAGE:
        {student_question}
        ---

        Your Simple, Friendly, Guiding Question (like the GOOD example):
        """

    def _retrieve_context(self, query: str, document_source: str, n_results: int = 5) -> str:
        """
        Searches Qdrant for the most relevant text chunks for a given query,
        filtered by a specific document source.
        """
        query_embedding = self.embedding_model.encode([query])
        
        # Search Qdrant using a filter on the 'source' metadata
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(), # Use the vectorized query
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=document_source),
                    )
                ]
            ),
            limit=n_results
        )
        
        # Combine the text from the retrieved chunks into a single context string
        context = "\n\n".join([hit.payload["text"] for hit in search_result])
        return context

    def generate_response(self, student_question: str, chat_history: list, document_source: str) -> str:
        """
        Generates a complete Socratic response by retrieving context and calling the LLM.
        """
        # 1. Retrieve relevant context from Qdrant
        context = self._retrieve_context(student_question, document_source)
        
        # If no relevant context is found, return a helpful message
        if not context.strip():
            return "I couldn't find specific information about that in the selected document. Perhaps try rephrasing your question or exploring a different topic?"

        # 2. Format the conversation history for the prompt
        history_lines = []
        for msg in chat_history:
            if 'user' in msg:
                history_lines.append(f"Student: {msg['user']}")
            if 'bot' in msg:
                history_lines.append(f"Tutor: {msg['bot']}")
        formatted_history = "\n".join(history_lines)
        
        # 3. Create the final prompt
        prompt = self.socratic_prompt_template.format(
            context=context,
            chat_history=formatted_history,
            student_question=student_question
        )
        
        # 4. Call the LLM to generate the Socratic question
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I'm sorry, I encountered an error while trying to formulate a response. Please try again."

