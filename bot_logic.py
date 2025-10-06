# bot_logic.py
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

class SocraticTutor:
    def __init__(self, api_key, collection_name="socratic_collection"):
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-latest')
        
        # Load the models and database connection
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_client = chromadb.PersistentClient(path="db")
        self.collection = self.db_client.get_or_create_collection(name=collection_name)
        
        # The core Socratic prompt
        self.socratic_prompt_template = """
        You are a Socratic Tutor for beginners. Your main goal is to make learning easy and clear. You are patient, friendly, and you always simplify concepts.

        Follow these rules:
        1.  NEVER give the direct answer.
        2.  Your questions must be very simple and direct. Point the student to the exact location in the text.
        3.  Always start with a friendly, encouraging phrase.
        4.  **Guideline on Relevance:** Carefully check if the CONTEXT can answer the student's question, even if the words are different. For example, the student might ask about "speeding," and the text might only mention "exceeding the speed limit" or refer to a specific section number. **You must try to connect these ideas.** Only if the context is completely unrelated (e.g., about vehicle registration when the question is about fines) should you say you can't find the information.

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

    def _retrieve_context(self, query, document_source, n_results=3):
        """Retrieves relevant text chunks from ChromaDB for a SPECIFIC source."""
        query_embedding = self.embedding_model.encode([query])
        
        # This 'where' filter is the magic that isolates the documents
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"source": document_source} # THE FILTERING HAPPENS HERE
        )
        return "\n".join(results['documents'][0])

    # bot_logic.py inside generate_response method (THE FIX)

    def generate_response(self, student_question, chat_history, document_source):
        """Generates the Socratic response using a specific document source."""
        # 1. Retrieve relevant context from the selected document
        context = self._retrieve_context(student_question, document_source)
        
        # ... (the rest of the method for formatting history and calling the LLM is exactly the same) ...
        history_lines = []
        for msg in chat_history:
            if 'user' in msg:
                history_lines.append(f"Student: {msg['user']}")
            if 'bot' in msg:
                history_lines.append(f"Tutor: {msg['bot']}")
        formatted_history = "\n".join(history_lines)
        
        prompt = self.socratic_prompt_template.format(
            context=context,
            chat_history=formatted_history,
            student_question=student_question
        )
        
        response = self.model.generate_content(prompt)
        return response.text