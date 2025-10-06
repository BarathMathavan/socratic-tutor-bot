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
        self.collection = self.db_client.get_collection(name=collection_name)
        
        # The core Socratic prompt
        self.socratic_prompt_template = """
        You are a Socratic Tutor Bot. Your goal is to guide students to their own conclusions without ever giving them a direct answer.
        You must adhere to the following rules:
        1.  NEVER provide a direct answer to the student's question.
        2.  Analyze the student's input and the provided context from the course materials.
        3.  Formulate a guiding question that prompts the student to think critically about the context.
        4.  If the student is correct, ask a follow-up question to test the depth of their understanding.
        5.  If the student is incorrect or confused, ask a simpler, scaffolding question to guide them back to the key concepts in the context.
        6.  Keep your responses concise and limited to one or two probing questions.

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

        Your Guiding Question:
        """

    def _retrieve_context(self, query, n_results=3):
        """Retrieves relevant text chunks from ChromaDB."""
        query_embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return "\n".join(results['documents'][0])

    # bot_logic.py inside generate_response method (THE FIX)

    def generate_response(self, student_question, chat_history):
        """Generates the Socratic response."""
        # 1. Retrieve relevant context
        context = self._retrieve_context(student_question)

        # 2. Format the chat history (ROBUST VERSION)
        history_lines = []
        for msg in chat_history:
            if 'user' in msg:
                history_lines.append(f"Student: {msg['user']}")
            if 'bot' in msg:
                history_lines.append(f"Tutor: {msg['bot']}")
        formatted_history = "\n".join(history_lines)
        
        # 3. Create the prompt
        prompt = self.socratic_prompt_template.format(
            context=context,
            chat_history=formatted_history,
            student_question=student_question
        )
        
        # 4. Call the LLM
        response = self.model.generate_content(prompt)
        return response.text