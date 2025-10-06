# Socratic Tutor Bot

An AI-powered conversational tutor that uses the Socratic method to guide students. The bot is grounded in instructor-provided PDF documents and avoids giving direct answers, fostering critical thinking.

## Features

-   **Socratic Dialogue:** Engages students with probing questions instead of direct answers.
-   **PDF-Grounded Knowledge:** Uses Retrieval-Augmented Generation (RAG) to ensure all responses are based on uploaded course materials.
-   **Instructor Portal:** A simple web interface for instructors to upload new PDF documents.
-   **Conversation Logging:** Saves chat sessions to a log file for instructor review and analytics.

## How to Run This Project

### 1. Prerequisites

-   Python 3.9+
-   Git

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/socratic-tutor-bot.git
    cd socratic-tutor-bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

1.  Create a `.env` file in the root of the project directory.
2.  Add your Google Generative AI API key to the `.env` file:
    ```
    GOOGLE_API_KEY='your_api_key_here'
    ```

### 4. Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```

2.  **Upload Documents:** Open your web browser and navigate to `http://127.0.0.1:5000/upload` to upload your PDF knowledge base.

3.  **Chat with the Tutor:** Navigate to `http://127.0.0.1:5000/` to start a conversation.