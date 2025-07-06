import os
from groq import Groq
from web_search import serper_search

class GoogleChatbot:
    def __init__(self, groq_api_key=None, serper_api_key=None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "8351c8d666a70eaf483cc0f2a0c120440aaa0b96")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set as env variable or provided.")
        self.groq_client = Groq(api_key=self.groq_api_key)

    def ask(self, user_query: str) -> str:
        # 1. Google search
        search_results = serper_search(user_query, num=5, api_key=self.serper_api_key)
        sources = []
        if "organic" in search_results:
            for result in search_results["organic"][:5]:
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
        # 2. Build context for LLM
        context = "Here are Google search results:\n"
        for i, src in enumerate(sources, 1):
            context += f"\n{i}. {src['title']}\n{src['url']}\n{src['snippet']}\n"
        # 3. Prompt LLM for a conversational, formatted answer
        prompt = (
            f"You are a helpful, concise, and professional chatbot. "
            f"Using ONLY the following Google search results, answer the user's question in a clear, well-formatted manner. "
            f"Provide bullet points or sections if helpful. If possible, cite the source titles. "
            f"\n\nUser question: {user_query}\n\n{context}"
        )
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful, concise, and professional chatbot that answers only using provided Google search results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

def main():
    print("🤖 Google Search Chatbot (Groq LLM)")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    bot = GoogleChatbot()
    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user_query:
                continue
            print("searching and thinking...\n")
            answer = bot.ask(user_query)
            print(f"\nBot:\n{answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
