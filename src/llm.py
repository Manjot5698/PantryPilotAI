from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMManager:
    """
    Handles response generation using a local Llama model.
    """

    def __init__(self):

        self.llm = ChatOllama(
            model="llama3:latest",
            temperature=0,)

        self.prompt = ChatPromptTemplate.from_template(
            """
                You are PantryPilot AI.
                Your task is to recommend the best recipe using ONLY the retrieved recipes.
                The user has also provided expiry information for their ingredients.
                Prefer recipes that use ingredients expiring sooner whenever possible.
                Instructions:
                    - Recommend ONE best recipe.
                    - Explain why it is the best choice.
                    - Give priority to ingredients expiring sooner.
                    - Mention any important missing ingredients.
                    - Briefly summarize the preparation steps.
                    - If no suitable recipe exists, clearly say so.
                    - Never invent recipes.

                User Ingredients & Expiry:
                {expiry_info}
                Retrieved Recipes:
                {context}
                User Question:
                {question}

                Answer:
            """)

        self.chain = (self.prompt| self.llm| StrOutputParser())

    def generate_response(self,question: str,documents: list[Document],expiry_info: dict[str, int],) -> str:
        """
        Generate a recipe recommendation.
        """

        context_parts = []

        for document in documents:

            title = document.metadata.get("title", "Unknown Recipe")

            ingredients = ", ".join(document.metadata.get("cleaned_ingredients", []))

            instructions = ""

            if "Instructions:" in document.page_content:
                instructions = (document.page_content.split("Instructions:")[1].strip()[:300])
            context_parts.append(f"""
Recipe: {title}

Ingredients:
{ingredients}

Instructions:
{instructions}
""".strip()
            )

        context = "\n\n".join(context_parts)

        expiry_text = "\n".join(f"- {ingredient}: {days} day(s) left" for ingredient, days in expiry_info.items())

        print("=" * 60)
        print(f"Context Length: {len(context)}")
        print("=" * 60)

        try:
            return self.chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "expiry_info": expiry_text,
                }
            )

        except Exception as e:
            return f"Error generating response: {e}"