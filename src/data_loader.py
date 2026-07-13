from pathlib import Path
from typing import List
import ast

import pandas as pd
from langchain_core.documents import Document


class DataLoader:
    """
    Loads the recipe dataset and converts each recipe
    into a LangChain Document.
    """

    REQUIRED_COLUMNS = [
        "Title",
        "Ingredients",
        "Instructions",
        "Cleaned_Ingredients",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the CSV file."""

        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} does not exist.")

        self.df = pd.read_csv(self.file_path)

        self._validate_columns()

        return self.df

    def _validate_columns(self):
        """Validate required columns."""

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    @staticmethod
    def _parse_list(value):
        """
        Convert a string representation of a Python list
        into an actual Python list.
        """

        if pd.isna(value):
            return []

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    def create_documents(self) -> List[Document]:
        """Convert every recipe into a LangChain Document."""

        if self.df is None:
            self.load_data()

        documents = []

        for index, row in self.df.iterrows():

            ingredients = self._parse_list(row["Ingredients"])
            cleaned_ingredients = self._parse_list(
                row["Cleaned_Ingredients"]
            )

            ingredients_text = "\n".join(
                f"- {ingredient}" for ingredient in ingredients
            )

            page_content = f"""
Recipe Name:
{row["Title"]}

Ingredients:
{ingredients_text}

Instructions:
{row["Instructions"]}
""".strip()

            document = Document(
                page_content=page_content,
                metadata={
                    "recipe_id": index,
                    "title": row["Title"],
                    "cleaned_ingredients": cleaned_ingredients,
                    "image": row.get("Image_Name", ""),
                },
            )

            documents.append(document)

        return documents

    def load_documents(self) -> List[Document]:
        """Public method used by the RAG pipeline."""

        self.load_data()
        return self.create_documents()