from pathlib import Path

import streamlit as st

from src.data_loader import DataLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.llm import LLMManager


@st.cache_resource
def load_pipeline():
    project_dir = Path(__file__).resolve().parent
    index_dir = project_dir / "faiss_index"
    index_files = (index_dir / "index.faiss", index_dir / "index.pkl")

    embedding_model = EmbeddingManager()

    vector_store = VectorStoreManager(embedding_model)
    if all(path.is_file() for path in index_files):
        vector_store.load(index_dir)
    else:
        dataset_path = project_dir / "data" / "13k-recipes.csv"
        documents = DataLoader(str(dataset_path)).load_documents()
        vector_store.create_vector_store(documents)
        vector_store.save(index_dir)

    retriever = vector_store.as_retriever(k=3)

    llm = LLMManager()

    return retriever, llm


st.set_page_config(
    page_title="PantryPilot",
    page_icon="🍳",
    layout="wide",
)

st.title("🍳 PantryPilot")
st.write(
    "Enter your available ingredients and their expiry dates."
)

retriever, llm = load_pipeline()


query = st.text_input(
    "Available Ingredients",
    placeholder="e.g. milk, tomatoes, chicken, rice",
)

expiry_input = st.text_area(
    "Expiry Information",
    placeholder="""
milk:1
tomatoes:2
chicken:5
rice:30
""".strip(),
)


def parse_expiry(text: str) -> dict[str, int]:

    expiry = {}

    for line in text.splitlines():

        if ":" not in line:
            continue

        ingredient, days = line.split(":", 1)

        ingredient = ingredient.strip()

        try:
            expiry[ingredient] = int(days.strip())
        except ValueError:
            continue

    return expiry


if st.button("Find Recipe"):

    if not query.strip():

        st.warning("Please enter your ingredients.")

        st.stop()

    expiry_info = parse_expiry(expiry_input)

    with st.spinner("Searching recipes..."):

        documents = retriever.invoke(query)

        response = llm.generate_response(
            question=query,
            documents=documents,
            expiry_info=expiry_info,
        )

    st.subheader("🍽 Recommendation")

    st.write(response)

    st.divider()

    with st.expander("Retrieved Recipes"):

        for doc in documents:

            st.subheader(doc.metadata["title"])

            st.write(doc.page_content)