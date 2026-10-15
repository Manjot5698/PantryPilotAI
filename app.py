from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_resource
def load_pipeline():
    from src.data_loader import DataLoader
    from src.embeddings import EmbeddingManager
    from src.llm import LLMManager
    from src.vector_store import VectorStoreManager

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

    return vector_store.as_retriever(k=3), LLMManager()


def build_pantry_context(
    pantry: pd.DataFrame, today: date | None = None
) -> tuple[str, dict[str, int], list[str]]:
    """Build the search query and days-to-expiry context from pantry rows."""
    reference_date = today or date.today()
    ingredients: list[str] = []
    expiry_info: dict[str, int] = {}
    errors: list[str] = []

    for row_number, row in enumerate(pantry.to_dict("records"), start=1):
        ingredient = str(row.get("Ingredient", "") or "").strip()
        expiry_date = row.get("Expiry date")
        if not ingredient:
            continue
        if expiry_date is None or pd.isna(expiry_date):
            errors.append(f"Add an expiry date for {ingredient}.")
            continue
        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()
        if not isinstance(expiry_date, date):
            errors.append(f"Enter a valid expiry date for {ingredient} (row {row_number}).")
            continue

        ingredients.append(ingredient)
        days_remaining = (expiry_date - reference_date).days
        expiry_info[ingredient] = min(
            days_remaining,
            expiry_info.get(ingredient, days_remaining),
        )

    return ", ".join(ingredients), expiry_info, errors


st.set_page_config(
    page_title="PantryPilot",
    page_icon="🍳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1050px; padding-top: 2.5rem; padding-bottom: 4rem;}
    .hero {
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color, #e4ebdf);
        border-radius: 22px;
        background: var(--secondary-background-color);
        box-shadow: 0 12px 34px rgba(36, 65, 39, .06);
    }
    .hero h1 {margin: 0 0 .45rem; color: var(--text-color); letter-spacing: -.04em;}
    .hero p {margin: 0; color: var(--text-color); opacity: .75; font-size: 1.05rem;}
    div[data-testid="stForm"] {
        padding: 1.25rem 1.4rem;
        border: 1px solid var(--border-color, #e6ebe3);
        border-radius: 18px;
        background: var(--secondary-background-color);
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border-radius: 11px;
        font-weight: 650;
        background: #36764b;
        border-color: #36764b;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🍳 PantryPilot</h1>
      <p>Turn what’s in your kitchen into something delicious—before it expires.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Your pantry")
st.caption("Add ingredients and choose their use-by dates. We’ll prioritize items expiring soon.")

today = date.today()
initial_pantry = pd.DataFrame(
    {
        "Ingredient": pd.Series(dtype="string"),
        "Expiry date": pd.Series(dtype="datetime64[ns]"),
    }
)

with st.form("recipe_search_form"):
    pantry = st.data_editor(
        initial_pantry,
        key="pantry_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Ingredient": st.column_config.TextColumn(
                "Ingredient",
                help="One ingredient per row, such as tomatoes or rice.",
                required=False,
            ),
            "Expiry date": st.column_config.DateColumn(
                "Expiry date",
                help="Pick the date on the package, or your best estimate.",
                format="MMM D, YYYY",
                min_value=today - timedelta(days=3650),
                max_value=today + timedelta(days=3650),
            ),
        },
    )
    preferences = st.text_input(
        "Optional preferences",
        placeholder="e.g. quick dinner, vegetarian, high-protein",
    )
    submitted = st.form_submit_button(
        "✨  Find recipes",
        type="primary",
        use_container_width=True,
    )

st.caption("Tip: use the **+** row in the table to add more ingredients.")

if submitted:
    query, expiry_info, validation_errors = build_pantry_context(pantry)
    if not query:
        st.error("Add at least one ingredient to your pantry.")
    elif validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        search_query = query
        if preferences.strip():
            search_query = f"{search_query}. Preferences: {preferences.strip()}"

        with st.spinner("Finding a recipe that makes the most of your pantry..."):
            retriever, llm = load_pipeline()
            documents = retriever.invoke(search_query)
            response = llm.generate_response(
                question=search_query,
                documents=documents,
                expiry_info=expiry_info,
            )

        st.markdown("### 🍽️ Your recommendation")
        st.success(response)

        with st.expander(f"See the {len(documents)} matching recipes"):
            for document in documents:
                st.markdown(f"#### {document.metadata.get('title', 'Recipe')}")
                st.write(document.page_content)
