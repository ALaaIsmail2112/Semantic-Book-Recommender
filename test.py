import pandas as pd 
import numpy as np 
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

books = pd.read_csv('Semantic Book Recommender/books_with_emotions.csv')

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail']
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

file_path = 'Semantic Book Recommender/books_descriptions.txt'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} does not exist. Please check the path and try again.")

try:
    raw_documents = TextLoader(file_path, encoding='utf-8').load()
except Exception as e:
    raise RuntimeError(f"Error loading {file_path}: {e}")

text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding=embeddings)

def retrieve_semantic_recommandation(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)
    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommandation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description'] if pd.notnull(row['description']) else 'No description available.'
        truncated_decs_split = description.split()
        truncated_description = " ".join(truncated_decs_split[:30]) + '...'

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"📘 {row['title']}\n👤 by {authors_str}\n📝 {truncated_description}"
        results.append((row['large_thumbnail'], caption))

    return results

categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background: #f5f5f5;
    color: #333;
}

#title {
    font-size: 36px;
    font-weight: bold;
    color: #4B008;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 40px;
}

#recommend-btn {
    background-color: #6a0dad !important;
    color: white !important;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
    text-align: center;
}

#recommend-btn:hover {
    background-color: #5a009d !important;
    transform: scale(1.05);
}


label {
    color: #4B0082;
    font-weight: bold;
    font-size: 14px;
}

#output-gallery .gallery-item {
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    background-color: white;
}

#form-section {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}
.recommend-btn {
    background-color: #6a0dad !important;
    color: white !important;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
    text-align: center;
}
.recommend-btn:hover {
    background-color: #5a009d !important;
    transform: scale(1.05);
}

"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:

    gr.Markdown("""<div id='title'>📚 Semantic Book Recommender with Emotions 💡 </div>""", elem_id='title')

    with gr.Group(elem_id='form-section'):
        with gr.Row():
            user_query = gr.Textbox(label='📖 Describe your dream book:', placeholder='e.g., A journey of redemption and hope')
        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label='📚 Pick a category:',
                value='All'
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label='🎭 Mood Preference:',
                value='All'
            )
            submit_button = gr.Button('🔍  Find Recommendations ', elem_classes="recommend-btn")

    gr.Markdown("## 🔍 Your Personalized Book Matches")
    output = gr.Gallery(label='📘 Recommendations', columns=4, rows=2, elem_id='output-gallery')

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == '__main__':
    dashboard.launch()
