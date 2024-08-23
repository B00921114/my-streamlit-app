import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
import torch
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from difflib import SequenceMatcher
import schedule
import time
from tabulate import tabulate
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Google Scholar search function using SerpAPI
def search_google_scholar(query, api_key, num_results=20):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_scholar",
        "q": query,
        "num": num_results,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Prepare the text data from search results
def prepare_text_data(results):
    texts = []
    meta_data = []
    for result in results.get("organic_results", []):  # Fixed empty loop issue
        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        abstract = result.get("snippet", "")
        full_text = title + " " + snippet
        cleaned_text = preprocess_text(full_text)

        # Assuming the results
        publication_year = int(result.get("publication_year", 2022))
        journal_impact_factor = float(result.get("journal_impact_factor", 0.0))
        author_reputation = float(result.get("author_reputation", 0.0))

        texts.append(cleaned_text)
        meta_data.append({
            "link": link,
            "abstract": abstract,
            'publication_year': publication_year,
            'journal_impact_factor': journal_impact_factor,
            'author_reputation': author_reputation
        })
    return texts, meta_data

# Automatic Data Refresh system
def refresh_data():
    print("Refreshing data...")
    global texts
    texts = []
    for query in queries:
        results = search_google_scholar(query, api_key, num_results=20)
        if isinstance(results, dict):
            t, m = prepare_text_data(results)
            texts += t
    print("Data refresh complete.")

schedule.every().day.at("02:00").do(refresh_data)

# Collecting Data Using the API
api_key = "84a7bc67f3ea3cb501fabfcb486cfcf5d34a565f222ad125255ce5f101ac8d52"  
queries = ["machine learning in healthcare", "artificial intelligence in medicine", "deep learning in medical imaging"]
texts = []
meta_data = []
m = []

for query in queries:
    results = search_google_scholar(query, api_key, num_results=20)
    if isinstance(results, dict):
        t, m = prepare_text_data(results)
        texts += t
        meta_data += m

# Tokenizing the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
for text in texts:
    encoded_dict = tokenizer(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Converting the lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor([0] * len(input_ids))

# Creating a DataLoader and a dataloader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Fine-Tune BERT with Batch Size and Epochs
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training the loop
model.train()
for epoch in range(training_args.num_train_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


def get_bert_embeddings(texts, model, tokenizer):
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Access the last_hidden_state
            last_hidden_state = outputs.last_hidden_state
            # Calculate the mean of the last hidden state
            embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)


# Example Usage
query = "neural networks in healthcare"
user_profile_embedding = get_bert_embeddings([query], model, tokenizer)

recommendations = search_google_scholar(query, api_key, num_results=20)

texts, _ = prepare_text_data(recommendations)
paper_embeddings = get_bert_embeddings(texts, model, tokenizer)
similarities = cosine_similarity(paper_embeddings, user_profile_embedding.reshape(1, -1))
ranked_indices = np.argsort(similarities[:, 0])[::-1]
recommended_papers = [(recommendations['organic_results'][i]['title'], recommendations['organic_results'][i]['link'], similarities[i][0]) for i in ranked_indices[:5]]

# Display
for title, link, score in recommended_papers:
    print(f"Title: {title}")
    print(f"Link: {link}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 80)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to calculate similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Get the exact link of the paper given its title
def get_paper_link_by_title(title):
    url = "https://api.crossref.org/works"
    params = {
        'query.title': title,
        'rows': 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['message']['items']:
            paper = data['message']['items'][0]
            doi = paper.get('DOI', None)
            if doi:
                return f"https://doi.org/{doi}"
            else:
                return "DOI not found for this paper."
        else:
            return "No results found."
    else:
        return f"Error: {response.status_code}"


def get_feedback(recommended_papers):
    feedback = []
    print("\nPlease rate the relevance of each recommended paper on a scale from 1 to 5 (1 = Not relevant, 5 = Very relevant):\n")
    for i, (title, link, score) in enumerate(recommended_papers, start=1):
        rating = int(input(f"Relevance of paper {i} (Title: {title}): "))
        feedback.append((title, link, score, rating))
    return feedback 

def get_user_query_and_recommend():
    # Prompt the user to enter their research topic
    user_query = input("Enter your research topic: ")

    # Perform a Google Scholar search based on the user query
    results = search_google_scholar(user_query, api_key, num_results=20)

    # Prepare the text data from the search results
    texts, meta_data = prepare_text_data(results)

    # Generate the BERT embeddings for the user query and the retrieved texts
    user_profile_embedding = get_bert_embeddings([user_query], model, tokenizer)
    paper_embeddings = get_bert_embeddings(texts, model, tokenizer)

    # Calculate the cosine similarity between the user's profile and the retrieved papers
    similarities = cosine_similarity(paper_embeddings, user_profile_embedding.reshape(1, -1))

    # Rank the papers based on their similarity to the user's query
    ranked_indices = np.argsort(similarities[:, 0])[::-1]

    # Prepare a list of recommended papers
    recommended_papers = []
    for i in ranked_indices[:5]:
        title = results['organic_results'][i]['title']
        link = results['organic_results'][i].get('link', 'Link not available')
        score = similarities[i][0]
        recommended_papers.append((title, link, score))

    return recommended_papers


#  Interaction loop
 
def hybrid_recommendation_system(user_query, user_profile_embedding, user_history, all_users_history, k=5):
    # Content-based recommendations using BERT
    recommendations = search_google_scholar(user_query, api_key, num_results=20)
    texts, meta_data = prepare_text_data(recommendations)
    paper_embeddings = get_bert_embeddings(texts, model, tokenizer)
    similarities = cosine_similarity(paper_embeddings, user_profile_embedding.reshape(1, -1))

    # Correctly defining content_based_papers
    content_based_papers = np.argsort(similarities[:, 0])[-k:][::-1]

    # Additional functionality for hybrid recommendation system
    # Example: collaborative filtering
    # collaborative_papers = collaborative_filtering(user_history, all_users_history, k)
    # final_recommendations = list(set(content_based_papers) | set(collaborative_papers))

    final_papers = [(recommendations['organic_results'][i]['title'], recommendations['organic_results'][i]['link']) for i in content_based_papers]

    for title, link in final_papers:
        print(f"Title: {title}")
        print(f"Link: {link}")
        print("-" * 80)

# Example Usage
user_query = "energy storage"
user_profile_embedding = get_bert_embeddings([user_query], model, tokenizer)
user_history = np.array([[1, 0, 0], [0, 1, 0]])
all_users_history = np.array([[1, 0, 1], [1, 0, 0]])

# Call the function once with the default value of k
hybrid_recommendation_system(user_query, user_profile_embedding, user_history, all_users_history)


import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Your existing functions (e.g., get_bert_embeddings, search_google_scholar, etc.) remain unchanged.

# Streamlit app title and description
st.title("Research Paper Recommendation System")
st.write("Enter your research topic to get paper recommendations:")

def hybrid_recommendation_system(user_query, user_profile_embedding, user_history, all_users_history, k=5):
    # Content-based recommendations using BERT
    recommendations = search_google_scholar(user_query, api_key, num_results=20)
    texts, meta_data = prepare_text_data(recommendations)
    paper_embeddings = get_bert_embeddings(texts, model, tokenizer)
    similarities = cosine_similarity(paper_embeddings, user_profile_embedding.reshape(1, -1))
    
    # Correctly defining content_based_papers
    content_based_papers = np.argsort(similarities[:, 0])[-k:][::-1]

    # Additional functionality for hybrid recommendation system
    # Example: collaborative filtering
    # collaborative_papers = collaborative_filtering(user_history, all_users_history, k)
    # final_recommendations = list(set(content_based_papers) | set(collaborative_papers))

    final_papers = [(recommendations['organic_results'][i]['title'], 
                     recommendations['organic_results'][i]['link']) 
                    for i in content_based_papers]

    st.subheader("Recommended Papers:")
    
    feedback = {}  # Dictionary to store feedback

    for idx, (title, link) in enumerate(final_papers):
        st.write(f"**Title**: {title}")
        st.write(f"**Link**: {link}")
        # Adding a slider for user feedback
        rating = st.slider(f"Rate this paper (1-5)", 1, 5, key=f"rating_{idx}")
        feedback[title] = rating
        st.write("---")

# Button to submit feedback
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        st.write("Your feedback:", feedback)
        # You can save the feedback to a file or a database here, if needed

# Example Usage (Streamlit Input)
user_query = st.text_input("Enter your research topic:", "energy storage")
user_profile_embedding = get_bert_embeddings([user_query], model, tokenizer)
user_history = np.array([[1, 0, 0], [0, 1, 0]])
all_users_history = np.array([[1, 0, 1], [1, 0, 0]])

if st.button("Get Recommendations"):
    hybrid_recommendation_system(user_query, user_profile_embedding, user_history, all_users_history)



