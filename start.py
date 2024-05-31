from googleapiclient.discovery import build
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Step 1: Fetch Comments from YouTube
def get_youtube_comments(video_id, api_key, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    ).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=max_results,
                textFormat='plainText'
            ).execute()
        else:
            break

    return comments

# Step 2: Convert Comments to Vectors
def comment_to_vector(comment, tokenizer, model):
    inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].numpy()
    return vector.flatten()

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Step 3: Store Vectors in Milvus
def insert_comment(collection, comment, vector):
    collection.insert([[None], [vector], [comment]])

def main():
    api_key = 'AIzaSyA-QbJi3qt86rbYbkfEZf-h1Ti3RUEDKoQ'  # Replace with your YouTube Data API key
    video_id = 'b6EjyQl-IMI' # Replace with your YouTube video ID
    comments = get_youtube_comments(video_id, api_key)
    print(comments)
    return

    comment_vectors = [comment_to_vector(comment, tokenizer, model) for comment in comments]

    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Define the schema
    fields = [
        FieldSchema(name="comment_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.STRING)
    ]
    schema = CollectionSchema(fields, "Comments collection")

    # Create the collection
    collection = Collection("comments", schema)

    # Insert all comments into the collection
    for comment, vector in zip(comments, comment_vectors):
        insert_comment(collection, comment, vector)

    # Example querying
    query_vector = comment_to_vector("This is a sample query comment.", tokenizer, model)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search([query_vector], "embedding", params=search_params, limit=5, expr=None)

    for result in results[0]:
        print(f"ID: {result.id}, Distance: {result.distance}, Text: {result.entity.get('text')}")

if __name__ == "__main__":
    main()
