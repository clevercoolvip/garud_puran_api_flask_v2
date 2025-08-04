import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from config import data_path
import time

start = time.perf_counter()
data = pd.read_csv(data_path)
df1 = data.iloc[:28, :]

model = SentenceTransformer('all-MiniLM-L6-v2')

class_texts = df1.iloc[:, 1].to_list()
class_names = df1.iloc[:, 0].to_list()
class_texts_embeddings = torch.load("class_embeddings.pt")

def get_result(query):
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, class_texts_embeddings)
    predicted_class = torch.argmax(cos_scores).item()
    return class_names[predicted_class]

if __name__=="__main__":
    print(data.iloc[35, 1])
    print(get_result(data.iloc[35, 1]))
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start}")