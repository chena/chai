import os
import json
import requests
import string
import ast
import tiktoken
import openai
import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.cluster import KMeans
from ast import literal_eval

MISO_APP_TOKEN = os.environ['MISO_OTHERS_API_KEY']
USER_ID = os.environ['USER_ID']
MISO_BASE_URL = 'https://api-edge.askmiso.com'
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_PATH = "data/comments_embeddings_talk.csv"

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))

def token_check(documents: list):
  doc_token_count = [num_tokens(d) for d in documents]
  print(sorted(doc_token_count))

def fetch_comments() -> list:
  # Get comments from Miso
  headers = {
    "Content-Type": "application/json",
    "X-API-KEY": MISO_APP_TOKEN
  }
  request_body = {
    "q": "*",
    "fl": ["description"],
    "rows": 700,
    "type": 'COMMENT',
    "anonymous_id": USER_ID,
    "fq": "custom_attributes.courseId:\"5fc4a352d375951a03cc0d45\""
  }
  res = json.loads(requests.post(MISO_BASE_URL + '/v1/search/search', json = request_body, headers=headers).text)
  documents = [a['description'].replace('\n', '') for a in res['data']['products']]
  translator = str.maketrans('', '', string.punctuation + "【】！？，。—")
  # seg_list = jieba.cut(doc.translate(translator))
  return [doc.translate(translator) for doc in documents]

def calculate_embeddings(documents: list):
  response = openai.Embedding.create(model=EMBEDDING_MODEL, input=documents)
  embeddings = [e['embedding'] for e in response['data']]
  df = pd.DataFrame({'text': documents, 'embedding': embeddings})
  df.to_csv(EMBEDDING_PATH, index=False)

def df_setup():
  df = pd.read_csv(EMBEDDING_PATH)
  # df['embedding'] = df['embedding'].apply(ast.literal_eval)
  df['embedding'] = df.embedding.apply(ast.literal_eval).apply(np.array)  # convert string to numpy array
  return df

def cluster():
  df = df_setup()
  matrix = np.vstack(df.embedding.values)
  print(matrix.shape)

  n_clusters = 4
  kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
  kmeans.fit(matrix)
  df['Cluster'] = kmeans.labels_

  for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")
    # print('Cluster {}'.format(i))
    comments = [
      q[:20] for q in df[df.Cluster == i]
        .sample(10, random_state=42)
        .text.values
    ]
    print(comments)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f'What do the following comments have in common?\n\nComments:\n"""\n{comments}\n"""\n\nTheme:',
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response["choices"][0]["text"].replace("\n", ""))
    print("-" * 100)

# step 1
documents = list(filter(lambda d: len(d) > 10, fetch_comments()))
# print([d for d in documents])
# print(len(documents))

# step 2
# token_check(comments)

# step 3
# calculate_embeddings(documents)

# step 4
df_setup()

# step 5
cluster()
