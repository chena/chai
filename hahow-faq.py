import os
import json
import requests
import string
import tiktoken
import openai
import pandas as pd

MISO_APP_TOKEN = os.environ['MISO_OTHERS_API_KEY']
USER_ID = os.environ['USER_ID']
MISO_BASE_URL = 'https://api-edge.askmiso.com'
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))

def token_check(document: list):
  doc_token_count = [num_tokens(d) for d in documents]
  print(sorted(doc_token_count))

def fetch_articles() -> list:
  # Get FAQ articles from Miso
  headers = {
    "Content-Type": "application/json",
    "X-API-KEY": MISO_APP_TOKEN
  }
  request_body = {
    "q": "*",
    "fl": ["title", "description"],
    "rows": 1000,
    "type": 'FAQ',
    "anonymous_id": USER_ID
  }
  res = json.loads(requests.post(MISO_BASE_URL + '/v1/search/search', json = request_body, headers=headers).text)
  documents = [a['title'] + ' ' + a['description'].replace('\n', '') for a in res['data']['products']]
  translator = str.maketrans('', '', string.punctuation + "【】！？，。—")
  # seg_list = jieba.cut(doc.translate(translator))
  return [doc.translate(translator) for doc in documents]

def calculate_embeddings(documents: list):
  response = openai.Embedding.create(model=EMBEDDING_MODEL, input=documents)
  embeddings = [e["embedding"] for e in response["data"]]
  df = pd.DataFrame({"text": documents, "embedding": embeddings})
  df.to_csv("data/faq_embeddings.csv", index=False)

documents = fetch_articles()
calculate_embeddings(documents)

# token check
# token_check(documents)



