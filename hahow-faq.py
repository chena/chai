import os
import json
import requests
import string
import ast
import tiktoken
import openai
import pandas as pd
from scipy import spatial

MISO_APP_TOKEN = os.environ['MISO_OTHERS_API_KEY']
USER_ID = os.environ['USER_ID']
MISO_BASE_URL = 'https://api-edge.askmiso.com'
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_PATH = "data/faq_embeddings.csv"

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
  df.to_csv(EMBEDDING_PATH, index=False)

def qa_without_rag():
  """
  q1 = 'Hahowise 學習助教是誰？'
  q2 = 'Hahowise 要怎麼使用？'
  q3 = 'Hahow 2023 有什麼新功能？'
  """
  q3 = 'Hahow 2023 有什麼新功能？'
  """
  很抱歉，我無法提供有關 Hahow 2023 的具體資訊，因為我是一個 AI 助手，無法預測未來的功能更新。建議您直接向 Hahow 官方網站或客服詢問，以獲得最準確的資訊。
  """
  response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about Hahow'},
        {'role': 'user', 'content': q3},
    ],
    model=GPT_MODEL,
    temperature=0,
  )
  print(response['choices'][0]['message']['content'])

def def_setup():
  df = pd.read_csv(EMBEDDING_PATH)
  df['embedding'] = df['embedding'].apply(ast.literal_eval)
  return df

def search_related_articles(
  query: str,
  df: pd.DataFrame,
  relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
  top_n: int = 10
) -> tuple[list[str], list[float]]:
  query_embedding_response = openai.Embedding.create(
    model=EMBEDDING_MODEL,
    input=query,
  )
  query_embedding = query_embedding_response["data"][0]["embedding"]
  strings_and_relatednesses = [
    (row["text"], relatedness_fn(query_embedding, row["embedding"]))
    for i, row in df.iterrows()
  ]
  strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
  strings, relatednesses = zip(*strings_and_relatednesses)
  return strings[:top_n], relatednesses[:top_n]

def query_message(
  query: str,
  df: pd.DataFrame,
  model: str,
  token_budget: int
) -> str:
  articles, relatednesses = search_related_articles(query, df)
  introduction = 'Use the below help articles from Hahow Help Center to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
  question = f"\n\nQuestion: {query}"
  message = introduction
  for article in articles:
    next_article = f'\n\nHelp Article:\n"""\n{article}\n"""'
    if (
        num_tokens(message + next_article + question, model=model)
        > token_budget
    ):
        break
    else:
        message += next_article
  return message + question

def ask(
  query: str,
  df: pd.DataFrame,
  model: str = GPT_MODEL,
  token_budget: int = 4096 - 500,
  print_message: bool = False,
) -> str:
  message = query_message(query, df, model=model, token_budget=token_budget)
  if print_message:
    print(message)
  messages = [
    {"role": "system", "content": "You are Hahow's customer support assistant. You answer questions Hahow's features and issues that users may encounter."},
    {"role": "user", "content": message},
  ]
  response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0
  )
  response_message = response["choices"][0]["message"]["content"]
  return response_message


"""step 1: fetch data"""
# documents = fetch_articles()

"""step 2: token check"""
# token_check(documents)

"""step 3: calculate embeddings"""
# calculate_embeddings(documents)

"""step 4: test without RAG"""
# qa_without_rag()

"""step 5: set up RAG"""
df = def_setup()

"""step 6: calculate relatedness"""
# strings, relatednesses = search_related_articles("要如何取得完課證明？", df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#   print(f"{relatedness=:.3f}")
#   print(string)

"""step 7: ask"""
question_bank = [
  'Hahowise 學習助教是誰？',
  'Hahowise 要怎麼使用？',
  '影片卡住怎麼辦？',
  '錯過直播怎麼辦？',
  '要如何取得完課證明？',
  '課程開課後，開始觀看前還可以退課嗎？',
  '退款後 coupon 會退嗎？',
  '募資失敗會退款嗎？',
  '老師不理我怎麼辦？',
  'Hahow 2024 會有什麼新功能？',
]
num = 0
for q in question_bank:
  num += 1
  print('{}. {}'.format(str(num), q))
  ans = ask(q, df)
  print(ans + '\n')