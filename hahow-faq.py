import os
import json
import requests
import numpy as np
import string
import jieba

MISO_APP_TOKEN = os.environ['MISO_OTHERS_API_KEY']
USER_ID = os.environ['USER_ID']
MISO_BASE_URL = 'https://api-edge.askmiso.com'
GPT_MODEL = "gpt-3.5-turbo"

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
documents = [a['title'] + ' ' + a['description'] for a in res['data']['products']]
doc_len = [len(d) for d in documents]
print(sum(doc_len))
# print(documents[-1])