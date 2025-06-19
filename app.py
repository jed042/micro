import os
import time
import requests
from flask import Flask, render_template
from newsapi import NewsApiClient
from groq import Groq

app = Flask(__name__)

class FusionBrainAPI:
    def __init__(self, api_key, secret_key):
        self.base_url = "https://api-key.fusionbrain.ai/"
        self.headers = {
            "X-Key": f"Key {api_key}",
            "X-Secret": f"Secret {secret_key}"
        }

    def get_pipelines(self):
        response = requests.get(self.base_url + "key/api/v1/pipelines", headers=self.headers)
        return response.json()

    def generate(self, pipeline_id, query, style=None):
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": 1024,
            "height": 1024,
            "generateParams": {
                "query": query,
                "style": style if style else "PHOTO"
            }
        }
        response = requests.post(self.base_url + f"key/api/v1/pipeline/{pipeline_id}/run", json=params, headers=self.headers)
        return response.json()

    def check_status(self, request_id):
        response = requests.get(self.base_url + f"key/api/v1/pipeline/status/{request_id}", headers=self.headers)
        return response.json()

@app.before_first_request
def initialize():
    newsapi_key = os.environ.get('NEWSAPI_KEY')
    groq_key = os.environ.get('GROQ_API_KEY')
    fb_api_key = os.environ.get('FB_API_KEY')
    fb_secret_key = os.environ.get('FB_SECRET_KEY')

    if not all([newsapi_key, groq_key, fb_api_key, fb_secret_key]):
        print("Missing API keys")
        return

    newsapi = NewsApiClient(api_key=newsapi_key)
    top_headlines = newsapi.get_top_headlines(category='general', language='en', page_size=10)

    groq_client = Groq(api_key=groq_key)
    fb_api = FusionBrainAPI(fb_api_key, fb_secret_key)
    pipelines = fb_api.get_pipelines()
    if pipelines['pipelines']:
        pipeline_id = pipelines['pipelines'][0]['id']
    else:
        print("No pipelines available")
        return

    for article in top_headlines['articles']:
        text = article.get('description', article.get('title', ''))
        if not text:
            continue
        # Summarize with Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
                {"role": "user", "content": f"Summarize the following news article in 50 words: {text}"}
            ],
            response_format={"type": "text"}
        )
        summary = response.choices[0].message.content
        # Generate image with FusionBrainAI
        task = fb_api.generate(pipeline_id, summary)
        request_id = task['id']
        while True:
            status = fb_api.check_status(request_id)
            if status['status'] == 'DONE':
                image_url = status['result'][0]['image']  # Assuming this is how it's returned
                break
            elif status['status'] == 'FAIL':
                print(f"Generation failed for {summary}")
                break
            time.sleep(5)
        else:
            # Download image
            image_data = requests.get(image_url).content
            timestamp = int(time.time())
            filename = f"image_{timestamp}.jpg"
            with open(os.path.join('static', 'images', filename), 'wb') as f:
                f.write(image_data)

@app.route('/')
def index():
    images = sorted(os.listdir('static/images'), reverse=True)[:10]
    return render_template('index.html', images=images)
