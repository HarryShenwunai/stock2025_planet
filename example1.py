from flask import Flask, render_template
import requests

app = Flask(__name__)

def fetch_api_data(url):
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()
    if isinstance(data, dict):
        return data.get('groups', []) or data.get('data', []) or data.get('results', []) or data.get('items', []) or data.get('events', [])
    elif isinstance(data, list):
        return data
    return []

def fetch_stock_data(url):
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data

@app.route('/')
def events():
    url = "https://dataserver.datasum.ai/techsum/api/v3/events"
    groups = fetch_api_data(url)
    return render_template("groups.html", groups=groups, page_title="TechSum Events Metrics", source_url=url)

@app.route('/aapl')
def aapl():
    url = "https://dataserver.datasum.ai/stock-info/api/v1/stock?symbol=AAPL"
    stock_data = fetch_stock_data(url)
    return render_template("stock.html", stock_data=stock_data, page_title="AAPL Stock Metrics", source_url=url)

if __name__ == '__main__':
    app.run(debug=True)