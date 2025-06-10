import os
import re
import time
import json
import random
import finnhub
import torch
import gradio as gr
import pandas as pd
import yfinance as yf
from pynvml import *
from peft import PeftModel
from collections import defaultdict
from datetime import date, datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


access_token = os.environ["HF_TOKEN"]
finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])

base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token,
    trust_remote_code=True,
    device_map="auto",  # ou "auto"
    torch_dtype=torch.float16,  # moins exigeant que float16
    offload_folder="offload/"
)
model = PeftModel.from_pretrained(
    base_model,
    'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
    offload_folder="offload/"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token
)

streamer = TextStreamer(tokenizer)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."


def print_gpu_utilization():
    
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_curday():
    
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):
    
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)

    return date.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):

    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    if len(stock_data) == 0:
        raise gr.Error(f"Failed to download stock price data for symbol {stock_symbol} from yfinance!")
    
#     print(stock_data)
    
    dates, prices = [], []
    available_dates = stock_data.index.format()
    
    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })


def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(f"Fetching news for {symbol}: {start_date} to {end_date}")
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        print(f"Found {len(weekly_news)} news items")
        if len(weekly_news) == 0:
            raise gr.Error(f"No company news found for symbol {symbol} from finnhub!")
        else :
            print(weekly_news)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data

def get_crypto_news(symbol, data):
    """
    Get crypto news for a given symbol using the CoinDesk News API
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        data: DataFrame with 'Start Date' and 'End Date' columns
    
    Returns:
        DataFrame with added 'News' column containing JSON strings of news
    """
    import requests
    from datetime import datetime, timedelta
    
    # Strip the '-USD' suffix if present
    clean_symbol = symbol.replace('-USD', '')
    
    # Mapping des symboles cryptos vers leurs noms complets pour la recherche
    crypto_names = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'ADA': 'Cardano',
        'XRP': 'Ripple',
        'DOT': 'Polkadot',
        'SOL': 'Solana',
        'AVAX': 'Avalanche',
        'MATIC': 'Polygon',
        'LINK': 'Chainlink',
        'UNI': 'Uniswap',
        'DOGE': 'Dogecoin',
        'LTC': 'Litecoin',
        'BCH': 'Bitcoin Cash',
        'ATOM': 'Cosmos',
        'ALGO': 'Algorand'
    }
    
    # Utiliser le nom complet si disponible, sinon le symbole
    search_term = crypto_names.get(clean_symbol.upper(), clean_symbol)
    
    news_list = []
    
    for i, row in data.iterrows():
        start_date = row['Start Date']
        end_date = row['End Date']
        
        # Convertir les dates en timestamps Unix
        start_ts = int(datetime.timestamp(start_date))
        end_ts = int(datetime.timestamp(end_date))
        
        # Utiliser l'API CoinDesk avec l'endpoint search
        url = "https://data-api.coindesk.com/news/v1/search"
        
        params = {
            'search_string': search_term,
            'lang': 'EN',
            'source_key': 'coindesk',
            'from_ts': start_ts,
            'to_ts': end_ts,
            'limit': 50,
            'sort_by': 'published_on',
            'sort_order': 'desc',
            "api_key": os.environ.get("COINDESK_API_KEY")
        }
        
        try:
            # Ajouter des headers pour l'API CoinDesk
            headers = {
                'User-Agent': 'FinGPT-Forecaster/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            print(f"URL: {response.url}")
            print(f"Status Code: {response.status_code}")
            
            response.raise_for_status()
            
            api_response = response.json()
            print(f"API Response keys: {api_response.keys()}")
            
            # Traiter les nouvelles selon la structure de l'API CoinDesk
            weekly_news = []
            
            # Vérifier la structure de réponse correcte
            if 'Data' in api_response and isinstance(api_response['Data'], list):
                all_news = api_response['Data']
            elif 'Data' in api_response and isinstance(api_response['Data'], dict):
                # Si Data est un objet, chercher les articles à l'intérieur
                all_news = api_response['Data'].get('articles', api_response['Data'].get('items', []))
            else:
                print(f"Unexpected API response structure: {api_response}")
                all_news = []
            
            print(f"Found {len(all_news)} raw news items")
            
            for news in all_news:
                # Utiliser PUBLISHED_ON selon la documentation
                news_ts = news.get('PUBLISHED_ON', 0)
                
                # Vérifier si la nouvelle est dans la plage de dates
                if start_ts <= news_ts <= end_ts:
                    # Extraire les champs selon la structure API
                    title = news.get('TITLE', '')
                    body = news.get('BODY', news.get('SUBTITLE', ''))
                    
                    # Limiter la longueur du résumé
                    summary = body[:200] + '...' if len(body) > 200 else body
                    
                    # Extraire l'URL
                    url_field = news.get('URL', '')
                    
                    # Extraire le sentiment si disponible
                    sentiment = news.get('SENTIMENT', '')
                    
                    # Extraire les mots-clés si disponibles
                    keywords = news.get('KEYWORDS', '')
                    
                    # Extraire les données de source
                    source_name = "CoinDesk"
                    if 'SOURCE_DATA' in news and news['SOURCE_DATA']:
                        source_name = news['SOURCE_DATA'].get('NAME', 'CoinDesk')
                    
                    weekly_news.append({
                        "date": datetime.fromtimestamp(news_ts).strftime('%Y%m%d%H%M%S'),
                        "headline": title,
                        "summary": summary,
                        "url": url_field,
                        "source": source_name,
                        "sentiment": sentiment,
                        "keywords": keywords,
                        "id": news.get('ID', ''),
                        "guid": news.get('GUID', '')
                    })
            
            # Trier par date (plus récent en premier)
            weekly_news.sort(key=lambda x: x['date'], reverse=True)
            
            # Si aucune nouvelle trouvée dans la plage exacte, essayer une recherche plus large
            if not weekly_news:
                print(f"No news found for {symbol} in exact date range, trying broader search")
                
                # Élargir la recherche de 7 jours
                extended_start = start_ts - (3 * 24 * 3600)
                extended_end = end_ts + (3 * 24 * 3600)
                
                params_extended = {
                    'search_string': search_term,
                    'lang': 'EN',
                    'source_key': 'coindesk',
                    'from_ts': extended_start,
                    'to_ts': extended_end,
                    'limit': 10,
                    "api_key": os.environ.get("COINDESK_API_KEY")
                }
                
                try:
                    response_extended = requests.get(url, params=params_extended, headers=headers, timeout=10)
                    if response_extended.status_code == 200:
                        api_response_extended = response_extended.json()
                        
                        if 'Data' in api_response_extended and isinstance(api_response_extended['Data'], list):
                            extended_news = api_response_extended['Data'][:5]  # Prendre seulement 5 nouvelles
                        else:
                            extended_news = []
                        
                        for news in extended_news:
                            news_ts = news.get('PUBLISHED_ON', 0)
                            
                            if news_ts:
                                title = news.get('TITLE', '')
                                body = news.get('BODY', news.get('SUBTITLE', ''))
                                summary = body[:200] + '...' if len(body) > 200 else body
                                
                                source_name = "CoinDesk"
                                if 'SOURCE_DATA' in news and news['SOURCE_DATA']:
                                    source_name = news['SOURCE_DATA'].get('NAME', 'CoinDesk')
                                
                                weekly_news.append({
                                    "date": datetime.fromtimestamp(news_ts).strftime('%Y%m%d%H%M%S'),
                                    "headline": title,
                                    "summary": summary,
                                    "url": news.get('URL', ''),
                                    "source": source_name,
                                    "sentiment": news.get('SENTIMENT', ''),
                                    "keywords": news.get('KEYWORDS', ''),
                                    "id": news.get('ID', ''),
                                    "guid": news.get('GUID', '')
                                })
                                
                except Exception as e:
                    print(f"Extended search failed for {symbol}: {str(e)}")
            
            print(f"Found {len(weekly_news)} news items for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            news_list.append(json.dumps(weekly_news))
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching news for {symbol}")
            news_list.append(json.dumps([]))
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching news for {symbol}: {str(e)}")
            print(f"Response content: {e.response.text if hasattr(e, 'response') and e.response else 'No response'}")
            # Fallback vers l'API CryptoCompare si CoinDesk échoue
            try:
                fallback_news = get_crypto_news_fallback(clean_symbol, start_date, end_date)
                news_list.append(json.dumps(fallback_news))
            except:
                news_list.append(json.dumps([]))
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            news_list.append(json.dumps([]))
            
        # Respecter les limites de taux
        time.sleep(1)  # 1 seconde entre les requêtes
    
    data['News'] = news_list
    return data


def get_crypto_news_fallback(clean_symbol, start_date, end_date):
    """
    Fonction de fallback utilisant CryptoCompare si CoinDesk échoue
    """
    try:
        import requests
        
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={clean_symbol}&excludeCategories=Sponsored"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            all_news = response.json().get('Data', [])
            start_ts = int(datetime.timestamp(start_date))
            end_ts = int(datetime.timestamp(end_date))
            
            fallback_news = []
            for news in all_news[:10]:  # Limiter à 10 nouvelles
                news_ts = news.get('published_on', 0)
                if start_ts <= news_ts <= end_ts:
                    fallback_news.append({
                        "date": datetime.fromtimestamp(news_ts).strftime('%Y%m%d%H%M%S'),
                        "headline": news.get('title', ''),
                        "summary": news.get('body', '')[:200] + '...',
                        "url": news.get('url', ''),
                        "source": "CryptoCompare"
                    })
            
            return fallback_news
        
    except Exception as e:
        print(f"Fallback also failed: {str(e)}")
    
    return []


def get_company_prompt(symbol):

    profile = finnhub_client.company_profile2(symbol=symbol)
    if not profile:
        raise gr.Error(f"Failed to find company profile for symbol {symbol} from finnhub!")
        
    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    if not basic_financials['series']:
        raise gr.Error(f"Failed to find basic financials for symbol {symbol} from finnhub!")
        
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic
    

def get_all_prompts_online(symbol, data, curday, with_basics=True):

    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."
        
    period = "{} to {}".format(curday, n_weeks_before(curday, -1))
    
    if with_basics:
        basics = get_current_basics(symbol, curday)
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."

    info = company_prompt + '\n' + prompt + '\n' + basics
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
        
    return info, prompt


def construct_prompt(ticker, curday, n_weeks, use_basics):

    try:
        steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    except Exception:
        raise gr.Error(f"Invalid date {curday}!")
        
    data = get_stock_data(ticker, steps)
    data = get_crypto_news(ticker, data) #get_news(ticker, data)
    data['Basics'] = [json.dumps({})] * len(data)
    # print(data)
    
    info, prompt = get_all_prompts_online(ticker, data, curday, use_basics)
    
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    # print(prompt)
    
    return info, prompt


def predict(ticker, date, n_weeks, use_basics):

    print_gpu_utilization()

    info, prompt = construct_prompt(ticker, date, n_weeks, use_basics)
      
    inputs = tokenizer(
        prompt, return_tensors='pt', padding=False
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    print("Inputs loaded onto devices.")
        
    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True, streamer=streamer
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

    torch.cuda.empty_cache()
    
    return info, answer


demo = gr.Interface(
    predict,
    inputs=[
        gr.Textbox(
            label="Ticker",
            value="AAPL",
            info="Companys from Dow-30 are recommended"
        ),
        gr.Textbox(
            label="Date",
            value=get_curday,
            info="Date from which the prediction is made, use format yyyy-mm-dd"
        ),
        gr.Slider(
            minimum=1,
            maximum=4,
            value=3,
            step=1,
            label="n_weeks",
            info="Information of the past n weeks will be utilized, choose between 1 and 4"
        ),
        gr.Checkbox(
            label="Use Latest Basic Financials",
            value=False,
            info="If checked, the latest quarterly reported basic financials of the company is taken into account."
        )
    ],
    outputs=[
        gr.Textbox(
            label="Information"
        ),
        gr.Textbox(
            label="Response"
        )
    ],
    title="FinGPT-Forecaster",
    description="""FinGPT-Forecaster takes random market news and optional basic financials related to the specified company from the past few weeks as input and responds with the company's **positive developments** and **potential concerns**. Then it gives out a **prediction** of stock price movement for the coming week and its **analysis** summary.
This model is finetuned on Llama2-7b-chat-hf with LoRA on the past year's DOW30 market data. Inference in this demo uses fp16 and **welcomes any ticker symbol**.
Company profile & Market news & Basic financials & Stock prices are retrieved using **yfinance & finnhub**.
This is just a demo showing what this model is capable of. Results inferred from randomly chosen news can be strongly biased.
For more detailed and customized implementation, refer to our FinGPT project: <https://github.com/AI4Finance-Foundation/FinGPT>
**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
"""
)

demo.launch()