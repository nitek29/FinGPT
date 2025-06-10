import os
import re
import csv
import math
import time
import json
import finnhub
from tqdm import tqdm
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
import datasets
from datasets import Dataset
from openai import OpenAI

# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

from indices import *
from prompt import get_all_prompts

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_KEY"))
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))


# ----------------------------------------------------------------------------------- #
# ---------------------------- RAW FINANCIAL ACQUISITION ---------------------------- #
# ----------------------------------------------------------------------------------- #

def bin_mapping(ret):
    
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    
    return up_down + (str(integer) if integer <= 5 else '5+')


def get_returns(stock_symbol, start_date, end_date):
    """
    Get weekly returns data for a given stock/crypto symbol
    """
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
    
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
        
    #print(f"Available columns: {stock_data.columns}")
    
    # Use 'Close' price if 'Adj Close' is not available
    price_col = 'Close' if 'Adj Close' not in stock_data.columns else 'Adj Close'
    
    # Resample to weekly data, getting the last price of each week
    weekly_prices = stock_data[price_col].resample('W-FRI').last()
    
    # Calculate returns and create the DataFrame with proper date alignment
    weekly_data = pd.DataFrame()
    weekly_data['End Date'] = weekly_prices.index[1:]  # Shift dates by 1 to align with returns
    weekly_data['Start Date'] = weekly_prices.index[:-1]  
    weekly_data['End Price'] = weekly_prices.values[1:]
    weekly_data['Start Price'] = weekly_prices.values[:-1]
    
    # Calculate returns properly using start and end prices
    weekly_data['Weekly Returns'] = (weekly_data['End Price'] - weekly_data['Start Price']) / weekly_data['Start Price']
    
    # Add the bin labels
    weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)
    
    # Reset index for consistency
    weekly_data = weekly_data.reset_index(drop=True)
    
    # Verify data quality
    if weekly_data.empty:
        raise ValueError(f"No weekly data could be calculated for {stock_symbol}")
    
    if weekly_data.isnull().any().any():
        print(f"Warning: NaN values found in {stock_symbol} data")
        weekly_data = weekly_data.dropna()
        
    return weekly_data


def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        # print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
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

def get_news_multi_source(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        # print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
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


def get_crypto_news_multi_source(symbol, data):
    """
    Get crypto news from multiple free sources
    """
    # Strip the '-USD' suffix if present
    clean_symbol = symbol.replace('-USD', '')
    
    news_list = []
    
    for i, row in data.iterrows():
        start_date = row['Start Date']
        end_date = row['End Date']
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        weekly_news = []
        
        # Source 1: CryptoCompare
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={clean_symbol}"
            response = requests.get(url)
            if response.status_code == 200:
                all_news = response.json().get('Data', [])
                
                # Convert dates to timestamps for comparison
                start_ts = int(datetime.timestamp(start_date))
                end_ts = int(datetime.timestamp(end_date))
                
                for news in all_news:
                    news_ts = news.get('published_on', 0)
                    if start_ts <= news_ts <= end_ts:
                        weekly_news.append({
                            "source": "CryptoCompare",
                            "date": datetime.fromtimestamp(news_ts).strftime('%Y%m%d%H%M%S'),
                            "headline": news.get('title', ''),
                            "summary": news.get('body', '')[:200] + '...'
                        })
        except Exception as e:
            print(f"CryptoCompare API error: {str(e)}")
        
        # Source 2: Coinpaprika (another free API)
        try:
            url = f"https://api.coinpaprika.com/v1/coins/{clean_symbol.lower()}/events"
            response = requests.get(url)
            if response.status_code == 200:
                events = response.json()
                for event in events:
                    event_date = datetime.strptime(event.get('date', ''), '%Y-%m-%dT%H:%M:%SZ')
                    if start_date <= event_date <= end_date:
                        weekly_news.append({
                            "source": "Coinpaprika",
                            "date": event_date.strftime('%Y%m%d%H%M%S'),
                            "headline": event.get('name', ''),
                            "summary": event.get('description', '')[:200] + '...'
                        })
        except Exception as e:
            print(f"Coinpaprika API error: {str(e)}")
            
        # Sort all collected news by date
        weekly_news.sort(key=lambda x: x['date'])
        
        news_list.append(json.dumps(weekly_news))
        time.sleep(0.5)  # Respect rate limits
    
    data['News'] = news_list
    return data


def get_basics(symbol, data, start_date, always=False):
    
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
            
    for i, row in data.iterrows():
        
        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = start_date if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
        
    data['Basics'] = final_basics
    
    return data
    

def prepare_data_for_symbol(symbol, data_dir, start_date, end_date, with_basics=True):
    
    data = get_returns(symbol, start_date, end_date)
    data = get_crypto_news(symbol, data) #get_news(symbol, data)
    
    if with_basics:
        data = get_basics(symbol, data, start_date)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}.csv")
    else:
        data['Basics'] = [json.dumps({})] * len(data)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv")
    
    return data


# ----------------------------------------------------------------------------------- #
# ---------------------------------- GPT4 ANALYSIS ---------------------------------- #
# ----------------------------------------------------------------------------------- #


def append_to_csv(filename, input_data, output_data):
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

        
def initialize_csv(filename):
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])


def query_gpt4_original(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    """
    Query GPT-4 API with proper rate limit handling and exponential backoff
    """
    import time
    import random
    import backoff  # pip install backoff
    from openai import OpenAI
    
    # Create OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Import the proper exception classes based on the OpenAI version
    import openai
    
    # Find out which exception classes we should use
    try:
        # For OpenAI 1.0.0+
        from openai import RateLimitError, APIError, APIConnectionError
        from openai import APITimeoutError as Timeout
        ServiceUnavailableError = APIError  # Use APIError as fallback
        InternalServerError = APIError  # Use APIError as fallback
    except ImportError:
        # For older versions
        from openai.error import RateLimitError, APIError, APIConnectionError, Timeout, ServiceUnavailableError
    
    # Fonction avec décorateur backoff pour gérer les limites de taux avec retries exponentiels
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APIConnectionError, Timeout),
        max_tries=5,
        max_time=180,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries")
    )
    def call_gpt_with_backoff(system_prompt, prompt):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt-4.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_gpt-4.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        
        # Vérification des prompts générés
        print(f"Generated {len(prompts)} prompts for {symbol}")
        if len(prompts) > 0:
            print(f"Sample prompt: {prompts[0][:200]}...")
        else:
            print(f"Warning: No prompts generated for {symbol}!")
            continue
        
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        
        for i, prompt in enumerate(prompts):
            
            if i < pre_done:
                continue
                
            # Ajouter une pause adaptative avant chaque appel
            if i > 0:
                # Pause plus longue tous les 5 appels
                if i % 5 == 0:
                    sleep_time = random.uniform(6, 12)
                else:
                    sleep_time = random.uniform(1, 3)
                
                print(f"Pausing for {sleep_time:.2f}s before next request...")
                time.sleep(sleep_time)
            
            try:
                # Utiliser la fonction avec backoff
                completion = call_gpt_with_backoff(system_prompt, prompt)
                answer = completion.choices[0].message.content
                
                # Enregistrer immédiatement
                append_to_csv(csv_file, prompt, answer)
                print(f"Processed {i+1}/{len(prompts)} for {symbol}")
                
            except Exception as e:
                error_message = str(e)
                print(f"❌ Error processing prompt {i} for {symbol}: {error_message}")
                
                # Vérifier si c'est une erreur de quota
                if "insufficient_quota" in error_message:
                    print("⚠️ API quota exceeded. Saving remaining prompts and exiting.")
                    
                    # Sauvegarder prompts restants
                    remaining_prompts = prompts[i:]
                    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts.json"
                    with open(remaining_file, "w") as f:
                        json.dump(remaining_prompts, f)
                    
                    print(f"Remaining prompts saved to {remaining_file}")
                    return  # Quitter la fonction
                
                print("Skipping this prompt and continuing...")
                append_to_csv(csv_file, prompt, f"ERROR: {error_message}")



# ----------------------------------------------------------------------------------- #
# -------------------------- TRANSFORM INTO TRAINING FORMAT ------------------------- #
# ----------------------------------------------------------------------------------- #

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPTS = {
    "company": "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",

    "crypto": "You are a seasoned crypto market analyst. Your task is to list the positive developments and potential concerns for cryptocurrencies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the cryptocurrencies price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",
}

def gpt4_to_llama(symbol, data_dir, start_date, end_date, with_basics=True):
    csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_mistral.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_mistral.csv'
    
    df = pd.read_csv(csv_file)
    
    prompts, answers, periods, labels = [], [], [], []
    
    for i, row in df.iterrows():
        try:
            # Ensure both prompt and answer are strings
            prompt, answer = str(row['prompt']), str(row['answer'])
            print(f"Processing row {i} for {symbol} - Prompt : {prompt}, Answer : {answer}")
            
            # Skip empty content
            if not prompt or not answer or prompt.lower() == 'nan' or answer.lower() == 'nan':
                print(f"Skipping row {i} - empty content")
                continue
            
            # Extract prediction period and label
            res = re.search(r"Then let's assume your prediction for next week \((.*)\) is ((:?up|down) by .*%).", prompt)
            if not res:
                print(f"Skipping row {i} - no prediction pattern found")
                continue
                
            period, label = res.group(1), res.group(2)
            
            # Transform prompt
            prompt = re.sub(
                r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((:?.*)%). Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.", 
                f"Then make your prediction of the {symbol} cryptocurrency price movement for next week ({period}). Provide a summary analysis to support your prediction.",
                prompt
            )
            
            # Transform answer
            if "[Prediction & Analysis]:" in answer:
                answer = re.sub(
                    r"\[Prediction & Analysis\]:\s*",
                    f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
                    answer
                )
            else:
                # Add header if missing
                answer = f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: {answer}"
                
            # Create final prompt with system prompts
            system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
            new_system_prompt = system_prompt.replace(':\n...', '\nPrediction: ...\nAnalysis: ...')
            
            prompt = B_INST + B_SYS + new_system_prompt + E_SYS + prompt + E_INST
            
            # Add to result lists
            prompts.append(prompt)
            answers.append(answer)
            periods.append(period)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing row {i} for {symbol}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
    return {
        "prompt": prompts,
        "answer": answers,
        "period": periods,
        "label": labels,
    }


def create_dataset(symbol_list, data_dir, start_date, end_date, train_ratio=0.8, with_basics=True):

    train_dataset_list = []
    test_dataset_list = []

    for symbol in symbol_list:

        #print("test1")
        data_dict = gpt4_to_llama(symbol, data_dir, start_date, end_date,  with_basics)
        #print("test2")
        # print(data_dict['prompt'][-1])
        # print(data_dict['answer'][-1])
        symbols = [symbol] * len(data_dict['label'])
        data_dict.update({"symbol": symbols})

        dataset = Dataset.from_dict(data_dict)
        train_size = round(float(train_ratio) * len(dataset))
        print(f"Processed {symbol} - Train size: {train_size}, Test size: {len(dataset) - train_size}")
        #print(dataset.select(range(train_size)))

        train_dataset_list.append(dataset.select(range(train_size)))
        if train_size >= len(dataset):
            continue
        test_dataset_list.append(dataset.select(range(train_size, len(dataset))))
        

    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list)

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset


def query_gpt4(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    """
    Query GPT-4 API with batch processing - single request with all prompts
    """
    import time
    import random
    import backoff
    from openai import OpenAI
    
    # Create OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Import exception classes
    try:
        from openai import RateLimitError, APIError, APIConnectionError
        from openai import APITimeoutError as Timeout
    except ImportError:
        from openai.error import RateLimitError, APIError, APIConnectionError, Timeout

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APIConnectionError, Timeout),
        max_tries=3,
        max_time=300,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries")
    )
    def call_gpt_batch(system_prompt, batch_prompt):
        return client.chat.completions.create(
            model="gpt-3.5-turbo",  # Utilisation d'un modèle moins coûteux
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_prompt}
            ],
            temperature=0.7,
            max_tokens=4000  # Augmenté pour traiter plusieurs prompts
        )

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt-4.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_gpt-4.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        
        # Vérification des prompts générés
        print(f"Generated {len(prompts)} prompts for {symbol}")
        if len(prompts) == 0:
            print(f"Warning: No prompts generated for {symbol}!")
            continue
            
        # Filtrer les prompts non traités
        remaining_prompts = prompts[pre_done:]
        if len(remaining_prompts) == 0:
            print(f"All prompts already processed for {symbol}")
            continue
            
        print(f"Processing {len(remaining_prompts)} remaining prompts for {symbol}")
        
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        
        # Traitement par batch pour éviter les limites de tokens
        batch_size = 51  # Ajustez selon vos besoins et limites de tokens
        
        for batch_start in range(0, len(remaining_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_prompts))
            batch_prompts = remaining_prompts[batch_start:batch_end]
            
            # Créer un prompt combiné avec des séparateurs clairs
            batch_prompt = "I will provide you with multiple analysis requests. Please respond to each one separately, maintaining the same format for each response.\n\n"
            
            for i, prompt in enumerate(batch_prompts):
                batch_prompt += f"=== ANALYSIS REQUEST {i+1} ===\n"
                batch_prompt += prompt + "\n\n"
            
            batch_prompt += "Please provide separate responses for each analysis request above, clearly numbered and formatted identically."
            
            try:
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(remaining_prompts) + batch_size - 1)//batch_size} for {symbol}")
                
                # Faire l'appel API avec le batch
                completion = call_gpt_batch(system_prompt, batch_prompt)
                full_response = completion.choices[0].message.content
                
                # Diviser la réponse en réponses individuelles
                responses = split_batch_response(full_response, len(batch_prompts))
                
                # Enregistrer chaque prompt et réponse individuellement
                for i, (prompt, answer) in enumerate(zip(batch_prompts, responses)):
                    append_to_csv(csv_file, prompt, answer)
                    print(f"Processed prompt {pre_done + batch_start + i + 1}/{len(prompts)} for {symbol}")
                
                # Pause entre les batches
                if batch_end < len(remaining_prompts):
                    sleep_time = random.uniform(2, 5)
                    print(f"Pausing for {sleep_time:.2f}s before next batch...")
                    time.sleep(sleep_time)
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Error processing batch for {symbol}: {error_message}")
                
                # Vérifier si c'est une erreur de quota
                if "insufficient_quota" in error_message:
                    print("⚠️ API quota exceeded. Saving remaining prompts and exiting.")
                    
                    # Sauvegarder prompts restants
                    remaining_after_error = remaining_prompts[batch_start:]
                    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts.json"
                    with open(remaining_file, "w") as f:
                        json.dump(remaining_after_error, f)
                    
                    print(f"Remaining prompts saved to {remaining_file}")
                    return
                
                # En cas d'autre erreur, enregistrer des erreurs pour ce batch
                for prompt in batch_prompts:
                    append_to_csv(csv_file, prompt, f"ERROR: {error_message}")


def split_batch_response(full_response, expected_count):
    """
    Divise une réponse en batch en réponses individuelles
    """
    # Tentative de division par les marqueurs d'analyse
    parts = full_response.split("=== ANALYSIS")
    
    if len(parts) > 1:
        # Supprime la première partie qui peut être vide ou contenir de l'intro
        responses = []
        for i, part in enumerate(parts[1:], 1):
            # Nettoie la réponse en supprimant les marqueurs
            clean_response = part.strip()
            if clean_response.startswith(f"REQUEST {i} ===") or clean_response.startswith("==="):
                # Trouve le début de la vraie réponse
                lines = clean_response.split('\n')
                response_start = 0
                for j, line in enumerate(lines):
                    if line.strip() and not line.startswith('===') and not line.startswith('REQUEST'):
                        response_start = j
                        break
                clean_response = '\n'.join(lines[response_start:]).strip()
            
            responses.append(clean_response)
    else:
        # Si la division par marqueurs échoue, essaie de diviser par patterns communs
        # Cherche les patterns de début de réponse
        response_patterns = [
            r'\[Positive Developments\]:',
            r'\[Potential Concerns\]:',
            r'Based on all the information'
        ]
        
        responses = []
        current_response = ""
        lines = full_response.split('\n')
        
        for line in lines:
            # Vérifie si cette ligne commence une nouvelle réponse
            is_new_response = any(re.search(pattern, line) for pattern in response_patterns)
            
            if is_new_response and current_response.strip():
                responses.append(current_response.strip())
                current_response = line + '\n'
            else:
                current_response += line + '\n'
        
        # Ajoute la dernière réponse
        if current_response.strip():
            responses.append(current_response.strip())
    
    # Assure-toi d'avoir le bon nombre de réponses
    while len(responses) < expected_count:
        responses.append("ERROR: Could not parse response from batch")
    
    # Tronque si on a trop de réponses
    return responses[:expected_count]


# ----------------------------------------------------------------------------------- #
# ---------------------------------- CLAUDE ANALYSIS ---------------------------------- #
# ----------------------------------------------------------------------------------- #

def query_claude(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    """
    Query Claude Sonnet 4 API with batch processing and proper rate limit handling for Claude Pro
    Claude Pro limits: ~8,000 requests per 8 hours
    """
    import time
    import random
    import backoff
    import anthropic
    
    # Create Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Claude Pro rate limits (being conservative)
    REQUESTS_PER_HOUR = 1000  # Conservative estimate (8000/8 hours)
    SECONDS_BETWEEN_REQUESTS = 3600 / REQUESTS_PER_HOUR  # ~3.6 seconds
    
    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.APIError, anthropic.APIConnectionError),
        max_tries=3,
        max_time=300,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries")
    )
    def call_claude_batch(system_prompt, batch_prompt):
        return client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Claude Sonnet 4
            max_tokens=4000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": batch_prompt}
            ]
        )

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_claude.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_claude.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        
        # Vérification des prompts générés
        print(f"Generated {len(prompts)} prompts for {symbol}")
        if len(prompts) == 0:
            print(f"Warning: No prompts generated for {symbol}!")
            continue
            
        # Filtrer les prompts non traités
        remaining_prompts = prompts[pre_done:]
        if len(remaining_prompts) == 0:
            print(f"All prompts already processed for {symbol}")
            continue
            
        print(f"Processing {len(remaining_prompts)} remaining prompts for {symbol}")
        
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        
        # Traitement par batch plus petit pour Claude (limites de contexte)
        batch_size = 3  # Plus petit que GPT pour être conservateur avec Claude
        
        total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size
        
        for batch_start in range(0, len(remaining_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_prompts))
            batch_prompts = remaining_prompts[batch_start:batch_end]
            
            # Créer un prompt combiné avec des séparateurs clairs
            batch_prompt = "I will provide you with multiple financial analysis requests. Please respond to each one separately, maintaining the exact same format for each response.\n\n"
            
            for i, prompt in enumerate(batch_prompts):
                batch_prompt += f"=== ANALYSIS REQUEST {i+1} ===\n"
                batch_prompt += prompt + "\n\n"
            
            batch_prompt += f"Please provide exactly {len(batch_prompts)} separate responses for each analysis request above. Each response should follow the format: [Positive Developments], [Potential Concerns], and [Prediction & Analysis]. Number each response clearly."
            
            try:
                current_batch = batch_start // batch_size + 1
                print(f"Processing batch {current_batch}/{total_batches} for {symbol} (requests {batch_start+1}-{batch_end})")
                
                # Respecter les limites de taux de Claude Pro
                if batch_start > 0:
                    # Pause plus longue pour respecter les limites Claude Pro
                    sleep_time = random.uniform(SECONDS_BETWEEN_REQUESTS, SECONDS_BETWEEN_REQUESTS * 1.5)
                    print(f"Pausing for {sleep_time:.2f}s to respect Claude Pro rate limits...")
                    time.sleep(sleep_time)
                
                # Faire l'appel API avec Claude
                completion = call_claude_batch(system_prompt, batch_prompt)
                full_response = completion.content[0].text
                
                # Diviser la réponse en réponses individuelles
                responses = split_claude_batch_response(full_response, len(batch_prompts))
                
                # Enregistrer chaque prompt et réponse individuellement
                for i, (prompt, answer) in enumerate(zip(batch_prompts, responses)):
                    append_to_csv(csv_file, prompt, answer)
                    print(f"Processed prompt {pre_done + batch_start + i + 1}/{len(prompts)} for {symbol}")
                
                # Afficher le progrès estimé
                progress = ((batch_start + batch_size) / len(remaining_prompts)) * 100
                print(f"Progress for {symbol}: {progress:.1f}%")
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Error processing batch for {symbol}: {error_message}")
                
                # Vérifier si c'est une erreur de quota ou rate limit
                if any(keyword in error_message.lower() for keyword in ["rate limit", "quota", "usage limit"]):
                    print("⚠️ Claude rate limit/quota exceeded. Saving remaining prompts and exiting.")
                    
                    # Sauvegarder prompts restants
                    remaining_after_error = remaining_prompts[batch_start:]
                    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts_claude.json"
                    with open(remaining_file, "w") as f:
                        json.dump(remaining_after_error, f)
                    
                    print(f"Remaining prompts saved to {remaining_file}")
                    print("You can resume later when your Claude Pro quota resets.")
                    return
                
                # En cas d'autre erreur, enregistrer des erreurs pour ce batch
                for prompt in batch_prompts:
                    append_to_csv(csv_file, prompt, f"ERROR: {error_message}")
                
                # Pause plus longue en cas d'erreur
                print("Pausing 30s after error before continuing...")
                time.sleep(30)

        print(f"✅ Completed processing all prompts for {symbol}")


def split_claude_batch_response(full_response, expected_count):
    """
    Divise une réponse en batch de Claude en réponses individuelles
    Claude tend à être plus structuré dans ses réponses
    """
    # Tentative de division par les marqueurs d'analyse
    parts = full_response.split("=== ANALYSIS")
    
    if len(parts) > 1:
        responses = []
        for i, part in enumerate(parts[1:], 1):
            clean_response = part.strip()
            
            # Supprimer les marqueurs de début
            if clean_response.startswith(f"REQUEST {i}"):
                lines = clean_response.split('\n')
                # Trouver où commence la vraie réponse
                response_start = 0
                for j, line in enumerate(lines):
                    if any(marker in line for marker in ["[Positive Developments]", "[Potential Concerns]"]):
                        response_start = j
                        break
                clean_response = '\n'.join(lines[response_start:]).strip()
            elif clean_response.startswith("==="):
                # Supprimer la ligne de séparateur
                lines = clean_response.split('\n')[1:]
                clean_response = '\n'.join(lines).strip()
            
            responses.append(clean_response)
    else:
        # Méthode alternative : chercher les patterns de début de réponse
        response_patterns = [
            r'\[Positive Developments\]:',
            r'### Analysis \d+',
            r'Analysis Request \d+',
            r'\*\*Analysis \d+\*\*'
        ]
        
        responses = []
        current_response = ""
        lines = full_response.split('\n')
        
        for line in lines:
            # Vérifier si cette ligne commence une nouvelle réponse
            is_new_response = any(re.search(pattern, line, re.IGNORECASE) for pattern in response_patterns)
            
            if is_new_response and current_response.strip() and "[Positive Developments]" in current_response:
                responses.append(current_response.strip())
                current_response = line + '\n'
            else:
                current_response += line + '\n'
        
        # Ajouter la dernière réponse
        if current_response.strip():
            responses.append(current_response.strip())
    
    # S'assurer d'avoir le bon nombre de réponses
    while len(responses) < expected_count:
        responses.append("ERROR: Could not parse response from Claude batch")
    
    # Tronquer si on a trop de réponses
    responses = responses[:expected_count]
    
    # Nettoyer les réponses pour s'assurer qu'elles ont le bon format
    cleaned_responses = []
    for response in responses:
        if not any(marker in response for marker in ["[Positive Developments]", "[Potential Concerns]"]):
            # Si la réponse n'a pas le bon format, ajouter une structure de base
            response = f"[Positive Developments]:\n{response}\n\n[Potential Concerns]:\n1. Market volatility\n\n[Prediction & Analysis]:\n{response}"
        cleaned_responses.append(response)
    
    return cleaned_responses


def resume_claude_from_remaining(symbol, data_dir, start_date, end_date, with_basics=True):
    """
    Fonction utilitaire pour reprendre le traitement Claude à partir des prompts sauvegardés
    """
    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts_claude.json"
    
    if not os.path.exists(remaining_file):
        print(f"No remaining prompts file found for {symbol}")
        return
        
    with open(remaining_file, 'r') as f:
        remaining_prompts = json.load(f)
    
    print(f"Resuming processing for {symbol} with {len(remaining_prompts)} remaining prompts")
    
    # Utiliser la fonction principale avec les prompts restants
    query_claude([symbol], data_dir, start_date, end_date, with_basics=with_basics)
    
    # Supprimer le fichier de prompts restants après traitement réussi
    try:
        os.remove(remaining_file)
        print(f"Removed remaining prompts file for {symbol}")
    except:
        pass


def query_mistral(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    """
    Query Mistral API with batch processing and proper rate limit handling
    Mistral rate limits: varies by plan, being conservative with timing
    """
    import time
    import random
    import backoff
    from mistralai import Mistral
    from mistralai.models import UserMessage, SystemMessage
    
    # Create Mistral client
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    
    # Mistral rate limits (being conservative)
    REQUESTS_PER_MINUTE = 60  # Conservative estimate
    SECONDS_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # ~1 second
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Mistral exceptions vary, catch all and handle specifically
        max_tries=3,
        max_time=300,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries")
    )
    def call_mistral_batch(system_prompt, batch_prompt):
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=batch_prompt)
        ]
        
        return client.chat.complete(
            model="mistral-large-latest",  # or "mistral-medium" for cheaper option
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_mistral.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_mistral.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        
        # Vérification des prompts générés
        print(f"Generated {len(prompts)} prompts for {symbol}")
        if len(prompts) == 0:
            print(f"Warning: No prompts generated for {symbol}!")
            continue
            
        # Filtrer les prompts non traités
        remaining_prompts = prompts[pre_done:]
        if len(remaining_prompts) == 0:
            print(f"All prompts already processed for {symbol}")
            continue
            
        print(f"Processing {len(remaining_prompts)} remaining prompts for {symbol}")
        
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        
        # Traitement par batch pour Mistral
        batch_size = 5  # Batch size modéré pour Mistral
        
        total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size
        
        for batch_start in range(0, len(remaining_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_prompts))
            batch_prompts = remaining_prompts[batch_start:batch_end]
            
            # Créer un prompt combiné avec des séparateurs clairs
            batch_prompt = "I will provide you with multiple financial analysis requests. Please respond to each one separately, maintaining the exact same format for each response.\n\n"
            
            for i, prompt in enumerate(batch_prompts):
                batch_prompt += f"=== ANALYSIS REQUEST {i+1} ===\n"
                batch_prompt += prompt + "\n\n"
            
            batch_prompt += f"Please provide exactly {len(batch_prompts)} separate responses for each analysis request above. Each response should follow the format: [Positive Developments], [Potential Concerns], and [Prediction & Analysis]. Number each response clearly."
            
            try:
                current_batch = batch_start // batch_size + 1
                print(f"Processing batch {current_batch}/{total_batches} for {symbol} (requests {batch_start+1}-{batch_end})")
                
                # Respecter les limites de taux de Mistral
                if batch_start > 0:
                    sleep_time = random.uniform(SECONDS_BETWEEN_REQUESTS, SECONDS_BETWEEN_REQUESTS * 1.5)
                    print(f"Pausing for {sleep_time:.2f}s to respect Mistral rate limits...")
                    time.sleep(sleep_time)
                
                # Faire l'appel API avec Mistral
                completion = call_mistral_batch(system_prompt, batch_prompt)
                full_response = completion.choices[0].message.content
                
                # Diviser la réponse en réponses individuelles
                responses = split_mistral_batch_response(full_response, len(batch_prompts))
                
                # Enregistrer chaque prompt et réponse individuellement
                for i, (prompt, answer) in enumerate(zip(batch_prompts, responses)):
                    append_to_csv(csv_file, prompt, answer)
                    print(f"Processed prompt {pre_done + batch_start + i + 1}/{len(prompts)} for {symbol}")
                
                # Afficher le progrès estimé
                progress = ((batch_start + batch_size) / len(remaining_prompts)) * 100
                print(f"Progress for {symbol}: {progress:.1f}%")
                    
            except Exception as e:
                error_message = str(e)
                print(f"❌ Error processing batch for {symbol}: {error_message}")
                
                # Vérifier si c'est une erreur de quota ou rate limit
                if any(keyword in error_message.lower() for keyword in ["rate limit", "quota", "usage limit", "too many requests"]):
                    print("⚠️ Mistral rate limit/quota exceeded. Saving remaining prompts and exiting.")
                    
                    # Sauvegarder prompts restants
                    remaining_after_error = remaining_prompts[batch_start:]
                    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts_mistral.json"
                    with open(remaining_file, "w") as f:
                        json.dump(remaining_after_error, f)
                    
                    print(f"Remaining prompts saved to {remaining_file}")
                    print("You can resume later when your Mistral quota resets.")
                    return
                
                # En cas d'autre erreur, enregistrer des erreurs pour ce batch
                for prompt in batch_prompts:
                    append_to_csv(csv_file, prompt, f"ERROR: {error_message}")
                
                # Pause plus longue en cas d'erreur
                print("Pausing 10s after error before continuing...")
                time.sleep(10)

        print(f"✅ Completed processing all prompts for {symbol}")


def split_mistral_batch_response(full_response, expected_count):
    """
    Divise une réponse en batch de Mistral en réponses individuelles
    Mistral tend à bien structurer ses réponses
    """
    import re
    
    # Tentative de division par les marqueurs d'analyse
    parts = full_response.split("=== ANALYSIS")
    
    if len(parts) > 1:
        responses = []
        for i, part in enumerate(parts[1:], 1):
            clean_response = part.strip()
            
            # Supprimer les marqueurs de début
            if clean_response.startswith(f"REQUEST {i}"):
                lines = clean_response.split('\n')
                # Trouver où commence la vraie réponse
                response_start = 0
                for j, line in enumerate(lines):
                    if any(marker in line for marker in ["[Positive Developments]", "[Potential Concerns]"]):
                        response_start = j
                        break
                clean_response = '\n'.join(lines[response_start:]).strip()
            elif clean_response.startswith("==="):
                # Supprimer la ligne de séparateur
                lines = clean_response.split('\n')[1:]
                clean_response = '\n'.join(lines).strip()
            
            responses.append(clean_response)
    else:
        # Méthode alternative : chercher les patterns de début de réponse
        response_patterns = [
            r'\[Positive Developments\]:',
            r'### Analysis \d+',
            r'Analysis Request \d+',
            r'\*\*Analysis \d+\*\*',
            r'Response \d+:',
            r'\d+\.\s*\[Positive Developments\]'
        ]
        
        responses = []
        current_response = ""
        lines = full_response.split('\n')
        
        for line in lines:
            # Vérifier si cette ligne commence une nouvelle réponse
            is_new_response = any(re.search(pattern, line, re.IGNORECASE) for pattern in response_patterns)
            
            if is_new_response and current_response.strip() and "[Positive Developments]" in current_response:
                responses.append(current_response.strip())
                current_response = line + '\n'
            else:
                current_response += line + '\n'
        
        # Ajouter la dernière réponse
        if current_response.strip():
            responses.append(current_response.strip())
    
    # S'assurer d'avoir le bon nombre de réponses
    while len(responses) < expected_count:
        responses.append("ERROR: Could not parse response from Mistral batch")
    
    # Tronquer si on a trop de réponses
    responses = responses[:expected_count]
    
    # Nettoyer les réponses pour s'assurer qu'elles ont le bon format
    cleaned_responses = []
    for response in responses:
        if not any(marker in response for marker in ["[Positive Developments]", "[Potential Concerns]"]):
            # Si la réponse n'a pas le bon format, ajouter une structure de base
            response = f"[Positive Developments]:\n{response}\n\n[Potential Concerns]:\n1. Market volatility\n\n[Prediction & Analysis]:\n{response}"
        cleaned_responses.append(response)
    
    return cleaned_responses


def resume_mistral_from_remaining(symbol, data_dir, start_date, end_date, with_basics=True):
    """
    Fonction utilitaire pour reprendre le traitement Mistral à partir des prompts sauvegardés
    """
    remaining_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_remaining_prompts_mistral.json"
    
    if not os.path.exists(remaining_file):
        print(f"No remaining prompts file found for {symbol}")
        return
        
    with open(remaining_file, 'r') as f:
        remaining_prompts = json.load(f)
    
    print(f"Resuming processing for {symbol} with {len(remaining_prompts)} remaining prompts")
    
    # Utiliser la fonction principale avec les prompts restants
    query_mistral([symbol], data_dir, start_date, end_date, with_basics=with_basics)
    
    # Supprimer le fichier de prompts restants après traitement réussi
    try:
        os.remove(remaining_file)
        print(f"Removed remaining prompts file for {symbol}")
    except:
        pass


# def claude_to_mistral_csv(symbol, data_dir, start_date, end_date, with_basics=True):
#     """
#     Convertit un fichier CSV Claude en format Mistral pour la compatibilité
#     """
#     claude_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_claude.csv' if with_basics else \
#                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_claude.csv'
    
#     mistral_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_mistral.csv' if with_basics else \
#                    f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_mistral.csv'
    
#     if not os.path.exists(claude_file):
#         print(f"Claude file {claude_file} not found")
#         return
    
#     # Copier le fichier Claude vers Mistral
#     import shutil
#     shutil.copy2(claude_file, mistral_file)
#     print(f"Copied {claude_file} to {mistral_file}")
