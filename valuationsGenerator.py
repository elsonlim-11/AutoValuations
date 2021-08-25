## System
import sys
import os
import streamlit as st
st.set_page_config(layout="wide")

## Data I/O
import pandas as pd 
import numpy as np
import yfinance as yf 
import requests
from bs4 import BeautifulSoup
import base64

## ML and Viz
import pyflux as pf
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
import math
import plotly.express as px
import plotly.graph_objects  as go


## Scraper
#---------
# Get sector data and ticker list
def standardizeSector(row):
    '''
    Standardizes sector descriptions between S&P and TSX companies
    '''
    sector = str(row['Sector'])

    if sector == 'Basic Materials':
        output = 'Materials'
    elif sector == 'Consumer Cyclical':
        output = 'Consumer Discretionary'
    elif sector == 'Consumer Defensive':
        output = 'Consumer Staples'
    elif sector == 'Financial Services':
        output = 'Financials'
    elif sector == 'Technology':
        output = 'Information Technology'
    elif sector == 'Healthcare':
        output = 'Health Care'
    else:
        output = sector

    return output

def getSPY500():
    ## Get list of tickers and industries from Wikipedia
    data = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = data[0][['Symbol','Security','GICS Sector','GICS Sub-Industry']].rename(columns={'GICS Sector':'Sector','Security':'Company','GICS Sub-Industry':'Industry'})
    df['Exchange'] = 'S&P'
    return df

def getTSX():
    ## Identify table by content; Scrape with BS4
    content = requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index").content
    soup = BeautifulSoup(content,'html.parser')
    table = soup.find(text='Symbol').find_parent('table')
    
    ## Extract all elements from each row in table; Append to listRows
    listRows = []
    for row in table.find_all('tr')[1:]:
        listRows.append([cell.get_text(strip=True) for cell in row.find_all('td')])
    
    ## Convert list of lists to df; Standardize values to match SPY; Add .TO suffix to tickers
    df = pd.DataFrame(listRows)
    df.columns = ['Symbol','Company','Sector','Industry']
    df['Exchange'] = 'TSX'
    df['Sector'] = df.apply(standardizeSector,axis=1)
    df['Symbol'] = df['Symbol'] + '.TO'
    
    return df

def getTickerList():
    ''' 
    Call both query functions; Union output
    '''
    spy = getSPY500()
    tsx = getTSX()
    tickers = pd.concat([spy,tsx]).reset_index(drop=True)

    return tickers

dfEligibleTickers = getTickerList()


## Sidebar
#---------
# Input sidebar: Ticker selection
st.sidebar.header('User Input')
inputTicker = st.sidebar.text_input("Input non-bank ticker here:", "AAPL")

if inputTicker.upper() not in dfEligibleTickers.Symbol.unique(): st.sidebar.write("**Ticker not found**")

## Query and cache ticker data from YF
@st.cache(allow_output_mutation=True)
def getTickerData(inputTicker):
    output = yf.Ticker(inputTicker)
    return output

ticker = getTickerData(inputTicker)
tickerHistoricals = ticker.history(period='5y',interval='1mo').dropna()
tickerFinancials = ticker.financials.transpose().reset_index()
tickerBalanceSheet = ticker.balance_sheet.transpose().reset_index()
tickerCashFlow = ticker.cashflow.transpose().reset_index()

def getShortNumber(num):
    '''
    Transforms number into a more readable format by shortening with K, M, B suffixes
    '''

    if num == None: 
        return None   

    adjustmentFactor = ['','K','M','B','T']
    adjustmentIndex = 0
    while abs(num) >= 1000:
        num  = num/1000
        adjustmentIndex += 1
    return f"{round(num,2)}{adjustmentFactor[adjustmentIndex]}"

## Display company summary 
st.sidebar.header('Company Summary')
st.sidebar.write(f"""
    **Name**: {ticker.info['longName']} \n
    **Industry**: {ticker.info['industry']} \n
    **Currency**: {ticker.info['financialCurrency']} \n
    **Current Price**: {ticker.info['currentPrice']} \n
    **52-week Range**: {round(ticker.info['fiftyTwoWeekLow'],2)} - {round(ticker.info['fiftyTwoWeekHigh'],2)} \n
    
    **Shares Outstanding**: {getShortNumber(ticker.info['sharesOutstanding'])} \n
    """)

## Display financials
st.sidebar.header('Financials')
st.sidebar.write(f"""
    **Revenue**: {getShortNumber(tickerFinancials.loc[0,'Total Revenue'])} \n
    **Gross Profit**: {getShortNumber(tickerFinancials.loc[0,'Gross Profit'])} \n
    **EBIT**: {getShortNumber(tickerFinancials.loc[0,'Ebit'])} \n
    **Net Income**: {getShortNumber(tickerFinancials.loc[0,'Net Income'])} \n
    """)


## Display valuations
def displayValuationsSidebar():
    if 'bank' in ticker.info['industry'].lower():
        textValuations = (f"""
            **Market Cap**: {getShortNumber(ticker.info['marketCap'])} \n
        """)

    else:
        textValuations = (f"""
            **Market Cap**: {getShortNumber(ticker.info['marketCap'])} \n
            **Enterprise Value**: {getShortNumber(ticker.info['enterpriseValue'])} \n
            **EV/Revenue**: {round(ticker.info['enterpriseToRevenue'],2)}*x* \n
            **EV/EBITDA**: {round(ticker.info['enterpriseToEbitda'],2)}*x* \n
        """)
    
    return textValuations

st.sidebar.header('Valuations')
st.sidebar.write(displayValuationsSidebar())

## Comps Table
#-------------
@st.cache(allow_output_mutation=True)
def getConstituents():
    '''
    Get list of tickers based on input ticker's sector
    Returns named tuple with yf.Ticker objects
    '''
    
    # Get list of tickers by input ticker sector 
    target = dfEligibleTickers.Sector[dfEligibleTickers.Symbol==inputTicker].values[0]
    df = dfEligibleTickers[dfEligibleTickers.Sector==target]

    # Convert list of tickers into tab-delimited string 
    strTickers = ''
    for ticker in df.Symbol.unique().tolist():
        strTickers = strTickers + ticker + ' '
    
    # Call yfinance API for selected tickers
    tickers = yf.Tickers(strTickers)
    return tickers

@st.cache(allow_output_mutation=True)
def getConstituentMetrics(constituent):
    '''
    Takes yf.Ticker object as input
    Returns dictionary containing key metrics of ticker
    '''
    try: 
        outputMetrics = {
            # Profitability 
                'roa': constituent.info['returnOnAssets'],
                'npm': constituent.info['profitMargins'],
            # Growth
                'growth': constituent.info['revenueGrowth'],  
            # Size
                'revenue': constituent.info['totalRevenue'],
                'ev': constituent.info['enterpriseValue'],
            # Capital structure
                'de': constituent.info['debtToEquity'],
                
            # Multiples
                'ev_revenue': constituent.info['enterpriseToRevenue'],
                'ev_ebitda': constituent.info['enterpriseToEbitda'],
                'price_earnings': constituent.info['trailingPE']
            }
    
    except:
        outputMetrics = {
                'roa': None,
                'npm': None,
                'growth': None,
                'revenue': None,
                'ev': None,
                'de': None,
                'ev_revenue': None,
                'ev_ebitda': None,
                'price_earnings': None
        }

    return outputMetrics


## Creates nested dictionary containing all potential comps and respective metrics
constituents = getConstituents()
dictConstituents = {}

## Initiates progress bar parameters (due to long runtimes)
progressStep = 0
progressBar = st.progress(progressStep)
endProgress = len(constituents.tickers)

for constituent in constituents.tickers.values():
    try:
        dictConstituents[constituent.info['symbol']] = getConstituentMetrics(constituent)
    except:
        continue
    ## Display progress
    progressStep += 1
    progressBar.progress(progressStep/endProgress)
progressBar.empty()

## Converts nested dictionary to dataframe
dfConstituents = pd.DataFrame(dictConstituents).transpose().dropna()

## Select comparables
#--------------------

def scaleData(df):
    '''
    Apply a min-max scaler to standardize data
    '''
    output = pd.DataFrame(
        data=MinMaxScaler().fit_transform(df),
        columns=df.columns,
        index=df.index
        )
    return output

def calculateMDS(df, n=15):
    '''
    Reduce input columns into two dimensions using multi-dimensional scaling (MDS)
    Calculate Euclidian distances between MDS coordinates
    Selects the top n companies based on shortest distance
    '''
    output = pd.DataFrame(
        data=MDS(2,random_state=42).fit_transform(df.drop(columns=['ev_revenue','ev_ebitda','price_earnings'])),
        columns=['mdsX','mdsY'],
        index=df.index)

    x, y = output.loc[inputTicker,'mdsX'], output.loc[inputTicker,'mdsY']
    
    output['euclidian'] = output.apply(lambda i: math.sqrt((i.mdsX - x)**2 + (i.mdsY - y)**2),axis=1)
    output['rank'] = output.euclidian.rank()
    output['status'] = output.apply(lambda x: 'Selected' if x['euclidian']==0 else 'Recommended' if x['rank']<=n else 'Other',axis=1)
    
    return output

def classifyComparables(df):
    '''
    Joins MDS calculations with ticker metrics
    '''
    mds = pd.merge(
        left=df,
        right=calculateMDS(scaleData(df)),
        left_index=True, right_index=True
        )

    output = pd.merge(
        left = dfEligibleTickers,
        right = mds.reset_index(drop=False).rename(columns={'index':'symbol'}),
        how = 'inner',
        left_on = 'Symbol',
        right_on = 'symbol',
    )

    return output.sort_values('rank',ascending=True).reset_index(drop=True)

dfClassified = classifyComparables(dfConstituents)


def downloadComparables(df):
    '''
    Creates link to download comps table as a csv
    '''
    csv = df.to_csv(index=False)
    b = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b}" download = "comparables_{inputTicker}.csv">Download as CSV</a>'
    return href


def visualizeMDS(df):
    '''
    Visualize classification model
    '''
    output = px.scatter(
        df,
        x = 'mdsX',
        y = 'mdsY',
        color = 'status',
        width=1280,
        height=720,
        hover_data={
            'symbol': True,
            'mdsX': False,
            'mdsY': False,
            'status': False,
            'growth': ':.2f',
            'revenue': True,
            'ev': True,
            'de': ':.2f',
            'ev_revenue': ':.2f',
            'ev_ebitda': ':.2f',
            'price_earnings': ':.2f'
            }
    ).update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })

    return output

#### RETURN HERE
## Forward multiples
#-------------------
def getAverageMultiples(df):
    '''
    Creates link to download comps table as a csv
    '''
    comps = df[df.status=='Recommended'][['Symbol','ev_revenue','ev_ebitda','price_earnings']]
    output = {
        'ev_revenue': comps.ev_revenue.mean(),
        'ev_ebitda': comps.ev_ebitda.mean(),
        'price_earnings': comps.price_earnings.mean()
    }
    
    return output

def simulateMetrics(n=5000):
    output = {}
    ## Get revenue and earnings
    for metric in ['Total Revenue', 'Net Income']:

        metricMU = tickerFinancials[metric].mean()
        metricSIGMA = tickerFinancials[metric].std()
        metricMIN = tickerFinancials[metric].min()
        metricMAX = tickerFinancials[metric].max()

        simulations = np.random.normal(loc=metricMU,scale=metricSIGMA,size=n)
        output[metric] = simulations

    ## Get EBITDA
    ebitda = tickerFinancials['Ebit'].add(tickerCashFlow['Depreciation'])
    ebitdaMU  = ebitda.mean()
    ebitdaSIGMA  = ebitda.std()
    ebitdaMIN  = ebitda.min()
    ebitdaMAX  = ebitda.max()

    simulations = np.random.normal(loc=ebitdaMU,scale=ebitdaSIGMA,size=n)

    output['EBITDA'] = simulations

    return output

def calculateEV(multiples,simulations):
    ev_revenue = simulations['Total Revenue'] * multiples['ev_revenue']
    ev_ebitda = simulations['EBITDA'] * multiples['ev_ebitda']
    #price_earnings = simulations['Net Income'] * multiples['price_earnings']

    #output = pd.DataFrame({'ev_revenue':ev_revenue,'ev_ebitda':ev_ebitda,'price_earnings':price_earnings})
    output = pd.DataFrame({'ev_revenue':ev_revenue,'ev_ebitda':ev_ebitda})
    
    return output

def visualizeValuations(df):
    fig = go.Figure()
    fig.add_trace(go.Box(x=df['ev_ebitda'],name='EV/EBITDA'))
    fig.add_trace(go.Box(x=df['ev_revenue'],name='EV/Revenue'))
    fig.update_layout(
        width=1280,
        height=720, 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
        )
    #fig.add_trace(go.Box(x=df['price_earnings'],name='Price/Earnings'))

    return fig

## Frontend
#----------
st.write(f"# {inputTicker} Valuation")
st.write(f'### Valuation Football Field')
st.write(visualizeValuations(
    calculateEV(
        multiples=getAverageMultiples(dfClassified),
        simulations=simulateMetrics())
        )
    )


st.write(f'### Comp Table')
st.write(dfClassified)
st.write(downloadComparables(dfClassified),unsafe_allow_html=True)


st.write('### FAQ')
with st.expander("How are comparable companies were selected?"):
    st.write("""
        ### Classification \n
        1. From the ticker given by the user, all companies in the S&P500 and TSX Composite in the same sector are extracted.
        2. Of these selected companies, the following metrics are extracted for each company:
            * Return on assets
            * Net profit margin
            * Revenue growth
            * Revenue
            * Enterprise value
            * Debt to equity

        3. These six metrics are then reduced into two dimensions using a multi-dimensional scaler.
        4. After the scaler reduces the metrics into two variables, the Euclidian distance between the selected company and other companies in the same sector is calculated.
        5. The nearest 15 companies are then assigned the Recommended status.
    # """)
    st.write(visualizeMDS(dfClassified))

with st.expander("How are forward multiples estimated?"):
    st.write(f"""
        ### Estimation
        1. Mean of multiples for Recommended companies is calculated.
        2. A Gaussian Monte Carlo simulation is run on the target company's denominator (e.g. Revenue, EBITDA).
        3. The output for the Monte Carlo is multipled against the calculated multiple in step 1.
    # """)


with st.expander("How is the data queried?"):
    st.write(f"""
        ### Data
        * Company Data: Queried directly from the yfinance Python library.
        * Exchange Data: Scraped from Wikipedia
    # """)