import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests
import plotly.graph_objects as go

def custom_format(value):
    if pd.isna(value):
        return None
    else:
        value_str = f"{value:,.2f}"
        value_str = value_str.replace(',', 'X').replace('.', ',').replace('X', '.')
        return value_str

def fetch_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'][-1]
        return price
    except Exception as e:
        raise Exception(f"Fehler beim Abrufen der Daten für {ticker}: {e}")

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    kgv = float(info.get('trailingPE')) if 'trailingPE' in info else None
    market_cap = float(info.get('marketCap')) if 'marketCap' in info else None
    dividend_yield = float(info.get('dividendYield')) if 'dividendYield' in info else None
    return

def load_data():
    try:
        url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/Mappe1.xlsx'
        df = pd.read_excel(url, names=['Datum', 'Name', 'Betrag'])
        return df
    except Exception as e:
        st.error(f"Fehler beim Lesen der Finanzdatendatei: {e}")
        return None

def load_stock_portfolio():
    try:
        url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/StockPortfolio.xlsx'
        stock_df = pd.read_excel(url, names=['Ticker', 'Quantity'])
        stock_df['CurrentPrice'] = stock_df['Ticker'].apply(fetch_current_price)
        stock_df.dropna(subset=['CurrentPrice'], inplace=True)
        stock_df = stock_df[stock_df['CurrentPrice'] != 0]
        stock_df['TotalValue'] = stock_df['Quantity'] * stock_df['CurrentPrice']
        stock_df['CurrentPrice'] = stock_df['CurrentPrice'].round(2).apply(custom_format)
        stock_df['TotalValue'] = stock_df['TotalValue'].round(2).apply(custom_format)
        return stock_df
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der Aktienportfolio-Datei: {e}")
        return None

def process_data(df):
    if df is not None and 'Datum' in df.columns:
        df['Betrag'] = pd.to_numeric(df['Betrag'], errors='coerce')
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datum'])
        df['YearMonth'] = df['Datum'].dt.to_period('M')
        df.dropna(subset=['Betrag', 'YearMonth'], inplace=True)
        return df
    else:
        st.error("Invalid or missing 'Datum' column in DataFrame")
        return None

def plot_portfolio_history(stock_df):
    end = datetime.now()
    start = end - pd.DateOffset(years=3)
    portfolio_history = pd.DataFrame()

    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        stock_data = yf.download(ticker, start=start, end=end, progress=False)
        stock_data['Value'] = stock_data['Close'] * row['Quantity']
        portfolio_history[ticker] = stock_data['Value']

    portfolio_history['TotalValue'] = portfolio_history.sum(axis=1)
    portfolio_history['TotalValue'].plot(title='Portfolio Value Over Last 3 Years')
    plt.xlabel('Date')
    plt.ylabel('Total Value')
    st.pyplot(plt)

def plot_financials(financial_df):
    plt.figure(figsize=(10, 6))
    financial_df['AdjustedAmount'] = financial_df.apply(lambda x: -x['Amount'] if x['Category'] == 'Expense' else x['Amount'], axis=1)
    for category in financial_df['Category'].unique():
        category_df = financial_df[financial_df['Category'] == category]
        plt.plot(category_df['YearMonth'].dt.to_timestamp(), category_df['AdjustedAmount'], marker='o', label=category)
    net_savings = financial_df.groupby('YearMonth')['AdjustedAmount'].sum()
    plt.plot(net_savings.index.to_timestamp(), net_savings.values, marker='o', label='Net Savings')
    plt.title('Monthly Financial Overview')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def kontenubersicht(df):
    st.title("Finanzdaten Analyse-App")
    aktueller_monat = datetime.now().strftime('%Y-%m')
    aktueller_monat_periode = pd.Period(aktueller_monat)
    if df is not None:
        df_sorted = df.sort_values(by='Datum', ascending=False)
        aktuelle_monatsdaten = df[df['Datum'].dt.to_period('M') == aktueller_monat_periode]
        aktuelle_monatsausgaben = aktuelle_monatsdaten[aktuelle_monatsdaten['Betrag'] < 0]['Betrag'].sum()
        aktuelle_monatseinnahmen = aktuelle_monatsdaten[aktuelle_monatsdaten['Betrag'] > 0]['Betrag'].sum()
        st.subheader(f"Ausgaben in {aktueller_monat}:")
        st.write(aktuelle_monatsausgaben)
        with st.expander("Letzte 10 Ausgaben anzeigen"):
            last_expenses = df_sorted[df_sorted['Betrag'] < 0].head(10)
            st.dataframe(last_expenses[['Datum', 'Name', 'Betrag']])
        st.subheader(f"Einnahmen in {aktueller_monat}:")
        st.write(aktuelle_monatseinnahmen)
        with st.expander("Letzte 10 Einnahmen anzeigen"):
            last_incomes = df_sorted[df_sorted['Betrag'] > 0].head(10)
            st.dataframe(last_incomes[['Datum', 'Name', 'Betrag']])
        gesamtausgaben = aktuelle_monatsdaten[aktuelle_monatsdaten['Betrag'] < 0]['Betrag'].sum()
        gesamteinnahmen = aktuelle_monatsdaten[aktuelle_monatsdaten['Betrag'] > 0]['Betrag'].sum()
        kontostand = gesamteinnahmen + gesamtausgaben
        st.subheader("Gesamtkontostand:")
        st.write(kontostand)

def show_add_ticker_form():
    with st.form("add_ticker_form"):
        st.subheader("Neue Aktie hinzufügen")
        ticker = st.text_input("Ticker")
        amount = st.number_input("Anzahl", min_value=1, step=1)
        submit_button = st.form_submit_button("Hinzufügen")
        if submit_button:
            url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/StockPortfolio.xlsx'
            add_ticker_to_excel(ticker, amount, "path_to_your_excel_file.xlsx")

def add_ticker_to_excel(ticker, amount, file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Ticker', 'Amount'])

    if ticker in df['Ticker'].values:
        df.loc[df['Ticker'] == ticker, 'Amount'] += amount
    else:
        new_row = pd.DataFrame({'Ticker': [ticker], 'Amount': [amount]})
        df = pd.concat([df, new_row], ignore_index=True)

    try:
        df.to_excel(file_path, index=False)
        st.success("Die Daten wurden erfolgreich in Excel hinzugefügt.")
    except Exception as e:
        st.error(f"Es gab einen Fehler beim Schreiben in Excel: {e}")


def add_entry_to_excel(date, name, amount, file_path):
    date_str = date.strftime('%d.%m.%Y')
    new_entry = pd.DataFrame({
        'Datum': [date_str], 
        'Name': [name], 
        'Betrag': [amount]
    })
    try:
        df = pd.read_excel(file_path)
        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Datum'])
        df['Datum'] = df['Datum'].dt.strftime('%d.%m.%Y')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Datum', 'Name', 'Betrag'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(file_path, index=False)


def analyse(df):
    st.title("Analyse")
    if df is not None:
        # ... Existing implementation ...

        # Sankey Chart Integration
        source = [0, 0, 1, 1, 2, 2, 3, 3]
        target = [4, 5, 6, 7, 8, 9, 10, 11]
        value = [8, 2, 2, 3, 4, 4, 2, 5]
        label = ["Income", "Expenses", "Savings", "Investments", 
                 "Salary", "Other Income", "Bills", "Entertainment", 
                 "Retirement Fund", "Stocks", "Bonds", "Savings Account"]

        fig = go.Figure(data=[go.Sankey(node=dict(pad=10, thickness=10, line=dict(color="black", width=0.5), label=label), link=dict(source=source, target=target, value=value))])
        fig.update_layout(title_text="Financial Flow - Sankey Diagram", font_size=10)
        st.plotly_chart(fig)

    else:
        st.error("Keine Daten zum Analysieren vorhanden.")


def make_recommendations(user_portfolio, asset_data):
    user_portfolio_df = pd.DataFrame(list(user_portfolio.items()), columns=['Ticker', 'Quantity'])
    merged_data = pd.merge(user_portfolio_df, asset_data, on='Ticker', how='left')
    asset_similarity = cosine_similarity(merged_data.drop(['Ticker', 'Quantity'], axis=1))
    weighted_scores = asset_similarity.dot(merged_data['Quantity'])
    recommendations = pd.DataFrame({'Ticker': asset_data['Ticker'], 'Score': weighted_scores})
    recommendations = recommendations.sort_values(by='Score', ascending=False)
    return recommendations

def empfehlung(df, stock_portfolio_df):
    st.title("Empfehlung")
    user_portfolio = dict(zip(stock_portfolio_df['Ticker'], stock_portfolio_df['Quantity']))
    asset_data = pd.read_excel('path_to_asset_data.xlsx')
    recommendations = make_recommendations(user_portfolio, asset_data)
    st.subheader("Investment Recommendations:")
    st.dataframe(recommendations)

def add_entry_to_excel(date, name, amount, file_path):
    date_str = date.strftime('%d.%m.%Y')
    new_entry = pd.DataFrame({
        'Datum': [date_str], 
        'Name': [name], 
        'Betrag': [amount]
    })
    try:
        df = pd.read_excel(file_path)
        df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['Datum'])
        df['Datum'] = df['Datum'].dt.strftime('%d.%m.%Y')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Datum', 'Name', 'Betrag'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(file_path, index=False)

def custom_format_large_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def display_fundamental_data(ticker, kurs):
    kgv, market_cap, dividend_yield = get_fundamental_data(ticker)
    formatierter_kurs = custom_format(kurs) if kurs is not None else 'N/A'
    kgv = f"{kgv:.2f}" if isinstance(kgv, float) else kgv
    market_cap = custom_format_large_number(market_cap) if market_cap != 'N/A' else 'N/A'
    dividend_yield = f"{dividend_yield:.2f}%" if isinstance(dividend_yield, float) else dividend_yield
    data = {
        "Kennzahl": ["Aktueller Kurs", "KGV", "Marktkapitalisierung", "Dividendenrendite"],
        "Wert": [formatierter_kurs, kgv, market_cap, dividend_yield]
    }
    df = pd.DataFrame(data)
    st.table(df)

def show_new_entry_form(df):
    with st.form("new_entry_form", clear_on_submit=True):
        st.subheader("Neue Buchung hinzufügen")
        date = st.date_input("Datum", datetime.today())
        name = st.text_input("Name")
        amount = st.number_input("Betrag", step=1.0)
        submitted = st.form_submit_button("Eintrag hinzufügen")
        if submitted:
            url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/Mappe1.xlsx'
            add_entry_to_excel(date, name, amount, "path_to_your_excel_file.xlsx")
            st.success("Buchung erfolgreich hinzugefügt.")
            if 'load_data' in st.session_state:
                del st.session_state['load_data']
            df = load_data()
    return df

def plot_stock_history(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")
    if data.empty:
        st.error(f"Keine Daten für {ticker} gefunden.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Schlusskurse')
    plt.title(f"Aktienkursverlauf der letzten 6 Monate: {ticker}")
    plt.xlabel('Datum')
    plt.ylabel('Kurs in €')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")['Close'][-1]
    except Exception as e:
        print(f"Fehler beim Abrufen der Daten für {ticker}: {e}")
        return None

def plot_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.title(f"5-Year Stock Price History for {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)
    except: 
        print("Stop")
        # Section for displaying stock information
def aktienkurse_app():
    st.title("Aktienkurse")
    aktien_ticker = st.text_input("Aktienticker eingeben:", "")
    if aktien_ticker:
        display_stock_data(aktien_ticker)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Wähle eine Seite", ["Kontenübersicht", "Analyse", "Empfehlung", "Aktienkurse"])
    st.title("Finanzdaten Analyse-App")
    if 'dataframe' not in st.session_state or page == "Kontenübersicht":
        st.session_state.dataframe = load_data()
    df = st.session_state.dataframe
    if df is not None:
        df = process_data(df)
    if page == "Kontenübersicht":
        kontenubersicht(df)
        df = show_new_entry_form(df)
        show_add_ticker_form()
        stock_portfolio_df = load_stock_portfolio()
        if stock_portfolio_df is not None:
            st.subheader("Mein Aktienportfolio")
            st.dataframe(stock_portfolio_df)
            total_portfolio_value = sum(stock_portfolio_df['TotalValue'].str.replace('.', '').str.replace(',', '.').astype(float))
            st.write(f"Gesamtwert des Portfolios: {custom_format(total_portfolio_value)}")
            plot_portfolio_history(stock_portfolio_df)
        st.session_state.dataframe = df
    elif page == "Analyse":
        analyse(df)
    elif page == "Empfehlung":
        empfehlung(df, load_stock_portfolio())
    elif page == "Aktienkurse":
        aktienkurse()

if __name__ == "__main__":
    main()
