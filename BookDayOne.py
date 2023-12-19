import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests

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
        print(f"Fehler beim Abrufen der Daten für {ticker}: {e}")
        return 0

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    kgv = float(info.get('trailingPE', 'N/A')) if 'trailingPE' in info else 'N/A'
    market_cap = float(info.get('marketCap', 'N/A')) if 'marketCap' in info else 'N/A'
    dividend_yield = float(info.get('dividendYield', 'N/A')) if 'dividendYield' in info else 'N/A'
    return kgv, market_cap, dividend_yield

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
            add_ticker_to_excel(ticker, amount, file_path)

def add_ticker_to_excel(ticker, amount, file_path):
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=['Ticker', 'Amount'])
        if ticker in df['Ticker'].values:
            df.loc[df['Ticker'] == ticker, 'Amount'] += amount
        else:
            new_row = pd.DataFrame({'Ticker': [ticker], 'Amount': [amount]})
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
        st.success("Die Daten wurden erfolgreich in Excel hinzugefügt.")
    except Exception as e:
        st.error(f"Es gab einen Fehler beim Schreiben in Excel: {e}")

def analyse(df):
    st.title("Analyse")
    if df is not None:
        financial_df = df.groupby(['YearMonth', 'Category'])['Amount'].sum().reset_index()
        plot_financials(financial_df)

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

def aktienkurse():
    st.title("Aktienkurse")
    aktien_name = st.text_input("Aktienname oder Tickersymbol eingeben:", "")
    if aktien_name:
        try:
            kurs = fetch_current_price(aktien_name)
            if kurs is not None:
                display_fundamental_data(aktien_name, kurs)
                plot_stock_history(aktien_name)
            else:
                st.error("Aktienkurs konnte nicht abgerufen werden.")
        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten: {e}")

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
