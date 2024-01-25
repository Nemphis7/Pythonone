import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from datetime import date

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
        raise Exception(f"Error fetching data for {ticker}: {e}")

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
        df = pd.read_excel(url, names=['Date', 'Name', 'Amount'])
        return df
    except Exception as e:
        st.error(f"Error reading financial data file: {e}")
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
        st.error(f"Error processing stock portfolio file: {e}")
        return None

def process_data(df):
    if df is not None and 'Date' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        df.dropna(subset=['Amount', 'YearMonth'], inplace=True)
        return df
    else:
        st.error("Invalid or missing 'Date' column in DataFrame")
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

def account_overview(df, stock_df):
    st.title("Financial Data Analysis App")
    current_month = datetime.now().strftime('%Y-%m')
    current_month_period = pd.Period(current_month)

    # Zeichnen Sie den Performance-Graphen der Aktien
    if stock_df is not None:
        historical_data = get_historical_data(stock_df, period="1y")  # Sie können den Zeitraum nach Bedarf anpassen
        plot_stock_performance(historical_data)

    if df is not None:
        df_sorted = df.sort_values(by='Date', ascending=False)
        current_month_data = df[df['Date'].dt.to_period('M') == current_month_period]
        current_month_expenses = current_month_data[current_month_data['Amount'] < 0]['Amount'].sum()
        current_month_income = current_month_data[current_month_data['Amount'] > 0]['Amount'].sum()
        
        st.subheader(f"Expenses in {current_month}:")
        st.write(current_month_expenses)
        with st.expander("Show last 10 expenses"):
            last_expenses = df_sorted[df_sorted['Amount'] < 0].head(10)
            st.dataframe(last_expenses[['Date', 'Name', 'Amount']])
        
        st.subheader(f"Income in {current_month}:")
        st.write(current_month_income)
        with st.expander("Show last 10 incomes"):
            last_incomes = df_sorted[df_sorted['Amount'] > 0].head(10)
            st.dataframe(last_incomes[['Date', 'Name', 'Amount']])
        
        total_expenses = current_month_data[current_month_data['Amount'] < 0]['Amount'].sum()
        total_income = current_month_data[current_month_data['Amount'] > 0]['Amount'].sum()
        account_balance = total_income + total_expenses
        
        st.subheader("Total account balance:")
        st.write(account_balance)



def analyse(df):
    st.title("Analyse")
    if df is not None and 'Betrag' in df.columns and 'Datum' in df.columns:
        # Debug: Show initial data
        st.write("Initial Data Sample:", df.head())

        df['Betrag'] = pd.to_numeric(df['Betrag'], errors='coerce')
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df.dropna(subset=['Betrag', 'Datum'], inplace=True)

        df['YearMonth'] = df['Datum'].dt.to_period('M')

        monthly_data = df.groupby('YearMonth')['Betrag'].sum().reset_index()
        monthly_income = monthly_data[monthly_data['Betrag'] > 0]['Betrag'].mean()
        monthly_expenses = monthly_data[monthly_data['Betrag'] < 0]['Betrag'].mean()
        average_savings = monthly_income + monthly_expenses

        # Debug: Show computed values
        st.write("Computed Monthly Data:", monthly_data)
        st.write("Average Monthly Income:", monthly_income)
        st.write("Average Monthly Expenses:", monthly_expenses)
        st.write("Average Monthly Savings:", average_savings)

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

def calculate_investment_period(current_age, retirement_age):
    return retirement_age - current_age

def generate_financial_recommendations(investment_period, stock_portfolio_df):
    # This is where you'd implement the logic for financial recommendations.
    # For example, you might adjust the stock/bond ratio based on the investment period.
    # This is just a placeholder:
    return f"Recommended investment strategy for an investment period of {investment_period} years."

def empfehlung(df, stock_portfolio_df):
    st.title("Empfehlung")

    # Inputs for age and retirement date
    current_age = st.number_input("Dein aktuelles Alter", min_value=18, max_value=100, step=1)
    retirement_age = st.number_input("Geplantes Rentenalter", min_value=current_age, max_value=100, step=1)

    if current_age and retirement_age:
        # Calculate the investment period
        investment_period = calculate_investment_period(current_age, retirement_age)

        # Generate and display financial recommendations
        recommendations = generate_financial_recommendations(investment_period, stock_portfolio_df)
        st.subheader("Personalisierte finanzielle Empfehlungen:")
        st.write(recommendations)
    else:
        st.write("Bitte geben Sie Ihr aktuelles Alter und das geplante Rentenalter ein, um Empfehlungen zu erhalten.")


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
    """Zeichnet den Kursverlauf der Aktie über die letzten 5 Jahre."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")
        if data.empty:
            st.error("Keine historischen Daten für diesen Ticker gefunden.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], label='Schlusskurs')
        plt.title(f"Aktienkursverlauf der letzten 5 Jahre: {ticker}")
        plt.xlabel('Datum')
        plt.ylabel('Kurs')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Daten für {ticker}: {e}")

def aktienkurse_app():
    """Streamlit App für die Anzeige der Aktienkurse."""
    st.title("Aktienkurse")
    aktien_ticker = st.text_input("Aktienticker eingeben:", "")
    if aktien_ticker:
        plot_stock_data(aktien_ticker)

def get_historical_data(stock_df, period="1y"):
    # Diese Funktion holt die historischen Daten für die Aktien im Portfolio
    historical_data = {}
    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        stock = yf.Ticker(ticker)
        historical_data[ticker] = stock.history(period=period)['Close']
    return historical_data

def plot_stock_performance(historical_data):
    # Verwenden Sie Streamlits integrierte Funktion, um den Graphen zu plotten
    for ticker, data in historical_data.items():
        st.line_chart(data, width=0, height=0, use_container_width=True)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Account Overview", "Analysis", "Recommendation", "Stock Prices"])

    st.title("Finance Data Analysis App")

    # Daten laden, wenn die App startet oder wenn "Account Overview" ausgewählt wird
    if 'dataframe' not in st.session_state or page == "Account Overview":
        st.session_state.dataframe = load_data()
        st.session_state.stock_df = load_stock_portfolio()

    df = st.session_state.dataframe
    stock_df = st.session_state.stock_df

    if page == "Account Overview":
        account_overview(df, stock_df)

    elif page == "Analysis":
        analyse(df)

    elif page == "Recommendation":
        empfehlung(df, stock_df)

    elif page == "Stock Prices":
        aktienkurse_app()

if __name__ == "__main__":
    main()
