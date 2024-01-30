import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from datetime import date
import numpy as np

INFLATION_RATE = 0.02

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
    except ValueError as e:
        st.error(f"Value Error: {e}")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")


def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'kgv': info.get('trailingPE', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A'),
        # Sie können hier weitere Daten hinzufügen, die Sie anzeigen möchten
    }


def load_data():
    try:
        url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/Mappe1.xlsx'
        df = pd.read_excel(url, names=['Date', 'Name', 'Amount', 'Category'])
        return df
    except Exception as e:
        st.error(f"Error reading financial data file: {e}")
        return None

def load_stock_portfolio():
    try:
        url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/StockPortfolio.xlsx'
        stock_df = pd.read_excel(url, names=['Ticker', 'Amount'])
        stock_df['CurrentPrice'] = stock_df['Ticker'].apply(fetch_current_price)
        stock_df.dropna(subset=['CurrentPrice'], inplace=True)
        stock_df = stock_df[stock_df['CurrentPrice'] != 0]
        stock_df['TotalValue'] = stock_df['Amount'] * stock_df['CurrentPrice']
        stock_df['CurrentPrice'] = stock_df['CurrentPrice'].round(2).apply(custom_format)
        stock_df['TotalValue'] = stock_df['TotalValue'].round(2).apply(custom_format)
        return stock_df
    except Exception as e:
        st.error(f"Error processing stock portfolio file: {e}")
        return None

def get_combined_historical_data(stock_df, period="1y"):
    portfolio_history = pd.DataFrame()
    
    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        Amount = row['Amount']  # Changed from 'Amount' to 'Amount'
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)['Close']
        portfolio_history[ticker] = hist * Amount
    
    portfolio_history['Total'] = portfolio_history.sum(axis=1)
    return portfolio_history['Total']


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

def plot_portfolio_performance(total_portfolio_history):
    if total_portfolio_history.empty:
        st.error("No data available to plot portfolio performance.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(total_portfolio_history.index, total_portfolio_history, label='Total Portfolio Value')
    plt.title('Total Portfolio Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Value')
    plt.legend()
    st.pyplot(plt)

def plot_portfolio_history(stock_df):
    end = datetime.now()
    start = end - pd.DateOffset(years=3)
    portfolio_history = pd.DataFrame()

    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        stock_data = yf.download(ticker, start=start, end=end, progress=False)
        stock_data['Value'] = stock_data['Close'] * row['Amount']
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
    st.title("Financial Data Analysis")
    current_month = datetime.now().strftime('%Y-%m')
    current_month_period = pd.Period(current_month)
    
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

    # Plot der Gesamtperformance am Ende der Account Overview
    if stock_df is not None:
        st.subheader("Portfolio:")
        total_portfolio_history = get_combined_historical_data(stock_df, period="1y")
        plot_portfolio_performance(total_portfolio_history)
         # Anzeige der Liste der Aktien unterhalb des Gesamtperformance-Charts
        st.subheader("Stocks in Portfolio:")
        st.table(stock_df[['Ticker', 'Amount', 'CurrentPrice', 'TotalValue']])
        

def analyse(df):
    st.title("Analyse")
    if df is not None and 'Amount' in df.columns and 'Date' in df.columns:
        # Debug: Show initial data
        st.write("Initial Data Sample:", df.head())

        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Amount', 'Date'], inplace=True)

        df['YearMonth'] = df['Date'].dt.to_period('M')

        monthly_data = df.groupby('YearMonth')['Amount'].sum().reset_index()
        monthly_income = monthly_data[monthly_data['Amount'] > 0]['Amount'].mean()
        monthly_expenses = monthly_data[monthly_data['Amount'] < 0]['Amount'].mean()
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
        st.error("No Data to analyse")

def adjust_for_inflation(value, years, inflation_rate):
    return value / ((1 + inflation_rate) ** years)
    
def calculate_real_monthly_income(total_investment, years_in_retirement):
    monthly_income = total_investment / (years_in_retirement * 12)
    return monthly_income

def calculate_portfolio_distribution(current_age):
    # Calculate stock percentage as 100 minus the age
    stock_percentage = 100 - current_age
    return stock_percentage, None  # The second value is a placeholder


def monte_carlo_simulation(start_balance, monthly_savings, stock_percentage, years_to_invest, inflation_rate, simulations=1000):
    # Assumed annual return rates (can be adjusted)
    avg_return_stock = 0.07
    avg_return_bond = 0.03
    std_dev_stock = 0.18
    std_dev_bond = 0.06

    # Preparing the simulation array
    results = np.zeros(simulations)

    for i in range(simulations):
        balance = start_balance
        for year in range(years_to_invest):
            annual_stock_return = np.random.normal(avg_return_stock, std_dev_stock)
            annual_bond_return = np.random.normal(avg_return_bond, std_dev_bond)
            weighted_return = (stock_percentage * annual_stock_return + (100 - stock_percentage) * annual_bond_return) / 100
            balance = balance * (1 + weighted_return) + monthly_savings * 12
            balance = adjust_for_inflation(balance, 1, inflation_rate)  # Adjust each year for inflation
        results[i] = balance

    return results
    def recommendation_page():
        st.title("Investment Recommendation")
    
        # User Inputs
        current_age = st.number_input("Your Current Age", min_value=18, max_value=100, step=1)
        retirement_age = st.number_input("Your Retirement Age", min_value=current_age+1, max_value=100, step=1)
        monthly_savings = st.number_input("Monthly Savings", min_value=0.0, step=1.0)
        inflation_rate = st.number_input("Expected Annual Inflation Rate", min_value=0.0, max_value=10.0, step=0.1, value=2.0) / 100
    
        if st.button("Calculate Investment Projection"):
            years_to_invest = retirement_age - current_age
            stock_percentage, _ = calculate_portfolio_distribution(current_age)
    
            # Run Monte Carlo Simulation
            simulation_results = monte_carlo_simulation(0, monthly_savings, stock_percentage, years_to_invest, inflation_rate)
    
            # Plot the results
            plt.figure(figsize=(10, 6))
            for simulation in simulation_results.T:
                plt.plot(simulation, linewidth=0.5, alpha=0.3)
            plt.title("Monte Carlo Simulation of Investment Over Time")
            plt.xlabel("Years")
            plt.ylabel("Portfolio Value")
            st.pyplot(plt)
    
            # Display Results
            median_projection = np.median(simulation_results, axis=1)[-1]
            lower_bound = np.percentile(simulation_results, 5, axis=1)[-1]
            upper_bound = np.percentile(simulation_results, 95, axis=1)[-1]
    
            st.write(f"Projected Investment Value at Retirement (Median): ${median_projection:,.2f}")
            st.write(f"95% Confidence Interval: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
            # Calculate Real Monthly Income
            years_in_retirement = 90 - retirement_age
            real_monthly_income = calculate_real_monthly_income(median_projection, years_in_retirement)
            st.write(f"Estimated Real Monthly Income in Today's Money: ${real_monthly_income:,.2f}")
        
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
        date = st.date_input("Date", datetime.today())
        name = st.text_input("Name")
        amount = st.number_input("Amount", step=1.0)
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
    plt.xlabel('Date')
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
        plt.xlabel('Date')
        plt.ylabel('Kurs')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Daten für {ticker}: {e}")

def Aktienkurse_app():
    st.title("Stock Price")
    aktien_ticker = st.text_input("Insert Stock Ticker:", "")
    if aktien_ticker:
        # Zeigen Sie die Stock Price an
        plot_stock_data(aktien_ticker)
        
        # Holen Sie sich die fundamentalen Daten
        fundamental_data = get_fundamental_data(aktien_ticker)
        
        # Überprüfen Sie die Daten in der Konsole
        print(fundamental_data)
        
        # Anzeige der fundamentalen Daten
        st.subheader(f"Aktuelle Informationen für: {aktien_ticker}")
        if fundamental_data:
            st.write(f"Kurs-Gewinn-Verhältnis (KGV): {fundamental_data['kgv']}")
            st.write(f"Marktkapitalisierung: {fundamental_data['market_cap']}")
            st.write(f"Dividendenrendite: {fundamental_data['dividend_yield']}")

def get_combined_historical_data(stock_df, period="1y"):
    # Holt die kombinierten historischen Daten für das Gesamtportfolio
    portfolio_history = pd.DataFrame()
    
    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        Amount = row['Amount']
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)['Close']
        portfolio_history[ticker] = hist * Amount
    
    # Summiere die Werte aller Aktien für jeden Tag
    portfolio_history['Total'] = portfolio_history.sum(axis=1)
    return portfolio_history['Total']

def plot_portfolio_performance(total_portfolio_history):
    # Plottet die Gesamtperformance des Portfolios
    plt.figure(figsize=(10, 5))
    plt.plot(total_portfolio_history.index, total_portfolio_history, label='Total Portfolio Value')
    plt.title('Total Portfolio Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Value')
    plt.legend()
    st.pyplot(plt)

def main():
    st.sidebar.title("Navigation")
    # Dark mode toggle switch
  

    page = st.sidebar.radio("Choose a page", ["Account Overview", "Analysis", "Recommendation", "Browse"])

    st.title("YouFinance")

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
        recommendation_page()

    elif page == "Browse":
        Aktienkurse_app()

if __name__ == "__main__":
    main()
