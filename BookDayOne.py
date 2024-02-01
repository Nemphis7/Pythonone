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
        
def monte_carlo_simulation(start_balance, monthly_savings, stock_percentage, years_to_invest, inflation_rate, simulations=1000):
    avg_return_stock = 0.07  # Average annual return rate for stocks
    avg_return_bond = 0.03  # Average annual return rate for bonds
    std_dev_stock = 0.18  # Standard deviation for stock returns
    std_dev_bond = 0.06  # Standard deviation for bond returns

    # Preparing a 2D array to store simulation results
    all_results = np.zeros((simulations, years_to_invest))

    for i in range(simulations):
        balance = start_balance
        for year in range(years_to_invest):
            # Generating random returns for stocks and bonds
            annual_stock_return = np.random.normal(avg_return_stock, std_dev_stock)
            annual_bond_return = np.random.normal(avg_return_bond, std_dev_bond)

            # Calculating weighted return based on stock and bond distribution
            weighted_return = (stock_percentage * annual_stock_return + (100 - stock_percentage) * annual_bond_return) / 100

            # Updating the balance with return and monthly savings
            balance = balance * (1 + weighted_return) + monthly_savings * 12

            # Adjusting for inflation
            balance = adjust_for_inflation(balance, 1, inflation_rate)

            # Storing the year-end balance in the results array
            all_results[i, year] = balance

    return all_results


def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Fetch the additional financial metrics
    roe = info.get('returnOnEquity', 'N/A')
    debt_to_equity = info.get('debtToEquity', 'N/A')
    price_to_book = info.get('priceToBook', 'N/A')

    return {
        'kgv': info.get('trailingPE', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A'),
        'roe': roe,
        'debt_to_equity': debt_to_equity,
        'price_to_book': price_to_book,
        # ... any other data you want to fetch
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

def calculate_cumulative_savings(monthly_savings, years_to_invest):
    total_savings = 0
    cumulative_savings = []
    for year in range(1, years_to_invest + 1):
        total_savings += monthly_savings * 12
        cumulative_savings.append(total_savings)
    return cumulative_savings

# Test the function with example values
test_savings = calculate_cumulative_savings(500, 30)  # Example: $500 per month over 30 years
print(test_savings)  # Check the output



def recommendation_page():
    st.title("Investment Recommendation")

    # User Inputs
    current_age = st.number_input("Your Current Age", min_value=18, max_value=100, step=1)
    retirement_age = st.number_input("Your Retirement Age", min_value=current_age + 1, max_value=100, step=1)
    monthly_savings = st.number_input("Monthly Savings", min_value=0.0, step=1.0)
    inflation_rate = st.number_input("Expected Annual Inflation Rate", min_value=0.0, max_value=10.0, step=0.1, value=2.0) / 100

    if st.button("Calculate Investment Projection"):
        years_to_invest = retirement_age - current_age

        # Calculate cumulative savings without investment
        cumulative_savings = calculate_cumulative_savings(monthly_savings, years_to_invest)

        # Plotting only cumulative savings
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, years_to_invest + 1), cumulative_savings, label='Cumulative Savings Without Investment', color='green', linestyle='--')
        plt.title("Cumulative Savings Over Time")
        plt.xlabel("Years")
        plt.ylabel("Total Savings")
        plt.legend()
        st.pyplot(plt)

        try:
            median_projection = np.median(simulation_results, axis=0)
            lower_bound = np.percentile(simulation_results, 5, axis=0)
            upper_bound = np.percentile(simulation_results, 95, axis=0)

            # Plotting simulations, median projection, and cumulative savings
            plt.figure(figsize=(10, 6))
            for simulation in simulation_results:
                plt.plot(range(1, years_to_invest + 1), simulation, linewidth=0.5, alpha=0.3, color='lightgray')
            plt.plot(range(1, years_to_invest + 1), median_projection, label='Median Projection', color='blue')
            plt.plot(range(1, years_to_invest + 1), cumulative_savings, label='Cumulative Savings Without Investment', color='green', linestyle='--')
            plt.fill_between(range(1, years_to_invest + 1), lower_bound, upper_bound, color='gray', alpha=0.5)

            plt.title("Investment Projection Over Time")
            plt.xlabel("Years")
            plt.ylabel("Portfolio Value")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")

def display_total_portfolio_value(stock_df):
    # Ensure numeric types and calculate the total value
    stock_df['Amount'] = pd.to_numeric(stock_df['Amount'], errors='coerce')
    stock_df['CurrentPrice'] = stock_df['CurrentPrice'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
    stock_df['TotalValue'] = stock_df['Amount'] * stock_df['CurrentPrice']
    total_portfolio_value = stock_df['TotalValue'].sum()
    formatted_total_portfolio_value = f"{total_portfolio_value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

    # Display the Total Portfolio Value with the 'total-row' class for dark blue background
    st.markdown(f"<div class='total-row' style='padding: 10px;'><strong>Total Portfolio Value: {formatted_total_portfolio_value}</strong></div>", unsafe_allow_html=True)

    
def account_overview(df, stock_df):
    st.title("Financial Data Analysis")
    current_month = datetime.now().strftime('%Y-%m')
    current_month_period = pd.Period(current_month)

    # Define the common table style
    table_style = """
    <style>
        .financial-table { font-size: 16px; margin-bottom: 20px; }
        .financial-table th, .financial-table td { text-align: left; padding: 8px; }
        .financial-table tr:nth-child(odd) { background-color: #f2f2f2; }
        .financial-table tr.highlight-row { background-color: lightblue; }
    </style>
    """
    
    # Apply the style at the beginning so it affects all tables
    st.markdown(table_style, unsafe_allow_html=True)

    if df is not None:
        df_sorted = df.sort_values(by='Date', ascending=False)
        current_month_data = df[df['Date'].dt.to_period('M') == current_month_period]
        current_month_expenses = current_month_data[current_month_data['Amount'] < 0]['Amount'].sum()
        current_month_income = current_month_data[current_month_data['Amount'] > 0]['Amount'].sum()

        total_expenses = current_month_data[current_month_data['Amount'] < 0]['Amount'].sum()
        total_income = current_month_data[current_month_data['Amount'] > 0]['Amount'].sum()
        account_balance = total_income + total_expenses

        # Creating an HTML table with styling for the financial summary
        html_table = f"""
        <table class='financial-table'>
            <tr><th>Category</th><th>Amount (€)</th></tr>
            <tr><td>Expenses</td><td>{current_month_expenses}</td></tr>
            <tr><td>Income</td><td>{current_month_income}</td></tr>
            <tr class='highlight-row'><td>Total Account Balance</td><td>{account_balance}</td></tr>
        </table>
        """

        # Display the table using markdown
        st.markdown(html_table, unsafe_allow_html=True)

        # Expander to show last ten expense bookings
        with st.expander("View Last 10 Expense Bookings"):
            last_10_expenses = df_sorted[df_sorted['Amount'] < 0].head(10)
            st.markdown(last_10_expenses.to_html(classes='financial-table'), unsafe_allow_html=True)

        # Expander to show last ten income bookings
        with st.expander("View Last 10 Income Bookings"):
            last_10_incomes = df_sorted[df_sorted['Amount'] > 0].head(10)
            st.markdown(last_10_incomes.to_html(classes='financial-table'), unsafe_allow_html=True)

    # [The rest of your code for plotting and displaying stock portfolio remains unchanged]


    # Plot der Gesamtperformance am Ende der Account Overview
    if stock_df is not None:
        st.subheader("Stocks in Portfolio:")
        
        # Calculate the total portfolio value
        total_portfolio_value = stock_df['TotalValue'].replace('[\.,]', '', regex=True).astype(float).sum()
        formatted_total_portfolio_value = f"{total_portfolio_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # Convert the DataFrame to HTML and use the 'financial-table' class for consistent styling
        html_stock_table = (
            stock_df.style
            .applymap(lambda x: "background-color: lightblue", subset=['TotalValue'])
            .set_table_attributes('class="financial-table"')
            .set_properties(**{'font-size': '16px'})  # Ensuring font size consistency
            .to_html(escape=False, index=False)
        )

    # Define the custom styling function for the 'TotalValue' column
    def highlight_total_value(val):
        color = 'lightblue' if pd.notnull(val) else 'white'
        return f'background-color: {color}'

    # Apply the custom styling to the DataFrame
    styled_df = stock_df.style.applymap(highlight_total_value, subset=['TotalValue']) \
                              .set_table_attributes('class="financial-table"') \
                              .set_properties(**{'font-size': '16px'})  # Ensuring font size consistency
    
    # Convert the styled DataFrame to HTML
    html_stock_table = styled_df.to_html(escape=False)

    # Display the styled table using markdown
    st.markdown(html_stock_table, unsafe_allow_html=True)

    # Display the Total Portfolio Value
    display_total_portfolio_value(stock_df)


 # Allow the user to select the time period for the historical data
    period = st.selectbox("Select the time period for the portfolio performance:",
                          options=['5y', '3y', '1y', '6mo'], index=2)

    # Retrieve and plot the historical data based on the selected time period
    portfolio_history = get_portfolio_historical_data(stock_df, period)
    plot_portfolio_history(portfolio_history)


def get_portfolio_historical_data(stock_df, period="1y"):
    portfolio_history = pd.DataFrame()
    for index, row in stock_df.iterrows():
        ticker = row['Ticker']
        amount = row['Amount']
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)['Close']
        portfolio_history[ticker] = hist * amount
    portfolio_history['Total'] = portfolio_history.sum(axis=1)
    return portfolio_history

# Function to plot the historical data
def plot_portfolio_history(portfolio_history):
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_history.index, portfolio_history['Total'], label='Total Portfolio Value')
    plt.title('Portfolio Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

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


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def recommendation_page():
    st.title("Investment Recommendation")

    # User Inputs
    current_age = st.number_input("Your Current Age", min_value=18, max_value=100, step=1)
    retirement_age = st.number_input("Your Retirement Age", min_value=current_age + 1, max_value=100, step=1)
    monthly_savings = st.number_input("Monthly Savings", min_value=0.0, step=1.0)
    inflation_rate = st.number_input("Expected Annual Inflation Rate", min_value=0.0, max_value=10.0, step=0.1, value=2.0) / 100

    if st.button("Calculate Investment Projection"):
        years_to_invest = retirement_age - current_age
        stock_percentage, _ = calculate_portfolio_distribution(current_age)

        # Run Monte Carlo Simulation
        simulation_results = monte_carlo_simulation(0, monthly_savings, stock_percentage, years_to_invest, inflation_rate)

        # Plot each simulation

        plt.figure(figsize=(10, 6))
        for simulation in simulation_results:
            plt.plot(range(years_to_invest), simulation, linewidth=0.5, alpha=0.3)
        plt.title("Monte Carlo Simulation of Investment Over Time")
        plt.xlabel("Years")
        plt.ylabel("Portfolio Value")
        st.pyplot(plt)

        try:
            # Calculate the median projection and bounds
            median_projection = np.median(simulation_results, axis=0)
            lower_bound = np.percentile(simulation_results, 5, axis=0)
            upper_bound = np.percentile(simulation_results, 95, axis=0)

            # Plot median and confidence interval
            plt.figure(figsize=(10, 6))
            plt.fill_between(range(years_to_invest), lower_bound, upper_bound, color='gray', alpha=0.5)
            plt.plot(median_projection, label='Median Projection')
            plt.title("Investment Projection Over Time")
            plt.xlabel("Years")
            plt.ylabel("Portfolio Value")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")

def custom_format_large_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def display_fundamental_data(ticker):
    fundamental_data = get_fundamental_data(ticker)
    
    # Format the market capitalization with commas
    market_cap = f"{fundamental_data['market_cap']:,}" if isinstance(fundamental_data['market_cap'], (int, float)) else fundamental_data['market_cap']
    
    # Format the dividend yield as a percentage
    dividend_yield = f"{fundamental_data['dividend_yield']:.2%}" if isinstance(fundamental_data['dividend_yield'], float) else fundamental_data['dividend_yield']
    
    # Format ROE as a percentage
    roe = f"{fundamental_data['roe']:.2%}" if isinstance(fundamental_data['roe'], float) else fundamental_data['roe']
    
    # Format Debt-to-Equity as a floating number with two decimal points
    debt_to_equity = f"{fundamental_data['debt_to_equity']:.2f}" if isinstance(fundamental_data['debt_to_equity'], float) else fundamental_data['debt_to_equity']
    
    # Format Price-to-Book as a floating number with two decimal points
    price_to_book = f"{fundamental_data['price_to_book']:.2f}" if isinstance(fundamental_data['price_to_book'], float) else fundamental_data['price_to_book']

    # Display the formatted fundamental data
    st.write(f"Price-earnings ratio (P/E ratio): {fundamental_data['kgv']}")
    st.write(f"Market capitalization: {market_cap}")
    st.write(f"Dividend yield: {dividend_yield}")
    st.write(f"Return on Equity (ROE): {roe}")
    st.write(f"Debt-to-Equity Ratio (D/E): {debt_to_equity}")
    st.write(f"Price-to-Book Ratio (P/B): {price_to_book}")


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
    plt.title(f"Share price performance over the last 6 months: {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Share price in €')
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

def plot_stock_data(ticker_a, ticker_b, period='5y'):
    """Plot the stock price percentage change over the selected period."""
    try:
        stock_a = yf.Ticker(ticker_a)
        stock_b = yf.Ticker(ticker_b)

        data_a = stock_a.history(period=period)['Close']
        data_b = stock_b.history(period=period)['Close']

        if data_a.empty or data_b.empty:
            st.error("No historical data found for one or both tickers.")
            return

        # Calculate the percentage change from the start of the period
        data_a_pct_change = data_a.pct_change().fillna(0).add(1).cumprod().sub(1)
        data_b_pct_change = data_b.pct_change().fillna(0).add(1).cumprod().sub(1)

        plt.figure(figsize=(10, 5))
        plt.plot(data_a_pct_change.index, data_a_pct_change, label=f'{ticker_a} Percentage Change', color='blue')
        plt.plot(data_b_pct_change.index, data_b_pct_change, label=f'{ticker_b} Percentage Change', color='orange')

        plt.title(f"Stock Price Percentage Change Comparison")
        plt.xlabel('Date')
        plt.ylabel('Percentage Change')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while fetching data for the tickers: {e}")


def display_comparison_table(ticker_a, ticker_b):
    data_a = get_fundamental_data(ticker_a)
    data_b = get_fundamental_data(ticker_b)

    # Helper function to format the financial metric
    def format_metric(metric):
        if isinstance(metric, float):
            if metric > 0.01:  # Assuming it's a ratio and not a percentage
                return f"{metric:.2f}"
            else:  # Assuming it's a percentage already and needs to be formatted as such
                return f"{metric:.2%}"
        elif isinstance(metric, int):
            return f"{metric:,}"  # For integers like market cap, use commas
        else:
            return metric

    # Creating a dictionary for each stock's financial data, formatted appropriately
    table_data = {
        f"{ticker_a}": [format_metric(data_a[key]) for key in ['kgv', 'market_cap', 'dividend_yield', 'roe', 'debt_to_equity', 'price_to_book']],
        f"{ticker_b}": [format_metric(data_b[key]) for key in ['kgv', 'market_cap', 'dividend_yield', 'roe', 'debt_to_equity', 'price_to_book']]
    }
    
    # Create and display the DataFrame for comparison
    comparison_df = pd.DataFrame(table_data, index=["P/E Ratio", "Market Cap", "Dividend Yield", "ROE", "D/E Ratio", "P/B Ratio"])
    st.table(comparison_df)


def Aktienkurse_app():
    st.title("Stock Price Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        aktien_ticker_a = st.text_input("Insert Stock Ticker for Stock A:", "")
    with col2:
        aktien_ticker_b = st.text_input("Insert Stock Ticker for Stock B:", "")
    with col3:
        period_option = st.selectbox("Select Period:", options=['5y', '3y', '1y', '6mo'], index=0)

    if aktien_ticker_a and aktien_ticker_b:
        # Show the stock price chart and comparison
        plot_stock_data(aktien_ticker_a, aktien_ticker_b, period=period_option)
        
        # Display the fundamental data comparison
        display_comparison_table(aktien_ticker_a, aktien_ticker_b)


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

    # Define the navigation options
    navigation_options = ["Account Overview", "Analysis", "Recommendation", "Browse"]

    # Radio buttons for navigation
    page_selection = st.sidebar.radio("Choose a page", navigation_options)

    st.title("YouFinance")

    # Load data when the app starts or when "Account Overview" is selected
    if 'dataframe' not in st.session_state or page_selection == "Account Overview":
        st.session_state.dataframe = load_data()
        st.session_state.stock_df = load_stock_portfolio()

    df = st.session_state.dataframe
    stock_df = st.session_state.stock_df

    # Call the corresponding function based on the selected page
    if page_selection == "Account Overview":
        account_overview(df, stock_df)
    elif page_selection == "Analysis":
        analyse(df)
    elif page_selection == "Recommendation":
        recommendation_page()
    elif page_selection == "Browse":
        Aktienkurse_app()

if __name__ == "__main__":
    main()
