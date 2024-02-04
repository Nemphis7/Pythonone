import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from datetime import date
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import time

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


# Function to plot the historical data with Plotly
        
def monte_carlo_simulation(start_balance, monthly_savings, stock_percentage, years_to_invest, inflation_rate, simulations=1000):
    avg_return_stock = 0.08  # Average annual return rate for stocks
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



def format_metric(metric):
    if isinstance(metric, float):
        # Check if the metric is less than 1 to decide on the percentage formatting
        if metric < 1:
            return f"{metric:.2%}"
        else:
            # For larger numbers, use separators
            return f"{metric:,.2f}"
    elif isinstance(metric, int):
        # For integers, use separators
        return f"{metric:,}"
    else:
        return metric

def display_comparison_table(ticker_a, ticker_b):
    data_a = get_fundamental_data(ticker_a)
    data_b = get_fundamental_data(ticker_b)

    # Define custom styles for the table
    custom_table_style = """
    <style>
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
        .styled-table tbody tr.active-row {
            font-weight: bold;
            color: #009879;
        }
    </style>
    """

    # Creating a dictionary for each stock's financial data, formatted appropriately
    table_data = {
        f"{ticker_a}": [format_metric(data_a[key]) for key in data_a],
        f"{ticker_b}": [format_metric(data_b[key]) for key in data_b]
    }

    # Create and display the DataFrame for comparison
    comparison_df = pd.DataFrame(table_data, index=list(data_a.keys()))

    # Convert the DataFrame to HTML and add custom styling
    html_comparison_table = comparison_df.to_html(classes="styled-table", escape=False, index=True)
    
    # Display the custom styled HTML table with Streamlit markdown
    st.markdown(custom_table_style + html_comparison_table, unsafe_allow_html=True)

def display_total_portfolio_value(stock_df):
    # Debugging: Display the stock_df DataFrame
    st.write("stock_df DataFrame:", stock_df)

    # Initialize formatted_total_portfolio_value
    formatted_total_portfolio_value = "N/A"

    # Check if required columns exist
    if 'Amount' in stock_df.columns and 'CurrentPrice' in stock_df.columns:
        # Calculate the total portfolio value
        stock_df['Amount'] = pd.to_numeric(stock_df['Amount'], errors='coerce')
        stock_df['CurrentPrice'] = pd.to_numeric(stock_df['CurrentPrice'], errors='coerce')
        stock_df['TotalValue'] = stock_df['Amount'] * stock_df['CurrentPrice']
        total_portfolio_value = stock_df['TotalValue'].sum()

        # Debugging: Display the total portfolio value
        st.write("Total Portfolio Value (before formatting):", total_portfolio_value)

        # Format the total portfolio value
        formatted_total_portfolio_value = f"{total_portfolio_value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        st.error("Required columns not found in stock_df")

    # Display the formatted total portfolio value
    st.write("Formatted Total Portfolio Value:", formatted_total_portfolio_value)

# Function to plot the historical data with Plotly
def plot_portfolio_history_plotly(portfolio_history):
    fig = px.line(
        portfolio_history, 
        x=portfolio_history.index, 
        y="Total", 
        title='Portfolio Performance Over Time',
        labels={'Total': 'Total Value', 'index': 'Date'},
        template="plotly_dark"  # Choose a template that suits your aesthetic needs
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='Total Value',
        legend_title_text='Trend',
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
        ),
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='white'
            ),
        )
    )
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)


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

        total_income = current_month_data[current_month_data['Amount'] > 0]['Amount'].sum()
        total_expenses = current_month_data[current_month_data['Amount'] < 0]['Amount'].sum()

        account_balance = total_income + total_expenses

        # Creating an HTML table with styling for the financial summary
        html_table = f"""
        <table class='financial-table'>
            <tr><th>Category</th><th>Amount (€)</th></tr>
            <tr><td>Income</td><td>{current_month_income}</td></tr>
            <tr><td>Expenses</td><td>{current_month_expenses}</td></tr>
            <tr class='highlight-row'><td>Total Account Balance</td><td>{account_balance}</td></tr>
        </table>
        """

        # Display the table using markdown
        st.markdown(html_table, unsafe_allow_html=True)

        with st.expander("View Last 10 Income Bookings"):
            last_10_incomes = df_sorted[df_sorted['Amount'] > 0].head(10)
            st.markdown(last_10_incomes.to_html(classes='financial-table'), unsafe_allow_html=True)
        
        with st.expander("View Last 10 Expense Bookings"):
            last_10_expenses = df_sorted[df_sorted['Amount'] < 0].head(10)
            st.markdown(last_10_expenses.to_html(classes='financial-table'), unsafe_allow_html=True)

      


    # Plot der Gesamtperformance am Ende der Account Overview
    if stock_df is not None:
        st.subheader("Stocks in Portfolio:")

        # Convert the stock dataframe to HTML and use style for left alignment
        html_stock_table = stock_df.to_html(index=False, escape=False, classes="table table-striped")
        html_stock_table = html_stock_table.replace('<table', '<table style="text-align: left;"')
        
        # Display the HTML table with Streamlit markdown
        st.markdown(html_stock_table, unsafe_allow_html=True)

    # Calculate and display the total portfolio value
    display_total_portfolio_value(stock_df)
    
    # Allow the user to select the time period for the historical data
    period = st.selectbox("Select the time period for the portfolio performance:",
                          options=['5y', '3y', '1y', '6mo'], index=2)
    portfolio_history = get_portfolio_historical_data(stock_df, period)
    
    # Plot the historical data with Plotly
    plot_portfolio_history_plotly(portfolio_history)


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

    # Define current_month_summary here so it's available in the local scope
    current_month = datetime.now().strftime('%Y-%m')
    current_month_period = pd.Period(current_month)
    current_month_summary = df[df['Date'].dt.to_period('M') == current_month_period].groupby('Category')['Amount'].sum().reset_index() if df is not None else pd.DataFrame()

    # This button will trigger the analysis
    if st.button("Start Analysis"):
        with st.spinner('Loading...'):
            time.sleep(1)  # Simulate a long-running operation

            if df is not None and 'Amount' in df.columns and 'Date' in df.columns:
                # Ensure correct data types
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

                # Exclude the current month's data for the last 6 months summary
                df_past_six_months = df[df['Date'].dt.to_period('M') < current_month_period].copy()
                df_past_six_months = df_past_six_months[df_past_six_months['Date'] >= df_past_six_months['Date'].max() - pd.DateOffset(months=6)]
                
                # Create 'Income' and 'Spent' columns
                df_past_six_months['Income'] = df_past_six_months['Amount'].apply(lambda x: x if x > 0 else 0)
                df_past_six_months['Spent'] = df_past_six_months['Amount'].apply(lambda x: x if x < 0 else 0)

                # Group by month and calculate the sums for 'Amount', 'Income', and 'Spent'
                monthly_summary = df_past_six_months.groupby(df_past_six_months['Date'].dt.to_period('M')).agg({'Amount': 'sum', 'Income': 'sum', 'Spent': 'sum'}).reset_index()

                # Sort by the actual date values
                monthly_summary['Date'] = monthly_summary['Date'].dt.to_timestamp()
                monthly_summary.sort_values(by='Date', inplace=True)
                monthly_summary['Date'] = monthly_summary['Date'].dt.strftime('%B %Y')  # Convert back to string for display

                # Calculate the average total for the last six months
                average_total = monthly_summary['Amount'].mean()
                st.session_state['average_monthly_savings'] = average_total

                # Create the total row separately without 'Date'
                total_values = {
                    'Amount': monthly_summary['Amount'].sum(),
                    'Income': monthly_summary['Income'].sum(),
                    'Spent': monthly_summary['Spent'].sum()
                }
                total_row = pd.DataFrame(total_values, index=['Total'])

                # Concatenate the total row and format the DataFrame for display
                monthly_summary_formatted = pd.concat([monthly_summary, total_row])
                
                # Display the last 6 months summary excluding the current month
                st.markdown("### Summary of Last 6 Months (Excluding Current Month)")
                st.dataframe(monthly_summary_formatted.style.format({"Amount": "{:.2f}", "Income": "{:.2f}", "Spent": "{:.2f}"}))
                st.markdown(f"**Average of Savings: {average_total:.2f}**")

                # Display the categories for the current month
                st.markdown("### Categories for the Current Month")
                st.dataframe(current_month_summary.style.format({"Amount": "{:.2f}"}))

                st.success('Analysis complete!')
            else:
                st.error("No data to analyse")



    if st.button("Generate Sankey Diagram"):
        st.header("Sankey Diagram")
        st.write("""
            A Sankey diagram is a type of flow diagram, in which the width of the arrows is proportional to the flow rate. 
            In financial analysis, it is used to depict the flow of income and shows how the initial income is allocated 
            to different expenses and savings.
        """)
        if not current_month_summary.empty:
            total_income = current_month_summary[current_month_summary['Amount'] > 0]['Amount'].sum()
            total_expenses = -current_month_summary[current_month_summary['Amount'] < 0]['Amount'].sum()
            net_savings = total_income - total_expenses

            source = []
            target = []
            value = []
            label = ['Income']

            for i, row in current_month_summary.iterrows():
                if row['Amount'] < 0:
                    source.append(0)
                    target.append(len(label))
                    value.append(-row['Amount'])
                    label.append(row['Category'])

            source.append(0)
            target.append(len(label))
            value.append(net_savings)
            label.append('Savings')

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    color=["#4CAF50", "#F44336", "#2196F3", "#FFC107", "#9C27B0", "#00BCD4", "#E91E63"],
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(76, 175, 80, 0.5)"
                )
            )])

            fig.update_layout(
                title_text="Financial Overview: Income, Expenses, and Savings Flows",
                font=dict(size=12, color='black'),
                paper_bgcolor='white',
                plot_bgcolor='white',
                width=1000,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available to generate Sankey Diagram")
        

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
    

    st.write("Please conduct your own research before opening an account with any of the listed brokers.")
    st.title("Savings Forecast")
    
    st.info("""
        **How to use the Tool and what is it providing us?:**

        - **Enter Your Current Age**: Your current age in years.
        - **Enter Your Retirement Age**: The age at which you plan to retire.
        - **Monthly Savings**: The amount you can save each month. By default, this is set to 70% of your average monthly savings calculated from the Analysis page.
        - **Expected Annual Inflation Rate**: Your expectation of the average annual increase in prices.
        
        Click on "Calculate Investment Projection" to view the potential growth of your investments over time based on a Monte Carlo simulation. 
        - You'll see the median projection as well as the 10th and 90th percentile ranges. 
        - After the calculation, you can choose to either plan your finances yourself or seek professional advisory. 
        - This will give you a comprehensive impression of what a part of your monthly saving could turn to in the future.
    """)

    default_monthly_savings = 0.0


    if 'average_monthly_savings' in st.session_state:
        # Set to 70% of the average monthly savings
        default_monthly_savings = st.session_state['average_monthly_savings'] * 0.7



    current_age = st.number_input("Your Current Age", min_value=18, max_value=100, step=1)
    retirement_age = st.number_input("Your Retirement Age", min_value=current_age + 15, max_value=100, step=1)
    monthly_savings = st.number_input("Monthly Savings", min_value=0.0, step=1.0, value=default_monthly_savings)
    inflation_rate = st.number_input("Expected Annual Inflation Rate", min_value=0.0, max_value=10.0, step=0.1, value=2.0) / 100


    if st.button("Calculate Investment Projection"):
        years_to_invest = retirement_age - current_age
        total_invested = sum([monthly_savings * 12 / ((1 + inflation_rate) ** year) for year in range(1, years_to_invest + 1)])
        stock_percentage, _ = calculate_portfolio_distribution(current_age)

        try:
            # Calculate the median projection and bounds
            simulation_results = monte_carlo_simulation(0, monthly_savings, stock_percentage, years_to_invest, inflation_rate)

            # Plot median and confidence interval (second graph)
            median_projection = np.median(simulation_results, axis=0)
            lower_bound = np.percentile(simulation_results, 10, axis=0)
            upper_bound = np.percentile(simulation_results, 90, axis=0)
            plt.figure(figsize=(10, 6))
            plt.fill_between(range(years_to_invest), lower_bound, upper_bound, color='gray', alpha=0.5)
            plt.plot(median_projection, label='Median')
            plt.title("Investment Projection Over Time")
            plt.xlabel("Years")
            plt.ylabel("Portfolio Value in €")
            plt.legend()
            st.pyplot(plt)

            # Display results as text below the second graph
            final_median_projection = median_projection[-1]
            final_lower_bound = lower_bound[-1]
            final_upper_bound = upper_bound[-1]
            st.write(f"Total amount invested over {years_to_invest} years (adjusted for inflation): € {total_invested:,.2f}")
            st.write(f"The median projected portfolio value at the end of the investment period (considering inflation) is: € {final_median_projection:,.2f}")
            st.write(f"The projected portfolio value range is from € {final_lower_bound:,.2f} to € {final_upper_bound:,.2f} (10th to 90th percentile)")
            
           
    
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

def resources_page():
    st.title("Resources")
    st.write("Download the initial Excel sheets as formatting examples.")

    # Links to your GitHub raw content
    portfolio_excel_url = "https://raw.githubusercontent.com/Nemphis7/Pythonone/main/StockPortfolio.xlsx"
    transaction_excel_url = "https://raw.githubusercontent.com/Nemphis7/Pythonone/main/Mappe1.xlsx"

    st.markdown(f"[Download Portfolio Excel Template]({portfolio_excel_url})", unsafe_allow_html=True)
    st.markdown(f"[Download Transaction Excel Template]({transaction_excel_url})", unsafe_allow_html=True)

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
    
    # Highlighted explanation box
    st.info("""
        **How to use the Stock Price Comparison tool:**

        1. **Enter Stock Tickers**: Type in the ticker symbols for the two companies you're interested in. For example, 'MSFT' for Microsoft Corporation.
        2. **Select Time Frame**: Choose the duration of the comparison.
        3. **Review Stock Performance Graph**: Observe the comparative index values on the graph, where the performance of each stock is normalized for a comparison over time.
        4. **Analyze Financial Metrics**: Key financial figures are shown below the chart.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        aktien_ticker_a = st.text_input("Insert Stock Ticker for Stock A:", "")
    with col2:
        aktien_ticker_b = st.text_input("Insert Stock Ticker for Stock B:", "")
    with col3:
        period_option = st.selectbox("Select Period:", options=['5y', '3y', '1y', '6mo'], index=0)

    if aktien_ticker_a and aktien_ticker_b:
        # Prepare a DataFrame for each stock with 'Ticker' and 'Amount' columns
        df_a = pd.DataFrame({'Ticker': [aktien_ticker_a], 'Amount': [1]})
        df_b = pd.DataFrame({'Ticker': [aktien_ticker_b], 'Amount': [1]})

        # Fetch the historical data for both stocks using the existing function
        portfolio_history_a = get_combined_historical_data(df_a, period=period_option)
        portfolio_history_b = get_combined_historical_data(df_b, period=period_option)

        # Normalize the starting prices to the same value for comparison (e.g., 100)
        portfolio_history_a = (portfolio_history_a / portfolio_history_a.iloc[0]) * 100
        portfolio_history_b = (portfolio_history_b / portfolio_history_b.iloc[0]) * 100

        # Combine the data into a single DataFrame for plotting
        combined_portfolio_history = pd.DataFrame({
            'Date': portfolio_history_a.index,
            aktien_ticker_a: portfolio_history_a.values,
            aktien_ticker_b: portfolio_history_b.values
        }).set_index('Date')

        # Plot the combined historical data using Plotly
        fig = px.line(combined_portfolio_history, x=combined_portfolio_history.index, y=[aktien_ticker_a, aktien_ticker_b])
        fig.update_traces(overwrite=True, marker=dict(size=10))

        # Update traces to set the color, assuming the first trace is for Stock A and the second is for Stock B
        fig.update_traces(overwrite=True, selector=dict(name=aktien_ticker_a), line=dict(color='blue'))
        fig.update_traces(overwrite=True, selector=dict(name=aktien_ticker_b), line=dict(color='orange'))

        fig.update_layout(
            title='Stock Performance Comparison Over Time',
            xaxis_title='Date',
            yaxis_title='Performance Index',
            template='plotly_dark',
            legend_title_text='Ticker'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display the comparison table using the existing function
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
    
def broker_overview_comparison():
    st.title("Broker Overview/Comparison")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Do the Financial Planning Yourself"):
            st.markdown("""
            Choosing the Right Broker: An Essential Step in Your Investment Journey
    
            The choice of broker is crucial for anyone starting in wealth building or for experienced investors looking for a new partner. Each broker offers a unique set of features, fees, and platforms, catering to different investor needs. While some may prioritize low fees, others might look for extensive research tools or a wide range of available securities. Here we compare Trade Republic, ING Bank, and DEGIRO, providing key facts to help you find the broker that best fits your investment strategy.
            """)


            brokers_info = {
                "Trade Republic": {
                    "description": "Trade Republic is a mobile-only broker that offers a simple, streamlined trading experience. It's known for its ease of use and low fees, making it a popular choice for new and casual investors.",
                    "fees": "Offers commission-free trades, with only a €1 external fee for settlement of orders. No account management fees.",
                    "platform": "Mobile app available on both iOS and Android, focused on simplicity and ease of use. Does not offer a desktop trading platform.",
                    "website": "https://www.traderepublic.com",
                    "key_features": [
                        "Wide range of ETFs and stocks available for trading.",
                        "Offers savings plans on ETFs completely commission-free.",
                        "Real-time prices and push notifications for executed orders."
                    ]
                },
                "ING Bank": {
                    "description": "ING Bank, through its brokerage arm ING-DiBa, offers a comprehensive online banking and brokerage experience with access to a wide range of investment products including stocks, ETFs, bonds, and more.",
                    "fees": "Fees vary by trade volume; for example, stock trades on German exchanges cost €4.90 + 0.25% of the order volume (minimum €9.90, maximum €69.90). No custody account fees.",
                    "platform": "Offers a web-based trading platform and a mobile app, providing a balance between functionality for experienced traders and simplicity for new users.",
                    "website": "https://www.ing.de",
                    "key_features": [
                        "Access to international markets.",
                        "Offers a broad selection of financial products beyond stocks and ETFs, including bonds, funds, and derivatives.",
                        "Provides extensive research tools and financial news."
                    ]
                },
              "DEGIRO": {
            "description": "DEGIRO is a Dutch online brokerage company that offers low-cost trading to retail investors worldwide. It's known for its affordable pricing structure and broad market access.",
            "fees": "Low trading fees compared to competitors, with specific fees depending on the market. For example, US stock trades are €0.50 + USD 0.004 per share.",
            "platform": "Web-based platform and mobile app available, focusing on functionality and offering tools for technical analysis.",
            "website": "https://www.degiro.eu",
            "key_features": [
                "Affordable pricing structure for trading across a wide range of markets.",
                "Offers an easy-to-use web-based platform and mobile app.",
                "Provides access to a broad spectrum of investment products."
                    ]
        },  # This comma was missing
        "JP Morgan": {
            "description": "JP Morgan offers a robust trading platform with a wide range of investment options, tailored for both novice and experienced investors looking for comprehensive financial services.",
            "fees": "Varies by account type and services used. Offers some commission-free options.",
            "platform": "Advanced web and mobile trading platforms with access to extensive research and tools.",
            "website": "https://www.jpmorgan.com",
            "key_features": [
                "Access to global markets and a wide range of investment products.",
                "Robust research and analysis tools.",
                "Personalized financial advisory services."
            ]
        },

                "Sparkasse": {
                    "description": "Sparkasse's brokerage arm provides a user-friendly trading experience, focusing on German and European markets with competitive fees for casual and intermediate investors.",
                    "fees": "Competitive trading fees, with special offers for savings plans on ETFs and stocks.",
                    "platform": "Web-based platform and mobile app offering easy access to trade executions and account management.",
                    "website": "https://www.sparkasse.de",
                    "key_features": [
                        "Easy access to European markets.",
                        "Offers a variety of savings and investment plans.",
                        "Reliable customer service with a strong local presence."
                    ]
                },
                "VR Bank": {
                    "description": "VR Bank provides comprehensive banking and brokerage services with a focus on cooperative values, offering personalized advice and a range of investment products for its members.",
                    "fees": "Fees depend on services and products chosen, often offering lower fees for members.",
                    "platform": "Combines online banking and brokerage services in a single platform, with mobile app support.",
                    "website": "https://www.vrbank.de",
                    "key_features": [
                        "Member-focused banking and investment services.",
                        "Access to a wide range of financial products, including sustainable investment options.",
                        "Personalized advisory services."
                    ]
                } ,
                "Deutsche Postbank": {
            "description": "Part of the Deutsche Bank Group, Postbank offers online brokerage services with a focus on affordability and accessibility, catering to the needs of everyday investors.",
            "fees": "Competitive pricing structure with low order fees and no custody account fees.",
            "platform": "User-friendly web and mobile platforms, designed for straightforward trading and account management.",
            "website": "https://www.postbank.de",
            "key_features": [
                "Affordable access to German and international markets.",
                "Simple and transparent fee structure.",
                "Integrates with Postbank's broader banking services."
                    ]
                },
        "eToro": {
            "description": "eToro is renowned for its social trading platform, allowing investors to copy trades of successful peers, and offering a wide range of cryptocurrencies, stocks, and other financial instruments.",
            "fees": "Zero-commission stock trading; other fees include spread fees for crypto and varying fees for other assets.",
            "platform": "Innovative platform that emphasizes social trading aspects, alongside traditional investment options.",
            "website": "https://www.etoro.com",
            "key_features": [
                "Unique social trading features.",
                "Wide range of cryptocurrencies alongside traditional investment options.",
                "User-friendly interface suitable for beginners and experienced traders alike."
                    ]
                }
            }


            broker_selection = st.selectbox("Select a Broker to Learn More:", list(brokers_info.keys()))
            broker = brokers_info[broker_selection]

            st.subheader(f"{broker_selection}")
            st.write(f"**Description**: {broker['description']}")
            st.write(f"**Fees**: {broker['fees']}")
            st.write(f"**Platform**: {broker['platform']}")
            st.markdown(f"**Website**: [Visit]({broker['website']})", unsafe_allow_html=True)
            st.write("**Key Features:**")
            for feature in broker['key_features']:
                st.markdown(f"- {feature}")
            st.markdown("""
        ### Personalized Financial Guidance

        After exploring your broker options, you might have specific questions or need guidance tailored to your financial situation and goals. A one-on-one meeting with a financial advisor can provide you with personalized advice, helping you make informed decisions about your investments.

        Whether you're just starting on your investment journey or looking to refine your strategy, a financial advisor can offer insights into:

        - Building a diversified investment portfolio
        - Understanding market risks and opportunities
        - Planning for long-term financial goals, such as retirement or wealth accumulation
        - Navigating complex financial situations and tax implications

        ### Schedule a Meeting

            If you're ready to take the next step in your financial journey, schedule a meeting below. You'll be able to choose a date and time that works best for you to discuss your investment needs and questions.

    with col2:
        if st.button("Get Professional Advisory"):
            # Calendly embed link
            calendly_embed_link = 'https://calendly.com/information-you-finance'  # Replace with your actual link
            calendly_html = f"""
            <div class="calendly-inline-widget" data-url="{calendly_embed_link}" style="min-width:320px;height:630px;"></div>
            <script type="text/javascript" src="https://assets.calendly.com/assets/external/widget.js"></script>
            """
            components.html(calendly_html, height=700)


def main():
    st.sidebar.title("Menu")

    # Updated to include "Brokers" as a new navigation option
    navigation_options = ["Account Overview", "Analysis", "Planning", "Wealth Advisory", "Browse","Resources"]

    page_selection = st.sidebar.radio("Choose a page", navigation_options)

    st.title("YouFinance")

    # File upload section
    with st.sidebar:
        uploaded_portfolio_file = st.file_uploader("Upload Portfolio Excel", type=['xlsx'])
        uploaded_transactions_file = st.file_uploader("Upload Transactions Excel", type=['xlsx'])

        if uploaded_portfolio_file is not None:
            st.session_state.stock_df = pd.read_excel(uploaded_portfolio_file)

        if uploaded_transactions_file is not None:
            st.session_state.dataframe = pd.read_excel(uploaded_transactions_file)

    # Load default data if not uploaded
    if 'dataframe' not in st.session_state or 'stock_df' not in st.session_state:
        st.session_state.dataframe = load_data() if 'dataframe' not in st.session_state else st.session_state.dataframe
        st.session_state.stock_df = load_stock_portfolio() if 'stock_df' not in st.session_state else st.session_state.stock_df

    df = st.session_state.dataframe
    stock_df = st.session_state.stock_df

    if page_selection == "Account Overview":
        account_overview(df, stock_df)
    elif page_selection == "Analysis":
        analyse(df)
    elif page_selection == "Planning":
        recommendation_page()
    elif page_selection == "Browse":
        Aktienkurse_app()
    elif page_selection == "Wealth Advisory":  # Corrected to match the navigation option
        broker_overview_comparison()  # Correct function call
    elif page_selection == "Resources":
        resources_page()

if __name__ == "__main__":
    main()
