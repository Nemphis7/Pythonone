import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests
import plotly.graph_objects as go
from datetime import date

def custom_format(value):
    if pd.isna(value):
@@ -21,7 +20,7 @@ def fetch_current_price(ticker):
        price = stock.history(period="1d")['Close'][-1]
        return price
    except Exception as e:
        raise Exception(f"Fehler beim Abrufen der Daten für {ticker}: {e}")
        raise Exception(f"Error fetching data for {ticker}: {e}")

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
@@ -34,10 +33,10 @@ def get_fundamental_data(ticker):
def load_data():
    try:
        url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/Mappe1.xlsx'
        df = pd.read_excel(url, names=['Datum', 'Name', 'Betrag'])
        df = pd.read_excel(url, names=['Date', 'Name', 'Amount'])
        return df
    except Exception as e:
        st.error(f"Fehler beim Lesen der Finanzdatendatei: {e}")
        st.error(f"Error reading financial data file: {e}")
        return None

def load_stock_portfolio():
@@ -52,19 +51,19 @@ def load_stock_portfolio():
        stock_df['TotalValue'] = stock_df['TotalValue'].round(2).apply(custom_format)
        return stock_df
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der Aktienportfolio-Datei: {e}")
        st.error(f"Error processing stock portfolio file: {e}")
        return None

def process_data(df):
    if df is not None and 'Datum' in df.columns:
        df['Betrag'] = pd.to_numeric(df['Betrag'], errors='coerce')
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datum'])
        df['YearMonth'] = df['Datum'].dt.to_period('M')
        df.dropna(subset=['Betrag', 'YearMonth'], inplace=True)
    if df is not None and 'Date' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        df.dropna(subset=['Amount', 'YearMonth'], inplace=True)
        return df
    else:
        st.error("Invalid or missing 'Datum' column in DataFrame")
        st.error("Invalid or missing 'Date' column in DataFrame")
        return None

def plot_portfolio_history(stock_df):
@@ -99,46 +98,46 @@ def plot_financials(financial_df):
    plt.grid(True)
    st.pyplot(plt)

def kontenubersicht(df):
    st.title("Finanzdaten Analyse-App")
    aktueller_monat = datetime.now().strftime('%Y-%m')
    aktueller_monat_periode = pd.Period(aktueller_monat)
def account_overview(df):
    st.title("Financial Data Analysis App")
    current_month = datetime.now().strftime('%Y-%m')
    current_month_period = pd.Period(current_month)
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

def show_add_ticker_form():
    with st.form("add_ticker_form"):
        st.subheader("Neue Aktie hinzufügen")
        st.subheader("Add new stock")
        ticker = st.text_input("Ticker")
        amount = st.number_input("Anzahl", min_value=1, step=1)
        submit_button = st.form_submit_button("Hinzufügen")
        quantity = st.number_input("Quantity", min_value=1, step=1)
        submit_button = st.form_submit_button("Add")
        if submit_button:
            url = 'https://raw.githubusercontent.com/Nemphis7/Pythonone/main/StockPortfolio.xlsx'
            add_ticker_to_excel(ticker, amount, "path_to_your_excel_file.xlsx")
            add_ticker_to_excel(ticker, quantity, "path_to_your_excel_file.xlsx")

def add_ticker_to_excel(ticker, amount, file_path):
def add_ticker_to_excel(ticker, quantity, file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Ticker', 'Amount'])
        df = pd.DataFrame

    if ticker in df['Ticker'].values:
        df.loc[df['Ticker'] == ticker, 'Amount'] += amount
@@ -173,8 +172,26 @@ def add_entry_to_excel(date, name, amount, file_path):

def analyse(df):
    st.title("Analyse")
    if df is not None:
        # ... Existing implementation ...
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
@@ -184,30 +201,39 @@ def analyse(df):
                 "Salary", "Other Income", "Bills", "Entertainment", 
                 "Retirement Fund", "Stocks", "Bonds", "Savings Account"]

        fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label), link=dict(source=source, target=target, value=value))])
        fig = go.Figure(data=[go.Sankey(node=dict(pad=10, thickness=10, line=dict(color="black", width=0.5), label=label), link=dict(source=source, target=target, value=value))])
        fig.update_layout(title_text="Financial Flow - Sankey Diagram", font_size=10)
        st.plotly_chart(fig)

    else:
        st.error("Keine Daten zum Analysieren vorhanden.")

def calculate_investment_period(current_age, retirement_age):
    return retirement_age - current_age

def make_recommendations(user_portfolio, asset_data):
    user_portfolio_df = pd.DataFrame(list(user_portfolio.items()), columns=['Ticker', 'Quantity'])
    merged_data = pd.merge(user_portfolio_df, asset_data, on='Ticker', how='left')
    asset_similarity = cosine_similarity(merged_data.drop(['Ticker', 'Quantity'], axis=1))
    weighted_scores = asset_similarity.dot(merged_data['Quantity'])
    recommendations = pd.DataFrame({'Ticker': asset_data['Ticker'], 'Score': weighted_scores})
    recommendations = recommendations.sort_values(by='Score', ascending=False)
    return recommendations
def generate_financial_recommendations(investment_period, stock_portfolio_df):
    # This is where you'd implement the logic for financial recommendations.
    # For example, you might adjust the stock/bond ratio based on the investment period.
    # This is just a placeholder:
    return f"Recommended investment strategy for an investment period of {investment_period} years."

def empfehlung(df, stock_portfolio_df):
    st.title("Empfehlung")
    user_portfolio = dict(zip(stock_portfolio_df['Ticker'], stock_portfolio_df['Quantity']))
    asset_data = pd.read_excel('path_to_asset_data.xlsx')
    recommendations = make_recommendations(user_portfolio, asset_data)
    st.subheader("Investment Recommendations:")
    st.dataframe(recommendations)

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
@@ -285,56 +311,61 @@ def get_stock_data(ticker):
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
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.title(f"5-Year Stock Price History for {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(data.index, data['Close'], label='Schlusskurs')
        plt.title(f"Aktienkursverlauf der letzten 5 Jahre: {ticker}")
        plt.xlabel('Datum')
        plt.ylabel('Kurs')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    except: 
        print("Stop")
        # Section for displaying stock information
def aktienkurse():
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Daten für {ticker}: {e}")

def aktienkurse_app():
    """Streamlit App für die Anzeige der Aktienkurse."""
    st.title("Aktienkurse")
    aktien_name = st.text_input("Aktienname oder Tickersymbol eingeben:", "")
    if aktien_name:
        try:
            kurs = get_stock_data(aktien_name)
            if kurs is not None:
                display_f
        except:
            print("Nö")
    aktien_ticker = st.text_input("Aktienticker eingeben:", "")
    if aktien_ticker:
        plot_stock_data(aktien_ticker)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Wähle eine Seite", ["Kontenübersicht", "Analyse", "Empfehlung", "Aktienkurse"])
    st.title("Finanzdaten Analyse-App")
    if 'dataframe' not in st.session_state or page == "Kontenübersicht":
    # Ensure the page titles here match the ones in the if-else conditions
    page = st.sidebar.radio("Choose a page", ["Account Overview", "Analysis", "Recommendation", "Stock Prices"])

    st.title("Finance Data Analysis App")

    # Load data when the app starts or when "Account Overview" is selected
    if 'dataframe' not in st.session_state or page == "Account Overview":
        st.session_state.dataframe = load_data()

    df = st.session_state.dataframe

    # Process data if it's not None
    if df is not None:
        df = process_data(df)
    if page == "Kontenübersicht":
        kontenubersicht(df)

    if page == "Account Overview":
        account_overview(df)
        df = show_new_entry_form(df)
        show_add_ticker_form()
        stock_portfolio_df = load_stock_portfolio()

        if stock_portfolio_df is not None:
            st.subheader("Mein Aktienportfolio")
            st.subheader("My Stock Portfolio")
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

            # Calculating the total portfolio value
            total_portfolio_value = sum(stock_portfolio_df['TotalValue'].str.replace('.', '').str.replace(',', '.').astype

if __name__ == "__main__":
    main()
