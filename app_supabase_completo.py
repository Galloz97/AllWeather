"""
Portfolio Monitor - Applicazione Streamlit completa multi-utente con Supabase
Versione con tutte le funzionalità: Monitoraggio, Analisi avanzata, FIRE, Leva, Monte Carlo
Pronto per GitHub e Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
from csv_importer import process_csv, TICKER_MAPPING
import warnings

warnings.filterwarnings('ignore')

from supabase_manager import (
    SupabaseManager,
    check_authentication,
    render_logout_button,
    init_supabase_auth
)

# ==================== CONFIGURAZIONE ====================

st.set_page_config(
    page_title="Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AUTENTICAZIONE ====================

check_authentication()
init_supabase_auth()

supabase = st.session_state.supabase
user_id = st.session_state.user.id

# ==================== UTILITY FUNCTIONS ====================

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="5y"):
    """Recupera dati storici da yfinance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data if not data.empty else None
    except:
        return None

@st.cache_data(ttl=3600)  # Cache per 1 ora
def get_stock_data_cached(ticker, period="5y"):
    """Scarica dati storici con cache"""
    try:
        data = yf.download(ticker, period=period, progress=False, timeout=10)
        if not data.empty and 'Close' in data.columns:
            return data['Close']
        return None
    except Exception as e:
        print(f"Errore download {ticker}: {e}")
        return None


@st.cache_data(ttl=3600)
def get_current_price(ticker):
    """Recupera prezzo attuale di un ticker"""
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.history(period='1d')['Close'].iloc[-1]
        return float(price)
    except:
        return None

def format_currency(value):
    """Formatta valore come valuta italiana"""
    if pd.isna(value) or value is None:
        return "€0,00"
    return f"€{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

def format_percentage(value):
    """Formatta valore come percentuale"""
    if pd.isna(value):
        return "0,00%"
    return f"{value:.2f}%"

def get_portfolio_metrics():
    """Calcola metriche del portafoglio con prezzo medio corretto (metodo weighted average)"""
    try:
        # Recupera tutte le transazioni
        transazioni_df = supabase.get_transazioni(user_id)
        
        if transazioni_df.empty:
            return None, pd.DataFrame()
        
        # Escludi liquidità
        transazioni_asset = transazioni_df[transazioni_df['ticker'] != 'LIQUIDITA'].copy()
        
        if transazioni_asset.empty:
            return None, pd.DataFrame()
        
        # Calcola posizione per ogni ticker
        portfolio_data = []
        
        for ticker in transazioni_asset['ticker'].unique():
            ticker_trans = transazioni_asset[transazioni_asset['ticker'] == ticker].copy()
            ticker_trans = ticker_trans.sort_values('data')
            
            # Tracking quantità e costo con metodo media mobile ponderata
            quantita_corrente = 0
            costo_medio_corrente = 0
            
            for _, trans in ticker_trans.iterrows():
                qty = float(trans['quantita'])
                prezzo = float(trans['prezzo_unitario'])
                
                if qty > 0:  # Buy
                    # Calcola nuovo prezzo medio ponderato
                    if quantita_corrente > 0:
                        # Media ponderata: (vecchio_costo*vecchia_qty + nuovo_costo*nuova_qty) / (vecchia_qty + nuova_qty)
                        costo_medio_corrente = (costo_medio_corrente * quantita_corrente + prezzo * qty) / (quantita_corrente + qty)
                    else:
                        # Prima acquisizione
                        costo_medio_corrente = prezzo
                    
                    quantita_corrente += qty
                    
                else:  # Sell (qty è negativo)
                    # Vendita: riduci quantità ma mantieni lo stesso prezzo medio
                    quantita_corrente += qty  # qty è già negativo
                    
                    # Se vendo tutto, resetta il prezzo medio
                    if quantita_corrente <= 0:
                        costo_medio_corrente = 0
            
            # IMPORTANTE: Salta se quantità <= 0
            if quantita_corrente <= 0.001:
                continue
            
            # Prezzo corrente di mercato
            prezzo_corrente = get_current_price(ticker)
            if prezzo_corrente is None:
                prezzo_corrente = costo_medio_corrente
            
            # Calcoli finali
            costo_totale = quantita_corrente * costo_medio_corrente
            valore_totale = quantita_corrente * prezzo_corrente
            guadagno = valore_totale - costo_totale
            guadagno_pct = (guadagno / costo_totale * 100) if costo_totale > 0 else 0
            
            # Determina asset class
            if ticker.endswith('.MI'):
                asset_class = 'ETF'
            else:
                asset_class = 'Azioni'
            
            portfolio_data.append({
                'Ticker': ticker,
                'Quantità': quantita_corrente,
                'Prezzo Acq.': costo_medio_corrente,
                'Prezzo Att.': prezzo_corrente,
                'Valore Totale': valore_totale,
                'Costo Totale': costo_totale,
                'P&L €': guadagno,
                'P&L %': guadagno_pct,
                'Asset Class': asset_class
            })
        
        if not portfolio_data:
            return None, pd.DataFrame()
        
        df = pd.DataFrame(portfolio_data)
        
        total_valore = df['Valore Totale'].sum()
        total_costo = df['Costo Totale'].sum()
        df['Peso %'] = (df['Valore Totale'] / total_valore * 100) if total_valore > 0 else 0
        
        total_guadagno = total_valore - total_costo
        total_guadagno_pct = (total_guadagno / total_costo * 100) if total_costo > 0 else 0
        
        metrics = {
            'valore_totale': total_valore,
            'costo_totale': total_costo,
            'guadagno_totale': total_guadagno,
            'guadagno_pct': total_guadagno_pct,
            'df': df
        }
        
        return metrics, df
        
    except Exception as e:
        print(f"Errore calcolo portafoglio: {e}")
        import traceback
        traceback.print_exc()
        return None, pd.DataFrame()


# =================== CALCOLO LIQUITIDA'====================

def get_saldo_liquidita():
    """Calcola il saldo della liquidità considerando TUTTE le transazioni"""
    try:
        transazioni_df = supabase.get_transazioni(user_id)
        
        if transazioni_df.empty:
            return 0.0
        
        saldo = 0.0
        
        for _, trans in transazioni_df.iterrows():
            ticker = trans['ticker']
            tipo = trans['tipo']
            importo = float(trans.get('importo', 0))
            commissioni = float(trans.get('commissioni', 0))
            
            # MOVIMENTI LIQUIDITÀ DIRETTI
            if ticker == 'LIQUIDITA':
                if tipo in ['Deposit', 'Bonifico']:
                    # Deposito: aumenta liquidità
                    saldo += abs(importo)
                elif tipo in ['Withdrawal', 'Prelievo']:
                    # Prelievo: riduce liquidità
                    saldo -= abs(importo)
                elif tipo in ['Tax', 'Imposta']:
                    # Imposta: riduce liquidità
                    saldo -= abs(importo)
            
            # MOVIMENTI TITOLI (impattano la liquidità)
            else:
                if tipo == 'Buy':
                    # Acquisto: pago importo + commissioni (riduce liquidità)
                    saldo -= abs(importo)
                    saldo -= commissioni
                elif tipo == 'Sell':
                    # Vendita: ricevo importo - commissioni (aumenta liquidità)
                    saldo += abs(importo)
                    saldo -= commissioni
                elif tipo == 'Dividend':
                    # Dividendo: ricevo importo (aumenta liquidità)
                    saldo += abs(importo)
        
        return saldo
        
    except Exception as e:
        print(f"Errore calcolo liquidità: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ==================== CALCOLI ANALITICI ====================

def calculate_volatility(ticker):
    """Calcola volatilità annualizzata"""
    data = get_stock_data(ticker)
    if data is None or len(data) == 0:
        return 0.15
    returns = data['Close'].pct_change().dropna()
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(df, risk_free_rate=0.025):
    """Calcola Sharpe Ratio del portafoglio"""
    try:
        portfolio_return = df['P&L %'].mean() / 100  # Media rendimenti
        
        # Calcola volatilità portfolio
        volatilities = []
        weights = df['Peso %'] / 100
        
        for ticker in df['Ticker']:
            try:
                vol = calculate_volatility(ticker)
                if vol is not None:
                    vol_float = float(vol)
                    if not np.isnan(vol_float) and vol_float > 0:
                        volatilities.append(vol_float)
                    else:
                        volatilities.append(0)
                else:
                    volatilities.append(0)
            except Exception as e:
                print(f"Errore volatilità Sharpe per {ticker}: {e}")
                volatilities.append(0)
        
        if not volatilities or all(v == 0 for v in volatilities):
            return 0
        
        portfolio_vol = np.average(volatilities, weights=weights)
        
        if portfolio_vol > 0:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return float(sharpe)
        
        return 0
    
    except Exception as e:
        print(f"Errore calcolo Sharpe Ratio: {e}")
        return 0


def calculate_max_drawdown(ticker):
    """Calcola maximum drawdown di un ticker"""
    data = get_stock_data(ticker)
    if data is None or len(data) < 2:
        return 0
    
    cumulative = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())

def calculate_ulcer_index(ticker):
    """Calcola Ulcer Index (misura il dolore dei drawdown)"""
    data = get_stock_data(ticker)
    if data is None or len(data) < 2:
        return 0
    
    cumulative = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown_pct = ((cumulative - running_max) / running_max * 100)
    
    negative_drawdowns = drawdown_pct[drawdown_pct < 0]
    if len(negative_drawdowns) == 0:
        return 0
    
    ulcer_index = np.sqrt(np.mean(negative_drawdowns ** 2))
    return float(ulcer_index)

def calculate_risk_parity_weights(df):
    """Calcola pesi Risk Parity basati sulla volatilità"""
    try:
        weights = []
        
        for _, row in df.iterrows():
            ticker = row['Ticker']
            try:
                vol = calculate_volatility(ticker)
                
                # Converti a float e verifica validità
                if vol is not None:
                    vol_float = float(vol)
                    if not np.isnan(vol_float) and vol_float > 0:
                        weights.append({
                            'Ticker': ticker,
                            'Volatilità': vol_float,
                            'Inv_Vol': 1 / vol_float
                        })
            except Exception as e:
                print(f"Errore volatilità per {ticker}: {e}")
                continue
        
        if not weights:
            return pd.DataFrame()
        
        weights_df = pd.DataFrame(weights)
        total_inv_vol = weights_df['Inv_Vol'].sum()
        weights_df['Risk Parity %'] = (weights_df['Inv_Vol'] / total_inv_vol * 100)
        
        return weights_df[['Ticker', 'Risk Parity %']]
    
    except Exception as e:
        print(f"Errore calcolo Risk Parity: {e}")
        return pd.DataFrame()


def calculate_relaxed_risk_parity(df, min_weight=0.05, max_weight=0.40):
    """Calcola Risk Parity con vincoli min/max"""
    try:
        risk_parity = calculate_risk_parity_weights(df)
        
        if risk_parity.empty:
            return pd.DataFrame()
        
        # Applica vincoli
        relaxed = risk_parity.copy()
        relaxed['Risk Parity %'] = relaxed['Risk Parity %'] / 100  # Converti a decimale
        
        # Clipping
        relaxed['Relaxed %'] = relaxed['Risk Parity %'].clip(min_weight, max_weight)
        
        # Rinormalizza
        total_weight = relaxed['Relaxed %'].sum()
        if total_weight > 0:
            relaxed['Relaxed %'] = (relaxed['Relaxed %'] / total_weight * 100)
        else:
            relaxed['Relaxed %'] = 0
        
        return relaxed[['Ticker', 'Relaxed %']]
    
    except Exception as e:
        print(f"Errore calcolo Relaxed Risk Parity: {e}")
        return pd.DataFrame()


def monte_carlo_simulation(df, n_simulations=1000, days=252, monthly_contribution=0, annual_contribution=0):
    """
    Simulazione Monte Carlo OTTIMIZZATA con versamenti periodici
    
    Args:
        df: DataFrame portafoglio
        n_simulations: Numero iterazioni
        days: Giorni di simulazione
        monthly_contribution: Versamento mensile (€)
        annual_contribution: Versamento annuale (€)
    """
    try:
        # Prepara dati
        tickers = df['Ticker'].tolist()
        weights = (df['Peso %'] / 100).values
        initial_value = df['Valore Totale'].sum()
        
        # OTTIMIZZAZIONE: Scarica dati UNA SOLA VOLTA
        returns = []
        vols = []
        
        print(f"📥 Download dati per {len(tickers)} ticker...")
        
        for ticker in tickers:
            try:
                close_prices = get_stock_data_cached(ticker, period="1y")
                
                if close_prices is not None and len(close_prices) > 20:
                    daily_returns = close_prices.pct_change().dropna()
                    ret = daily_returns.mean()
                    vol = daily_returns.std() * np.sqrt(252)
                    
                    returns.append(float(ret) if not np.isnan(ret) else 0.0005)
                    vols.append(float(vol) if not np.isnan(vol) and vol > 0 else 0.15)
                else:
                    returns.append(0.0005)
                    vols.append(0.15)
                    
            except Exception as e:
                print(f"Errore {ticker}: {e}")
                returns.append(0.0005)
                vols.append(0.15)
        
        returns = np.array(returns)
        vols = np.array(vols)
        
        print(f"🎲 Simulazione con versamenti: {monthly_contribution}€/mese, {annual_contribution}€/anno")
        
        # Pre-genera random numbers
        np.random.seed(42)
        random_matrix = np.random.normal(0, 1, size=(n_simulations, days, len(tickers)))
        
        # Calcola rendimenti giornalieri
        daily_returns_matrix = (
            returns / 252 +
            vols / np.sqrt(252) * random_matrix
        )
        
        # Pesa i rendimenti
        portfolio_returns = np.sum(daily_returns_matrix * weights, axis=2)
        
        # NUOVO: Calcola versamenti per ogni giorno
        contributions_schedule = np.zeros(days)
        
        # Versamenti mensili (ogni 21 giorni di trading ≈ 1 mese)
        if monthly_contribution > 0:
            for day in range(0, days, 21):  # ~12 volte/anno
                contributions_schedule[day] = monthly_contribution
        
        # Versamenti annuali (ogni 252 giorni di trading = 1 anno)
        if annual_contribution > 0:
            for day in range(0, days, 252):
                contributions_schedule[day] += annual_contribution
        
        # Simula portafoglio con versamenti
        simulations = np.zeros((n_simulations, days))
        
        for sim in range(n_simulations):
            portfolio_value = initial_value
            
            for day in range(days):
                # Aggiungi versamento del giorno
                portfolio_value += contributions_schedule[day]
                
                # Applica rendimento
                portfolio_value *= (1 + portfolio_returns[sim, day])
                
                simulations[sim, day] = portfolio_value
        
        # Calcola capitale versato totale
        total_contributions = np.sum(contributions_schedule)
        
        print(f"✅ Simulazione completata! Capitale versato: €{total_contributions:,.0f}")
        
        return simulations, total_contributions
        
    except Exception as e:
        print(f"Errore simulazione Monte Carlo: {e}")
        import traceback
        traceback.print_exc()
        return np.array([[]]), 0

def bootstrap_simulation(df, n_simulations=1000, days=252, monthly_contribution=200, annual_contribution=1000):
    tickers = df['Ticker'].tolist()
    weights = (df['Peso %'] / 100).values
    initial_value = df['Valore Totale'].sum()
    print("DEBUG: Tickers:", tickers)
    print("DEBUG: Pesi (percentuali):", weights)
    print("DEBUG: Valore iniziale totale:", initial_value)
    
    historical_returns = {}
    for ticker in tickers:
        close_prices = get_stock_data_cached(ticker, period="5y")
        if close_prices is not None and not close_prices.empty:
            # Se close_prices è DataFrame, estrai la colonna prezzi chiusura come Serie
            if isinstance(close_prices, pd.DataFrame):
                if 'Close' in close_prices.columns:
                    close_series = close_prices['Close']
                else:
                    # Se è DataFrame senza 'Close', prendi la prima colonna
                    close_series = close_prices.iloc[:, 0]
            elif isinstance(close_prices, pd.Series):
                close_series = close_prices
            else:
                close_series = None
        
            if close_series is not None:
                daily_returns = close_series.pct_change().dropna()
                if len(daily_returns) < 30:
                    daily_returns_array = np.array([0.0005] * 252)
                else:
                    daily_returns_array = daily_returns.values
            else:
                daily_returns_array = np.array([0.0005] * 252)
        else:
            daily_returns_array = np.array([0.0005] * 252)
        
        historical_returns[ticker] = daily_returns_array

    contributions_schedule = np.zeros(days)
    for day in range(0, days, 21):
        contributions_schedule[day] = monthly_contribution
    for day in range(0, days, 252):
        contributions_schedule[day] += annual_contribution

    simulations = np.zeros((n_simulations, days))
    np.random.seed(42)

    for sim in range(n_simulations):
        portfolio_value = initial_value
        for day in range(days):
            portfolio_value += contributions_schedule[day]
            portfolio_return = 0.0
            for ticker, weight in zip(tickers, weights):
                sampled_return = np.random.choice(historical_returns[ticker])
                portfolio_return += weight * sampled_return
            portfolio_value *= (1 + portfolio_return)
            simulations[sim, day] = portfolio_value

    total_contributions = np.sum(contributions_schedule)
    return simulations, total_contributions

# ==================== PAGINA MONITORAGGIO ====================

def page_monitoraggio():
    """Pagina principale - Monitoraggio Portafoglio"""
    st.title("📊 Monitoraggio Portafoglio")
    
    metrics, df = get_portfolio_metrics()
    
    if metrics is None:
        st.info("📭 Nessun asset nel portafoglio. Aggiungi un asset per iniziare!")
        
        st.subheader("➕ Aggiungi Asset al Portafoglio")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", key="new_ticker")
        with col2:
            quantita = st.number_input("Quantità", min_value=0.0, key="new_quantita")
        with col3:
            prezzo = st.number_input("Prezzo Acquisto (€)", min_value=0.0, key="new_prezzo")
        with col4:
            asset_class = st.selectbox("Classe Asset", ["Azioni", "ETF", "Obbligazioni", "Cripto", "Liquidità"], key="new_class")
        
        if st.button("✅ Aggiungi Asset"):
            if ticker and quantita > 0 and prezzo > 0:
                result = supabase.add_portafoglio_asset(
                    user_id=user_id,
                    ticker=ticker.upper(),
                    quantita=quantita,
                    prezzo_acquisto=prezzo,
                    asset_class=asset_class
                )
                
                if result['success']:
                    st.success(f"✓ {ticker.upper()} aggiunto al portafoglio!")
                    st.rerun()
                else:
                    st.error(f"Errore: {result['error']}")
            else:
                st.warning("Inserisci valori validi")
        return
    
    # KPI Principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Valore Totale",
            format_currency(metrics['valore_totale']),
            delta=format_currency(metrics['guadagno_totale'])
        )
    
    with col2:
        st.metric(
            "Guadagno/Perdita %",
            format_percentage(metrics['guadagno_pct']),
            delta=f"{metrics['guadagno_pct']:.2f}%"
        )
    
    with col3:
        st.metric("Numero Asset", len(df))
    
    with col4:
        stats = supabase.get_statistiche_transazioni(user_id)
        st.metric("Commissioni Totali", format_currency(stats.get('commissioni_totale', 0)))
    
        st.divider()
    
    # Calcola e mostra sempre la liquidità
    saldo_liquidita = get_saldo_liquidita()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Composizione Portafoglio")
    
    with col2:
        st.metric("💰 Liquidità Disponibile", format_currency(saldo_liquidita))
    
    # Se ci sono asset, mostra la tabella
    if metrics:
        # Aggiungi liquidità al dataframe
        liquidita_row = pd.DataFrame([{
            'Ticker': 'LIQUIDITA',
            'Quantità': saldo_liquidita,
            'Prezzo Acq.': 1.0,
            'Prezzo Att.': 1.0,
            'Valore Totale': saldo_liquidita,
            'Costo Totale': saldo_liquidita,
            'P&L €': 0,
            'P&L %': 0,
            'Asset Class': 'Cash',
            'Peso %': 0
        }])
        
        df = pd.concat([df, liquidita_row], ignore_index=True)
        
        # Ricalcola pesi con liquidità inclusa
        totale_con_liquidita = metrics['valore_totale'] + saldo_liquidita
        df['Peso %'] = (df['Valore Totale'] / totale_con_liquidita * 100)
        
        # Mostra tabella
        display_df = df.copy()
        display_df['Prezzo Acq.'] = display_df['Prezzo Acq.'].apply(lambda x: f"€{x:.2f}")
        display_df['Prezzo Att.'] = display_df['Prezzo Att.'].apply(lambda x: f"€{x:.2f}")
        display_df['Valore Totale'] = display_df['Valore Totale'].apply(format_currency)
        display_df['Costo Totale'] = display_df['Costo Totale'].apply(format_currency)
        display_df['P&L €'] = display_df['P&L €'].apply(format_currency)
        display_df['P&L %'] = display_df['P&L %'].apply(format_percentage)
        display_df['Peso %'] = display_df['Peso %'].apply(format_percentage)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Grafici
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df,
                values='Valore Totale',
                names='Ticker',
                title="Composizione Portafoglio (inclusa liquidità)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df[df['Ticker'] != 'LIQUIDITA'].sort_values('P&L %', ascending=False),
                x='Ticker',
                y='P&L %',
                title="Rendimento per Asset",
                color='P&L %',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Solo liquidità disponibile
        if saldo_liquidita > 0:
            st.info(f"💰 Hai {format_currency(saldo_liquidita)} di liquidità disponibile ma nessun asset in portafoglio.")
        else:
            st.info("📭 Nessun asset in portafoglio e nessuna liquidità disponibile.")
    
    st.divider()

    st.info("ℹ️ **Nota:** Il portafoglio viene calcolato automaticamente dalle transazioni. Per aggiungere asset, vai alla sezione 'Storico' o 'Importa CSV'.")

    # Leva Finanziaria
    st.subheader("💰 Leva Finanziaria (Credit Lombard)")
    
    leverage = st.slider("Moltiplicatore Leva", min_value=1.0, max_value=2.0, value=1.0, step=0.1)

    if leverage > 1.0:
        # Helper per normalizzare percentuali
        def normalize_percentage(value_str, default="0.035"):
            """Converte valore a decimale gestendo sia decimali che percentuali"""
            try:
                val = float(value_str)
                
                # Se > 10, è chiaramente un errore (es. 210 invece di 2.1)
                if val > 10.0:
                    return float(default)
                
                # Se > 1, è espresso in % (es. 3.5)
                if val > 1.0:
                    return val / 100
                
                # Se 0-1, è già decimale (es. 0.035)
                return val
                
            except:
                return float(default)
        
        # Leggi e normalizza
        euribor_decimal = normalize_percentage(
            supabase.get_config(user_id, "euribor_3m", "0.035"),
            "0.035"
        )
        
        spread_decimal = normalize_percentage(
            supabase.get_config(user_id, "spread_credit_lombard", "0.02"),
            "0.02"
        )
        
        # Costo totale in decimale
        costo_leva_totale_decimal = euribor_decimal + spread_decimal
        costo_leva_totale_percent = costo_leva_totale_decimal * 100
        
        # Calcoli
        valore_base = metrics['valore_totale']
        valore_con_leva = valore_base * leverage
        importo_prestito = valore_con_leva - valore_base
        costo_leva_annuale = importo_prestito * costo_leva_totale_decimal
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Valore Base Portafoglio", format_currency(valore_base))
            st.metric("Importo Prestito", format_currency(importo_prestito))
        
        with col_b:
            st.metric("Valore con Leva", format_currency(valore_con_leva))
            st.metric(
                "Costo Annuale Leva", 
                format_currency(costo_leva_annuale), 
                f"{costo_leva_totale_percent:.2f}% p.a."
            )
        
        # Breakdown dettagliato
        with st.expander("🔍 Dettaglio Costo Leva"):
            st.write(f"**Composizione Tasso:**")
            st.write(f"• Euribor 3M: {euribor_decimal * 100:.2f}%")
            st.write(f"• Spread Lombard: {spread_decimal * 100:.2f}%")
            st.write(f"• **Totale:** {costo_leva_totale_percent:.2f}%")
            st.write("")
            st.write(f"**Calcolo Costo Annuale:**")
            st.write(f"• Prestito: €{importo_prestito:,.2f}")
            st.write(f"• Tasso annuo: {costo_leva_totale_percent:.2f}%")
            st.write(f"• Formula: €{importo_prestito:,.2f} × {costo_leva_totale_decimal:.4f}")
            st.write(f"• **Costo:** €{costo_leva_annuale:,.2f}")
            
            # Costo mensile
            costo_mensile = costo_leva_annuale / 12
            st.write("")
            st.write(f"💡 **Costo mensile:** €{costo_mensile:,.2f}")
            
            # Warning
            if euribor_decimal > 0.10 or spread_decimal > 0.10:
                st.warning("⚠️ I tassi sembrano molto alti. Vai in Configurazione per verificare i parametri.")
                
                if st.button("🔧 Apri Configurazione"):
                    st.switch_page("pages/configurazione.py")



# ==================== PAGINA STORICO TRANSAZIONI ====================

def page_storico_transazioni():
    """Pagina Storico Transazioni"""
    st.title("📋 Storico Transazioni")
    
    # Filtri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Range Date",
            value=(datetime(2024, 1, 1), datetime.now()),
            key="date_range"
        )
    
    with col2:
        ticker_filter = st.text_input("Filtra per Ticker (vuoto = tutti)", "", key="ticker_filter")
    
    with col3:
        tipo_filter = st.selectbox("Filtra per Tipo", 
                               ["Tutti", "Buy", "Sell", "Dividend", "Deposit", "Withdrawal", "Tax"], 
                               key="tipo_filter")

    
    # Recupera transazioni
    transazioni_df = supabase.get_transazioni(
        user_id=user_id,
        ticker=ticker_filter.upper() if ticker_filter else None,
        data_inizio=date_range[0].strftime('%Y-%m-%d'),
        data_fine=date_range[1].strftime('%Y-%m-%d')
    )
    
    if not transazioni_df.empty:
        if tipo_filter != "Tutti":
            transazioni_df = transazioni_df[transazioni_df['tipo'] == tipo_filter]
    
    # Visualizza tabella
    st.subheader("Transazioni Storiche")
    
    if not transazioni_df.empty:
        display_cols = ['data', 'ticker', 'tipo', 'quantita', 'prezzo_unitario', 'importo', 'commissioni', 'note']
        display_df = transazioni_df[display_cols].copy()
        display_df['prezzo_unitario'] = display_df['prezzo_unitario'].apply(lambda x: f"€{x:.2f}")
        display_df['importo'] = display_df['importo'].apply(format_currency)
        display_df['commissioni'] = display_df['commissioni'].apply(format_currency)
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Nessuna transazione trovata")
    
    st.divider()
    
    # Statistiche
    st.subheader("Statistiche Transazioni")
    
    stats = supabase.get_statistiche_transazioni(user_id)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Totale Transazioni", stats.get('totale_transazioni', 0))
    with col2:
        st.metric("Acquisti", stats.get('buy_count', 0))
    with col3:
        st.metric("Vendite", stats.get('sell_count', 0))
    with col4:
        st.metric("Valore Totale", format_currency(stats.get('valore_totale', 0)))
    with col5:
        st.metric("Commissioni", format_currency(stats.get('commissioni_totale', 0)))
    
    st.divider()
    
    # Aggiungi Transazione
    st.subheader("➕ Aggiungi Nuova Transazione")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trans_data = st.date_input("Data", key="trans_data")
    with col2:
        trans_ticker = st.text_input("Ticker", key="trans_ticker")
    with col3:
        trans_tipo = st.selectbox("Tipo", ["Buy", "Sell", "Dividend"], key="trans_tipo")
    with col4:
        trans_quantita = st.number_input("Quantità", min_value=0.0, key="trans_quantita")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trans_prezzo = st.number_input("Prezzo Unitario (€)", min_value=0.0, key="trans_prezzo")
    with col2:
        trans_commissioni = st.number_input("Commissioni (€)", min_value=0.0, key="trans_comm")
    with col3:
        trans_note = st.text_input("Note", key="trans_note")
    
    if st.button("✅ Aggiungi Transazione"):
        if trans_ticker and trans_quantita > 0 and trans_prezzo > 0:
            result = supabase.add_transazione(
                user_id=user_id,
                data=trans_data.strftime('%Y-%m-%d'),
                ticker=trans_ticker.upper(),
                tipo=trans_tipo,
                quantita=trans_quantita,
                prezzo_unitario=trans_prezzo,
                commissioni=trans_commissioni,
                note=trans_note
            )
            
            if result['success']:
                st.success("✓ Transazione aggiunta!")
                st.rerun()
            else:
                st.error(f"Errore: {result['error']}")
        else:
            st.warning("Inserisci valori validi")

    st.divider()
    
    # Gestione Liquidità
    st.subheader("💰 Gestione Liquidità")
    
    tab1, tab2 = st.tabs(["💵 Deposito", "💸 Prelievo"])
    
    with tab1:
        st.write("**Aggiungi Deposito di Liquidità**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            deposit_data = st.date_input("Data Deposito", key="deposit_data")
        with col2:
            deposit_importo = st.number_input("Importo (€)", min_value=0.0, key="deposit_importo")
        with col3:
            deposit_note = st.text_input("Note", "Deposito bonifico", key="deposit_note")
        
        if st.button("✅ Aggiungi Deposito"):
            if deposit_importo > 0:
                result = supabase.add_transazione(
                    user_id=user_id,
                    data=deposit_data.strftime('%Y-%m-%d'),
                    ticker='LIQUIDITA',
                    tipo='Deposit',
                    quantita=deposit_importo,
                    prezzo_unitario=1.0,
                    commissioni=0,
                    note=deposit_note
                )
                
                if result['success']:
                    st.success(f"✓ Deposito di {format_currency(deposit_importo)} aggiunto!")
                    st.rerun()
                else:
                    st.error(f"Errore: {result['error']}")
            else:
                st.warning("Inserisci un importo valido")
    
    with tab2:
        st.write("**Aggiungi Prelievo di Liquidità**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            withdrawal_data = st.date_input("Data Prelievo", key="withdrawal_data")
        with col2:
            withdrawal_importo = st.number_input("Importo (€)", min_value=0.0, key="withdrawal_importo")
        with col3:
            withdrawal_note = st.text_input("Note", "Prelievo", key="withdrawal_note")
        
        if st.button("✅ Aggiungi Prelievo"):
            if withdrawal_importo > 0:
                result = supabase.add_transazione(
                    user_id=user_id,
                    data=withdrawal_data.strftime('%Y-%m-%d'),
                    ticker='LIQUIDITA',
                    tipo='Withdrawal',
                    quantita=-withdrawal_importo,
                    prezzo_unitario=1.0,
                    commissioni=0,
                    note=withdrawal_note
                )
                
                if result['success']:
                    st.success(f"✓ Prelievo di {format_currency(withdrawal_importo)} aggiunto!")
                    st.rerun()
                else:
                    st.error(f"Errore: {result['error']}")
            else:
                st.warning("Inserisci un importo valido")



# ==================== PAGINA ANALISI PORTAFOGLIO ====================

def page_analisi_portafoglio():
    """Pagina Analisi di Portafoglio Avanzata"""
    st.title("🔍 Analisi Portafoglio Avanzata")
    
    metrics, df = get_portfolio_metrics()
    
    if metrics is None:
        st.info("📭 Nessun dato portafoglio da analizzare")
        return
    
    # IMPORTANTE: Escludi LIQUIDITA dalle analisi (non ha dati storici)
    df_analysis = df[df['Ticker'] != 'LIQUIDITA'].copy()
    
    if df_analysis.empty:
        st.info("📭 Nessun asset da analizzare (solo liquidità disponibile)")
        return
    
    # Metriche Principali
    st.subheader("📊 Metriche Portafoglio")
    
    risk_free_rate = float(supabase.get_config(user_id, "tasso_risk_free", "0.025"))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calcola volatilità solo per asset reali (non liquidità)
        volatilities = []
        for ticker in df_analysis['Ticker']:
            try:
                vol = calculate_volatility(ticker)
                # Converti a float e verifica che sia valido
                if vol is not None and not np.isnan(float(vol)) and float(vol) > 0:
                    volatilities.append(float(vol))
            except Exception as e:
                print(f"Errore calcolo volatilità per {ticker}: {e}")
                continue
        
        if volatilities:
            avg_vol = np.mean(volatilities)
            st.metric("Volatilità Annualizzata", f"{avg_vol*100:.2f}%")
        else:
            st.metric("Volatilità Annualizzata", "N/A")
    
    with col2:
        try:
            if volatilities:
                sharpe = calculate_sharpe_ratio(df_analysis, risk_free_rate)
                if sharpe is not None and not np.isnan(float(sharpe)):
                    st.metric("Sharpe Ratio", f"{float(sharpe):.3f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            else:
                st.metric("Sharpe Ratio", "N/A")
        except Exception as e:
            print(f"Errore calcolo Sharpe Ratio: {e}")
            st.metric("Sharpe Ratio", "N/A")
    
    with col3:
        max_dds = []
        for ticker in df_analysis['Ticker']:
            try:
                dd = calculate_max_drawdown(ticker)
                if dd is not None and not np.isnan(float(dd)):
                    max_dds.append(float(dd))
            except Exception as e:
                print(f"Errore calcolo drawdown per {ticker}: {e}")
                continue
        
        if max_dds:
            avg_dd = np.mean(max_dds)
            st.metric("Max Drawdown", f"{avg_dd*100:.2f}%")
        else:
            st.metric("Max Drawdown", "N/A")
    
    with col4:
        ulcers = []
        for ticker in df_analysis['Ticker']:
            try:
                ulc = calculate_ulcer_index(ticker)
                if ulc is not None and not np.isnan(float(ulc)):
                    ulcers.append(float(ulc))
            except Exception as e:
                print(f"Errore calcolo Ulcer Index per {ticker}: {e}")
                continue
        
        if ulcers:
            avg_ulcer = np.mean(ulcers)
            st.metric("Ulcer Index", f"{avg_ulcer:.2f}%")
        else:
            st.metric("Ulcer Index", "N/A")


    
    st.divider()
    
    # Strategie di Allocazione
    st.subheader("📈 Strategie di Allocazione Ottimale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Parity vs Allocazione Attuale**")
        risk_parity = calculate_risk_parity_weights(df_analysis)
        
        if not risk_parity.empty:
            comparison_df = df_analysis[['Ticker', 'Peso %']].merge(risk_parity, on='Ticker', how='left')
            comparison_df['Risk Parity %'] = comparison_df['Risk Parity %'].fillna(0)
            
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Dati insufficienti per calcolare Risk Parity")
    
    with col2:
        st.write("**Relaxed Risk Parity (5% - 40%)**")
        relaxed_rp = calculate_relaxed_risk_parity(df_analysis, min_weight=0.05, max_weight=0.40)
        
        if not relaxed_rp.empty:
            st.dataframe(relaxed_rp, use_container_width=True)
        else:
            st.info("Dati insufficienti per calcolare Relaxed Risk Parity")
    
    st.divider()

    # Simulazione Bootstrap
    st.subheader("🎲 Simulazione Bootstrap")
    st.info("📊 Simulazione basata su ricampionamento dei rendimenti storici reali (non parametrica)")
    
    # Controlli simulazione
    col1, col2 = st.columns(2)
    
    with col1:
        n_years = st.slider(
            "Orizzonte temporale (anni)",
            min_value=1,
            max_value=50,
            value=30,
            step=1
        )
    
    with col2:
        n_simulations = st.select_slider(
            "Numero di iterazioni",
            options=[100, 500, 1000, 2500, 5000, 10000],
            value=5000
        )
    
    # Versamenti periodici
    st.subheader("💰 Piano di Accumulo Capitale (PAC)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_contribution = st.number_input(
            "Versamento mensile (€)",
            min_value=0,
            max_value=100000,
            value=0,
            step=100
        )
    
    with col2:
        annual_contribution = st.number_input(
            "Versamento annuale (€)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000
        )
    
    with col3:
        years_contributions = (monthly_contribution * 12 + annual_contribution) * n_years
        st.metric(
            "💵 Capitale versato totale",
            format_currency(years_contributions)
        )
    
    days = n_years * 252
    
    st.write(f"**Configurazione:** {n_simulations:,} simulazioni su **{n_years} anni** ({days} giorni)")
    
    if st.button("🚀 Esegui Simulazione Bootstrap", type="primary"):
        with st.spinner(f"Esecuzione {n_simulations:,} simulazioni Bootstrap..."):
            try:
                # Esegui Bootstrap
                simulations, total_contributed = bootstrap_simulation(
                    df_analysis, 
                    n_simulations=n_simulations, 
                    days=days,
                    monthly_contribution=monthly_contribution,
                    annual_contribution=annual_contribution
                )
                
                if simulations.size > 0:
                    # Calcola percentili 5°, 75°, 95°
                    percentili = np.percentile(simulations, [5, 75, 95], axis=0)
                    
                    # Asse X in anni
                    x_axis = np.arange(days) / 252
                    
                    # Grafico
                    fig = go.Figure()
                    
                    # 5° Percentile (pessimistico)
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=percentili[0],
                        name='5° Percentile',
                        mode='lines',
                        line=dict(color='red', width=2),
                        hovertemplate='Anno: %{x:.1f}<br>Valore: €%{y:,.0f}<extra></extra>'
                    ))
                    
                    # 75° Percentile
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=percentili[1],
                        name='75° Percentile',
                        mode='lines',
                        line=dict(color='blue', width=3),
                        hovertemplate='Anno: %{x:.1f}<br>Valore: €%{y:,.0f}<extra></extra>'
                    ))
                    
                    # 95° Percentile (ottimistico)
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=percentili[2],
                        name='95° Percentile',
                        mode='lines',
                        line=dict(color='green', width=2),
                        hovertemplate='Anno: %{x:.1f}<br>Valore: €%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Area tra 5° e 95°
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=percentili[2],
                        fill=None, mode='lines',
                        line=dict(width=0),
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=percentili[0],
                        fill='tonexty', mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(0, 100, 255, 0.1)',
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    pac_info = ""
                    if monthly_contribution > 0 or annual_contribution > 0:
                        pac_info = f"<br><sub>Con PAC: {monthly_contribution}€/mese + {annual_contribution}€/anno</sub>"
                    
                    fig.update_layout(
                        title=f"Simulazione Bootstrap - {n_years} anni{pac_info}",
                        xaxis_title="Anni",
                        yaxis_title="Valore Portafoglio (€)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiche
                    final_values = simulations[:, -1]
                    initial_value = metrics['valore_totale']
                    
                    st.subheader(f"📊 Risultati dopo {n_years} anni")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("💰 Capitale Iniziale", format_currency(initial_value))
                    
                    with col2:
                        st.metric("💵 Versato (PAC)", format_currency(total_contributed))
                    
                    with col3:
                        p5_final = np.percentile(final_values, 5)
                        st.metric("📉 5° Percentile", format_currency(p5_final))
                    
                    with col4:
                        p75_final = np.percentile(final_values, 75)
                        st.metric("📊 75° Percentile", format_currency(p75_final))
                    
                    with col5:
                        p95_final = np.percentile(final_values, 95)
                        st.metric("📈 95° Percentile", format_currency(p95_final))
                    
                    st.divider()
                    
                    # Rendimenti e CAGR
                    st.subheader("📈 Analisi Rendimenti")
                    
                    total_invested = initial_value + total_contributed
                    
                    col1, col2, col3 = st.columns(3)
                    
                    def calculate_cagr_on_initial(final_val, initial_val, contributions, years):
                        if initial_val > 0:
                            growth = final_val - contributions
                            return ((growth / initial_val) ** (1 / years) - 1) * 100
                        return 0
                    
                    with col1:
                        cagr_5 = calculate_cagr_on_initial(p5_final, initial_value, total_contributed, n_years)
                        st.metric("CAGR 5° (Pessimistico)", f"{cagr_5:.2f}% /anno")
                    
                    with col2:
                        cagr_75 = calculate_cagr_on_initial(p75_final, initial_value, total_contributed, n_years)
                        st.metric("🎯 CAGR 75° (Base)", f"{cagr_75:.2f}% /anno")
                    
                    with col3:
                        cagr_95 = calculate_cagr_on_initial(p95_final, initial_value, total_contributed, n_years)
                        st.metric("CAGR 95° (Ottimistico)", f"{cagr_95:.2f}% /anno")
                    
                    # Salva CAGR per FIRE
                    st.session_state['analysis_cagr_5'] = cagr_5
                    st.session_state['analysis_cagr_75'] = cagr_75
                    st.session_state['analysis_cagr_95'] = cagr_95
                    st.session_state['analysis_years'] = n_years
                    st.session_state['analysis_iterations'] = n_simulations
                    
                else:
                    st.error("Errore: simulazione non ha prodotto risultati")
            
            except Exception as e:
                st.error(f"Errore durante la simulazione: {e}")
                import traceback
                st.code(traceback.format_exc())
    

# ==================== PAGINA SIMULAZIONE FIRE ====================

def page_simulazione_fire():
    """Pagina Simulazione FIRE con validazione"""
    st.title("🔥 Simulazione FIRE (Financial Independence, Retire Early)")
    
    metrics, df = get_portfolio_metrics()
    
    if metrics is None:
        st.info("📭 Nessun dato portafoglio per simulazione FIRE")
        return
    
    # Helper per validare valori
    def safe_float(config_value, default, min_val=None, max_val=None):
        """Leggi e valida un valore float dalla config"""
        try:
            val = float(config_value)
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except:
            return float(default)
    
    def safe_int(config_value, default, min_val=None, max_val=None):
        """Leggi e valida un valore int dalla config"""
        try:
            val = int(config_value)
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except:
            return int(default)
    
    st.info("💡 **FIRE Calculator**: Valori di default dalla Configurazione")
    
    # LEGGI e VALIDA parametri dalla configurazione
    inflazione_decimal = safe_float(supabase.get_config(user_id, "inflazione", "0.02"), "0.02", 0.0, 0.2)
    versamento_mensile_default = safe_float(supabase.get_config(user_id, "versamento_mensile", "500"), "500", 0, 50000)
    
    # Parametri FIRE
    tasso_prelievo_default = safe_float(supabase.get_config(user_id, "tasso_prelievo_fire", "4.0"), "4.0", 1.0, 10.0)
    spese_annue_default = safe_float(supabase.get_config(user_id, "spese_annue_fire", "30000"), "30000", 0, 500000)
    eta_attuale_default = safe_int(supabase.get_config(user_id, "eta_attuale", "30"), "30", 18, 80)
    cagr_accumulo_default = safe_float(supabase.get_config(user_id, "cagr_accumulo", "4.0"), "4.0", 0.0, 20.0)
    cagr_prelievi_default = safe_float(supabase.get_config(user_id, "cagr_prelievi", "3.0"), "3.0", 0.0, 20.0)
    
    # Parametri FIRE
    st.subheader("💰 Parametri Simulazione")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monthly_contribution = st.number_input(
            "Versamento mensile (€)",
            min_value=0,
            max_value=50000,
            value=500,
            step=100,
            help="Aggiungi capitale ogni mese"
        )
    
    with col2:
        annual_contribution_fire = st.number_input(
            "Versamento annuale (€)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000,
            help="Aggiungi capitale una volta all’anno (es. bonus, 13a)"
        )
    
    with col3:
        annual_expenses = st.number_input(
            "Spese annue desiderate (€)",
            min_value=0,
            max_value=500000,
            value=30000,
            step=1000,
            help="Questa voce serve a valutare sostenibilità in FIRE. È il fabbisogno annuo da coprire con prelievi"
        )
    
    with col4:
        withdrawal_rate = st.number_input(
            "Tasso Prelievo Annuo (%)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Percentuale massima di patrimonio da prelevare ogni anno per essere sostenibili"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cagr_accumulation = st.number_input(
            "CAGR Accumulo (%)",
            min_value=0.0,
            max_value=20.0,
            value=cagr_accumulo_default,
            step=0.1,
            help="Tasso di crescita annuo durante accumulo"
        )
    
    with col2:
        cagr_withdrawal = st.number_input(
            "CAGR Prelievi (%)",
            min_value=0.0,
            max_value=20.0,
            value=cagr_prelievi_default,
            step=0.1,
            help="Tasso di crescita annuo durante prelievi"
        )
    
    with col3:
        withdrawal_rate = st.number_input(
            "Tasso Prelievo Annuo (%)",
            min_value=1.0,
            max_value=10.0,
            value=tasso_prelievo_default,  # Ora validato
            step=0.5,
            help="Regola del 4%"
        )
    
    with col4:
        inflation_rate = st.number_input(
            "Inflazione annua (%)",
            min_value=0.0,
            max_value=10.0,
            value=inflazione_decimal * 100,  # Converti da decimale a %
            step=0.1,
            help="Tasso inflazione per indicizzare le spese"
        )
    
    # Input età
    fire_age = st.number_input(
        "Età attuale",
        min_value=18,
        max_value=80,
        value=eta_attuale_default,
        help="Inserisci la tua età attuale"
    )
    
    fire_number = annual_expenses / (withdrawal_rate / 100)
    
    st.divider()
    st.subheader("🎯 Risultati Simulazione FIRE")
    
    if st.button("🚀 Simula FIRE", type="primary"):
        # Inizializza years_to_fire a None
        years_to_fire = None
        portfolio_value = current_value
        monthly_rate = cagr_accumulation / 100 / 12
        
        accumulation_data = []
        max_years = 50
        fire_reached = False
        
        # Fase Accumulo
        for year in range(max_years):
            
            portfolio_value += annual_contribution_fire
            
            for month in range(12):
                portfolio_value = portfolio_value * (1 + monthly_rate) + monthly_contribution
            
            accumulation_data.append({'year': year, 'value': portfolio_value})
            
            if not fire_reached and portfolio_value >= fire_number:
                years_to_fire = year + 1
                fire_reached = True
        
        # Se non raggiunto, imposta a max_years
        if years_to_fire is None:
            years_to_fire = max_years
        
        # Fase Prelievi
        withdrawal_data = []
        fire_portfolio = portfolio_value
        annual_withdrawal = annual_expenses
        withdrawal_years = 40
        
        for year in range(withdrawal_years):
            fire_portfolio -= annual_withdrawal
            fire_portfolio *= (1 + cagr_withdrawal / 100)
            fire_portfolio = max(fire_portfolio, 0)
            
            withdrawal_data.append({
                'year': years_to_fire + year,
                'value': fire_portfolio
            })
            
            if fire_portfolio <= 0:
                break
        
        # Metriche Risultati
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 FIRE Number", format_currency(fire_number))
        
        with col2:
            if fire_reached:
                st.metric("⏱️ Anni per FIRE", f"{years_to_fire} anni")
            else:
                st.metric("⏱️ Anni per FIRE", f"{years_to_fire} anni (non raggiunto)")
        
        with col3:
            # Calcola età al FIRE in modo sicuro
            if fire_reached:
                eta_fire_value = fire_age + years_to_fire
                st.metric("🎂 Età al FIRE", f"{eta_fire_value} anni")
            else:
                st.metric("🎂 Età al FIRE", f"{fire_age + years_to_fire}+ anni")
        
        with col4:
            total_contributed = (monthly_contribution * 12 + annual_contribution_fire)* years_to_fire
            st.metric("💰 Totale Versato", format_currency(total_contributed))
        
        # Warning se non raggiunto
        if not fire_reached:
            st.warning(f"⚠️ FIRE Number non raggiunto in {max_years} anni. Considera di aumentare i versamenti o ridurre le spese desiderate.")
        
        st.divider()
        
        # Grafico
        st.subheader("📈 Evoluzione Portafoglio: Accumulo → FIRE → Prelievi")
        
        accumulation_years = [d['year'] for d in accumulation_data]
        accumulation_values = [d['value'] for d in accumulation_data]
        withdrawal_years_list = [d['year'] for d in withdrawal_data]
        withdrawal_values = [d['value'] for d in withdrawal_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=accumulation_years,
            y=accumulation_values,
            name='Fase Accumulo',
            mode='lines',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)',
            hovertemplate='Anno: %{x}<br>Valore: €%{y:,.0f}<extra></extra>'
        ))
        
        if withdrawal_data:
            fig.add_trace(go.Scatter(
                x=withdrawal_years_list,
                y=withdrawal_values,
                name='Fase Prelievi',
                mode='lines',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)',
                hovertemplate='Anno: %{x}<br>Valore: €%{y:,.0f}<extra></extra>'
            ))
        
        fig.add_hline(
            y=fire_number,
            line_dash="dash",
            line_color="red",
            annotation_text=f"FIRE Number: €{fire_number:,.0f}",
            annotation_position="right"
        )
        
        if fire_reached:
            fig.add_trace(go.Scatter(
                x=[years_to_fire],
                y=[fire_portfolio],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='FIRE Raggiunto!',
                hovertemplate=f'Anno {years_to_fire}<br>FIRE! €{fire_portfolio:,.0f}<extra></extra>'
            ))
        
        title_suffix = "raggiunta" if fire_reached else "non raggiunta"
        fig.update_layout(
            title=f"Simulazione FIRE - Indipendenza {title_suffix} in {years_to_fire} anni (CAGR: {cagr_accumulation}%)",
            xaxis_title="Anni da Oggi",
            yaxis_title="Valore Portafoglio (€)",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Riepilogo
        st.divider()
        st.subheader("📊 Riepilogo Simulazione")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fase Accumulo:**")
            st.write(f"• Portafoglio iniziale: {format_currency(current_value)}")
            st.write(f"• Versamenti mensili: {format_currency(monthly_contribution)}")
            st.write(f"• CAGR: {cagr_accumulation:.2f}%")
            st.write(f"• Durata: {years_to_fire} anni")
            st.write(f"• Portafoglio finale: {format_currency(portfolio_value)}")
        
        with col2:
            st.write("**Fase Prelievi (FIRE):**")
            st.write(f"• Portafoglio iniziale: {format_currency(portfolio_value)}")
            st.write(f"• Prelievo annuo: {format_currency(annual_expenses)}")
            st.write(f"• CAGR: {cagr_withdrawal:.2f}%")
            st.write(f"• Tasso prelievo: {withdrawal_rate}%")
            if len(withdrawal_data) > 29:
                st.write(f"• Portafoglio dopo 30 anni: {format_currency(withdrawal_data[29]['value'])}")


# ==================== PAGINA IMPORTA TRANSAZIONI CSV============

def page_import_csv():
    """Pagina per importare transazioni da CSV"""
    st.title("📤 Importa Transazioni da CSV")
    
    if "user" not in st.session_state:
        st.warning("⚠️ Devi essere loggato per importare.")
        return
    
    user_id = st.session_state.user.id
    st.info(f"👤 Importerai le transazioni per: {st.session_state.user.email}")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Carica il CSV", type="csv")
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head(10))
        
        st.divider()
        
        if st.button("📥 Importa su Supabase", type="primary"):
            temp_path = "/tmp/transazioni_import.csv"
            uploaded_file.seek(0)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Importazione..."):
                result = process_csv(temp_path, user_id, supabase.client)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("✓ Importate", result['importate'])
            col2.metric("✗ Errori", result['errori'])
            col3.metric("⚠️ Non mappate", result['non_mappate'])
            
            if result['importate'] > 0:
                st.success(f"✅ {result['importate']} transazioni importate!")


# ==================== PAGINA CONFIGURAZIONE ====================

def page_configurazione():
    """Pagina Configurazione con gestione coerente percentuali"""
    st.title("⚙️ Configurazione")
    
    # Helper per validare valori
    def safe_float(config_value, default, min_val=None, max_val=None):
        """Leggi e valida un valore float dalla config"""
        try:
            val = float(config_value)
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except:
            return float(default)
    
    def safe_int(config_value, default, min_val=None, max_val=None):
        """Leggi e valida un valore int dalla config"""
        try:
            val = int(config_value)
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except:
            return int(default)
    
    # Parametri generali
    st.subheader("📊 Parametri Generali")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Leggi in formato decimale, mostra come %
        tasso_risk_free_decimal = safe_float(
            supabase.get_config(user_id, "tasso_risk_free", "0.025"),
            "0.025", 0.0, 0.2
        )
        tasso_risk_free = st.number_input(
            "Tasso Risk-Free (%)",
            min_value=0.0,
            max_value=20.0,
            value=tasso_risk_free_decimal * 100,  # Converti a %
            step=0.1,
            help="Tasso di interesse senza rischio (es. 2.5%)"
        )
    
    with col2:
        inflazione_decimal = safe_float(
            supabase.get_config(user_id, "inflazione", "0.02"),
            "0.02", 0.0, 0.2
        )
        inflazione = st.number_input(
            "Inflazione (%)",
            min_value=0.0,
            max_value=20.0,
            value=inflazione_decimal * 100,  # Converti a %
            step=0.1,
            help="Tasso inflazione annuale (es. 2.0%)"
        )
    
    with col3:
        orizzonte_val = safe_int(
            supabase.get_config(user_id, "orizzonte_temporale", "30"),
            "30", 1, 50
        )
        orizzonte_temporale = st.number_input(
            "Orizzonte temporale (anni)",
            min_value=1,
            max_value=50,
            value=orizzonte_val,
            step=1
        )
    
    # PAC
    st.subheader("💰 Piano di Accumulo Capitale (PAC)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        versamento_val = safe_float(
            supabase.get_config(user_id, "versamento_mensile", "500"),
            "500", 0, 50000
        )
        versamento_mensile = st.number_input(
            "Versamento mensile (€)",
            min_value=0,
            max_value=50000,
            value=int(versamento_val),
            step=100
        )
    
    with col2:
        versamento_annuo_val = safe_float(
            supabase.get_config(user_id, "versamento_annuale", "0"),
            "0", 0, 1000000
        )
        versamento_annuale = st.number_input(
            "Versamento annuale (€)",
            min_value=0,
            max_value=1000000,
            value=int(versamento_annuo_val),
            step=1000
        )
    
    # FIRE
    st.subheader("🔥 Parametri FIRE")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        eta_val = safe_int(
            supabase.get_config(user_id, "eta_attuale", "30"),
            "30", 18, 80
        )
        eta_attuale = st.number_input(
            "Età attuale",
            min_value=18,
            max_value=80,
            value=eta_val
        )
    
    with col2:
        spese_val = safe_float(
            supabase.get_config(user_id, "spese_annue_fire", "30000"),
            "30000", 0, 500000
        )
        spese_annue_fire = st.number_input(
            "Spese annue desiderate (€)",
            min_value=0,
            max_value=500000,
            value=int(spese_val),
            step=1000
        )
    
    with col3:
        # IMPORTANTE: Salva come %, leggi come %
        tasso_prelievo_val = safe_float(
            supabase.get_config(user_id, "tasso_prelievo_fire", "4.0"),
            "4.0", 1.0, 10.0
        )
        tasso_prelievo_fire = st.number_input(
            "Tasso prelievo FIRE (%)",
            min_value=1.0,
            max_value=10.0,
            value=tasso_prelievo_val,  # GIÀ in %
            step=0.5,
            help="Regola del 4% standard"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # IMPORTANTE: Salva come %, leggi come %
        cagr_acc_val = safe_float(
            supabase.get_config(user_id, "cagr_accumulo", "4.0"),
            "4.0", 0.0, 20.0
        )
        cagr_accumulo = st.number_input(
            "CAGR Accumulo (%)",
            min_value=0.0,
            max_value=20.0,
            value=cagr_acc_val,  # GIÀ in %
            step=0.1,
            help="Rendimento atteso in fase accumulo (es. 4.0%)"
        )
    
    with col2:
        cagr_prel_val = safe_float(
            supabase.get_config(user_id, "cagr_prelievi", "3.0"),
            "3.0", 0.0, 20.0
        )
        cagr_prelievi = st.number_input(
            "CAGR Prelievi (%)",
            min_value=0.0,
            max_value=20.0,
            value=cagr_prel_val,  # GIÀ in %
            step=0.1,
            help="Rendimento atteso in fase prelievi (es. 3.0%)"
        )
    
    # NUOVO: Parametri Leva Finanziaria
    st.subheader("📈 Parametri Leva Finanziaria")

    col1, col2 = st.columns(2)

    with col1:
        # Leggi euribor
        euribor_saved = supabase.get_config(user_id, "euribor_3m", "0.035")
        euribor_val = safe_float(euribor_saved, "0.035", 0.0, 10.0)  # Max 10.0 per sicurezza
        
        # Auto-detect: se > 1, è stato salvato come % invece che decimale
        if euribor_val > 1.0:
            euribor_val = euribor_val / 100
        
        # IMPORTANTE: Limita il valore a range ragionevole (0-10%)
        euribor_val = max(0.0, min(0.10, euribor_val))  # Max 10% = 0.10
        
        euribor_3m = st.number_input(
            "Euribor 3M (%)",
            min_value=0.0,
            max_value=10.0,
            value=euribor_val * 100,  # Converti a % per display
            step=0.1,
            help="Tasso Euribor 3 mesi (es. 3.5%)"
        )

    with col2:
        # Leggi spread
        spread_saved = supabase.get_config(user_id, "spread_credit_lombard", "0.02")
        spread_val = safe_float(spread_saved, "0.02", 0.0, 10.0)
        
        # Auto-detect
        if spread_val > 1.0:
            spread_val = spread_val / 100
        
        # IMPORTANTE: Limita il valore
        spread_val = max(0.0, min(0.10, spread_val))  # Max 10% = 0.10
        
        spread_lombard = st.number_input(
            "Spread Credito Lombard (%)",
            min_value=0.0,
            max_value=10.0,
            value=spread_val * 100,  # Converti a % per display
            step=0.1,
            help="Spread sul credito lombard (es. 2.0%)"
        )

    # Info tasso totale
    tasso_totale_leva = euribor_3m + spread_lombard
    st.info(f"💰 **Costo totale leva:** {tasso_totale_leva:.2f}% p.a. (Euribor {euribor_3m:.2f}% + Spread {spread_lombard:.2f}%)")

    
    st.divider()
    
    # Bottone Salva
    if st.button("💾 Salva Configurazione", type="primary"):
        # Salva parametri generali
        supabase.set_config(user_id, "tasso_risk_free", str(tasso_risk_free / 100))
        supabase.set_config(user_id, "inflazione", str(inflazione / 100))
        supabase.set_config(user_id, "orizzonte_temporale", str(orizzonte_temporale))
        
        # Salva PAC
        supabase.set_config(user_id, "versamento_mensile", str(versamento_mensile))
        supabase.set_config(user_id, "versamento_annuale", str(versamento_annuale))
        
        # Salva FIRE
        supabase.set_config(user_id, "eta_attuale", str(eta_attuale))
        supabase.set_config(user_id, "spese_annue_fire", str(spese_annue_fire))
        supabase.set_config(user_id, "tasso_prelievo_fire", str(tasso_prelievo_fire))
        supabase.set_config(user_id, "cagr_accumulo", str(cagr_accumulo))
        supabase.set_config(user_id, "cagr_prelievi", str(cagr_prelievi))
        
        # NUOVO: Salva parametri leva come DECIMALI
        supabase.set_config(user_id, "euribor_3m", str(euribor_3m / 100))
        supabase.set_config(user_id, "spread_credit_lombard", str(spread_lombard / 100))
        
        st.success("✅ Configurazione salvata!")
        
        # Mostra riepilogo
        with st.expander("📋 Riepilogo valori salvati"):
            st.write("**Parametri Generali:**")
            st.write(f"• Tasso Risk-Free: {tasso_risk_free}% (salvato: {tasso_risk_free/100:.4f})")
            st.write(f"• Inflazione: {inflazione}% (salvato: {inflazione/100:.4f})")
            st.write(f"• Orizzonte: {orizzonte_temporale} anni")
            
            st.write("")
            st.write("**Parametri PAC:**")
            st.write(f"• Versamento mensile: €{versamento_mensile:,.0f}")
            st.write(f"• Versamento annuale: €{versamento_annuale:,.0f}")
            
            st.write("")
            st.write("**Parametri FIRE:**")
            st.write(f"• Età attuale: {eta_attuale} anni")
            st.write(f"• Spese annue: €{spese_annue_fire:,.0f}")
            st.write(f"• Tasso prelievo: {tasso_prelievo_fire}%")
            st.write(f"• CAGR accumulo: {cagr_accumulo}%")
            st.write(f"• CAGR prelievi: {cagr_prelievi}%")
            
            st.write("")
            st.write("**Parametri Leva:**")
            st.write(f"• Euribor 3M: {euribor_3m}% (salvato: {euribor_3m/100:.4f})")
            st.write(f"• Spread Lombard: {spread_lombard}% (salvato: {spread_lombard/100:.4f})")
            st.write(f"• Costo totale leva: {tasso_totale_leva:.2f}%")

    st.success("✓ Tutte le configurazioni vengono salvate automaticamente su Supabase!")

# ==================== MAIN ====================

def main():
    """Funzione principale dell'applicazione"""
    
    st.sidebar.title("📊 Portfolio Monitor")
    
    # Bottone Logout
    render_logout_button()
    
    st.sidebar.divider()
    
    # Navigazione Pagine
    pages = {
        "📊 Monitoraggio": page_monitoraggio,
        "📋 Storico": page_storico_transazioni,
        "🔍 Analisi": page_analisi_portafoglio,
        "🔥 FIRE": page_simulazione_fire,
        "📤 Importa CSV": page_import_csv,
        "⚙️ Configurazione": page_configurazione,
    }
    
    selected_page = st.sidebar.radio("Navigazione", list(pages.keys()))
    
    st.sidebar.divider()
    st.sidebar.info(
        "**Portfolio Monitor v3.0**\n\n"
        "Streamlit + Supabase\n\n"
        f"👤 Utente: {st.session_state.user.email}"
    )
    
    # Esegui pagina selezionata
    pages[selected_page]()

if __name__ == "__main__":
    main()
