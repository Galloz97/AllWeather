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
def get_stock_data(ticker, period="1y"):
    """Recupera dati storici da yfinance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data if not data.empty else None
    except:
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

def calculate_sharpe_ratio(df, risk_free_rate):
    """Calcola Sharpe Ratio del portafoglio"""
    volatilities = []
    for _, row in df.iterrows():
        vol = calculate_volatility(row['Ticker'])
        volatilities.append(vol)
    
    if not volatilities or sum(volatilities) == 0:
        return 0.0
    
    portfolio_vol = np.mean(volatilities)
    avg_return = np.mean(df['P&L %']) / 100
    
    if portfolio_vol == 0:
        return 0.0
    
    return (avg_return - risk_free_rate) / portfolio_vol

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
    """Calcola pesi Risk Parity (inversamente proporzionali a volatilità)"""
    inv_vols = []
    tickers = []
    
    for _, row in df.iterrows():
        vol = calculate_volatility(row['Ticker'])
        if vol > 0:
            inv_vols.append(1 / vol)
            tickers.append(row['Ticker'])
    
    if not inv_vols:
        return pd.DataFrame()
    
    total_inv_vols = sum(inv_vols)
    weights = [iv / total_inv_vols * 100 for iv in inv_vols]
    
    return pd.DataFrame({'Ticker': tickers, 'Risk Parity %': weights})

def calculate_relaxed_risk_parity(df, min_weight=0.05, max_weight=0.40):
    """Calcola Relaxed Risk Parity con vincoli min/max"""
    rp = calculate_risk_parity_weights(df)
    
    if rp.empty:
        return pd.DataFrame()
    
    rp['Peso Clip %'] = rp['Risk Parity %'].clip(lower=min_weight*100, upper=max_weight*100)
    total = rp['Peso Clip %'].sum()
    rp['Relaxed Risk Parity %'] = (rp['Peso Clip %'] / total * 100)
    
    return rp[['Ticker', 'Relaxed Risk Parity %']]

def monte_carlo_simulation(df, n_simulations=1000, days=252):
    """Simulazione Monte Carlo del portafoglio per 12 mesi"""
    returns_list = []
    volatilities_list = []
    
    for _, row in df.iterrows():
        data = get_stock_data(row['Ticker'])
        if data is None or len(data) < 2:
            continue
        
        daily_returns = data['Close'].pct_change().dropna()
        returns_list.append(daily_returns.mean() * 252)
        volatilities_list.append(daily_returns.std() * np.sqrt(252))
    
    avg_return = np.mean(returns_list) if returns_list else 0.07
    avg_volatility = np.mean(volatilities_list) if volatilities_list else 0.15
    
    portfolio_value = df['Valore Totale'].sum()
    simulations = []
    
    for _ in range(n_simulations):
        daily_returns = np.random.normal(avg_return / 252, avg_volatility / np.sqrt(252), days)
        values = portfolio_value * np.cumprod(1 + daily_returns)
        simulations.append(values)
    
    return np.array(simulations)

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
        euribor = float(supabase.get_config(user_id, "euribor_3m", "0.035"))
        spread = float(supabase.get_config(user_id, "spread_credit_lombard", "0.02"))
        costo_leva_totale = euribor + spread
        
        valore_base = metrics['valore_totale']
        valore_con_leva = valore_base * leverage
        importo_prestito = valore_con_leva - valore_base
        costo_leva_annuale = importo_prestito * costo_leva_totale
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Valore Base Portafoglio", format_currency(valore_base))
            st.metric("Importo Prestito", format_currency(importo_prestito))
        
        with col_b:
            st.metric("Valore con Leva", format_currency(valore_con_leva))
            st.metric("Costo Annuale Leva", format_currency(costo_leva_annuale), f"{costo_leva_totale*100:.2f}% p.a.")

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
    
    # Simulazione Monte Carlo
    st.subheader("🎲 Simulazione Monte Carlo (1000 iterazioni, 12 mesi)")
    
    if st.button("Esegui Simulazione"):
        with st.spinner("Esecuzione simulazione..."):
            try:
                simulations = monte_carlo_simulation(df_analysis, n_simulations=1000, days=252)
                
                percentili = np.percentile(simulations, [10, 25, 50, 75, 90], axis=0)
                
                fig = go.Figure()
                
                for i, (p, label) in enumerate([(10, '10°'), (25, '25°'), (50, '50°'), (75, '75°'), (90, '90°')]):
                    fig.add_trace(go.Scatter(
                        y=percentili[i],
                        name=f'Percentile {label}',
                        mode='lines',
                        hovertemplate='Giorno: %{x}<br>Valore: €%{y:,.2f}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Simulazione Monte Carlo - Valore Portafoglio (12 mesi)",
                    xaxis_title="Giorni",
                    yaxis_title="Valore Portafoglio (€)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiche finali
                final_values = simulations[:, -1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Valore Iniziale", format_currency(metrics['valore_totale']))
                with col2:
                    st.metric("Mediana (50°)", format_currency(np.median(final_values)))
                with col3:
                    st.metric("10° Percentile", format_currency(np.percentile(final_values, 10)))
                with col4:
                    st.metric("90° Percentile", format_currency(np.percentile(final_values, 90)))
            
            except Exception as e:
                st.error(f"Errore durante la simulazione: {e}")
                st.info("Verifica che tutti i ticker abbiano dati storici disponibili su Yahoo Finance")


# ==================== PAGINA SIMULAZIONE FIRE ====================

def page_simulazione_fire():
    """Pagina Simulazione FIRE"""
    st.title("🔥 Simulazione FIRE (Financial Independence Retire Early)")
    
    # Parametri da configurazione
    tasso_prelievo = float(supabase.get_config(user_id, "tasso_prelievo_fire", "0.04"))
    inflazione = float(supabase.get_config(user_id, "tasso_inflazione", "0.02"))
    
    st.subheader("📋 Parametri di Simulazione")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        obiettivo_fire = st.number_input(
            "Obiettivo FIRE (€)",
            min_value=10000,
            value=500000,
            step=10000,
            key="fire_obiettivo"
        )
    
    with col2:
        spesa_annuale = st.number_input(
            "Spesa Annuale Attuale (€)",
            min_value=1000,
            value=30000,
            step=1000,
            key="fire_spesa"
        )
    
    with col3:
        tasso_prelievo_input = st.slider(
            "Tasso Prelievo Sostenibile (%)",
            min_value=2.0,
            max_value=5.0,
            value=tasso_prelievo * 100,
            step=0.1,
            key="fire_prelievo"
        ) / 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pac_mensile = st.number_input(
            "PAC Mensile (€)",
            min_value=0,
            value=500,
            step=50,
            key="fire_pac"
        )
    
    with col2:
        inflazione_input = st.slider(
            "Inflazione Attesa (%)",
            min_value=0.5,
            max_value=5.0,
            value=inflazione * 100,
            step=0.1,
            key="fire_inflazione"
        ) / 100
    
    with col3:
        anni_obiettivo = st.number_input(
            "Anni Obiettivo",
            min_value=1,
            value=20,
            step=1,
            key="fire_anni"
        )
    
    st.divider()
    
    # Calcoli FIRE
    st.subheader("📊 Analisi FIRE")
    
    obiettivo_finanziario = spesa_annuale / tasso_prelievo_input
    spesa_futura = spesa_annuale * (1 + inflazione_input) ** anni_obiettivo
    
    metrics, _ = get_portfolio_metrics()
    valore_attuale = metrics['valore_totale'] if metrics else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Obiettivo Finanziario",
            format_currency(obiettivo_finanziario),
            f"Spesa / {tasso_prelievo_input*100:.1f}%"
        )
    
    with col2:
        st.metric(
            f"Spesa fra {anni_obiettivo} anni",
            format_currency(spesa_futura),
            f"Inflazione: {inflazione_input*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Portafoglio Attuale",
            format_currency(valore_attuale)
        )
    
    st.divider()
    
    # Simulazione Scenari
    st.subheader("📈 Traiettoria verso FIRE")
    
    def simulate_fire_path(val_init, pac, anni, rendimento):
        """Simula crescita portafoglio verso FIRE"""
        valori = [val_init]
        for _ in range(anni):
            val_prev = valori[-1]
            val_new = val_prev * (1 + rendimento) + pac * 12
            valori.append(val_new)
        return valori
    
    # Scenari con diversi rendimenti
    rendimento_pessimistico = -0.05
    rendimento_base = 0.07
    rendimento_ottimistico = 0.10
    
    anni_list = list(range(0, anni_obiettivo + 1))
    
    valori_pessimistico = simulate_fire_path(valore_attuale, pac_mensile, anni_obiettivo, rendimento_pessimistico)
    valori_base = simulate_fire_path(valore_attuale, pac_mensile, anni_obiettivo, rendimento_base)
    valori_ottimistico = simulate_fire_path(valore_attuale, pac_mensile, anni_obiettivo, rendimento_ottimistico)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=anni_list,
        y=valori_pessimistico,
        name='Pessimistico (-5%)',
        line=dict(color='red', dash='dash'),
        hovertemplate='Anno: %{x}<br>Valore: €%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_list,
        y=valori_base,
        name='Base (7%)',
        line=dict(color='blue'),
        hovertemplate='Anno: %{x}<br>Valore: €%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_list,
        y=valori_ottimistico,
        name='Ottimistico (+10%)',
        line=dict(color='green', dash='dash'),
        hovertemplate='Anno: %{x}<br>Valore: €%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=obiettivo_finanziario,
        line_dash="dot",
        line_color="orange",
        annotation_text="Target FIRE",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Traiettoria verso FIRE",
        xaxis_title="Anni",
        yaxis_title="Valore Portafoglio (€)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Tabella Anno per Anno
    st.subheader("📋 Dettaglio Anno per Anno (Scenario Base)")
    
    tabella_fire = pd.DataFrame({
        'Anno': anni_list,
        'PAC Cumulativo': [pac_mensile * 12 * anno for anno in anni_list],
        'Valore Portafoglio': valori_base,
        'Gap dall\'Obiettivo': [max(0, obiettivo_finanziario - v) for v in valori_base]
    })
    
    # Formattazione
    display_fire = tabella_fire.copy()
    display_fire['PAC Cumulativo'] = display_fire['PAC Cumulativo'].apply(format_currency)
    display_fire['Valore Portafoglio'] = display_fire['Valore Portafoglio'].apply(format_currency)
    display_fire['Gap dall\'Obiettivo'] = display_fire['Gap dall\'Obiettivo'].apply(format_currency)
    
    st.dataframe(display_fire, use_container_width=True)

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
    """Pagina Configurazione Parametri Personali"""
    st.title("⚙️ Configurazione")
    
    st.subheader("🔧 Parametri Globali")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parametri Economici")
        
        tasso_risk_free = st.slider(
            "Tasso Risk-Free (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(supabase.get_config(user_id, "tasso_risk_free", "2.5")) / 100,
            step=0.1,
            key="config_risk_free"
        ) * 100
        
        if st.button("Salva Tasso Risk-Free"):
            supabase.set_config(user_id, "tasso_risk_free", str(tasso_risk_free))
            st.success("✓ Salvato")
        
        st.divider()
        
        tasso_inflazione = st.slider(
            "Inflazione Attesa (%)",
            min_value=0.5,
            max_value=5.0,
            value=float(supabase.get_config(user_id, "tasso_inflazione", "2.0")) / 100,
            step=0.1,
            key="config_inflation"
        ) * 100
        
        if st.button("Salva Inflazione"):
            supabase.set_config(user_id, "tasso_inflazione", str(tasso_inflazione))
            st.success("✓ Salvato")
        
        st.divider()
        
        tasso_prelievo = st.slider(
            "Tasso Prelievo FIRE (%)",
            min_value=2.0,
            max_value=5.0,
            value=float(supabase.get_config(user_id, "tasso_prelievo_fire", "4.0")) / 100,
            step=0.1,
            key="config_fire_rate"
        ) * 100
        
        if st.button("Salva Tasso Prelievo FIRE"):
            supabase.set_config(user_id, "tasso_prelievo_fire", str(tasso_prelievo))
            st.success("✓ Salvato")
    
    with col2:
        st.subheader("Credit Lombard (Leva)")
        
        euribor = st.slider(
            "Euribor 3M (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(supabase.get_config(user_id, "euribor_3m", "3.5")) / 100,
            step=0.1,
            key="config_euribor"
        ) * 100
        
        if st.button("Salva Euribor"):
            supabase.set_config(user_id, "euribor_3m", str(euribor))
            st.success("✓ Salvato")
        
        st.divider()
        
        spread = st.slider(
            "Spread Banca (%)",
            min_value=0.5,
            max_value=3.0,
            value=float(supabase.get_config(user_id, "spread_credit_lombard", "2.0")) / 100,
            step=0.1,
            key="config_spread"
        ) * 100
        
        if st.button("Salva Spread"):
            supabase.set_config(user_id, "spread_credit_lombard", str(spread))
            st.success("✓ Salvato")
        
        st.divider()
        
        costo_totale_leva = (euribor + spread)
        st.info(f"💰 **Costo Totale Leva: {costo_totale_leva:.2f}% annuo**\n\nEuribor 3M: {euribor:.2f}% + Spread: {spread:.2f}%")
    
    st.divider()
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
