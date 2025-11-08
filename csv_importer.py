"""
CSV Importer - Importa transazioni dal CSV a Supabase
Converte ticker da formato BIT: a Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from supabase import create_client, Client
from datetime import datetime
import streamlit as st

# Mapping ticker BIT: → Yahoo Finance
TICKER_MAPPING = {
    "BIT:CSSPX": "CSSPX.MI",      # iShares Core S&P 500 UCITS ETF USD Acc
    "BIT:CMOD": "CMOD.MI",        # iShares Core MSCI World UCITS ETF
    "BIT:ITPS": "ITPS.MI",        # iShares MSCI Japan UCITS ETF
    "BIT:SMEA": "SMEA.MI",        # Saifirst EMP Asia Equity Fund
    "BIT:TPRO": "TPRO.MI",        # Thematic ETF Portfolio
    "BIT:CSEMAS": "CSEMAS.MI",    # iShares MSCI EM Asia UCITS ETF
    "BIT:SGLD-ETFP": "SGLD.MI",   # iShares Physical Gold ETC
    "BIT:CSBGE7": "CSBGE7.MI",    # iShares Bond UCITS ETF
    "BIT:AT1": "AT1.MI",          # Additional Tier 1 Bond ETF
    "BIT:SWDA": "SWDA.MI",        # iShares Core MSCI World UCITS ETF USD Acc
}

def clean_csv_value(value):
    """Pulisce valori dal CSV (virgole decimali italiane)"""
    if pd.isna(value) or value == "":
        return None
    
    if isinstance(value, str):
        # Sostituisci virgola con punto
        value = value.replace(",", ".")
        try:
            return float(value)
        except:
            return None
    
    return float(value)

def parse_date(date_str):
    """Converte data italiana in ISO format"""
    if pd.isna(date_str) or date_str == "":
        return None
    
    try:
        return pd.to_datetime(date_str, format="%d/%m/%Y").strftime("%Y-%m-%d")
    except:
        return None

def get_ticker_mapping(ticker_bit):
    """Ritorna il ticker Yahoo Finance dal ticker BIT"""
    return TICKER_MAPPING.get(ticker_bit, None)

def process_csv(csv_file_path, user_id: str, supabase_client: Client):
    """Processa il CSV e importa le transazioni su Supabase"""
    
    # Leggi il CSV
    df = pd.read_csv(csv_file_path, sep=",")
    
    # Rinomina colonne (rimuovi spazi e caratteri speciali)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Mappa colonne
    column_map = {
        'inserisci la data dell\'operazione': 'data',
        'inserisci l\'operazione': 'tipo',
        'inserisci il ticket dello strumento': 'ticker',
        'inserisci la quantità': 'quantita',
        'inserisci il pmc': 'prezzo_unitario',
        'inserisci le commissioni': 'commissioni',
        'totale': 'totale'
    }
    
    df = df.rename(columns=column_map)
    
    # Filtra solo Buy e Sell
    df = df[df['tipo'].isin(['Buy', 'Sell'])]
    
    # Pulisci dati
    df['data'] = df['data'].apply(parse_date)
    df['quantita'] = df['quantita'].apply(clean_csv_value)
    df['prezzo_unitario'] = df['prezzo_unitario'].apply(clean_csv_value)
    df['commissioni'] = df['commissioni'].apply(clean_csv_value)
    df['commissioni'] = df['commissioni'].fillna(0)
    
    # Filtra righe con ticker validi
    df = df[df['ticker'].notna()]
    
    # Converti ticker
    df['ticker_yahoo'] = df['ticker'].apply(get_ticker_mapping)
    
    # Rimuovi righe senza ticker mapping
    df_valide = df[df['ticker_yahoo'].notna()].copy()
    df_non_mappate = df[df['ticker_yahoo'].isna()].copy()
    
    transazioni_importate = 0
    errori = []
    
    # Importa transazioni valide
    for idx, row in df_valide.iterrows():
        try:
            if pd.isna(row['data']) or pd.isna(row['quantita']) or pd.isna(row['prezzo_unitario']):
                continue
            
            importo = row['quantita'] * row['prezzo_unitario']
            
            response = supabase_client.table("transazioni").insert({
                "user_id": user_id,
                "data": row['data'],
                "ticker": row['ticker_yahoo'],
                "tipo": row['tipo'],
                "quantita": float(row['quantita']),
                "prezzo_unitario": float(row['prezzo_unitario']),
                "importo": float(importo),
                "commissioni": float(row['commissioni']) if row['commissioni'] else 0,
                "note": f"Importato da CSV - {row['ticker']}",
                "created_at": datetime.now().isoformat()
            }).execute()
            
            transazioni_importate += 1
            print(f"✓ {row['data']} | {row['tipo']} | {row['ticker_yahoo']} | {row['quantita']} unità")
            
        except Exception as e:
            errori.append(f"Errore riga {idx}: {str(e)}")
            print(f"✗ Errore riga {idx}: {e}")
    
    # Report
    print("\n" + "="*60)
    print("REPORT IMPORTAZIONE CSV")
    print("="*60)
    print(f"✓ Transazioni importate: {transazioni_importate}")
    print(f"✗ Errori: {len(errori)}")
    
    if df_non_mappate.shape[0] > 0:
        print(f"\n⚠️ Ticker non mappati ({df_non_mappate.shape[0]}):")
        for ticker_non_mappato in df_non_mappate['ticker'].unique():
            print(f"  - {ticker_non_mappato}")
    
    if errori:
        print("\nErrori:")
        for errore in errori[:5]:
            print(f"  - {errore}")
    
    return {
        'importate': transazioni_importate,
        'errori': len(errori),
        'non_mappate': df_non_mappate.shape[0]
    }

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="CSV Importer", page_icon="📤", layout="wide")
    st.title("📤 Importa Transazioni da CSV")
    
    # Carica credenziali Supabase
    try:
        supabase_url = st.secrets["supabase"]["supabase_url"]
        supabase_key = st.secrets["supabase"]["supabase_key"]
        supabase_client = create_client(supabase_url, supabase_key)
    except:
        st.error("❌ Supabase non configurato. Configura i secrets.")
        return
    
    # Carica user ID dal session state (se loggato)
    if "user" not in st.session_state:
        st.warning("⚠️ Devi essere loggato per importare transazioni.")
        return
    
    user_id = st.session_state.user.id
    st.info(f"👤 Importerai le transazioni per: {st.session_state.user.email}")
    
    st.divider()
    
    # Upload CSV
    st.subheader("1️⃣ Seleziona il file CSV")
    uploaded_file = st.file_uploader("Carica il tuo CSV di transazioni", type="csv")
    
    if uploaded_file is not None:
        # Preview CSV
        st.subheader("2️⃣ Anteprima CSV")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head(10), use_container_width=True)
        
        st.divider()
        
        # Mapping ticker
        st.subheader("3️⃣ Mapping Ticker (BIT: → Yahoo Finance)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Ticker mappati automaticamente:**")
            for bit_ticker, yahoo_ticker in TICKER_MAPPING.items():
                st.write(f"• {bit_ticker} → **{yahoo_ticker}**")
        
        with col2:
            st.write("**Aggiungi nuovi mapping:**")
            new_bit = st.text_input("Ticker BIT:", key="new_bit")
            new_yahoo = st.text_input("Ticker Yahoo Finance:", key="new_yahoo")
            
            if st.button("Aggiungi mapping"):
                if new_bit and new_yahoo:
                    TICKER_MAPPING[new_bit] = new_yahoo
                    st.success(f"✓ Aggiunto: {new_bit} → {new_yahoo}")
        
        st.divider()
        
        # Importa
        st.subheader("4️⃣ Importa su Supabase")
        
        if st.button("📥 Importa Transazioni", type="primary"):
            # Salva file temporaneo
            temp_path = "/tmp/transazioni_import.csv"
            uploaded_file.seek(0)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Importazione in corso..."):
                result = process_csv(temp_path, user_id, supabase_client)
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("✓ Importate", result['importate'])
            col2.metric("✗ Errori", result['errori'])
            col3.metric("⚠️ Non mappate", result['non_mappate'])
            
            if result['importate'] > 0:
                st.success(f"✅ {result['importate']} transazioni importate con successo!")
                st.info("Vai alla pagina 'Storico' per vedere le transazioni importate.")

if __name__ == "__main__":
    main()
