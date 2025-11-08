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
    "BIT:CMOD": "CMOD.MI",        # Invesco Bloomberg Commodity UCITS ETF
    "BIT:ITPS": "ITPS.MI",        # iShares $ TIPS UCITS ETF USD Acc
    "BIT:SMEA": "SMEA.MI",        # iShares Core MSCI Europe UCITS ETF
    "BIT:SEMA": "SEMA.MI",        # iShares MSCI EM UCITS ETF USD Acc
    "BIT:TPRO": "TPRO.MI",        # Technoprobe Spa
    "BIT:CSEMAS": "CSEMAS.MI",    # iShares MSCI EM Asia UCITS ETF
    "BIT:SGLD-ETFP": "SGLD.MI",   # iShares Physical Gold ETC
    "BIT:VGEA": "VGEA.MI",        # Vanguard EUR Eurozone Government Bond UCITS ETF EUR Acc
    "BIT:X710": "X710.MI",        # Xtrackers II Eurozone Government Bond 7-10 UCITS ETF
    "BIT:CSBGE7": "CSBGE7.MI",    # iShares € Govt Bond 3-7yr ETF EUR Acc
    "BIT:CSBGU3": "CSBGU3.MI",    # iShares $ Treasury Bd 1-3y ETF USD Acc
    "BIT:AT1": "AT1.MI",          # Invesco AT1 Capital Bond ETF
    "BIT:CONV": "CONV.MI",          # SPDR FTSE Global Convertible Bond UCITS ETF
    "BIT:SWDA": "SWDA.MI",        # iShares Core MSCI World UCITS ETF USD Acc
    "BIT:XDEB": "XDEB.MI",        # Xtrackers MSCI World Minimum Volatility UCITS ETF
    "BIT:XDEM": "XDEM.MI",        # Xtrackers MSCI World Momentum UCITS ETF
    "BIT:INFL-ETFP": "INFL.MI",   # Amundi Euro Inflation Expectations 2-10Y UCITS ETF Acc
    "BIT:IBCI-ETFP": "IBCI.MI",   # iShares € Inflation Linked Govt Bond UCITS ETF EUR Acc
    "BIT:LEONIA": "LEONIA.MI",    # Amundi EUR Overnight Return UCITS ETF Acc
    "BIT:XEON": "XEON.MI",        # Xtrackers II EUR Overnight Rate Swap UCITS ETF
    "SWX:BITC": "BITC.SW",        # CoinShares Physical Bitcoin
    "BTCEUR": "BTC-EUR",          # Bitcoin/Euro
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

def process_csv(csv_file_path, user_id: str, supabase_client):
    """Processa il CSV e importa le transazioni su Supabase"""
    df = pd.read_csv(csv_file_path, sep=",")
    df.columns = [col.strip().lower() for col in df.columns]
    column_map = {
        "inserisci la data dell'operazione": "data",
        "inserisci l'operazione": "tipo",
        "inserisci il ticket dello strumento": "ticker",
        "inserisci la quantità": "quantita",
        "inserisci il pmc": "prezzo_unitario",
        "inserisci le commissioni": "commissioni",
        "totale": "totale"
    }
    df = df.rename(columns=column_map)

    transazioni_importate = 0
    errori = []
    non_mappate = 0

    def clean(val):
        if pd.isna(val) or val == "" or val is None:
            return None
        if isinstance(val, (float, int)):
            return float(val)
        
        val = str(val).replace("€", "").strip()
        
        # Gestione formato italiano: 1.234,56 → 1234.56
        # Gestione formato inglese: 1,234.56 → 1234.56
        
        # Se c'è sia punto che virgola, dipende dall'ordine
        if "," in val and "." in val:
            # Se la virgola è dopo il punto → formato italiano (1.234,56)
            if val.rindex(",") > val.rindex("."):
                val = val.replace(".", "")  # Rimuovi separatore migliaia
                val = val.replace(",", ".")  # Virgola diventa punto decimale
            # Altrimenti formato inglese (1,234.56)
            else:
                val = val.replace(",", "")  # Rimuovi separatore migliaia
        # Solo virgola → assume formato italiano (1234,56)
        elif "," in val:
            val = val.replace(",", ".")
        # Solo punto → già formato corretto (1234.56)
        
        try:
            return float(val)
        except Exception:
            return None


    def parse_date(x):
        try:
            return pd.to_datetime(str(x), format="%d/%m/%Y").strftime("%Y-%m-%d")
        except:
            return None

    for idx, row in df.iterrows():
        tipo = str(row.get("tipo", "")).strip()
        data = parse_date(row.get("data", ""))
        ticker = str(row.get("ticker", "")).strip()
        quantita = clean(row.get("quantita", None))
        pmc = clean(row.get("prezzo_unitario", None))
        commissioni = clean(row.get("commissioni", None)) or 0
        totale = clean(row.get("totale", None))

        # LIQUIDITÀ
        if tipo.lower() in ["bonifico", "deposito"]:
            ticker_finale = "LIQUIDITA"
            tipo_finale = "Deposit"
            quantita_finale = abs(totale) if totale else 0
            prezzo_finale = 1.0
            importo_finale = quantita_finale
            note = "Deposito Liquidità"
            
        elif tipo.lower() == "prelievo":
            ticker_finale = "LIQUIDITA"
            tipo_finale = "Withdrawal"
            quantita_finale = -abs(totale) if totale else 0
            prezzo_finale = 1.0
            importo_finale = quantita_finale
            note = "Prelievo Liquidità"
            
        elif tipo.lower() == "imposta":
            ticker_finale = "LIQUIDITA"
            tipo_finale = "Tax"
            quantita_finale = -abs(totale) if totale else 0
            prezzo_finale = 1.0
            importo_finale = quantita_finale
            note = "Imposta/Tassa"
            
        elif tipo in ["Buy", "Sell"]:
            if isinstance(ticker, str) and ticker.startswith("BIT:"):
                ticker_finale = ticker.replace("BIT:", "") + ".MI"
            elif ticker and ticker.upper() not in ["EURO", "NAN", ""]:
                ticker_finale = ticker
            else:
                non_mappate += 1
                continue
                
            tipo_finale = tipo
            if quantita is None or pmc is None:
                non_mappate += 1
                continue
                
            quantita_finale = quantita if tipo == "Buy" else -abs(quantita)
            prezzo_finale = pmc
            importo_finale = quantita * pmc
            note = f"Importato da CSV: {ticker}"
        else:
            non_mappate += 1
            continue

        # Skip se data non valida
        if data is None:
            errori.append(f"Riga {idx}: Data non valida")
            continue

        try:
            response = supabase_client.table("transazioni").insert({
                "user_id": user_id,
                "data": data,
                "ticker": ticker_finale,
                "tipo": tipo_finale,
                "quantita": float(quantita_finale),
                "prezzo_unitario": float(prezzo_finale),
                "importo": float(importo_finale),
                "commissioni": float(commissioni),
                "note": note,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            transazioni_importate += 1
            
        except Exception as e:
            errori.append(f"Riga {idx}: {str(e)}")
            if len(errori) <= 5:
                print(f"❌ Riga {idx}: {str(e)}")

    return {
        'importate': transazioni_importate,
        'errori': len(errori),
        'non_mappate': non_mappate
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
