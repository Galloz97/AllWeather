"""
Supabase Integration - Modulo per gestire connessione a Supabase
Autenticazione utente e database multi-tenant
"""

import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict

class SupabaseManager:
    """Gestisce connessione e operazioni Supabase"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Inizializza client Supabase
        
        Args:
            supabase_url: URL progetto Supabase
            supabase_key: Chiave API Supabase (anon key)
        """
        # Prova a leggere dai secrets
        try:
            if supabase_url is None:
                supabase_url = st.secrets["supabase"]["supabase_url"]
            if supabase_key is None:
                supabase_key = st.secrets["supabase"]["supabase_key"]
        except (KeyError, FileNotFoundError):
            st.error("❌ Errore: Secrets non configurati correttamente!")
            st.info("""
            Per risolvere:
            1. Vai a https://share.streamlit.io
            2. Seleziona la tua app
            3. Clicca Settings → Secrets
            4. Incolla:
            ```
            [supabase]
            supabase_url = "https://YOUR_PROJECT.supabase.co"
            supabase_key = "eyJhbGc..."
            ```
            """)
            raise ValueError("Supabase URL e KEY non configurate in secrets")
        
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("✓ Connesso a Supabase")
        except Exception as e:
            st.error(f"❌ Errore connessione Supabase: {e}")
            raise
    
    def signup(self, email: str, password: str) -> Dict:
        """Registra nuovo utente"""
        try:
            response = self.client.auth.sign_up({"email": email, "password": password})
            return {"success": True, "user": response.user}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def login(self, email: str, password: str) -> Dict:
        """Login utente"""
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            return {
                "success": True,
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def logout(self) -> Dict:
        """Logout utente"""
        try:
            self.client.auth.sign_out()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_user(self, session=None) -> Optional[Dict]:
        """Recupera utente corrente"""
        try:
            if session:
                user = self.client.auth.get_user(session.access_token)
            else:
                user = self.client.auth.get_user()
            return user
        except:
            return None
    
    # ==================== TRANSAZIONI ====================
    
    def add_transazione(self, user_id: str, data: str, ticker: str, tipo: str,
                       quantita: float, prezzo_unitario: float, 
                       commissioni: float = 0, note: str = "") -> Dict:
        """Aggiunge transazione associata all'utente"""
        try:
            importo = quantita * prezzo_unitario
            
            response = self.client.table("transazioni").insert({
                "user_id": user_id,
                "data": data,
                "ticker": ticker,
                "tipo": tipo,
                "quantita": quantita,
                "prezzo_unitario": prezzo_unitario,
                "importo": importo,
                "commissioni": commissioni,
                "note": note,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_transazioni(self, user_id: str, ticker: str = None,
                       data_inizio: str = None, data_fine: str = None) -> pd.DataFrame:
        """Recupera transazioni dell'utente"""
        try:
            query = self.client.table("transazioni").select("*").eq("user_id", user_id)
            
            if ticker:
                query = query.eq("ticker", ticker)
            
            if data_inizio:
                query = query.gte("data", data_inizio)
            
            if data_fine:
                query = query.lte("data", data_fine)
            
            response = query.order("data", desc=True).execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
        except Exception as e:
            print(f"✗ Errore recupero transazioni: {e}")
            return pd.DataFrame()
    
    def update_transazione(self, id: int, user_id: str, **kwargs) -> Dict:
        """Aggiorna transazione"""
        try:
            response = self.client.table("transazioni").update({
                **kwargs,
                "updated_at": datetime.now().isoformat()
            }).eq("id", id).eq("user_id", user_id).execute()
            
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_transazione(self, id: int, user_id: str) -> Dict:
        """Elimina transazione"""
        try:
            self.client.table("transazioni").delete().eq("id", id).eq("user_id", user_id).execute()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== PORTAFOGLIO ====================
    
    def add_portafoglio_asset(self, user_id: str, ticker: str, quantita: float,
                             prezzo_acquisto: float, asset_class: str) -> Dict:
        """Aggiunge asset al portafoglio"""
        try:
            response = self.client.table("portafoglio").upsert({
                "user_id": user_id,
                "ticker": ticker,
                "quantita": quantita,
                "prezzo_acquisto": prezzo_acquisto,
                "asset_class": asset_class,
                "updated_at": datetime.now().isoformat()
            }).execute()
            
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_portafoglio(self, user_id: str) -> pd.DataFrame:
        """Recupera portafoglio dell'utente"""
        try:
            response = self.client.table("portafoglio").select("*").eq("user_id", user_id).execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
        except Exception as e:
            print(f"✗ Errore recupero portafoglio: {e}")
            return pd.DataFrame()
    
    def delete_portafoglio_asset(self, user_id: str, ticker: str) -> Dict:
        """Elimina asset dal portafoglio"""
        try:
            self.client.table("portafoglio").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== CONFIGURAZIONE ====================
    
    def set_config(self, user_id: str, chiave: str, valore: str) -> Dict:
        """Salva configurazione per utente"""
        try:
            response = self.client.table("configurazione").upsert({
                "user_id": user_id,
                "chiave": chiave,
                "valore": valore,
                "updated_at": datetime.now().isoformat()
            }).execute()
            
            return {"success": True, "data": response.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_config(self, user_id: str, chiave: str, default: str = None) -> Optional[str]:
        """Recupera configurazione per utente"""
        try:
            response = self.client.table("configurazione").select("valore").eq("user_id", user_id).eq("chiave", chiave).execute()
            
            if response.data:
                return response.data[0]['valore']
            return default
        except Exception as e:
            print(f"✗ Errore recupero config: {e}")
            return default
    
    def get_all_config(self, user_id: str) -> Dict:
        """Recupera tutte le configurazioni per utente"""
        try:
            response = self.client.table("configurazione").select("chiave, valore").eq("user_id", user_id).execute()
            
            if response.data:
                return {row['chiave']: row['valore'] for row in response.data}
            return {}
        except Exception as e:
            print(f"✗ Errore recupero config: {e}")
            return {}
    
    # ==================== STATISTICHE ====================
    
    def get_statistiche_transazioni(self, user_id: str) -> Dict:
        """Calcola statistiche transazioni per utente"""
        try:
            response = self.client.table("transazioni").select("tipo, importo, commissioni").eq("user_id", user_id).execute()
            
            if not response.data:
                return {}
            
            df = pd.DataFrame(response.data)
            
            stats = {
                'totale_transazioni': len(df),
                'buy_count': len(df[df['tipo'] == 'Buy']),
                'sell_count': len(df[df['tipo'] == 'Sell']),
                'valore_totale': float(df['importo'].sum()) if 'importo' in df else 0,
                'commissioni_totale': float(df['commissioni'].sum()) if 'commissioni' in df else 0
            }
            
            return stats
        except Exception as e:
            print(f"✗ Errore statistiche: {e}")
            return {}

# ==================== AUTHENTICATION ====================

def init_supabase_auth():
    """Inizializza autenticazione Supabase in Streamlit"""
    
    if "supabase" not in st.session_state:
        st.session_state.supabase = SupabaseManager()
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user" not in st.session_state:
        st.session_state.user = None
    
    if "session" not in st.session_state:
        st.session_state.session = None

def render_login_page():
    """Render pagina login/signup"""
    
    st.set_page_config(page_title="Portfolio Monitor - Login", page_icon="📊")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🔐 Login")
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Accedi", key="login_btn"):
            if email and password:
                result = st.session_state.supabase.login(email, password)
                
                if result['success']:
                    st.session_state.authenticated = True
                    st.session_state.user = result['user']
                    st.session_state.session = result['session']
                    st.success("✓ Accesso effettuato!")
                    st.rerun()
                else:
                    st.error(f"✗ Errore: {result['error']}")
            else:
                st.warning("Inserisci email e password")
    
    with col2:
        st.header("📝 Registrazione")
        
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Conferma Password", type="password", key="confirm_pass")
        
        if st.button("Registrati", key="signup_btn"):
            if new_email and new_password and confirm_pass:
                if new_password != confirm_pass:
                    st.error("Le password non coincidono")
                else:
                    result = st.session_state.supabase.signup(new_email, new_password)
                    
                    if result['success']:
                        st.success("✓ Registrazione effettuata! Controlla email per la verifica.")
                    else:
                        st.error(f"✗ Errore: {result['error']}")
            else:
                st.warning("Inserisci tutti i campi")

def check_authentication():
    """Verifica autenticazione. Se non autenticato, mostra login"""
    
    init_supabase_auth()
    
    if not st.session_state.authenticated:
        render_login_page()
        st.stop()

def render_logout_button():
    """Render bottone logout in sidebar"""
    
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        if st.session_state.user:
            st.sidebar.text(f"👤 {st.session_state.user.email}")
    
    with col2:
        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            st.session_state.supabase.logout()
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session = None
            st.rerun()
