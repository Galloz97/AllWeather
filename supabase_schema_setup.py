"""
Supabase SQL Schema Setup
Script per creare le tabelle nel database Supabase
Esegui questo codice una volta per inizializzare il database
"""

# ==================== SQL DA ESEGUIRE SU SUPABASE ====================

SCHEMA_SQL = """

-- Tabella Transazioni (multi-tenant per user_id)
CREATE TABLE IF NOT EXISTS transazioni (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    ticker TEXT NOT NULL,
    tipo TEXT NOT NULL,  -- 'Buy', 'Sell', 'Dividend'
    quantita REAL NOT NULL,
    prezzo_unitario REAL NOT NULL,
    importo REAL NOT NULL,
    commissioni REAL DEFAULT 0,
    note TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, id)
);

-- Index per performance
CREATE INDEX idx_transazioni_user_id ON transazioni(user_id);
CREATE INDEX idx_transazioni_ticker ON transazioni(ticker);
CREATE INDEX idx_transazioni_data ON transazioni(data);

-- Tabella Portafoglio (multi-tenant per user_id)
CREATE TABLE IF NOT EXISTS portafoglio (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    quantita REAL NOT NULL,
    prezzo_acquisto REAL NOT NULL,
    asset_class TEXT NOT NULL,
    data_aggiunta TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, ticker)
);

-- Index per performance
CREATE INDEX idx_portafoglio_user_id ON portafoglio(user_id);
CREATE INDEX idx_portafoglio_ticker ON portafoglio(ticker);

-- Tabella Configurazione (multi-tenant per user_id)
CREATE TABLE IF NOT EXISTS configurazione (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    chiave TEXT NOT NULL,
    valore TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, chiave)
);

-- Index per performance
CREATE INDEX idx_configurazione_user_id ON configurazione(user_id);

-- ==================== ROW LEVEL SECURITY ====================

-- Abilita RLS su tutte le tabelle
ALTER TABLE transazioni ENABLE ROW LEVEL SECURITY;
ALTER TABLE portafoglio ENABLE ROW LEVEL SECURITY;
ALTER TABLE configurazione ENABLE ROW LEVEL SECURITY;

-- Policy per transazioni: ogni utente vede solo i propri dati
CREATE POLICY "Users can view their own transazioni"
  ON transazioni FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own transazioni"
  ON transazioni FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own transazioni"
  ON transazioni FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own transazioni"
  ON transazioni FOR DELETE
  USING (auth.uid() = user_id);

-- Policy per portafoglio: ogni utente vede solo i propri dati
CREATE POLICY "Users can view their own portafoglio"
  ON portafoglio FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own portafoglio"
  ON portafoglio FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own portafoglio"
  ON portafoglio FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own portafoglio"
  ON portafoglio FOR DELETE
  USING (auth.uid() = user_id);

-- Policy per configurazione: ogni utente vede solo i propri dati
CREATE POLICY "Users can view their own configurazione"
  ON configurazione FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own configurazione"
  ON configurazione FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own configurazione"
  ON configurazione FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own configurazione"
  ON configurazione FOR DELETE
  USING (auth.uid() = user_id);

"""

ISTRUZIONI = """
╔════════════════════════════════════════════════════════════════╗
║        SETUP DATABASE SUPABASE PER PORTFOLIO MONITOR           ║
╚════════════════════════════════════════════════════════════════╝

PASSO 1: Accedi a Supabase Dashboard
├─ Vai su https://app.supabase.com
├─ Seleziona il tuo progetto
└─ Clicca su "SQL Editor"

PASSO 2: Esegui lo schema SQL
├─ Copia tutto il codice SQL qui sotto
├─ Vai alla scheda "SQL Editor" in Supabase
├─ Clicca "New Query"
├─ Incolla il codice SQL
└─ Clicca "Run" (eseguirai tutte le tabelle)

PASSO 3: Verifica tabelle create
├─ Vai su "Table Editor"
├─ Dovresti vedere 3 tabelle:
│  ├─ transazioni
│  ├─ portafoglio
│  └─ configurazione
└─ Tutte con colonna "user_id" per multi-tenant

PASSO 4: Ottieni credenziali Supabase
├─ Vai su "Settings" → "API"
├─ Copia "Project URL" (URL)
├─ Copia "anon public" key
└─ Salva in .streamlit/secrets.toml

PASSO 5: Configura Streamlit Secrets
├─ Crea file .streamlit/secrets.toml
├─ Aggiungi:
│  [supabase]
│  supabase_url = "https://your-project.supabase.co"
│  supabase_key = "eyJhbGc..."
└─ Non fare commit su GitHub!

PASSO 6: Installa dipendenze Python
└─ pip install supabase python-gotrue

PASSO 7: Esegui app
└─ streamlit run app_supabase.py

════════════════════════════════════════════════════════════════

TABELLE SCHEMA:

transazioni (multi-tenant):
┌─ id (BIGSERIAL PRIMARY KEY)
├─ user_id (UUID, foreign key auth.users)
├─ data (TEXT: YYYY-MM-DD)
├─ ticker (TEXT)
├─ tipo (TEXT: Buy, Sell, Dividend)
├─ quantita (REAL)
├─ prezzo_unitario (REAL)
├─ importo (REAL)
├─ commissioni (REAL)
├─ note (TEXT)
├─ created_at (TIMESTAMP)
└─ updated_at (TIMESTAMP)

portafoglio (multi-tenant):
┌─ id (BIGSERIAL PRIMARY KEY)
├─ user_id (UUID, foreign key auth.users)
├─ ticker (TEXT, UNIQUE per user)
├─ quantita (REAL)
├─ prezzo_acquisto (REAL)
├─ asset_class (TEXT)
├─ data_aggiunta (TIMESTAMP)
└─ updated_at (TIMESTAMP)

configurazione (multi-tenant):
┌─ id (BIGSERIAL PRIMARY KEY)
├─ user_id (UUID, foreign key auth.users)
├─ chiave (TEXT, UNIQUE per user)
├─ valore (TEXT)
└─ updated_at (TIMESTAMP)

════════════════════════════════════════════════════════════════

ROW LEVEL SECURITY (RLS):

✓ Automaticamente abilitato
✓ Ogni utente vede SOLO i suoi dati
✓ Policies configurate per SELECT, INSERT, UPDATE, DELETE
✓ auth.uid() = user_id è il controllo di sicurezza

════════════════════════════════════════════════════════════════

INDICI CREATI (Performance):

transazioni:
├─ idx_transazioni_user_id (ricerca veloce per utente)
├─ idx_transazioni_ticker (ricerca veloce per ticker)
└─ idx_transazioni_data (ricerca veloce per data)

portafoglio:
├─ idx_portafoglio_user_id (ricerca veloce per utente)
└─ idx_portafoglio_ticker (ricerca veloce per ticker)

configurazione:
└─ idx_configurazione_user_id (ricerca veloce per utente)

════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(ISTRUZIONI)
    print("\n\nCODICE SQL DA ESEGUIRE:\n")
    print(SCHEMA_SQL)
