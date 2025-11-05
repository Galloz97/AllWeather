# 🚀 GUIDA COMPLETA - INTEGRAZIONE SUPABASE

## 🎯 Cosa farai

Integrerai **Supabase** (database PostgreSQL gratuito nel cloud) con la tua app Streamlit per avere:
- ✅ **Login/Registrazione** per ogni utente
- ✅ **Database multi-tenant** - ogni utente ha i SUOI dati isolati
- ✅ **Sincronizzazione cloud** - dati sempre disponibili
- ✅ **Security** - Row Level Security integrato
- ✅ **Scalabilità** - Free tier per sempre per progetti personali

---

## STEP 1: Crea Progetto Supabase

### 1.1 Registrazione
1. Vai su https://supabase.com
2. Clicca "Start your project"
3. Scegli "Sign up with GitHub" o email
4. Verifica email

### 1.2 Crea Nuovo Progetto
1. Dashboard → "New project"
2. Nome: `portfolio-monitor` (o il nome che preferisci)
3. Database password: **salva in un posto sicuro**
4. Region: Scegli la più vicina (es. eu-west-1 per Europa)
5. Clicca "Create new project"

Attendi 2-3 minuti il provisioning del database.

---

## STEP 2: Configura Database Schema

### 2.1 Recupera Schema SQL

File fornito: `supabase_schema_setup.py`

```bash
python supabase_schema_setup.py
```

Questo stamperà le istruzioni e il codice SQL da eseguire.

### 2.2 Esegui Schema SQL su Supabase

1. Nel dashboard Supabase, vai a **"SQL Editor"**
2. Clicca **"New query"**
3. Copia tutto il codice SQL dal file `supabase_schema_setup.py` (nella sezione `SCHEMA_SQL`)
4. Incolla in Supabase
5. Clicca **"Run"**

Dovresti vedere:
- ✓ 3 tabelle create: `transazioni`, `portafoglio`, `configurazione`
- ✓ Row Level Security (RLS) abilitato
- ✓ Indici creati per performance

### 2.3 Verifica Tabelle

Vai a **"Table Editor"** nel dashboard Supabase.
Dovresti vedere 3 tabelle con colonna `user_id` per multi-tenant.

---

## STEP 3: Recupera Credenziali Supabase

### 3.1 Ottieni URL e KEY

1. Nel dashboard Supabase, vai a **"Settings" → "API"**
2. Cerca la sezione "Project API keys"
3. Copia:
   - **Project URL** (es: `https://your-project.supabase.co`)
   - **anon public** key (la chiave lunga)

### 3.2 Crea File Secrets Locale

Crea `.streamlit/secrets.toml`:

```toml
[supabase]
supabase_url = "https://your-project.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**IMPORTANTE**: Non fare commit su GitHub! Aggiungi a `.gitignore`:
```
.streamlit/secrets.toml
```

---

## STEP 4: Setup Locale (Sviluppo)

### 4.1 Installa Dipendenze

```bash
pip install -r requirements_supabase.txt
```

Questo installa:
- streamlit
- supabase (client Python)
- python-gotrue (autenticazione)
- yfinance, pandas, plotly

### 4.2 Testa Connessione Locale

```bash
streamlit run app_supabase.py
```

Dovresti vedere:
1. Pagina login/registrazione
2. Registrati con una email
3. Accedi
4. Dashboard portafoglio vuoto (primo accesso)
5. Aggiungi un asset
6. I dati vengono salvati su Supabase

---

## STEP 5: Deploy su Streamlit Cloud + GitHub

### 5.1 Carica su GitHub

```bash
git add .
git commit -m "Add Supabase integration"
git push origin main
```

### 5.2 Deploy su Streamlit Cloud

1. Vai a https://share.streamlit.io/deploy
2. Seleziona il repository GitHub
3. Branch: `main`
4. Main file: `app_supabase.py`
5. Clicca "Deploy"

### 5.3 Aggiungi Secrets in Streamlit Cloud

1. Nel dashboard Streamlit Cloud, seleziona la tua app
2. Clicca **"⋮" (menu)** → **"Settings"**
3. Vai a sezione **"Secrets"**
4. Copia il contenuto di `.streamlit/secrets.toml` (URL e KEY)
5. Incolla nella textarea dei secrets
6. Salva

L'app ricarica automaticamente con i secrets configurati.

---

## 📁 File Forniti

| File | Descrizione |
|------|-------------|
| `supabase_manager.py` | Classe SupabaseManager per tutte le operazioni |
| `supabase_schema_setup.py` | Schema SQL + istruzioni setup |
| `app_supabase.py` | App Streamlit integrata con Supabase |
| `requirements_supabase.txt` | Dipendenze Python |
| `.streamlit/secrets.toml.example` | Esempio file secrets |

---

## 🔐 Architettura Sicurezza

### Authentication Flow
```
1. Utente accede → Supabase Auth
2. Genera JWT token + session
3. Token memorizzato in session state Streamlit
4. Ogni query include user_id dal token
```

### Row Level Security (RLS)
```
transazioni:
- User A vede SOLO le sue transazioni (user_id = auth.uid())
- User B vede SOLO le sue transazioni (user_id = auth.uid())
- È impossibile per User A vedere i dati di User B
```

Questo è configurato automaticamente nel database con le Policies SQL.

---

## 🔑 Classe SupabaseManager

### Autenticazione
```python
supabase = SupabaseManager()

# Login
result = supabase.login("user@example.com", "password")
if result['success']:
    user = result['user']
    session = result['session']

# Registrazione
supabase.signup("user@example.com", "password")

# Logout
supabase.logout()
```

### Transazioni
```python
# Aggiungi
supabase.add_transazione(
    user_id=user.id,
    data="2024-11-05",
    ticker="AAPL",
    tipo="Buy",
    quantita=10,
    prezzo_unitario=150.00,
    commissioni=1.50,
    note="Acquisto"
)

# Recupera
df = supabase.get_transazioni(
    user_id=user.id,
    ticker="AAPL",
    data_inizio="2024-01-01",
    data_fine="2024-12-31"
)

# Aggiorna
supabase.update_transazione(
    id=1,
    user_id=user.id,
    quantita=15
)

# Elimina
supabase.delete_transazione(id=1, user_id=user.id)
```

### Portafoglio
```python
# Aggiungi asset
supabase.add_portafoglio_asset(
    user_id=user.id,
    ticker="VWRL.DE",
    quantita=50,
    prezzo_acquisto=85.30,
    asset_class="ETF"
)

# Recupera portafoglio
df = supabase.get_portafoglio(user_id=user.id)

# Elimina asset
supabase.delete_portafoglio_asset(user_id=user.id, ticker="VWRL.DE")
```

### Configurazione (per utente)
```python
# Salva parametro personale
supabase.set_config(user_id=user.id, chiave="euribor_3m", valore="0.035")

# Recupera
value = supabase.get_config(user_id=user.id, chiave="euribor_3m", default="0.03")

# Recupera tutte le config
all_config = supabase.get_all_config(user_id=user.id)
```

---

## 🛠️ Troubleshooting

### ❌ "Supabase URL e KEY non configurate"
**Soluzione:**
- Crea `.streamlit/secrets.toml`
- Verifica che supabase_url e supabase_key siano corretti
- Restart Streamlit: `Ctrl+C` e `streamlit run app_supabase.py`

### ❌ "relation 'transazioni' does not exist"
**Soluzione:**
- Schema SQL non eseguito
- Esegui il codice SQL da `supabase_schema_setup.py` nel SQL Editor Supabase

### ❌ "Login fallisce / Utente non creato"
**Soluzione:**
- Email non valida
- Password < 6 caratteri
- Controlla logs: vai a Supabase → "Auth" → "Users"

### ❌ "403 permission denied"
**Soluzione:**
- RLS policies non configurate correttamente
- Esegui di nuovo lo schema SQL completo
- Verifica che le policies siano create

### ❌ App lenta
**Soluzione:**
- Aumenta cache ttl: `@st.cache_data(ttl=7200)`
- Aggiungi `st.set_page_config(initial_sidebar_state="collapsed")`
- Riduci numero query database

---

## 📊 Multi-Utente - Come Funziona

### Scenario: 3 Utenti

```
User A (alice@example.com)
├─ Portafoglio
│  ├─ VWRL.DE: 50 azioni
│  └─ AAPL: 10 azioni
└─ Transazioni
   ├─ Buy VWRL.DE: 2024-01-15
   └─ Buy AAPL: 2024-02-01

User B (bob@example.com)
├─ Portafoglio
│  ├─ IUSN.DE: 30 azioni
│  └─ BND.L: 100 obbligazioni
└─ Transazioni
   ├─ Buy IUSN.DE: 2024-03-10
   └─ Buy BND.L: 2024-04-05

User C (charlie@example.com)
├─ Portafoglio
│  └─ Vuoto (primo accesso)
└─ Transazioni
   └─ Vuoto
```

Ogni utente vede SOLO i propri dati grazie a:
- JWT token con user_id
- RLS Policies nel database
- Row filtering automatico

---

## 🚀 Prossimi Passi

### Da Aggiungere (Optional)
- [ ] Two-factor authentication (2FA)
- [ ] Social login (Google, GitHub)
- [ ] Esportazione dati personali (GDPR)
- [ ] Backup automatici
- [ ] Rate limiting API
- [ ] Logging e audit trail

### Scaling
- Free tier Supabase: illimitato per progetti personali
- A 100k righe, considerare database potenziato
- Per multi-millione di righe, upgrade piano Pro

---

## 📞 Support

### Documentazione
- Supabase: https://supabase.com/docs
- Streamlit: https://docs.streamlit.io
- Python Client: https://github.com/supabase/supabase-py

### Forum
- Supabase Discussions: https://github.com/supabase/supabase/discussions
- Streamlit Forum: https://discuss.streamlit.io

---

## ✅ Checklist Finale

- [ ] Progetto Supabase creato
- [ ] Schema SQL eseguito (3 tabelle + RLS)
- [ ] Credenziali Supabase recuperate
- [ ] `.streamlit/secrets.toml` creato (locale)
- [ ] `requirements_supabase.txt` installato
- [ ] `streamlit run app_supabase.py` funziona
- [ ] Registrazione utente funziona
- [ ] Login/logout funziona
- [ ] Aggiunta asset funziona
- [ ] Dati salvati su Supabase
- [ ] GitHub push completato
- [ ] Streamlit Cloud deploy completato
- [ ] Secrets aggiunti in Streamlit Cloud
- [ ] App cloud funziona con login

Fatto! 🎉 Hai una app multi-utente completamente sicura su Supabase!
