Loto — sugestões estatísticas + ML para a Lotaria Nacional de Angola (5/90).

Interfaces:
  streamlit run lotaria_ml_streamlit.py
  python lotaria_ml_tkinter.py

Backend:
  BACKEND=csv       → resultados CSV + sugestões em sugestoes_bloqueadas.json
  BACKEND=supabase  → tabelas resultados + sugestoes_bloqueadas (ver supabase/schema.sql)

Secrets (.streamlit/secrets.toml):
  BACKEND = "supabase"
  SUPABASE_URL = "..."
  SUPABASE_KEY = "..."
  APP_PASSWORD = "..."   # opcional, protege escrita

Supabase:
  Executar uma vez o ficheiro supabase/schema.sql no SQL Editor.
  Não são necessárias tabelas extra para Estatísticas, Algoritmos ou Assistente
  (calculam em memória a partir de resultados + sugestoes_bloqueadas).

Páginas principais:
  Dashboard, Análise, Grelha, Estatísticas, Algoritmos, Histórico,
  Histórico Sugestões, Verificar números, Modelo ML, Assistente, Relatório Semanal
