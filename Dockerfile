# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Sistema SUP (Dr. Ajuda)
# Python 3.10-slim + pré-cache do modelo de embedding (~400MB)
# 1 worker Uvicorn — ChromaDB local não suporta múltiplos processos
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# Metadados da imagem
LABEL maintainer="SUP — Sistema de Apoio Clínico"
LABEL description="Sistema RAG de apoio à decisão clínica (Dr. Ajuda)"
LABEL uso="Exclusivo para profissionais de saúde habilitados"

# ── Dependências do sistema ───────────────────────────────────────────────────
# build-essential: compilar extensões C (lxml, chromadb)
# curl: usado pelo HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# ── Instalar dependências Python (camada cacheável separada) ──────────────────
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# ── Pré-baixar modelo de embedding (evita download no primeiro request) ────────
# O modelo paraphrase-multilingual-MiniLM-L12-v2 (~400MB) é baixado aqui,
# durante o build, para que o container suba sem acesso à internet em produção.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Baixando modelo de embedding...'); \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('Modelo em cache. OK.')"

# ── Copiar código da aplicação ────────────────────────────────────────────────
COPY super.py web_api.py ./
COPY templates/ templates/

# Porta exposta pelo Uvicorn
EXPOSE 8000

# ── Health check para DigitalOcean App Platform e Docker ─────────────────────
# /health responde sem tocar no banco (< 200ms)
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Comando de inicialização ──────────────────────────────────────────────────
# --workers 4  — ChromaDB agora roda como servidor HTTP separado (container chroma),
#               portanto múltiplos workers Uvicorn são seguros.
#               Regra geral: 2× o número de CPUs disponíveis.
#               Ajuste conforme o hardware do servidor.
# --timeout-keep-alive 90 — pipeline clínico pode demorar até ~60s com retry Groq.
CMD ["uvicorn", "web_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "90"]
