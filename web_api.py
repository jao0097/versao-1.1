"""
web_api.py — Camada web do sistema SUP (Dr. Ajuda)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API FastAPI que expõe o pipeline RAG clínico para profissionais de saúde.

⚠️  USO RESTRITO: apenas profissionais de saúde habilitados.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Carrega .env antes de qualquer outra configuração
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO DE LOG
# ══════════════════════════════════════════════════════════════════════════════

_LOG_FORMAT = os.getenv("LOG_FORMAT", "text").strip().lower()

if _LOG_FORMAT == "json":
    # Formato estruturado para Datadog / plataformas de observabilidade
    import json as _json_module

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_obj = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                log_obj["exc"] = self.formatException(record.exc_info)
            return _json_module.dumps(log_obj, ensure_ascii=False)

    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logging.root.addHandler(_handler)
    logging.root.setLevel(logging.INFO)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("sup.web")

# ══════════════════════════════════════════════════════════════════════════════
#  SENTRY (opcional — só inicializa se SENTRY_DSN estiver definido)
# ══════════════════════════════════════════════════════════════════════════════

_SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            traces_sample_rate=0.2,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            # Nunca enviar conteúdo de perguntas/respostas ao Sentry
            before_send=lambda event, hint: _sanitize_sentry_event(event),
        )
        logger.info("Sentry inicializado com sucesso.")
    except ImportError:
        logger.warning("sentry-sdk não instalado — monitoramento Sentry desabilitado.")


def _sanitize_sentry_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Remove dados sensíveis antes de enviar ao Sentry (LGPD)."""
    # Remove corpo de requisições para proteger conteúdo de consultas médicas
    event.pop("request", None)
    return event


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES DA API
# ══════════════════════════════════════════════════════════════════════════════

API_VERSION = "1.0.0"

# Token obrigatório em produção — sem valor padrão para forçar configuração explícita
_WEB_TOKEN = os.getenv("WEB_TOKEN", "").strip()
if not _WEB_TOKEN:
    logger.critical(
        "WEB_TOKEN não configurado! "
        "Para desenvolvimento, defina WEB_TOKEN=dev-token-local no .env"
    )
    # Não encerra aqui — permite subir para diagnóstico, mas todos os endpoints
    # protegidos retornarão 503 se o token estiver vazio.

# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN: inicializa o banco UMA VEZ ao subir o servidor
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Inicializa ChromaDB + embeddings via _ensure_db() durante o startup.
    Um único worker Uvicorn — ChromaDB não suporta múltiplos processos.
    """
    logger.info("Inicializando banco vetorial e embeddings... (pode levar ~30s)")
    import super as sup  # importa o módulo RAG

    try:
        sup._ensure_db()  # carrega ChromaDB + SentenceTransformer
        app.state.sup = sup
        logger.info(
            "Base pronta. Chunks disponíveis: %s",
            sup.colecao.count() if sup.colecao else "desconhecido",
        )
    except Exception as exc:
        logger.exception("Falha ao inicializar banco vetorial: %s", exc)
        raise

    yield  # servidor em execução

    # Cleanup (ChromaDB persistente não precisa de flush explícito)
    logger.info("Encerrando servidor SUP.")


# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)

# ══════════════════════════════════════════════════════════════════════════════
#  APLICAÇÃO FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SUP — Sistema de Apoio Clínico Dr. Ajuda",
    description=(
        "API de apoio à decisão clínica baseada em RAG. "
        "⚠️ Uso exclusivo de profissionais de saúde habilitados."
    ),
    version=API_VERSION,
    docs_url=None,   # desabilita Swagger público em produção
    redoc_url=None,
    lifespan=lifespan,
)

# Registra o handler de rate limit
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Templates Jinja2 (pasta templates/ ou fallback inline)
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR) if os.path.isdir(_TEMPLATES_DIR) else None

# ══════════════════════════════════════════════════════════════════════════════
#  AUTENTICAÇÃO: Bearer Token estático
# ══════════════════════════════════════════════════════════════════════════════

_bearer_scheme = HTTPBearer(auto_error=False)


def _verificar_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """
    Valida o token Bearer. Retorna 401 se inválido ou ausente.
    Nunca loga o token recebido.
    """
    if not _WEB_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servidor mal configurado: WEB_TOKEN não definido.",
        )
    token = credentials.credentials if credentials else None
    if not token or token != _WEB_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de acesso inválido ou ausente. Contate o administrador.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class ConsultaRequest(BaseModel):
    pergunta: str

    @field_validator("pergunta")
    @classmethod
    def pergunta_nao_vazia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("A pergunta não pode ser vazia.")
        if len(v) > 4000:
            raise ValueError("Pergunta muito longa (máximo 4000 caracteres).")
        return v


class ConsultaResponse(BaseModel):
    resposta: str
    modelo_usado: str
    tempo_resposta_s: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check (sem autenticação)",
    tags=["Infraestrutura"],
)
def health_check() -> HealthResponse:
    """
    Liveness probe — responde sem tocar no banco.
    Usado pelo DigitalOcean App Platform e Docker HEALTHCHECK.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/status",
    summary="Estatísticas da base de conhecimento",
    tags=["Infraestrutura"],
    dependencies=[Depends(_verificar_token)],
)
def status_base(request: Request) -> JSONResponse:
    """
    Retorna métricas do banco vetorial. Pode ser lento — não use como health check.
    Requer autenticação Bearer token.
    """
    sup = request.app.state.sup
    try:
        chunks_total = sup.colecao.count() if sup.colecao else 0
    except Exception:
        chunks_total = -1

    # Contagem de URLs processadas (sem logar conteúdo)
    try:
        processadas = sup.carregar_processadas()
        urls_total = len(processadas)
        artigos = sum(1 for u in processadas if "youtube" not in u)
        videos = sum(1 for u in processadas if "youtube" in u)
    except Exception:
        urls_total = artigos = videos = -1

    return JSONResponse({
        "api_version": API_VERSION,
        "status": "online",
        "banco": {
            "pasta_chroma": sup.PASTA_CHROMA,
            "pasta_saida": sup.PASTA_SAIDA,
            "chunks_totais": chunks_total,
            "urls_processadas": urls_total,
            "artigos": artigos,
            "videos_youtube": videos,
        },
        "modelos_groq": {
            "resposta": sup.MODELO_POTENTE,
            "classificacao": sup.MODELO_RAPIDO,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.post(
    "/consultar",
    response_model=ConsultaResponse,
    summary="Consulta clínica (Q&A)",
    tags=["Consulta"],
    dependencies=[Depends(_verificar_token)],
)
@limiter.limit("10/minute")
def consultar(
    request: Request,
    body: ConsultaRequest,
) -> ConsultaResponse:
    """
    Executa o pipeline RAG clínico e retorna a resposta em Markdown.
    - Timeout efetivo: 90s (configurado no Uvicorn)
    - Rate limit: 10 req/min por IP
    - Nunca loga o conteúdo da pergunta ou resposta (LGPD)
    """
    sup = request.app.state.sup
    ip_addr = get_remote_address(request)

    # Valida que o banco está disponível
    try:
        chunks_disponiveis = sup.colecao.count() if sup.colecao else 0
    except Exception as exc:
        logger.error("Erro ao verificar banco vetorial: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de conhecimento indisponível. Tente novamente em instantes.",
        )

    if chunks_disponiveis == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Base de conhecimento vazia. Indexe conteúdo antes de consultar.",
        )

    # Loga apenas metadados — NUNCA o conteúdo da pergunta
    logger.info(
        "Consulta iniciada | ip=%s | pergunta_chars=%d",
        ip_addr,
        len(body.pergunta),
    )

    t0 = time.time()
    try:
        resposta = sup.pipeline_perguntar(body.pergunta)
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(
            "Erro no pipeline | ip=%s | elapsed=%.1fs | erro=%s",
            ip_addr,
            elapsed,
            type(exc).__name__,  # nunca expõe detalhes do erro
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Serviço temporariamente indisponível. "
                "O sistema de IA pode estar sobrecarregado — tente novamente em alguns instantes."
            ),
        )

    elapsed = time.time() - t0
    logger.info(
        "Consulta concluída | ip=%s | pergunta_chars=%d | resposta_chars=%d | "
        "tempo=%.1fs | modelo=%s",
        ip_addr,
        len(body.pergunta),
        len(resposta),
        elapsed,
        sup.MODELO_POTENTE,
    )

    return ConsultaResponse(
        resposta=resposta,
        modelo_usado=sup.MODELO_POTENTE,
        tempo_resposta_s=round(elapsed, 2),
    )


@app.get(
    "/consultar/stream",
    summary="Consulta clínica via Server-Sent Events (SSE)",
    tags=["Consulta"],
)
@limiter.limit("10/minute")
def consultar_stream(
    request: Request,
    pergunta: str,
    token: str,
) -> StreamingResponse:
    """
    Executa o pipeline RAG e transmite a resposta progressivamente via SSE.
    Parâmetros: ?pergunta=<string>&token=<bearer_token>

    Como o pipeline_perguntar é bloqueante, a resposta é enviada de uma vez
    ao final — mas o SSE mantém conexão aberta e evita timeout do browser.
    """
    # Autenticação via query param (SSE não suporta headers no browser)
    if not _WEB_TOKEN or token != _WEB_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
        )

    pergunta = (pergunta or "").strip()
    if not pergunta:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Pergunta não pode ser vazia.",
        )
    if len(pergunta) > 4000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Pergunta muito longa (máximo 4000 caracteres).",
        )

    sup = request.app.state.sup
    ip_addr = get_remote_address(request)

    def _event_generator():
        # Sinaliza ao cliente que o processamento começou
        yield "data: {\"tipo\": \"inicio\"}\n\n"

        t0 = time.time()
        logger.info(
            "SSE consulta iniciada | ip=%s | pergunta_chars=%d",
            ip_addr,
            len(pergunta),
        )

        try:
            resposta = sup.pipeline_perguntar(pergunta)
            elapsed = time.time() - t0

            logger.info(
                "SSE consulta concluída | ip=%s | tempo=%.1fs | modelo=%s",
                ip_addr,
                elapsed,
                sup.MODELO_POTENTE,
            )

            payload = json.dumps({
                "tipo": "resposta",
                "resposta": resposta,
                "modelo_usado": sup.MODELO_POTENTE,
                "tempo_resposta_s": round(elapsed, 2),
            }, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "SSE erro | ip=%s | elapsed=%.1fs | erro=%s",
                ip_addr,
                elapsed,
                type(exc).__name__,
            )
            erro_payload = json.dumps({
                "tipo": "erro",
                "mensagem": (
                    "Serviço temporariamente indisponível. "
                    "Tente novamente em alguns instantes."
                ),
            }, ensure_ascii=False)
            yield f"data: {erro_payload}\n\n"

        yield "data: {\"tipo\": \"fim\"}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # desativa buffer no Nginx
        },
    )


@app.get(
    "/",
    summary="Interface web",
    tags=["Interface"],
    response_class=HTMLResponse,
)
def interface_web(request: Request) -> HTMLResponse:
    """
    Serve a interface HTML do sistema SUP.
    A autenticação é feita no lado do cliente via sessionStorage + Bearer token.
    """
    # Lê o HTML do template se disponível, senão usa o embutido
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse(content=_HTML_EMBUTIDO, status_code=200)


# ══════════════════════════════════════════════════════════════════════════════
#  TRATAMENTO GLOBAL DE EXCEÇÕES (não expõe stack trace)
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": (
                "Limite de consultas excedido (10 por minuto por IP). "
                "Aguarde antes de tentar novamente."
            )
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erro não tratado em %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Erro interno do servidor. A equipe técnica foi notificada."
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HTML EMBUTIDO (fallback se pasta templates/ não existir)
# ══════════════════════════════════════════════════════════════════════════════

_HTML_EMBUTIDO = open(
    os.path.join(os.path.dirname(__file__), "templates", "index.html"),
    encoding="utf-8",
).read() if os.path.exists(
    os.path.join(os.path.dirname(__file__), "templates", "index.html")
) else """<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>SUP — Dr. Ajuda</title></head>
<body>
<p>⚠️ Template não encontrado. Crie templates/index.html ou execute docker compose.</p>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT LOCAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8000"))
    uvicorn.run(
        "web_api:app",
        host=host,
        port=port,
        workers=1,  # ChromaDB local não suporta múltiplos workers
        timeout_keep_alive=90,
        reload=False,
    )
