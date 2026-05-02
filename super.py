"""
sup — Sistema de Apoio Clínico Dr. Ajuda
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistema RAG de apoio à decisão clínica para médicos, baseado no conhecimento
extraído do canal e site do Dr. Ajuda.

Permite que médicos consultem a base de conhecimento para:
  • Raciocínio diagnóstico e diagnóstico diferencial
  • Condutas clínicas e escolha de tratamentos
  • Interpretação de exames e achados
  • Formulação e revisão de laudos médicos
  • Consulta de protocolos e conceitos clínicos

Fontes suportadas:
  • Sites / blogs   → crawl automático de listagem, raspa artigos
  • YouTube         → transcrição automática de vídeos (1 ou vários)
  • Arquivos locais → processa pares .txt + .json já salvos em disco

⚠️  AVISO: Este sistema é de uso exclusivo de profissionais de saúde habilitados.
    As respostas não substituem o julgamento clínico do médico.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALAÇÃO
  pip install requests beautifulsoup4 chromadb groq sentence-transformers youtube-transcript-api
  export GROQ_API_KEY='sua_chave'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — CONSULTAR (uso principal para médicos)

  Consulta clínica interativa (menu):
    sup

  Consulta direta por linha de comando:
    sup --pergunta "quais os critérios diagnósticos para síndrome metabólica?"
    sup --pergunta "condutas para hipertensão resistente"

  Ver status e estatísticas:
    sup --status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — INDEXAR (fase de configuração — banco já completo em produção)

  Crawl de site (descobre e indexa todos os artigos):
    sup --crawl "https://site.com/blog"
    sup --crawl "https://site.com/blog" --filtro "/posts/"
    sup --crawl "https://site.com/blog" --sem-paginacao

  Artigos (uma ou várias URLs):
    sup --artigos "https://a.com/p1 https://a.com/p2"
    sup --artigos lista.txt

  Vídeo(s) do YouTube:
    sup --video "https://youtube.com/watch?v=ID"
    sup --video "url1" "url2" "url3"
    sup --video lista.txt

  Arquivos locais (.txt + .json gerados pelo sistema):
    sup --local

  Localizar arquivo local por nome / tema:
    sup --buscar-local

  Forçar reindexação de uma URL já processada:
    sup --reindexar "https://..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIÁVEIS DE AMBIENTE (opcionais)
  RAG_PASTA_SAIDA          pasta de saída dos arquivos
  RAG_PASTA_CHROMA         pasta do banco vetorial
  RAG_DEBUG_LOG            caminho para log de debug (desativado se vazio)
  GROQ_MODELO_RAPIDO       modelo para metadados/classificação  (padrão: llama-3.1-8b-instant)
  GROQ_MODELO_POTENTE      modelo para respostas finais         (padrão: llama-3.3-70b-versatile)
  GROQ_MODELO_LIMPEZA      modelo para limpeza de transcrição   (padrão: llama-3.1-8b-instant)
  CHROMA_CHUNK_PALAVRAS    tamanho do chunk em palavras         (padrão: 500)
  CHROMA_CHUNK_OVERLAP     sobreposição entre chunks            (padrão: 50)
  LIMPEZA_TRANSCRICAO      local | ia | nenhuma                 (padrão: local)
  GROQ_LIMPEZA_MAX_CHARS   máx. chars por bloco de limpeza IA  (padrão: 7500)
  GROQ_LIMPEZA_PAUSA_S     pausa entre blocos de limpeza        (padrão: 1.25)
  GROQ_METADADOS_MAX_CHARS máx. chars enviados para metadados   (padrão: 4000)
  TIMEOUT_REQUISICAO       timeout HTTP em segundos             (padrão: 15)
  PAUSA_ENTRE_PAGINAS      pausa entre requisições web          (padrão: 1.0)
  GROQ_MAX_RETRIES         tentativas em caso de rate-limit     (padrão: 10)
  RAG_ARTIGOS_FETCH_THREADS  downloads paralelos no lote de artigos (padrão: 1; ex.: 4)
  RAG_HTML_PARSER          lxml | html.parser | html5lib       (padrão: lxml se instalado)
  RAG_MIN_CHARS_ARTIGO     mín. de caracteres no texto raspado (padrão: 0 = aceita curtos)
  RAG_YT_PAUSA_ENTRE_VIDEOS  pausa base entre cada vídeo no lote (padrão: 4s)
  RAG_YT_PAUSA_JITTER        atraso aleatório extra 0..N s (padrão: 2)
  RAG_YT_APOS_TRANSCRICAO    pausa entre transcrição e oEmbed (padrão: 0.8s)
  RAG_YT_PRE_FETCH           pausa antes do 1º pedido de transcrição por vídeo (padrão: 0.5s)
  RAG_YT_FETCH_RETRIES       tentativas se a transcrição falhar (padrão: 5)
  RAG_YT_FETCH_BACKOFF       base do backoff exponencial entre tentativas (padrão: 2.5s)
  RAG_YT_STOP_ON_IPBLOCK     para o lote ao detectar IpBlocked (padrão: 1)
  RAG_YT_MAX_CONSEC_IPBLOCK  quantos IpBlocked seguidos para parar (padrão: 1)
  RAG_YT_IPBLOCK_COOLDOWN_S  sugestão de espera antes de tentar de novo (padrão: 3600)
  RAG_YT_DLP_ENABLED         fallback via yt-dlp para legendas (padrão: 0)
  RAG_YT_DLP_COOKIES_FILE    caminho opcional para cookies.txt do yt-dlp
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import concurrent.futures
import difflib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS GROQ (centralizados)
# ══════════════════════════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVISO LEGAL — SISTEMA DE APOIO À DECISÃO CLÍNICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este sistema é destinado EXCLUSIVAMENTE a profissionais de saúde habilitados.
Todas as respostas são geradas com base no conhecimento extraído do canal e
site do Dr. Ajuda e têm finalidade de APOIO à decisão — não substituem o
julgamento clínico, o exame físico do paciente nem as diretrizes oficiais das
sociedades médicas. O profissional de saúde é inteiramente responsável pela
decisão diagnóstica e terapêutica final.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Aviso médico anexado ao final de toda resposta gerada ao médico
AVISO_MEDICO = (
    "\n\n---\n"
    "⚠️  **Aviso de apoio clínico**: Esta resposta é produzida por um sistema de suporte à decisão "
    "baseado no conteúdo do Dr. Ajuda e destina-se exclusivamente a profissionais de saúde habilitados. "
    "Não substitui o julgamento clínico, o exame físico do paciente nem as diretrizes das sociedades "
    "médicas. O médico assistente é o único responsável pela conduta diagnóstica e terapêutica final."
)

PROMPT_METADADOS_SISTEMA = (
    "Você é um indexador clínico especializado em preparar conteúdo médico para recuperação semântica em RAG.\n"
    "Sua saída alimenta diretamente o ChromaDB — metadados precisos aumentam a recuperabilidade dos chunks.\n\n"
    "Raciocine em ordem antes de preencher cada campo:\n"
    "1. Leia o conteúdo identificando a condição/tema central e as entidades clínicas (fármacos, CIDs, procedimentos).\n"
    "2. Extraia apenas o que está explícito — nunca infira diagnósticos, doses ou afirmações não presentes.\n"
    "3. Preserve terminologia médica exata; prefira nomes genéricos de fármacos.\n"
    "4. Se um campo não puder ser preenchido com segurança: \"\" ou [].\n"
    "5. Retorne APENAS JSON válido, sem markdown, sem texto fora do objeto.\n"
)

PROMPT_METADADOS_USUARIO_HEADER = (
    "Retorne JSON com estas chaves (sem extras):\n"
    "{\n"
    '  "resumo": string,            // 1–3 frases clínicas fiéis; sem extrapolação\n'
    '  "palavras_chave": string[],  // 3–8 termos médicos do texto\n'
    '  "tema_principal": string,    // 1 frase: "Diagnóstico e tratamento de X"\n'
    '  "topicos_abordados": string[], // 3–8 tópicos clínicos reais\n'
    '  "condicoes_clinicas": string[], // patologias/síndromes citadas ([] se nenhuma)\n'
    '  "medicamentos": string[],    // nomes genéricos dos fármacos ([] se nenhum)\n'
    '  "procedimentos": string[],   // exames, cirurgias, escalas ([] se nenhum)\n'
    '  "especialidade": string,     // área médica principal\n'
    '  "nivel_evidencia": "opinião de especialista"|"revisão narrativa"|"estudo clínico"|"diretriz"|"outro",\n'
    '  "nivel_tecnico": "iniciante"|"intermediário"|"avançado",\n'
    '  "linguagem": "pt"|"en"|"outro"\n'
    "}\n"
)

PROMPT_LIMPEZA_SISTEMA = (
    "Você é um editor de transcrições médicas preparando texto para indexação semântica em RAG.\n"
    "Meta: maximizar a recuperabilidade clínica dos chunks — termos médicos precisos e frases completas aumentam hits relevantes.\n\n"
    "PRESERVAR (prioridade máxima):\n"
    "- Toda terminologia médica, nomes de fármacos, doses, valores laboratoriais, CIDs, escalas e índices.\n"
    "- Números sempre com seu contexto ('5 mg/kg/dia', não apenas '5').\n"
    "- Nomes próprios, siglas médicas e acrônimos.\n\n"
    "CORRIGIR:\n"
    "- Pontuação e capitalização óbvias.\n"
    "- Repetições de vício de fala claramente redundantes ('é é é', 'então então').\n"
    "- Marcações de ruído: [música], [aplausos], [risadas], [inaudível] e similares.\n"
    "- Inserir parágrafo a cada mudança clara de tema clínico.\n\n"
    "PROIBIDO:\n"
    "- Resumir, omitir ou condensar qualquer trecho.\n"
    "- Adicionar informações, interpretações ou diagnósticos não presentes.\n\n"
    "Retorne APENAS o texto limpo, sem comentários, cabeçalhos ou markdown.\n"
    "Se indicado 'Trecho N de M', edite apenas esse trecho."
)

PROMPT_CLASSIFICADOR_SISTEMA = (
    "Você é o roteador de busca de um sistema RAG clínico. Sua decisão determina quais chunks serão recuperados.\n"
    "Uma classificação errada desperdiça tokens ou perde informação — seja preciso.\n\n"
    "Raciocine nesta ordem antes de classificar:\n"
    "1. ESPECIFICA: algum título da lista corresponde diretamente à pergunta? Se sim → tipo='especifica', fonte_alvo=título exato.\n"
    "2. COMPARATIVA: a pergunta compara ≥2 condições, tratamentos, fármacos ou abordagens? Se sim → tipo='comparativa'.\n"
    "3. FORA_DE_ESCOPO: a pergunta não tem nenhuma relação com medicina, saúde ou ciências clínicas? Se sim → tipo='fora_de_escopo'.\n"
    "4. Caso contrário → tipo='geral'.\n\n"
    "n_chunks obrigatório: especifica=4 | geral=6 | comparativa=8 | fora_de_escopo=0.\n"
    "Se não tiver certeza do título exato em 'especifica', use 'geral' com fonte_alvo=null.\n"
    "Retorne APENAS JSON válido, sem markdown.\n"
)

PROMPT_ROUTER_CONVERSA_SISTEMA = (
    "Você é o roteador de intenções de um CLI de RAG clínico (Dr. Ajuda). Retorne APENAS JSON válido, sem markdown.\n\n"
    "AÇÕES disponíveis → parâmetros obrigatórios:\n"
    "consultar          → pergunta_clinica: string\n"
    "artigos_lote       → entradas: string[]  (URLs http/https não-YouTube ou caminho .txt)\n"
    "videos_lote        → entradas: string[]  (URLs/IDs YouTube ou caminho .txt)\n"
    "crawl_site         → url_listagem: string, filtro_path: string|null, sem_paginacao: boolean\n"
    "processar_local    → (sem parâmetros)\n"
    "buscar_local       → consulta: string|null\n"
    "reindexar          → url: string\n"
    "status             → (sem parâmetros)\n"
    "ajuda              → (sem parâmetros)\n"
    "sair               → (sem parâmetros)\n"
    "perguntar_clarificacao → pergunta: string\n\n"
    "PRIORIDADE de decisão (avalie nesta ordem):\n"
    "1. Contém dúvida/caso clínico → consultar\n"
    "2. Contém URL YouTube ou ID de vídeo → videos_lote\n"
    "3. Contém URL http(s) não-YouTube → artigos_lote (exceto se pedir 'crawl'/'varrer'/'descobrir links' → crawl_site)\n"
    "4. Pede 'reindexar' + URL → reindexar\n"
    "5. Pede 'processar' arquivos locais → processar_local\n"
    "6. Pede 'buscar'/'achar' arquivo local → buscar_local\n"
    "7. Pede status/estatísticas → status\n"
    "8. Parâmetro essencial ausente → perguntar_clarificacao (nunca invente URLs)\n"
)

PROMPT_RESPOSTA_SISTEMA = (
    "Você é um consultor clínico sênior baseado EXCLUSIVAMENTE no conhecimento indexado do Dr. Ajuda.\n"
    "Interlocutor: sempre um MÉDICO habilitado buscando apoio ao raciocínio clínico.\n\n"
    "ANTES DE ESCREVER, identifique mentalmente:\n"
    "- Qual é a questão clínica central implícita na pergunta?\n"
    "- Quais fontes são mais relevantes para respondê-la?\n"
    "- As fontes convergem ou divergem? Há lacunas?\n"
    "Essa análise interna melhora a síntese — não a exiba, use-a para escrever melhor.\n\n"
    "REGRAS\n"
    "1. Use APENAS as fontes fornecidas. Nunca extrapole, invente doses, diagnósticos ou referências.\n"
    "2. Ausência de informação → declare: 'Não localizado na base do Dr. Ajuda.'\n"
    "3. Cite *(Fonte N — Título)* ao lado de cada dado clínico. Divergência entre fontes → '⚠️ Divergência:'.\n"
    "4. Preserve exatamente: fármacos, doses, valores laboratoriais, escalas, CIDs.\n"
    "5. Linguagem técnica. Nunca orientações diretas a pacientes. Sem disclaimers.\n\n"
    "ESTRUTURA — omita seções genuinamente inaplicáveis; nunca omita 🔍 nem 📚.\n\n"
    "🔍 **Análise da Consulta**\n"
    "Reenquadre a pergunta: qual é a questão clínica real, qual contexto está implícito, qual interpretação você responde.\n\n"
    "🧠 **Síntese Clínica**\n"
    "Responda diretamente. Interprete e conecte — não liste o que as fontes dizem, mostre o que significam juntas.\n"
    "Parágrafos densos, sem listas soltas. Cite *(Fonte N)* a cada dado relevante.\n\n"
    "🩺 **Raciocínio Diagnóstico / DD** *(quando aplicável)*\n"
    "Hipóteses com critérios, achados que aproximam/afastam cada uma, diagnóstico mais provável justificado.\n\n"
    "💊 **Conduta Clínica** *(quando aplicável)*\n"
    "Sequência prática: investigação → tratamento (fármacos + doses das fontes) → monitoramento → critérios de encaminhamento.\n\n"
    "⚠️ **Pontos de Atenção / Red Flags** *(quando aplicável)*\n"
    "Alarmes, contraindicações, interações relevantes, situações que exigem ação imediata.\n\n"
    "🔎 **Lacunas na Base** *(quando houver)*\n"
    "O que as fontes não cobriram. Explicitar é mais útil ao médico do que omitir.\n\n"
    "📚 **Fontes Consultadas**\n"
    "[N] Título — URL\n"
)
from urllib.parse import urljoin, urlparse, urlencode
from urllib.request import urlopen

# ── Dependências de terceiros ──────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Debug NDJSON — ativo apenas se RAG_DEBUG_LOG estiver definido como caminho válido
DEBUG_LOG_PATH = os.getenv("RAG_DEBUG_LOG", "").strip()
DEBUG_SESSION_ID = "sup"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not DEBUG_LOG_PATH:
        return
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES  ← edite aqui ou via variáveis de ambiente
# ══════════════════════════════════════════════════════════════════════════════

# ── Suporte a múltiplas chaves Groq com rotação automática ────────────────────
# Opção 1 — variável única com vírgulas: GROQ_API_KEYS=chave1,chave2,chave3
# Opção 2 — variáveis numeradas:         GROQ_API_KEY=chave1  GROQ_API_KEY_2=chave2 ...
# As duas formas podem ser combinadas; duplicatas são removidas.
def _carregar_groq_keys() -> List[str]:
    """Lê todas as chaves Groq configuradas e retorna lista sem duplicatas."""
    chaves: List[str] = []
    # Opção 1: GROQ_API_KEYS separadas por vírgula
    multi = os.getenv("GROQ_API_KEYS", "").strip()
    if multi:
        chaves.extend(k.strip() for k in multi.split(",") if k.strip())
    # Opção 2: GROQ_API_KEY + GROQ_API_KEY_2, GROQ_API_KEY_3, …
    for sufixo in ["", "_2", "_3", "_4", "_5"]:
        k = os.getenv(f"GROQ_API_KEY{sufixo}", "").strip()
        if k and k not in chaves:
            chaves.append(k)
    # Remove placeholder e duplicatas mantendo ordem
    validas = [k for k in chaves if k and k != "sua_chave_groq_aqui"]
    seen: List[str] = []
    for k in validas:
        if k not in seen:
            seen.append(k)
    return seen

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "sua_chave_groq_aqui")  # mantido p/ compat.
_GROQ_KEYS: List[str] = _carregar_groq_keys()

_PASTA_PADRAO       = os.path.expanduser("~/Documentos/rag_base")
PASTA_SAIDA         = os.getenv("RAG_PASTA_SAIDA",  _PASTA_PADRAO)
PASTA_CHROMA        = os.getenv("RAG_PASTA_CHROMA", os.path.join(PASTA_SAIDA, "chroma_db"))
ARQUIVO_PROCESSADAS = os.path.join(PASTA_SAIDA, ".urls_processadas.json")

# Modelos Groq — 3 papéis distintos para controle de custo/velocidade
MODELO_RAPIDO  = os.getenv("GROQ_MODELO_RAPIDO",  "llama-3.1-8b-instant")   # metadados, classificação
MODELO_POTENTE = os.getenv("GROQ_MODELO_POTENTE", "llama-3.3-70b-versatile") # resposta final
MODELO_LIMPEZA = os.getenv("GROQ_MODELO_LIMPEZA", "llama-3.1-8b-instant")   # limpeza em lotes

# Chunking
CHUNK_PALAVRAS      = int(os.getenv("CHROMA_CHUNK_PALAVRAS", "500"))
CHUNK_OVERLAP       = int(os.getenv("CHROMA_CHUNK_OVERLAP",  "50"))

# Limpeza de transcrição: "local" (0 tokens) | "ia" (Groq) | "nenhuma" (bruto)
LIMPEZA_TRANSCRICAO = os.getenv("LIMPEZA_TRANSCRICAO", "local").strip().lower()
LIMPEZA_MAX_CHARS   = int(os.getenv("GROQ_LIMPEZA_MAX_CHARS", "7500"))   # max chars por bloco IA
LIMPEZA_PAUSA_S     = float(os.getenv("GROQ_LIMPEZA_PAUSA_S", "1.25"))   # pausa entre blocos

# Metadados
METADADOS_MAX_CHARS = int(os.getenv("GROQ_METADADOS_MAX_CHARS", "4000"))

# Economia de tokens (consulta)
# - Quando ligado, reduz contexto enviado e evita enviar lista completa de títulos ao classificador.
GROQ_ECONOMIA = os.getenv("GROQ_ECONOMIA", "0").strip().lower() in ("1", "true", "yes", "on")
GROQ_CLASSIF_TITULOS_MAX = max(0, int(os.getenv("GROQ_CLASSIF_TITULOS_MAX", "30")))
GROQ_CONTEXTO_CHUNK_MAX_CHARS = max(0, int(os.getenv("GROQ_CONTEXTO_CHUNK_MAX_CHARS", "1200")))
GROQ_CONTEXTO_TOTAL_MAX_CHARS = max(0, int(os.getenv("GROQ_CONTEXTO_TOTAL_MAX_CHARS", "9000")))
GROQ_MAX_N_CHUNKS = max(0, int(os.getenv("GROQ_MAX_N_CHUNKS", "0")))  # 0 = sem limite extra

# Requisições web
TIMEOUT           = int(os.getenv("TIMEOUT_REQUISICAO", "15"))
PAUSA_ENTRE_REQS  = float(os.getenv("PAUSA_ENTRE_PAGINAS", "1.0"))
GROQ_MAX_RETRIES  = int(os.getenv("GROQ_MAX_RETRIES", "10"))

# YouTube: reduzir bloqueio por excesso de requisições (sem proxy)
RAG_YT_PAUSA_ENTRE_VIDEOS = float(os.getenv("RAG_YT_PAUSA_ENTRE_VIDEOS", "4.0"))
RAG_YT_PAUSA_JITTER       = float(os.getenv("RAG_YT_PAUSA_JITTER", "2.0"))
RAG_YT_APOS_TRANSCRICAO   = float(os.getenv("RAG_YT_APOS_TRANSCRICAO", "0.8"))
RAG_YT_PRE_FETCH          = float(os.getenv("RAG_YT_PRE_FETCH", "0.5"))
RAG_YT_FETCH_RETRIES      = max(1, int(os.getenv("RAG_YT_FETCH_RETRIES", "5")))
RAG_YT_FETCH_BACKOFF      = float(os.getenv("RAG_YT_FETCH_BACKOFF", "2.5"))

# Se bater IpBlocked/RequestBlocked, é bloqueio anti-bot (não é "falta de legenda").
RAG_YT_STOP_ON_IPBLOCK        = os.getenv("RAG_YT_STOP_ON_IPBLOCK", "1").strip().lower() not in ("0", "false", "no")
RAG_YT_IPBLOCK_COOLDOWN_S     = float(os.getenv("RAG_YT_IPBLOCK_COOLDOWN_S", "3600"))  # 1h padrão
RAG_YT_MAX_CONSEC_IPBLOCK     = max(1, int(os.getenv("RAG_YT_MAX_CONSEC_IPBLOCK", "1")))

# Fallback opcional via yt-dlp (não exige proxy, mas pode precisar cookies dependendo do caso)
RAG_YT_DLP_ENABLED            = os.getenv("RAG_YT_DLP_ENABLED", "0").strip().lower() in ("1", "true", "yes")
RAG_YT_DLP_COOKIES_FILE       = os.getenv("RAG_YT_DLP_COOKIES_FILE", "").strip()  # caminho para cookies.txt (opcional)

# Download paralelo só na fase HTTP+parse do lote de artigos (1 = sequencial, como antes)
RAG_ARTIGOS_FETCH_THREADS = max(1, min(32, int(os.getenv("RAG_ARTIGOS_FETCH_THREADS", "1"))))

# Mínimo de caracteres no corpo do artigo após raspagem (0 = só rejeita vazio; antigo era ~150)
RAG_MIN_CHARS_ARTIGO = max(0, int(os.getenv("RAG_MIN_CHARS_ARTIGO", "0")))


def _pick_bs_parser() -> str:
    """lxml costuma ser bem mais rápido que html.parser; fallback automático."""
    override = os.getenv("RAG_HTML_PARSER", "").strip().lower()
    if override in ("lxml", "html.parser", "html5lib"):
        return override
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


BS_PARSER = _pick_bs_parser()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

# Seletores de conteúdo principal (ordem de prioridade)
SELETORES_CONTEUDO = [
    "article", "main",
    "[class*='post-content']", "[class*='article-content']",
    "[class*='entry-content']", "[class*='article-body']",
    "[class*='post-body']",    "[class*='content']",
    "div#content", "div#main",
]

# Elementos removidos antes de extrair o texto
SELETORES_LIXO = [
    "nav", "header", "footer", "aside", "form",
    "script", "style", "noscript", "iframe",
    "[class*='sidebar']",    "[class*='menu']",       "[class*='social']",
    "[class*='share']",      "[class*='comment']",    "[class*='related']",
    "[class*='newsletter']", "[class*='popup']",      "[class*='cookie']",
    "[class*='banner']",     "[class*='ad-']",        "[class*='ads']",
]


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO E INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

GROQ_ENABLED = bool(_GROQ_KEYS)
if not GROQ_ENABLED:
    print(
        "⚠️  Nenhuma GROQ_API_KEY configurada — recursos de IA ficam indisponíveis.\n"
        "   Para habilitar: defina GROQ_API_KEY (e opcionalmente GROQ_API_KEY_2) no .env\n",
        file=sys.stderr,
    )
else:
    _n = len(_GROQ_KEYS)
    print(
        f"✅  Groq: {_n} chave{'s' if _n > 1 else ''} carregada{'s' if _n > 1 else ''} "
        f"({'rotação automática ativa' if _n > 1 else 'chave única'}).",
        file=sys.stderr,
    )

# groq_client mantido para compatibilidade de imports externos
groq_client = Groq(api_key=_GROQ_KEYS[0]) if GROQ_ENABLED else None


import threading as _threading


class _GroqKeyPool:
    """
    Pool thread-safe de clientes Groq com rotação automática em rate-limit.

    Comportamento:
    - Enquanto houver chaves não tentadas no round atual: rotaciona imediatamente (sem wait).
    - Quando todas as chaves do round falharam com 429/503: espera e recomeça.
    - Repete por GROQ_MAX_RETRIES rounds completos antes de desistir.
    """

    def __init__(self, keys: List[str]) -> None:
        self._clients: List[Groq] = [Groq(api_key=k) for k in keys]
        self._idx: int = 0
        self._lock = _threading.Lock()

    def _rotate(self) -> None:
        with self._lock:
            self._idx = (self._idx + 1) % len(self._clients)

    def create(self, **kwargs):
        n = len(self._clients)
        ultimo: Optional[BaseException] = None
        round_count = 0

        while round_count < GROQ_MAX_RETRIES:
            keys_tried = 0
            while keys_tried < n:
                try:
                    return self._clients[self._idx].chat.completions.create(**kwargs)
                except Exception as e:
                    ultimo = e
                    if not _groq_retryavel(e):
                        raise
                    keys_tried += 1
                    if keys_tried < n:
                        ant = self._idx + 1
                        self._rotate()
                        print(
                            f"   🔄  Rate-limit chave {ant}/{n} → chave {self._idx + 1}/{n}",
                            file=sys.stderr,
                        )
            # Todas as chaves falharam neste round
            round_count += 1
            if round_count >= GROQ_MAX_RETRIES:
                break
            assert ultimo is not None
            w = _retry_wait(ultimo, round_count)
            print(
                f"   ⏸️  Todas as {n} chave(s) em rate-limit — "
                f"aguardando {w:.1f}s (round {round_count}/{GROQ_MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(w)
            self._rotate()

        assert ultimo is not None
        raise ultimo


_groq_pool: Optional[_GroqKeyPool] = _GroqKeyPool(_GROQ_KEYS) if GROQ_ENABLED else None

http: Optional[requests.Session] = None
embedding_fn = None
chroma = None
colecao = None


def _ensure_db() -> None:
    """Inicializa recursos pesados apenas quando necessário (DB/embeddings/HTTP)."""
    global http, embedding_fn, chroma, colecao
    if colecao is not None and chroma is not None and embedding_fn is not None and http is not None:
        return

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    os.makedirs(PASTA_CHROMA, exist_ok=True)

    if http is None:
        http = requests.Session()
        http.headers.update(HEADERS)

    if embedding_fn is None:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    if chroma is None:
        _chroma_host = os.getenv("CHROMA_HOST")
        _chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        if _chroma_host:
            chroma = chromadb.HttpClient(host=_chroma_host, port=_chroma_port)
        else:
            chroma = chromadb.PersistentClient(path=PASTA_CHROMA)

    if colecao is None:
        colecao = chroma.get_or_create_collection(
            name="base_conhecimento",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICAÇÃO DE URLs
# ══════════════════════════════════════════════════════════════════════════════

def carregar_processadas() -> Set[str]:
    _ensure_db()
    if os.path.exists(ARQUIVO_PROCESSADAS):
        with open(ARQUIVO_PROCESSADAS, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def marcar_processada(url: str, processadas: Set[str]) -> None:
    processadas.add(url)
    with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
        json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ — chamada com retry automático e backoff exponencial
# ══════════════════════════════════════════════════════════════════════════════

def _retry_wait(erro: BaseException, tentativa: int) -> float:
    """Extrai o tempo de espera da mensagem da API ou usa backoff exponencial."""
    m = re.search(r"try again in ([\d.]+)\s*s", str(erro), re.I)
    if m:
        return float(m.group(1)) + 0.75
    ra = getattr(getattr(erro, "response", None), "headers", {}).get("retry-after")
    try:
        return float(ra) + 0.75
    except (TypeError, ValueError):
        pass
    return min(120.0, 2.5 * (1.65 ** tentativa))


def _groq_retryavel(erro: BaseException) -> bool:
    status = getattr(erro, "status_code", None)
    if status in (429, 503):
        return True
    if status == 413:
        body = str(erro).lower()
        return "rate_limit" in body or "tokens per minute" in body or "tpm" in body
    return False


def chamar_groq(
    sistema: str,
    usuario: str,
    modelo: str = MODELO_RAPIDO,
    json_mode: bool = False,
    temperatura: float = 0.1,
) -> str:
    """
    Chama a API Groq com rotação automática de chaves em rate-limit.

    Com múltiplas chaves (GROQ_API_KEY + GROQ_API_KEY_2, ...):
      • 429 numa chave → troca para a próxima imediatamente (sem esperar).
      • Todas as chaves em 429 → aguarda backoff exponencial e recomeça.
    """
    if not GROQ_ENABLED or _groq_pool is None:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. Defina a variável de ambiente GROQ_API_KEY para usar recursos de IA."
        )
    kwargs: dict = {
        "model": modelo,
        "temperature": temperatura,
        "messages": [
            {"role": "system", "content": sistema},
            {"role": "user",   "content": usuario},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return _groq_pool.create(**kwargs).choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS GERAIS
# ══════════════════════════════════════════════════════════════════════════════

def slugify(texto: str, limite: int = 60) -> str:
    s = unicodedata.normalize("NFKD", texto)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", s.lower()))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:limite] or "item"


def chunkar(texto: str) -> List[str]:
    """Divide em chunks de palavras com sobreposição configurável."""
    t = (texto or "").strip()
    if not t:
        return []
    palavras = t.split()
    step = max(1, CHUNK_PALAVRAS - CHUNK_OVERLAP)
    out = [
        " ".join(palavras[i : i + CHUNK_PALAVRAS])
        for i in range(0, len(palavras), step)
        if palavras[i : i + CHUNK_PALAVRAS]
    ]
    if not out and t:
        return [t]
    return out


def gerar_metadados(titulo: str, conteudo: str, tipo: str) -> dict:
    """
    Gera metadados semânticos enriquecidos via Groq.
    Funciona para artigos web e transcrições de vídeo.
    """
    raw = chamar_groq(
        sistema=PROMPT_METADADOS_SISTEMA,
        usuario=(
            f"{PROMPT_METADADOS_USUARIO_HEADER}\n\n"
            f"Tipo: {tipo}\n"
            f"Título: {titulo}\n\n"
            "Conteúdo (trecho):\n"
            f"{conteudo[:METADADOS_MAX_CHARS]}"
        ),
        json_mode=True,
    )
    fallback = {
        "resumo": conteudo[:200],
        "palavras_chave": [],
        "tema_principal": "",
        "topicos_abordados": [],
        "condicoes_clinicas": [],
        "medicamentos": [],
        "procedimentos": [],
        "especialidade": "",
        "nivel_evidencia": "outro",
        "nivel_tecnico": "intermediário",
        "linguagem": "pt",
    }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    return parsed


def indexar_conteudo(
    conteudo: str,
    titulo: str,
    url: str,
    tipo: str,
    meta_extra: dict,
    meta_ia: Optional[dict] = None,
) -> int:
    """
    Divide em chunks e faz upsert no ChromaDB.
    Armazena chunk_index, total_chunks e todos os campos de meta_ia.
    Retorna o número de chunks indexados.
    """
    _ensure_db()
    chunks = chunkar(conteudo)
    if not chunks:
        return 0

    slug  = slugify(url)[:80]
    total = len(chunks)
    ids   = [f"{slug}_c{i}" for i in range(total)]

    # Campos IA achatados para o ChromaDB (apenas strings)
    ia_fields: dict = {}
    if meta_ia:
        ia_fields["resumo"]              = str(meta_ia.get("resumo", ""))[:500]
        ia_fields["palavras_chave"]      = ", ".join(meta_ia.get("palavras_chave", []))
        ia_fields["tema_principal"]      = str(meta_ia.get("tema_principal", ""))
        ia_fields["topicos_abordados"]   = ", ".join(meta_ia.get("topicos_abordados", []))
        ia_fields["nivel_tecnico"]       = str(meta_ia.get("nivel_tecnico", ""))
        ia_fields["linguagem"]           = str(meta_ia.get("linguagem", "pt"))
        # Campos clínicos (Dr. Ajuda)
        ia_fields["condicoes_clinicas"]  = ", ".join(meta_ia.get("condicoes_clinicas", []))
        ia_fields["medicamentos"]        = ", ".join(meta_ia.get("medicamentos", []))
        ia_fields["procedimentos"]       = ", ".join(meta_ia.get("procedimentos", []))
        ia_fields["especialidade"]       = str(meta_ia.get("especialidade", ""))
        ia_fields["nivel_evidencia"]     = str(meta_ia.get("nivel_evidencia", ""))

    metas = [
        {
            "titulo":       titulo,
            "url":          url,
            "tipo":         tipo,
            "data":         date.today().isoformat(),
            "chunk_index":  i,
            "total_chunks": total,
            **{k: str(v) for k, v in meta_extra.items()},
            **ia_fields,
        }
        for i in range(total)
    ]

    colecao.upsert(ids=ids, documents=chunks, metadatas=metas)
    return total


def salvar_arquivos(
    titulo: str,
    url: str,
    conteudo: str,
    meta_ia: dict,
    tipo: str,
    extra: dict,
) -> None:
    """Salva .txt e .json em PASTA_SAIDA. Evita colisão de slug entre URLs diferentes."""
    slug = slugify(titulo)
    base = os.path.join(PASTA_SAIDA, slug)

    sufixo = 1
    while os.path.exists(f"{base}.json"):
        if _ler_url_json(f"{base}.json") == url:
            break  # mesma URL, sobrescreve normalmente
        base = os.path.join(PASTA_SAIDA, f"{slug}_{sufixo}")
        sufixo += 1

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(
            f"TIPO: {tipo}\n"
            f"TÍTULO: {titulo}\n"
            f"URL: {url}\n"
            f"DATA: {date.today().isoformat()}\n"
            f"RESUMO: {meta_ia.get('resumo', '')}\n"
            f"[CONTEÚDO]\n"
            f"{conteudo}"
        )

    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tipo":              tipo,
                "titulo":            titulo,
                "url":               url,
                "data_coleta":       date.today().isoformat(),
                "resumo":            meta_ia.get("resumo", ""),
                "palavras_chave":    meta_ia.get("palavras_chave", []),
                "tema_principal":    meta_ia.get("tema_principal", ""),
                "topicos_abordados": meta_ia.get("topicos_abordados", []),
                "nivel_tecnico":     meta_ia.get("nivel_tecnico", ""),
                "linguagem":         meta_ia.get("linguagem", "pt"),
                # Campos clínicos (Dr. Ajuda)
                "condicoes_clinicas": meta_ia.get("condicoes_clinicas", []),
                "medicamentos":       meta_ia.get("medicamentos", []),
                "procedimentos":      meta_ia.get("procedimentos", []),
                "especialidade":      meta_ia.get("especialidade", ""),
                "nivel_evidencia":    meta_ia.get("nivel_evidencia", ""),
                "tamanho_chars":      len(conteudo),
                "indexado_chroma":    True,
                **extra,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _ler_url_json(caminho: str) -> str:
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f).get("url", "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DA PASTA CENTRAL
# ══════════════════════════════════════════════════════════════════════════════

def salvar_catalogo() -> None:
    """
    Gera (ou atualiza) o arquivo _CATALOGO.md na PASTA_SAIDA com um resumo
    legível de todo o conteúdo indexado: artigos por domínio, vídeos por canal,
    estatísticas gerais e data da última atualização.
    Chamado automaticamente após cada indexação e com --status.
    """
    _ensure_db()
    if not os.path.exists(PASTA_SAIDA):
        return

    arquivos_json = [
        f for f in os.listdir(PASTA_SAIDA)
        if f.endswith(".json") and not f.startswith("_")
    ]

    artigos: list = []
    videos:  list = []

    for nome in sorted(arquivos_json):
        caminho = os.path.join(PASTA_SAIDA, nome)
        try:
            with open(caminho, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if not isinstance(meta, dict):
            continue

        tipo   = meta.get("tipo", "")
        titulo = meta.get("titulo", nome[:-5])
        url    = meta.get("url", "")
        resumo = meta.get("resumo", "")
        data   = meta.get("data_coleta", "")
        chars  = meta.get("tamanho_chars", 0)
        tags   = ", ".join(meta.get("palavras_chave", [])) or "—"
        nivel  = meta.get("nivel_tecnico", "")

        entrada = {
            "titulo": titulo,
            "url":    url,
            "resumo": resumo,
            "data":   data,
            "chars":  chars,
            "tags":   tags,
            "nivel":  nivel,
        }

        if tipo == "video_youtube":
            entrada["canal"]   = meta.get("canal", "")
            entrada["duracao"] = meta.get("duracao", "")
            videos.append(entrada)
        else:
            entrada["dominio"] = meta.get("dominio", urlparse(url).netloc if url else "")
            artigos.append(entrada)

    # ── Agrupa artigos por domínio ────────────────────────────────────────
    dominios: dict = {}
    for a in artigos:
        d = a["dominio"] or "local"
        dominios.setdefault(d, []).append(a)

    # ── Agrupa vídeos por canal ───────────────────────────────────────────
    canais: dict = {}
    for v in videos:
        c = v["canal"] or "Sem canal"
        canais.setdefault(c, []).append(v)

    # ── Escreve o Markdown ────────────────────────────────────────────────
    linhas: list = []
    def w(s=""): linhas.append(s)

    w("# 📚 Catálogo da Base de Conhecimento")
    w()
    w(f"> Gerado automaticamente por **sup** em {date.today().isoformat()}")
    w(f"> Pasta: `{PASTA_SAIDA}`")
    w()
    w("---")
    w()
    w("## 📊 Resumo")
    w()
    w(f"| Item | Quantidade |")
    w(f"|------|-----------|")
    w(f"| Artigos indexados | {len(artigos)} |")
    w(f"| Vídeos indexados  | {len(videos)} |")
    w(f"| Domínios de artigos | {len(dominios)} |")
    w(f"| Canais de vídeo   | {len(canais)} |")
    w(f"| Chunks no banco   | {colecao.count()} |")
    w()

    if dominios:
        w("---")
        w()
        w("## 📄 Artigos")
        w()
        for dominio, itens in sorted(dominios.items()):
            w(f"### 🌐 {dominio} ({len(itens)} artigo(s))")
            w()
            for a in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{a['titulo']}]({a['url']})" if a["url"] else a["titulo"]
                w(f"#### {link}")
                if a["resumo"]:
                    w(f"> {a['resumo']}")
                meta_line = []
                if a["data"]:   meta_line.append(f"📅 {a['data']}")
                if a["chars"]:  meta_line.append(f"📝 {a['chars']:,} chars")
                if a["nivel"]:  meta_line.append(f"🎓 {a['nivel']}")
                if a["tags"] != "—": meta_line.append(f"🏷️ {a['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    if canais:
        w("---")
        w()
        w("## 🎬 Vídeos YouTube")
        w()
        for canal, itens in sorted(canais.items()):
            w(f"### 📺 {canal} ({len(itens)} vídeo(s))")
            w()
            for v in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{v['titulo']}]({v['url']})" if v["url"] else v["titulo"]
                w(f"#### {link}")
                if v["resumo"]:
                    w(f"> {v['resumo']}")
                meta_line = []
                if v["data"]:    meta_line.append(f"📅 {v['data']}")
                if v["duracao"]: meta_line.append(f"⏱️ {v['duracao']}")
                if v["chars"]:   meta_line.append(f"📝 {v['chars']:,} chars")
                if v["nivel"]:   meta_line.append(f"🎓 {v['nivel']}")
                if v["tags"] != "—": meta_line.append(f"🏷️ {v['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    caminho_catalogo = os.path.join(PASTA_SAIDA, "_CATALOGO.md")
    with open(caminho_catalogo, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"  📋 Catálogo atualizado: {caminho_catalogo}")

_RE_TAG     = re.compile(r"\[[^\]]+\]")
_RE_ESPACO  = re.compile(r"[ \t]+")
_RE_NEWLINE = re.compile(r"\n{3,}")
_RE_REPETE  = re.compile(r"(\b\w{1,16}\b)(\s+\1\b)+", re.IGNORECASE)


def _limpar_local(texto: str) -> str:
    """Limpeza por regex — sem tokens Groq, muito rápida."""
    if not texto:
        return ""
    t = _RE_TAG.sub(" ", texto)
    t = _RE_ESPACO.sub(" ", t)
    linhas = [ln.strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in linhas if ln)
    t = _RE_NEWLINE.sub("\n\n", t)
    t = _RE_REPETE.sub(r"\1", t)
    return t.strip()


def _dividir_para_limpeza(texto: str) -> List[str]:
    """
    Parte em blocos de até LIMPEZA_MAX_CHARS chars, quebrando em newlines
    para preservar contexto. Evita 413 / esgotamento de TPM na API Groq.
    """
    if len(texto) <= LIMPEZA_MAX_CHARS:
        return [texto]
    blocos: List[str] = []
    i, n = 0, len(texto)
    while i < n:
        end = min(i + LIMPEZA_MAX_CHARS, n)
        if end < n:
            trecho = texto[i:end]
            nl = trecho.rfind("\n")
            if nl > LIMPEZA_MAX_CHARS // 3:
                end = i + nl + 1
        blocos.append(texto[i:end])
        i = end
    return blocos



def limpar_transcricao(texto_bruto: str) -> str:
    """
    Limpa a transcrição de vídeo conforme LIMPEZA_TRANSCRICAO:
      local  (padrão) → regex, 0 tokens, muito rápido
      ia              → Groq em blocos, melhor qualidade
      nenhuma         → texto bruto
    """
    modo = LIMPEZA_TRANSCRICAO
    if modo in ("nenhuma", "none", "off", "raw"):
        print("  ⏭️  Limpeza desativada — texto bruto.")
        return (texto_bruto or "").strip()
    if modo in ("local", "rapida", "fast", "regex"):
        print("  🧹 Limpeza local (regex, 0 tokens)...")
        return _limpar_local(texto_bruto)

    # modo ia
    print("  🧹 Limpando com Groq (modo ia)...")
    partes = _dividir_para_limpeza(texto_bruto)
    total  = len(partes)
    if total > 1:
        print(f"     ({total} blocos de até {LIMPEZA_MAX_CHARS} chars)")
    saidas: List[str] = []
    for idx, bloco in enumerate(partes, start=1):
        cab = f"Trecho {idx} de {total}.\n\n" if total > 1 else ""
        saidas.append(chamar_groq(
            PROMPT_LIMPEZA_SISTEMA,
            f"{cab}Limpe:\n\n{bloco}",
            modelo=MODELO_LIMPEZA,
        ))
        if idx < total:
            time.sleep(LIMPEZA_PAUSA_S)
    return "\n\n".join(s.strip() for s in saidas if s.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: ARTIGOS WEB
# ══════════════════════════════════════════════════════════════════════════════

def _get_soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = http.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, BS_PARSER)
    except requests.RequestException as e:
        print(f"   ⚠️  GET falhou ({url}): {e}")
        return None


def _raspar_de_soup(url: str, soup: BeautifulSoup) -> Optional[dict]:
    """Extrai título e texto a partir de um BeautifulSoup já carregado."""
    for sel in SELETORES_LIXO:
        for el in soup.select(sel):
            el.decompose()

    h1    = soup.find("h1")
    title = soup.find("title")
    titulo = (
        h1.get_text(strip=True)    if h1    else
        title.get_text(strip=True) if title else
        urlparse(url).path.split("/")[-1] or "Sem título"
    )

    el = None
    for s in SELETORES_CONTEUDO:
        el = soup.select_one(s)
        if el is not None:
            break
    if el is None:
        el = soup.find("body") or soup

    conteudo = re.sub(r"\s{2,}", " ", el.get_text(separator=" ", strip=True)).strip()

    if not conteudo:
        print("   ⚠️  Conteúdo vazio após extração — pulando")
        return None

    if RAG_MIN_CHARS_ARTIGO > 0 and len(conteudo) < RAG_MIN_CHARS_ARTIGO:
        print(
            f"   ⚠️  Conteúdo abaixo do mínimo ({len(conteudo)} < {RAG_MIN_CHARS_ARTIGO} chars) — pulando "
            f"(ajuste RAG_MIN_CHARS_ARTIGO ou use 0 para aceitar textos curtos)"
        )
        return None

    return {"titulo": titulo, "conteudo": conteudo, "url": url}


def raspar_artigo(url: str) -> Optional[dict]:
    """Raspa artigo web com seletores inteligentes e remoção de lixo."""
    soup = _get_soup(url)
    if soup is None:
        return None
    return _raspar_de_soup(url, soup)


def _baixar_e_raspar_artigo(url: str) -> Tuple[str, Optional[dict]]:
    """
    GET + parse em sessão própria (seguro para ThreadPoolExecutor).
    Usado no lote de artigos quando RAG_ARTIGOS_FETCH_THREADS > 1.
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)
    try:
        r = sess.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, BS_PARSER)
        return url, _raspar_de_soup(url, soup)
    except requests.RequestException as e:
        print(f"   ⚠️  GET falhou ({url}): {e}")
        return url, None


def processar_artigo(
    url: str,
    processadas: Set[str],
    *,
    atualizar_catalogo: bool = True,
    artigo: Optional[dict] = None,
) -> int:
    """
    Raspa, gera metadados, indexa e salva um artigo.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro.

    ``artigo``: se já vier raspado (ex.: download paralelo), pula novo GET.
    ``atualizar_catalogo``: em lotes grandes, use False e chame salvar_catalogo() ao final.
    """
    _ensure_db()
    if url in processadas:
        return 0

    if artigo is None:
        artigo = raspar_artigo(url)
    if not artigo:
        return -1

    print(f"   ✓ \"{artigo['titulo'][:65]}\" — {len(artigo['conteudo'])} chars")

    meta_ia = gerar_metadados(artigo["titulo"], artigo["conteudo"], "artigo")
    dominio = urlparse(url).netloc

    n = indexar_conteudo(
        conteudo=artigo["conteudo"],
        titulo=artigo["titulo"],
        url=url,
        tipo="artigo",
        meta_extra={"dominio": dominio},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=artigo["titulo"],
        url=url,
        conteudo=artigo["conteudo"],
        meta_ia=meta_ia,
        tipo="artigo",
        extra={"dominio": dominio},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    if atualizar_catalogo:
        salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

_RE_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


def extrair_video_id(url: str) -> Optional[str]:
    """Aceita URL completa, youtu.be, shorts ou ID puro de 11 chars."""
    s = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _RE_YT_ID.search(s)
    return m.group(1) if m else None


def _metadados_yt(video_id: str) -> dict:
    """Obtém título e canal via oEmbed — sem precisar de API key."""
    query = urlencode({
        "url":    f"https://www.youtube.com/watch?v={video_id}",
        "format": "json",
    })
    try:
        with urlopen(f"https://www.youtube.com/oembed?{query}", timeout=10) as r:
            data = json.loads(r.read().decode())
        return {
            "titulo": data.get("title",       video_id),
            "canal":  data.get("author_name", ""),
        }
    except Exception:
        return {"titulo": video_id, "canal": ""}


def _pausa_entre_videos_no_lote() -> None:
    """Espaça pedidos ao YouTube entre um vídeo e o próximo (base + jitter aleatório)."""
    base = RAG_YT_PAUSA_ENTRE_VIDEOS
    jit = RAG_YT_PAUSA_JITTER
    if base <= 0 and jit <= 0:
        return
    extra = random.uniform(0.0, jit) if jit > 0 else 0.0
    time.sleep(max(0.0, base + extra))


def _fetch_transcricao_youtube(video_id: str):
    """Busca legenda com retentativas e backoff (menos bloqueio por rajada de erros)."""
    ultimo: Optional[BaseException] = None
    for tentativa in range(RAG_YT_FETCH_RETRIES):
        try:
            if tentativa == 0 and RAG_YT_PRE_FETCH > 0:
                time.sleep(RAG_YT_PRE_FETCH)
            api = YouTubeTranscriptApi()
            return api.fetch(video_id, languages=["pt", "pt-BR", "en"])
        except Exception as e:
            ultimo = e
            nome = type(e).__name__
            if nome in ("IpBlocked", "RequestBlocked"):
                # Bloqueio anti-bot: retry curto tende a piorar.
                break
            if tentativa >= RAG_YT_FETCH_RETRIES - 1:
                break
            w = min(
                120.0,
                RAG_YT_FETCH_BACKOFF * (2 ** tentativa) + random.uniform(0.25, 1.25),
            )
            print(
                f"   ⏸️  Falha ao obter transcrição — aguardando {w:.1f}s "
                f"(tentativa {tentativa + 1}/{RAG_YT_FETCH_RETRIES})"
            )
            time.sleep(w)
    assert ultimo is not None
    raise ultimo


def _eh_bloqueio_yt(e: BaseException) -> bool:
    n = type(e).__name__
    if n in ("IpBlocked", "RequestBlocked"):
        return True
    msg = str(e).lower()
    return "ipblocked" in msg or "requestblocked" in msg or "youtube is blocking requests" in msg


def _obter_transcricao_ytdlp(url: str) -> Optional[str]:
    """
    Fallback via yt-dlp: tenta baixar legendas (manual/auto) e retornar texto.
    Requer yt-dlp instalado no sistema. Pode usar cookies.txt opcional.
    """
    if not RAG_YT_DLP_ENABLED:
        return None
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("   ⚠️  yt-dlp não encontrado; fallback desativado.")
        return None

    tmp_dir = os.path.join(PASTA_SAIDA, "_tmp_ytdlp_subs")
    os.makedirs(tmp_dir, exist_ok=True)
    out_tpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "pt,pt-BR,en",
        "--sub-format", "vtt",
        "-o", out_tpl,
        url,
    ]
    if RAG_YT_DLP_COOKIES_FILE:
        cmd = cmd[:1] + ["--cookies", RAG_YT_DLP_COOKIES_FILE] + cmd[1:]

    try:
        subprocess.run(cmd, cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("   ⚠️  yt-dlp falhou ao obter legendas.")
        return None

    vid = extrair_video_id(url) or ""
    candidatos = []
    if vid:
        for nome in os.listdir(tmp_dir):
            if nome.startswith(vid) and nome.endswith(".vtt"):
                candidatos.append(os.path.join(tmp_dir, nome))
    if not candidatos:
        return None

    # Preferir pt/pt-BR quando disponível
    candidatos.sort(key=lambda p: (".pt" not in p and ".pt-BR" not in p, len(p)))
    try:
        with open(candidatos[0], encoding="utf-8", errors="ignore") as f:
            vtt = f.read()
    except Exception:
        return None

    linhas = []
    for ln in vtt.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.startswith(("WEBVTT", "NOTE")):
            continue
        if "-->" in t:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        linhas.append(t)
    return " ".join(linhas).strip() or None


def processar_video(
    url: str,
    processadas: Set[str],
    *,
    atualizar_catalogo: bool = True,
) -> int:
    """
    Transcreve, limpa, gera metadados, indexa e salva um vídeo do YouTube.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro.
    """
    _ensure_db()
    if url in processadas:
        return 0

    video_id = extrair_video_id(url)
    if not video_id:
        print(f"   ⚠️  Não foi possível extrair ID do vídeo: {url}")
        return -1

    print(f"   📹 Baixando transcrição: {url}")
    try:
        transcript = _fetch_transcricao_youtube(video_id)
        texto_bruto = " ".join(
            t["text"] if isinstance(t, dict) else t.text for t in transcript
        )
    except Exception as e:
        if _eh_bloqueio_yt(e):
            print("   ⛔ YouTube bloqueou requisições deste IP (IpBlocked/RequestBlocked).")
            if RAG_YT_DLP_ENABLED:
                print("   ↪ Tentando fallback por yt-dlp (legendas)...")
                alt = _obter_transcricao_ytdlp(url)
                if alt:
                    texto_bruto = alt
                    transcript = [{"text": alt, "start": 0.0, "duration": 0.0}]
                else:
                    print(f"   ⏸️  Recomendado aguardar ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tentar novamente.")
                    return -2
            else:
                print(f"   ⏸️  Recomendado aguardar ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tentar novamente.")
                print("   Dica: export RAG_YT_DLP_ENABLED=1 (requer yt-dlp) e opcional RAG_YT_DLP_COOKIES_FILE=cookies.txt")
                return -2
        print(f"   ⚠️  Transcrição indisponível: {e}")
        return -1

    # Duração estimada a partir do último item da transcrição
    ultimo  = transcript[-1]
    start   = float(ultimo["start"]    if isinstance(ultimo, dict) else getattr(ultimo, "start",    0))
    dur     = float(ultimo["duration"] if isinstance(ultimo, dict) else getattr(ultimo, "duration", 0))
    dur_s   = int(math.ceil(start + dur))
    duracao = f"{dur_s // 3600:02d}:{(dur_s % 3600) // 60:02d}:{dur_s % 60:02d}"

    if RAG_YT_APOS_TRANSCRICAO > 0:
        time.sleep(RAG_YT_APOS_TRANSCRICAO)

    meta_yt = _metadados_yt(video_id)
    titulo  = meta_yt["titulo"]
    canal   = meta_yt["canal"]
    print(f"   ✓ \"{titulo[:65]}\" — {duracao}")

    conteudo = limpar_transcricao(texto_bruto)
    meta_ia  = gerar_metadados(titulo, conteudo, "video_youtube")

    n = indexar_conteudo(
        conteudo=conteudo,
        titulo=titulo,
        url=url,
        tipo="video_youtube",
        meta_extra={"canal": canal, "duracao": duracao, "video_id": video_id},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=titulo,
        url=url,
        conteudo=conteudo,
        meta_ia=meta_ia,
        tipo="video_youtube",
        extra={"canal": canal, "duracao": duracao, "video_id": video_id},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    if atualizar_catalogo:
        salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: PROCESSAMENTO DE ARQUIVOS LOCAIS (.txt + .json)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_processar_pasta() -> None:
    """
    Varre PASTA_SAIDA e processa pares .txt + .json ainda não indexados.
    Compatível com arquivos gerados por versões anteriores do sistema.
    """
    _ensure_db()
    if not os.path.exists(PASTA_SAIDA):
        print(f"Pasta não encontrada: {PASTA_SAIDA}")
        return

    txts = [a for a in os.listdir(PASTA_SAIDA) if a.endswith(".txt")]
    if not txts:
        print("Nenhum .txt encontrado na pasta.")
        return

    print(f"\n📂 {len(txts)} arquivo(s) encontrado(s)\n{'─'*50}")
    processadas_set = carregar_processadas()

    for nome_txt in txts:
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            print(f"⚠️  JSON não encontrado para: {nome_txt} — pulando")
            continue

        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  JSON inválido ({caminho_json}): {e} — pulando")
            continue

        if meta.get("indexado_chroma") is True:
            print(f"⏭️  Já indexado: {meta.get('titulo', nome_base)}")
            continue

        titulo = meta.get("titulo", nome_base)
        tipo   = meta.get("tipo", "video_youtube")
        print(f"\n🎬 Processando: {titulo}")

        with open(caminho_txt, encoding="utf-8") as f:
            texto_bruto = f.read()

        texto_limpo = limpar_transcricao(texto_bruto)
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
        extra   = {k: v for k, v in meta.items()
                   if k in ("canal", "dominio", "duracao", "video_id")}

        n = indexar_conteudo(
            conteudo=texto_limpo,
            titulo=titulo,
            url=meta.get("url", ""),
            tipo=tipo,
            meta_extra=extra,
            meta_ia=meta_ia,
        )

        meta.update(meta_ia)
        meta["indexado_chroma"] = True
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Registra no histórico de URLs processadas (se tiver URL válida)
        url_meta = meta.get("url", "")
        if url_meta:
            marcar_processada(url_meta, processadas_set)

        print(f"  ✅ {titulo} — {n} chunks indexados")

    print(f"\n🏁 Concluído! Total de chunks no banco: {colecao.count()}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: LOTE DE VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar_url_yt(url: str) -> str:
    """Normaliza ID puro (11 chars) para URL completa."""
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return f"https://www.youtube.com/watch?v={url}"
    return url


def _ler_urls_arquivo(caminho: str) -> List[str]:
    """
    Lê URLs de um arquivo de texto — uma por linha.
    Ignora linhas em branco e comentários iniciados com '#'.
    """
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}")
        return []
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    urls = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        urls.append(_normalizar_url_yt(linha))
    return urls


_RE_URL_HTTP = re.compile(r"https?://[^\s<>\'\"`,;)]+", re.I)


def _trim_sufixo_url(u: str) -> str:
    return u.rstrip(").,;]\"'>\r\n\t ")


def extrair_urls_http_em_texto(texto: str) -> List[str]:
    """Encontra URLs http(s) em um bloco colado (várias linhas, espaços ou vírgulas)."""
    out: List[str] = []
    seen: Set[str] = set()
    for m in _RE_URL_HTTP.finditer(texto or ""):
        u = _trim_sufixo_url(m.group(0))
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _resolver_entradas_para_urls_artigos(entradas: List[str], *, run_id: str = "pre-fix-paste") -> List[str]:
    """
    Converte linhas coladas + texto livre em lista de URLs de artigo.
    Hipóteses de debug: H1=arquivo; H2=regex multi-link; H3=fallback linha única.
    """
    if not entradas:
        # region agent log
        _debug_log(run_id, "H1", "sup.py:_resolver_entradas_para_urls_artigos", "empty entradas", {})
        # endregion
        return []

    if len(entradas) == 1:
        lone = entradas[0].strip()
        if os.path.exists(lone) and not lone.startswith(("http://", "https://")):
            urls = _ler_urls_arquivo(lone)
            # region agent log
            _debug_log(
                run_id,
                "H1",
                "sup.py:_resolver_entradas_para_urls_artigos",
                "resolved from file path",
                {"count": len(urls)},
            )
            # endregion
            return urls

    blob = "\n".join(entradas)
    extraidas = extrair_urls_http_em_texto(blob)
    if extraidas:
        # region agent log
        _debug_log(
            run_id,
            "H2",
            "sup.py:_resolver_entradas_para_urls_artigos",
            "resolved via http regex",
            {"blob_len": len(blob), "url_count": len(extraidas), "line_count": len(entradas)},
        )
        # endregion
        return extraidas

    out: List[str] = []
    seen: Set[str] = set()
    for linha in entradas:
        t = linha.strip()
        if t.startswith(("http://", "https://")) and t not in seen:
            seen.add(t)
            out.append(t)
    # region agent log
    _debug_log(
        run_id,
        "H3",
        "sup.py:_resolver_entradas_para_urls_artigos",
        "fallback raw lines only",
        {"blob_len": len(blob), "url_count": len(out), "line_count": len(entradas)},
    )
    # endregion
    return out


def _parece_youtube(s: str) -> bool:
    """Heurística rápida: retorna True se a string parece ser URL ou ID de YouTube."""
    s = (s or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return True
    low = s.lower()
    return "youtube.com" in low or "youtu.be" in low


def _resolver_entradas_para_urls_video(entradas: List[str], *, run_id: str = "pre-fix-paste") -> List[str]:
    """Extrai links YouTube e IDs a partir de texto colado ou lista de linhas."""
    if not entradas:
        # region agent log
        _debug_log(run_id, "H4", "sup.py:_resolver_entradas_para_urls_video", "empty entradas", {})
        # endregion
        return []

    if len(entradas) == 1:
        lone = entradas[0].strip()
        if os.path.exists(lone) and not _parece_youtube(lone):
            urls = _ler_urls_arquivo(lone)
            # region agent log
            _debug_log(
                run_id,
                "H4",
                "sup.py:_resolver_entradas_para_urls_video",
                "resolved from file",
                {"count": len(urls)},
            )
            # endregion
            return urls

    blob = "\n".join(entradas)
    out: List[str] = []
    seen: Set[str] = set()

    for u in extrair_urls_http_em_texto(blob):
        u2 = _normalizar_url_yt(u)
        if _eh_url_youtube(u2) and u2 not in seen:
            seen.add(u2)
            out.append(u2)

    for raw in re.split(r"[\s,;|]+", blob):
        tok = raw.strip().strip("()[]<>'\"")
        if not tok:
            continue
        if _parece_youtube(tok):
            u2 = _normalizar_url_yt(tok)
            if _eh_url_youtube(u2) and u2 not in seen:
                seen.add(u2)
                out.append(u2)

    # region agent log
    _debug_log(
        run_id,
        "H4",
        "sup.py:_resolver_entradas_para_urls_video",
        "video URL resolution done",
        {"blob_len": len(blob), "url_count": len(out), "line_count": len(entradas)},
    )
    # endregion
    return out


def pipeline_artigos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs de artigos em sequência.
    Pula URLs inválidas e já indexadas, com resumo ao final.
    """
    _ensure_db()
    urls_validas: List[str] = []
    for url in urls:
        u = url.strip()
        if not _eh_url(u):
            print(f"⚠️  Ignorando (URL inválida): {u}")
            continue
        if _eh_url_youtube(u):
            print(f"⚠️  Ignorando (é YouTube, use opção de vídeos): {u}")
            continue
        urls_validas.append(u.rstrip("/"))

    if not urls_validas:
        print("Nenhuma URL de artigo válida para processar.")
        return

    processadas  = carregar_processadas()
    novas        = [u for u in urls_validas if u not in processadas]
    ja_indexadas = len(urls_validas) - len(novas)
    total        = len(novas)

    print(f"\n{'═'*55}")
    print(f"  📄 LOTE DE ARTIGOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")

    if not novas:
        print("✅ Todos os artigos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros = 0

    threads = RAG_ARTIGOS_FETCH_THREADS
    por_url: Dict[str, Optional[dict]] = {}

    if threads > 1 and len(novas) >= 2:
        w = min(threads, len(novas))
        print(f"  ⚡ Download paralelo: {w} thread(s) | parser HTML: {BS_PARSER}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as ex:
            futuros = [ex.submit(_baixar_e_raspar_artigo, u) for u in novas]
            for fut in concurrent.futures.as_completed(futuros):
                u, dados = fut.result()
                por_url[u] = dados

        for i, url in enumerate(novas, start=1):
            print(f"\n[{i}/{total}] 📄 {url}")
            artigo = por_url.get(url)
            if not artigo:
                erros += 1
                continue
            n = processar_artigo(
                url, processadas, atualizar_catalogo=False, artigo=artigo,
            )
            if n < 0:
                erros += 1
            else:
                total_chunks += n
            if i < total:
                time.sleep(PAUSA_ENTRE_REQS)
    else:
        for i, url in enumerate(novas, start=1):
            print(f"\n[{i}/{total}] 📄 {url}")
            n = processar_artigo(url, processadas, atualizar_catalogo=False)
            if n < 0:
                erros += 1
            else:
                total_chunks += n
            if i < total:
                time.sleep(PAUSA_ENTRE_REQS)

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")
    salvar_catalogo()


def pipeline_videos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs/IDs de vídeos do YouTube em sequência.
    Pula URLs já indexadas, exibe progresso e resumo ao final.
    """
    _ensure_db()
    # Normaliza todas as URLs e filtra inválidas
    urls_normalizadas = []
    for url in urls:
        url = _normalizar_url_yt(url)
        if not _eh_url_youtube(url):
            print(f"⚠️  Ignorando (não é YouTube): {url}")
            continue
        urls_normalizadas.append(url)

    if not urls_normalizadas:
        print("Nenhuma URL válida para processar.")
        return

    processadas   = carregar_processadas()
    novas         = [u for u in urls_normalizadas if u not in processadas]
    ja_indexadas  = len(urls_normalizadas) - len(novas)
    total         = len(novas)

    print(f"\n{'═'*55}")
    print(f"  🎬 LOTE DE VÍDEOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")
    if total >= 2:
        print(
            f"  ⏱️  Pausa entre vídeos: {RAG_YT_PAUSA_ENTRE_VIDEOS}s + aleatório 0–{RAG_YT_PAUSA_JITTER}s "
            f"(export RAG_YT_PAUSA_ENTRE_VIDEOS / RAG_YT_PAUSA_JITTER para ajustar)"
        )

    if not novas:
        print("✅ Todos os vídeos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0
    ipblock_seq  = 0

    for i, url in enumerate(novas, start=1):
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas, atualizar_catalogo=False)
        if n == -2:
            ipblock_seq += 1
            erros += 1
            if RAG_YT_STOP_ON_IPBLOCK and ipblock_seq >= RAG_YT_MAX_CONSEC_IPBLOCK:
                print(
                    f"\n⛔ Bloqueio de IP detectado ({ipblock_seq}x). Parando lote para não agravar."
                    f"\n   Aguarde ~{int(RAG_YT_IPBLOCK_COOLDOWN_S)}s e tente novamente."
                )
                break
        elif n < 0:
            erros += 1
        else:
            ipblock_seq = 0
            total_chunks += n
        if i < total:
            _pausa_entre_videos_no_lote()

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: BUSCA FUZZY DE ARQUIVO LOCAL (indexação por nome/tema)
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().strip().split())


def _tokenizar(texto: str) -> Set[str]:
    tokens: Set[str] = set()
    for parte in _normalizar(texto).split():
        limpo = "".join(ch for ch in parte if ch.isalnum())
        if len(limpo) >= 3:
            tokens.add(limpo)
    return tokens


def _score_relevancia(consulta: str, candidato: dict) -> float:
    """
    Score combinado:
      70% sobreposição de tokens (palavras com ≥3 chars em comum)
      30% similaridade de string via difflib
    """
    alvo = _normalizar(
        f"{candidato['titulo']} {candidato['nome_txt']} {candidato.get('amostra', '')}"
    )
    cons = _normalizar(consulta)
    if not cons or not alvo:
        return 0.0
    ratio       = difflib.SequenceMatcher(None, cons, alvo).ratio()
    t_cons      = _tokenizar(cons)
    t_alvo      = _tokenizar(alvo)
    token_score = len(t_cons & t_alvo) / max(1, len(t_cons))
    return 0.7 * token_score + 0.3 * ratio


def _candidatos_locais() -> List[dict]:
    """Retorna arquivos locais (.txt + .json) ainda não indexados."""
    candidatos = []
    if not os.path.exists(PASTA_SAIDA):
        return candidatos

    for nome_txt in os.listdir(PASTA_SAIDA):
        if not nome_txt.endswith(".txt"):
            continue
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            continue
        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("indexado_chroma") is True:
            continue

        try:
            with open(caminho_txt, encoding="utf-8") as f:
                amostra = f.read(2500)
        except OSError:
            amostra = ""

        candidatos.append({
            "titulo":       meta.get("titulo", nome_base),
            "nome_txt":     nome_txt,
            "caminho_txt":  caminho_txt,
            "caminho_json": caminho_json,
            "amostra":      amostra,
        })
    return candidatos


def _selecionar_candidato(candidatos: List[dict], consulta: str) -> Optional[dict]:
    """
    Seleciona o candidato mais relevante por score combinado.
    Abaixo do limiar de confiança (0.35), apresenta as opções ao usuário.
    """
    if not candidatos or not consulta.strip():
        return None

    ranqueados = sorted(
        ((_score_relevancia(consulta, c), c) for c in candidatos),
        reverse=True,
    )
    melhor_score, melhor = ranqueados[0]

    if melhor_score >= 0.35:
        return melhor

    print("\nMatch automático incerto. Candidatos mais próximos:")
    top = ranqueados[:5]
    for i, (score, c) in enumerate(top, start=1):
        print(f"  {i}. [{score:.2f}] {c['titulo']} ({c['nome_txt']})")

    escolha = input("Número correto (ou Enter para cancelar): ").strip()
    if escolha.isdigit():
        idx = int(escolha) - 1
        if 0 <= idx < len(top):
            return top[idx][1]
    return None


def pipeline_indexar_video_por_nome() -> None:
    """
    Pede uma descrição do arquivo local, encontra via busca fuzzy
    e indexa interativamente após confirmação do usuário.
    """
    _ensure_db()
    candidatos = _candidatos_locais()
    if not candidatos:
        print(f"Nenhum arquivo não indexado encontrado em: {PASTA_SAIDA}")
        return

    consulta  = input("Descreva o vídeo/artigo (nome, tema, palavras do conteúdo): ").strip()
    candidato = _selecionar_candidato(candidatos, consulta)

    if not candidato:
        print("Não foi possível identificar o arquivo.")
        return

    print(f"\nEncontrado: {candidato['titulo']} ({candidato['nome_txt']})")
    confirma = input("É esse? (s/n): ").strip().lower()
    if confirma not in ("s", "sim", "y", "yes"):
        print("Cancelado.")
        return

    try:
        with open(candidato["caminho_json"], encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")
        return

    titulo = meta.get("titulo", candidato["nome_txt"])
    tipo   = meta.get("tipo", "video_youtube")
    print(f"\n🎬 Processando: {titulo}")

    with open(candidato["caminho_txt"], encoding="utf-8") as f:
        texto_bruto = f.read()

    texto_limpo = limpar_transcricao(texto_bruto)
    with open(candidato["caminho_txt"], "w", encoding="utf-8") as f:
        f.write(texto_limpo)

    meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
    extra   = {k: v for k, v in meta.items()
               if k in ("canal", "dominio", "duracao", "video_id")}

    n = indexar_conteudo(
        conteudo=texto_limpo,
        titulo=titulo,
        url=meta.get("url", ""),
        tipo=tipo,
        meta_extra=extra,
        meta_ia=meta_ia,
    )
    meta.update(meta_ia)
    meta["indexado_chroma"] = True
    with open(candidato["caminho_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Registra no histórico de URLs processadas (se tiver URL válida)
    url_meta = meta.get("url", "")
    if url_meta:
        processadas_set = carregar_processadas()
        marcar_processada(url_meta, processadas_set)

    print(f"\n🏁 Concluído! {n} chunks indexados. Total no banco: {colecao.count()}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: DESCOBERTA DE LINKS (CRAWLER WEB)
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url_youtube(url: str) -> bool:
    return (
        "youtube.com/watch" in url or
        "youtu.be/" in url or
        "youtube.com/shorts/" in url
    )


def _eh_paginacao(tag_a, href: str, url_base: str) -> bool:
    """Detecta links de paginação por texto, classe CSS ou padrão de URL."""
    texto   = tag_a.get_text(strip=True).lower()
    classes = " ".join(tag_a.get("class", [])).lower()
    gatilhos_texto = {"próximo", "proximo", "next", "seguinte", "›", "»", ">", "mais"}
    if any(p in texto for p in gatilhos_texto):
        return True
    if any(p in classes for p in ("next", "pagination", "pager", "proximo")):
        return True
    if re.search(r"(/page/\d+|[?&]p(?:age)?=\d+)$", href):
        return True
    base_path = urlparse(url_base).path.rstrip("/")
    if re.match(rf"^{re.escape(base_path)}/\d+$", urlparse(href).path.rstrip("/")):
        return True
    return False


def descobrir_links(
    url_base: str,
    filtro_path: Optional[str] = None,
    seguir_paginacao: bool = True,
) -> dict:
    """
    Varre a página de listagem (com suporte a paginação) e separa links em:
      - artigos : páginas do mesmo domínio que passam no filtro de path
      - videos  : links do YouTube
    Retorna {"artigos": [...], "videos": [...]}
    """
    dominio   = urlparse(url_base).netloc
    visitadas: Set[str] = set()
    artigos:   Set[str] = set()
    videos:    Set[str] = set()
    fila      = [url_base]

    print(f"\n🔍 Descobrindo links em: {url_base}")
    if filtro_path:
        print(f"   Filtro de path: '{filtro_path}'")

    while fila:
        url_atual = fila.pop(0)
        if url_atual in visitadas:
            continue
        visitadas.add(url_atual)

        soup = _get_soup(url_atual)
        if soup is None:
            continue

        for tag_a in soup.find_all("a", href=True):
            href = urljoin(url_atual, tag_a["href"]).split("#")[0].rstrip("/")
            if not href:
                continue

            if _eh_url_youtube(href):
                videos.add(href)
                continue

            parsed = urlparse(href)
            if parsed.netloc != dominio:
                continue
            if href == url_atual or href in visitadas:
                continue

            if filtro_path and filtro_path not in href:
                if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                    fila.append(href)
                continue

            # Página de paginação: enfileira para crawl mas NÃO indexa como artigo
            if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                fila.append(href)
                continue

            artigos.add(href)

        time.sleep(PAUSA_ENTRE_REQS)

    print(f"   ✓ {len(artigos)} artigo(s) | {len(videos)} vídeo(s) YouTube")
    return {"artigos": sorted(artigos), "videos": sorted(videos)}


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: Q&A COM CLASSIFICAÇÃO INTELIGENTE DE PERGUNTAS
# ══════════════════════════════════════════════════════════════════════════════

def _buscar_titulos() -> List[str]:
    """Retorna todos os títulos únicos presentes no ChromaDB."""
    _ensure_db()
    resultado = colecao.get(include=["metadatas"])
    if not resultado or not resultado.get("metadatas"):
        return []
    return list({m.get("titulo", "") for m in resultado["metadatas"] if m.get("titulo")})


def _buscar_titulos_candidatos(pergunta: str, k: int) -> List[str]:
    """
    Retorna até k títulos candidatos (únicos) usando uma busca vetorial rápida
    nos chunks já existentes. Evita enviar uma lista enorme de títulos ao Groq.
    """
    _ensure_db()
    if not pergunta.strip() or k <= 0:
        return []

    total_docs = colecao.count()
    if total_docs <= 0:
        return []

    n = min(max(5, k * 2), total_docs)  # pega mais chunks para formar k títulos únicos
    try:
        res = colecao.query(query_texts=[pergunta], n_results=n)
    except Exception:
        return []

    metas = (res.get("metadatas") or [[]])[0] or []
    out: List[str] = []
    seen: Set[str] = set()
    for m in metas:
        t = (m.get("titulo") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def _classificar_pergunta(pergunta: str, titulos: List[str]) -> dict:
    """
    Usa Groq para classificar a pergunta em:
      especifica   → filtra por fonte_alvo, busca 4 chunks
      geral        → busca ampla, 6 chunks
      comparativa  → busca ampla, 8 chunks
      fora_de_escopo → não busca, resposta direta
    """
    print("  🔀 Classificando consulta...")
    lista = "\n".join(f"- {t}" for t in titulos)
    raw = chamar_groq(
        sistema=PROMPT_CLASSIFICADOR_SISTEMA,
        usuario=(
            f"Títulos disponíveis:\n{lista}\n\n"
            f"Pergunta: {pergunta}\n\n"
            '{"tipo":"especifica"|"geral"|"comparativa"|"fora_de_escopo","fonte_alvo":string|null,"n_chunks":number,"raciocinio":string}'
        ),
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"tipo": "geral", "fonte_alvo": None, "n_chunks": 6, "raciocinio": "fallback"}


def _buscar_chunks(pergunta: str, classificacao: dict) -> List[dict]:
    _ensure_db()
    n    = max(1, classificacao.get("n_chunks", 5))
    tipo = classificacao.get("tipo", "geral")
    alvo = classificacao.get("fonte_alvo")

    # ChromaDB lança exceção se n_results > total de documentos na coleção
    total_docs = colecao.count()
    if total_docs == 0:
        return []
    n = min(n, total_docs)

    kwargs: dict = {"query_texts": [pergunta], "n_results": n}
    if tipo == "especifica" and alvo:
        kwargs["where"] = {"titulo": alvo}

    try:
        resultado = colecao.query(**kwargs)
    except Exception:
        # Se o filtro falhar (ex.: fonte_alvo não encontrada), tenta sem filtro
        resultado = colecao.query(query_texts=[pergunta], n_results=n)

    docs  = resultado.get("documents", [[]])[0]
    metas = resultado.get("metadatas", [[]])[0]
    return [
        {
            "texto":  doc,
            "titulo": meta.get("titulo", ""),
            "url":    meta.get("url", ""),
            "tipo":   meta.get("tipo", ""),
            "chunk":  meta.get("chunk_index", 0),
        }
        for doc, meta in zip(docs, metas)
    ]


def _deduplicar_fontes_consultadas(resposta: str) -> Tuple[str, dict]:
    """Deduplica itens da seção 'Fontes Consultadas' por URL (ou linha normalizada)."""
    texto = resposta or ""
    header_match = re.search(r"(?is)(?:^|\r?\n)((?:📚\s*)?\*\*fontes consultadas\*\*:?)\s*\r?\n", texto)
    if not header_match:
        return resposta, {"section_found": False, "reason": "header_not_found"}

    start = header_match.end()
    tail = texto[start:]
    end_match = re.search(r"(?is)\r?\n(?:---|⚠️\s*\*\*Aviso)", tail)
    end = (start + end_match.start()) if end_match else len(texto)
    header = texto[:start]
    body = texto[start:end]
    trailer = texto[end:]
    raw_lines = [ln for ln in body.splitlines() if ln.strip()]

    dedup_items: List[str] = []
    seen: Set[str] = set()
    for ln in raw_lines:
        clean = ln.strip()
        content = re.sub(r"^\s*(?:[-*]\s*|\d+\.\s*|\[\d+\]\s*)", "", clean)
        urls = re.findall(r"https?://[^\s\)\]]+", content, flags=re.I)
        key = (urls[0].strip().lower() if urls else re.sub(r"\s+", " ", content.lower()).strip())
        if not key or key in seen:
            continue
        seen.add(key)
        dedup_items.append(content)

    if not dedup_items:
        return resposta, {
            "section_found": True,
            "before_count": len(raw_lines),
            "after_count": 0,
        }

    new_body = "\n".join(f"- [{i}] {item}" for i, item in enumerate(dedup_items, 1))
    patched = header + new_body + trailer
    return patched, {
        "section_found": True,
        "end_found": bool(end_match),
        "before_count": len(raw_lines),
        "after_count": len(dedup_items),
        "removed_count": len(raw_lines) - len(dedup_items),
    }


def _forcar_secao_fontes_consultadas(resposta: str, fontes_unicas: List[dict]) -> Tuple[str, dict]:
    """Reescreve a seção de fontes com a lista única do pipeline."""
    texto = resposta or ""
    header_match = re.search(r"(?is)(?:^|\r?\n)((?:📚\s*)?\*\*fontes consultadas\*\*:?)\s*\r?\n", texto)
    if not header_match:
        return resposta, {"section_found": False, "reason": "header_not_found"}

    start = header_match.end()
    tail = texto[start:]
    end_match = re.search(r"(?is)\r?\n(?:---|⚠️\s*\*\*Aviso)", tail)
    end = (start + end_match.start()) if end_match else len(texto)
    prefix = texto[:start]
    suffix = texto[end:]

    linhas = []
    for i, f in enumerate(fontes_unicas, 1):
        tipo_label = "Vídeo" if f.get("tipo") == "video_youtube" else "Artigo"
        linhas.append(f"{i}. {tipo_label}: {f.get('titulo', '').strip()} — {f.get('url', '').strip()}")
    novo_corpo = "\n".join(linhas)

    return prefix + novo_corpo + suffix, {
        "section_found": True,
        "canonical_count": len(linhas),
        "end_found": bool(end_match),
    }


def pipeline_perguntar(pergunta: str) -> str:
    """
    Q&A com classificação inteligente:
    1. Classifica a pergunta (específica / geral / comparativa / fora de escopo)
    2. Busca chunks no ChromaDB (com ou sem filtro de fonte)
    3. Responde usando Groq com o contexto encontrado, citando fontes
    4. Anexa aviso médico obrigatório ao final de toda resposta
    """
    # Em modo econômico, evita enviar uma lista enorme de títulos ao Groq:
    # primeiro obtém candidatos via busca vetorial e só cai para a lista completa se necessário.
    titulos: List[str] = []
    if GROQ_ECONOMIA:
        titulos = _buscar_titulos_candidatos(pergunta, GROQ_CLASSIF_TITULOS_MAX)
    if not titulos:
        titulos = _buscar_titulos()
    if not titulos:
        return "⚠️  Base de conhecimento vazia. Indexe algum conteúdo primeiro (use --artigos, --video ou --local)."

    t0 = time.time()
    classificacao = _classificar_pergunta(pergunta, titulos)
    print(f"  Tipo: {classificacao['tipo']} | {classificacao.get('raciocinio', '')}")

    if classificacao["tipo"] == "fora_de_escopo":
        return (
            "⚠️  **Consulta fora do escopo médico-clínico da base do Dr. Ajuda.**\n\n"
            "O sistema não encontrou relação clínica suficiente entre essa pergunta e o conteúdo indexado.\n\n"
            "**Sugestões para reformular:**\n"
            "  • Especifique a condição, síndrome ou doença (ex.: \"critérios diagnósticos para X\")\n"
            "  • Inclua o contexto clínico (ex.: \"paciente com Y, como conduzir Z?\")\n"
            "  • Mencione o fármaco, dose, exame ou procedimento de interesse\n"
            "  • Use terminologia médica — o sistema reconhece CID, nomes genéricos e escalas clínicas\n\n"
            "Se a dúvida for genuinamente não-médica, este sistema não é o recurso adequado."
        )

    # Ajustes de economia pós-classificação (sem remover qualidade quando desligado)
    if GROQ_ECONOMIA:
        tipo = str(classificacao.get("tipo", "geral"))
        padrao = {"especifica": 4, "geral": 6, "comparativa": 8}.get(tipo, 6)
        alvo = {"especifica": 3, "geral": 4, "comparativa": 6}.get(tipo, max(3, min(6, padrao)))
        try:
            n_req = int(classificacao.get("n_chunks", padrao))
        except Exception:
            n_req = padrao
        classificacao["n_chunks"] = max(1, min(n_req, alvo))

    if GROQ_MAX_N_CHUNKS > 0:
        try:
            classificacao["n_chunks"] = max(1, min(int(classificacao.get("n_chunks", 6)), GROQ_MAX_N_CHUNKS))
        except Exception:
            classificacao["n_chunks"] = min(6, GROQ_MAX_N_CHUNKS)

    chunks = _buscar_chunks(pergunta, classificacao)
    if not chunks:
        return (
            "⚠️  **Nenhum conteúdo relevante encontrado** para essa consulta na base do Dr. Ajuda.\n\n"
            "Possíveis causas:\n"
            "  • O tema ainda não foi indexado — use `--artigos` ou `--video` para adicionar conteúdo\n"
            "  • A consulta usa termos muito diferentes dos presentes na base\n"
            "  • Tente reformular com sinônimos clínicos ou termos mais específicos\n\n"
            f"  Base atual: {len(titulos)} fonte(s) indexada(s)."
        )

    # Monta o contexto com limites (economiza tokens sem perder rastreabilidade)
    chunk_max = GROQ_CONTEXTO_CHUNK_MAX_CHARS
    total_max = GROQ_CONTEXTO_TOTAL_MAX_CHARS
    total_chars = 0

    blocos = []
    for i, c in enumerate(chunks):
        tipo_label = "Vídeo" if c["tipo"] == "video_youtube" else "Artigo"
        texto = c["texto"] or ""
        if GROQ_ECONOMIA and chunk_max > 0 and len(texto) > chunk_max:
            texto = texto[:chunk_max].rstrip() + "\n…"

        bloco = (
            f"[Fonte {i+1} — {tipo_label}]\n"
            f"Título: {c['titulo']}\n"
            f"URL: {c['url']}\n\n"
            f"{texto}"
        )

        if GROQ_ECONOMIA and total_max > 0:
            prox = (len(bloco) + (5 if blocos else 0))  # separador aproximado
            if total_chars + prox > total_max:
                break
            total_chars += prox

        blocos.append(
            bloco
        )
    contexto = "\n\n---\n".join(blocos)

    # Fontes únicas para _forcar_secao_fontes_consultadas
    fontes_vistas: dict = {}
    fontes_unicas: List[dict] = []
    for c in chunks:
        chave = (c["titulo"], c["url"])
        if chave not in fontes_vistas:
            fontes_vistas[chave] = len(fontes_vistas) + 1
            fontes_unicas.append({"titulo": c["titulo"], "url": c["url"], "tipo": c["tipo"]})

    mensagem_usuario = (
        f"CONSULTA: {pergunta}\n\n"
        f"TRECHOS DA BASE ({len(fontes_vistas)} fonte(s), {len(chunks)} trecho(s)):\n\n"
        f"{contexto}\n\n"
        "Sintetize com base exclusivamente nos trechos acima."
    )

    print("  💬 Gerando resposta com Groq...")
    # Roteia modelo por complexidade (economia sem perder qualidade quando necessário)
    modelo_resposta = MODELO_POTENTE
    if GROQ_ECONOMIA:
        tipo = str(classificacao.get("tipo", "geral"))
        pergunta_len = len((pergunta or "").strip())
        # simples e curta → modelo rápido; comparativa/específica longa → mantém potente
        if tipo in ("geral",) and pergunta_len <= 220:
            modelo_resposta = MODELO_RAPIDO

    resposta = chamar_groq(
        sistema=PROMPT_RESPOSTA_SISTEMA,
        usuario=mensagem_usuario,
        modelo=modelo_resposta,
    )
    elapsed = time.time() - t0
    print(f"  ✅ Resposta gerada em {elapsed:.1f}s")
    resposta, _ = _deduplicar_fontes_consultadas(resposta or "")
    resposta, _ = _forcar_secao_fontes_consultadas(resposta or "", fontes_unicas)
    # Evita duplicar o aviso caso o modelo o inclua por conta própria.
    if re.search(r"aviso\s+de\s+apoio\s+cl[ií]nico|n[aã]o\s+substitui\s+o\s+julgamento\s+cl[ií]nico", resposta, re.I):
        return resposta
    return resposta + AVISO_MEDICO


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL: CRAWLER
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_crawler(
    url_listagem: str,
    filtro_path: Optional[str] = None,
    sem_paginacao: bool = False,
) -> None:
    """Descobre todos os links da listagem e processa artigos + vídeos."""
    processadas = carregar_processadas()

    encontrados = descobrir_links(
        url_listagem,
        filtro_path=filtro_path,
        seguir_paginacao=not sem_paginacao,
    )
    artigos = [u for u in encontrados["artigos"] if u not in processadas]
    videos  = [u for u in encontrados["videos"]  if u not in processadas]
    pulados = (
        (len(encontrados["artigos"]) - len(artigos)) +
        (len(encontrados["videos"])  - len(videos))
    )

    print(f"\n📋 Novos: {len(artigos)} artigo(s) + {len(videos)} vídeo(s)"
          f"  |  {pulados} já processado(s)")

    if not artigos and not videos:
        print("✅ Tudo já indexado. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0
    total        = len(artigos) + len(videos)
    i            = 0

    for url in artigos:
        i += 1
        print(f"\n[{i}/{total}] 📄 {url}")
        n = processar_artigo(url, processadas, atualizar_catalogo=False)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    for url in videos:
        i += 1
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas, atualizar_catalogo=False)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            _pausa_entre_videos_no_lote()

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"  💾 Arquivos em: {PASTA_SAIDA}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS DA BASE
# ══════════════════════════════════════════════════════════════════════════════

def status_indice() -> None:
    _ensure_db()
    processadas   = carregar_processadas()
    arquivos_json = (
        [f for f in os.listdir(PASTA_SAIDA) if f.endswith(".json")]
        if os.path.exists(PASTA_SAIDA) else []
    )
    artigos_count = sum(1 for u in processadas if "youtube" not in u)
    videos_count  = sum(1 for u in processadas if "youtube" in u)
    dominios: dict = {}
    for u in processadas:
        if "youtube" not in u:
            d = urlparse(u).netloc
            dominios[d] = dominios.get(d, 0) + 1

    print(f"\n{'═'*55}")
    print("  📊 STATUS DA BASE DE CONHECIMENTO")
    print(f"{'═'*55}")
    print(f"  Pasta            : {PASTA_SAIDA}")
    print(f"  Banco vetorial   : {PASTA_CHROMA}")
    print(f"  Artigos indexados: {artigos_count}")
    print(f"  Vídeos YT        : {videos_count}")
    print(f"  Arquivos .json   : {len(arquivos_json)}")
    print(f"  Chunks no banco  : {colecao.count()}")
    print(f"  Modelo resposta  : {MODELO_POTENTE}")
    print(f"  Modelo rápido    : {MODELO_RAPIDO}")
    if dominios:
        print(f"\n  {'Domínio':<42} {'Artigos':>7}")
        print(f"  {'─'*50}")
        for d, n in sorted(dominios.items(), key=lambda x: -x[1]):
            print(f"  {d:<42} {n:>7}")
    catalogo = os.path.join(PASTA_SAIDA, "_CATALOGO.md")
    if os.path.exists(catalogo):
        import datetime
        ts = os.path.getmtime(catalogo)
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        print(f"\n  📋 Catálogo atualizado em: {dt}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MENU INTERATIVO
# ══════════════════════════════════════════════════════════════════════════════

def menu() -> None:
    L = 52  # largura do menu
    def _linha(c="═"): print(f"  {c * L}")
    def _titulo(t): print(f"  {t}")
    def _item(n, icone, label, detalhe=""):
        detalhe_fmt = f"  {detalhe}" if detalhe else ""
        print(f"    [{n}] {icone}  {label}{detalhe_fmt}")
    def _sep(): print(f"  {'─' * L}")

    while True:
        print()
        _linha()
        _titulo("🩺  APOIO CLÍNICO DR. AJUDA  |  Base de Conhecimento Médico")
        _linha()
        print()
        _titulo("  INDEXAR CONTEÚDO")
        _sep()
        _item("1", "📄", "Artigos / posts médicos por URL",  "cole várias URLs: linhas, espaço ou vírgula")
        _item("3", "🎬", "Vídeo(s) do YouTube (Dr. Ajuda)", "um ou vários (URL, ID ou arquivo .txt)")
        _item("4", "📂", "Arquivos locais",                  "processa .txt + .json da pasta de saída")
        _item("5", "🔍", "Buscar arquivo local",             "localiza por nome ou tema")
        _item("6", "🔄", "Forçar reindexação",               "reprocessa URL já indexada")
        print()
        _titulo("  CONSULTA CLÍNICA")
        _sep()
        _item("7", "💬", "Consulta (diagnóstico / conduta / laudo)")
        _item("8", "📊", "Status da base de conhecimento")
        print()
        _linha()
        _item("0", "🚪", "Sair")
        _linha()
        print()
        opcao = input("  Opção: ").strip()
        print()

        if opcao == "0":
            print("  Até logo!")
            break

        # ── 1. Artigos por URL (lote) ─────────────────────────────────────
        elif opcao == "1":
            print("  Cole as URLs dos artigos em qualquer formato:")
            print("    • uma por linha, ou várias na mesma linha (espaço / vírgula), ou bloco com links;")
            print("    • ou caminho de um .txt com uma URL por linha.")
            print("  ── linha vazia para confirmar ──")
            entradas: List[str] = []
            while True:
                linha = input("  > ").strip()
                if not linha:
                    break
                entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            urls_lote = _resolver_entradas_para_urls_artigos(entradas, run_id="menu-artigos")

            try:
                pipeline_artigos_em_lote(urls_lote)
            except Exception as e:
                print(f"  ❌ Erro ao processar lote de artigos: {e}")

        # ── 3. Vídeo(s) do YouTube ────────────────────────────────────────
        elif opcao == "3" or _parece_youtube(opcao):
            # Permite colar a URL/ID direto como opção do menu
            if _parece_youtube(opcao):
                entradas = [opcao]
            else:
                print("  Cole links ou IDs do YouTube (linha única com vários, ou uma por linha,")
                print("  ou caminho de um .txt com uma entrada por linha).")
                print("  ── linha vazia para confirmar ──")
                entradas = []
                while True:
                    linha = input("  > ").strip()
                    if not linha:
                        break
                    entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            # Se for um único caminho de arquivo, lê o arquivo
            urls_lote = _resolver_entradas_para_urls_video(entradas, run_id="menu-video")

            if len(urls_lote) == 1:
                url = _normalizar_url_yt(urls_lote[0])
                processadas = carregar_processadas()
                n = processar_video(url, processadas)
                _exibir_resultado_avulso(n)
                print(f"  📦 Total de chunks no banco: {colecao.count()}")
            else:
                pipeline_videos_em_lote(urls_lote)

        # ── 4. Arquivos locais ────────────────────────────────────────────
        elif opcao == "4":
            pipeline_processar_pasta()

        # ── 5. Buscar arquivo local ───────────────────────────────────────
        elif opcao == "5":
            pipeline_indexar_video_por_nome()

        # ── 6. Forçar reindexação ─────────────────────────────────────────
        elif opcao == "6":
            url = input("  URL para reindexar: ").strip()
            processadas = carregar_processadas()
            processadas.discard(url)
            with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
                json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
            url = _normalizar_url_yt(url)
            if _eh_url_youtube(url):
                n = processar_video(url, processadas)
            elif _eh_url(url):
                n = processar_artigo(url, processadas)
            else:
                print("  ❌ URL inválida.")
                continue
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")

        # ── 7. Consulta clínica ───────────────────────────────────────────
        elif opcao == "7":
            print("  Descreva o caso, condição, medicamento, exame ou dúvida clínica:")
            pergunta = input("  > ").strip()
            if not pergunta:
                continue
            print(f"\n  🩺 Consultando base do Dr. Ajuda...")
            print(f"  {'─' * L}")
            print(pipeline_perguntar(pergunta))

        # ── 8. Status da base ─────────────────────────────────────────────
        elif opcao == "8":
            status_indice()

        else:
            print("  ❌ Opção inválida. Tente novamente.")


def _intro_conversa() -> None:
    L = 55
    print(f"\n  {'═' * L}")
    print("  🩺  SUP — Apoio Clínico Dr. Ajuda")
    print(f"  {'─' * L}")
    print("  Fale em linguagem natural. Exemplos:")
    print("    • consultar: critérios de síndrome metabólica")
    print("    • indexar https://youtu.be/ID")
    print("    • indexar artigos https://site.com/post1 https://site.com/post2")
    print("    • crawlear o blog https://site.com/blog com filtro /posts/")
    print("    • status | reindexar https://... | processar locais | buscar local")
    print(f"  {'─' * L}")
    print("  'ajuda' repete este resumo  ·  'sair' encerra")
    print(f"  {'═' * L}\n")


def _rotear_mensagem_conversa(msg: str) -> dict:
    msg = (msg or "").strip()
    if not msg:
        return {"acao": "perguntar_clarificacao", "pergunta": "O que você gostaria de fazer?"}
    low = msg.lower().strip()
    if low in ("sair", "exit", "quit", "q"):
        return {"acao": "sair"}
    if low in ("ajuda", "help", "h", "?"):
        return {"acao": "ajuda"}

    # Fallback heurístico quando IA não estiver disponível
    if not GROQ_ENABLED:
        if re.search(r"\bstatus\b|\bestat[ií]stic", low):
            return {"acao": "status"}
        if re.search(r"\breindex", low):
            m = re.search(r"(https?://\S+)", msg)
            return {"acao": "reindexar", "url": (m.group(1) if m else "")}
        if _parece_youtube(msg):
            return {"acao": "videos_lote", "entradas": [msg]}
        if re.search(r"https?://", msg) and not _parece_youtube(msg):
            return {"acao": "artigos_lote", "entradas": [msg]}
        if re.search(r"\bcrawl\b|\bvarrer\b|\bcrawlear\b", low):
            m = re.search(r"(https?://\S+)", msg)
            return {"acao": "crawl_site", "url_listagem": (m.group(1) if m else ""), "filtro_path": None, "sem_paginacao": False}
        if re.search(r"\blocal\b|\barquivos\b", low):
            return {"acao": "processar_local"}
        return {"acao": "perguntar_clarificacao", "pergunta": "Configure o GROQ_API_KEY para consulta clínica. Para indexação, cole URLs/IDs."}

    try:
        raw = chamar_groq(
            sistema=PROMPT_ROUTER_CONVERSA_SISTEMA,
            usuario=f"Mensagem do usuário:\n{msg}\n",
            modelo=MODELO_RAPIDO,
            json_mode=True,
        )
        d = json.loads(raw)
        if isinstance(d, dict) and d.get("acao"):
            return d
    except Exception:
        pass

    return {"acao": "consultar", "pergunta_clinica": msg}


def conversa() -> None:
    _intro_conversa()
    while True:
        try:
            msg = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Até logo!")
            return

        if not msg:
            continue

        r = _rotear_mensagem_conversa(msg)
        acao = r.get("acao")

        if acao == "sair":
            print("  Até logo!")
            return

        if acao == "ajuda":
            _intro_conversa()
            continue

        if acao == "status":
            status_indice()
            continue

        if acao == "consultar":
            pergunta = (r.get("pergunta_clinica") or "").strip()
            if not pergunta:
                print("  Qual é a sua dúvida clínica?")
                continue
            print(f"\n  🩺 Consultando base do Dr. Ajuda...\n  {'─'*55}")
            print(pipeline_perguntar(pergunta))
            continue

        if acao == "processar_local":
            pipeline_processar_pasta()
            continue

        if acao == "buscar_local":
            consulta = (r.get("consulta") or "").strip()
            if consulta:
                # reutiliza o pipeline existente (interativo) como fallback;
                # quando há consulta, tentamos passar pela variável global via prompt ao usuário.
                print("  Vou abrir a busca local. Se necessário, cole/ajuste a descrição.")
            pipeline_indexar_video_por_nome()
            continue

        if acao == "artigos_lote":
            entradas = r.get("entradas") or []
            if isinstance(entradas, str):
                entradas = [entradas]
            if not entradas:
                entradas = [msg]
            urls = _resolver_entradas_para_urls_artigos(entradas, run_id="conversa-artigos")
            if not urls:
                print("  Não encontrei URLs de artigos válidas nessa mensagem. Cole links http(s) ou um arquivo .txt.")
                continue
            pipeline_artigos_em_lote(urls)
            continue

        if acao == "videos_lote":
            entradas = r.get("entradas") or []
            if isinstance(entradas, str):
                entradas = [entradas]
            if not entradas:
                entradas = [msg]
            urls = _resolver_entradas_para_urls_video(entradas, run_id="conversa-video")
            if not urls:
                print("  Não encontrei URLs/IDs de YouTube válidos nessa mensagem. Cole um link/ID ou um arquivo .txt.")
                continue
            if len(urls) == 1:
                url = _normalizar_url_yt(urls[0])
                processadas = carregar_processadas()
                print(f"\n  🎬 Indexando vídeo: {url}")
                n = processar_video(url, processadas)
                _exibir_resultado_avulso(n)
                print(f"  📦 Total de chunks no banco: {colecao.count()}")
            else:
                pipeline_videos_em_lote(urls)
            continue

        if acao == "crawl_site":
            url_listagem = (r.get("url_listagem") or "").strip()
            if not url_listagem:
                print("  Qual é a URL base/listagem do site para crawl?")
                continue
            filtro_path = r.get("filtro_path")
            sem_paginacao = bool(r.get("sem_paginacao", False))
            pipeline_crawler(url_listagem, filtro_path=filtro_path, sem_paginacao=sem_paginacao)
            continue

        if acao == "reindexar":
            url = (r.get("url") or "").strip()
            if not url:
                print("  Qual URL você quer reindexar?")
                continue
            url = _normalizar_url_yt(url)
            processadas = carregar_processadas()
            processadas.discard(url)
            with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
                json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
            print(f"\n  🔄 Reindexando: {url}")
            if _eh_url_youtube(url):
                n = processar_video(url, processadas)
            elif _eh_url(url):
                n = processar_artigo(url, processadas)
            else:
                print("  ❌ URL inválida.")
                continue
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")
            continue

        if acao == "perguntar_clarificacao":
            pergunta = (r.get("pergunta") or "Como posso ajudar?").strip()
            print(f"  {pergunta}")
            continue

        print("  Não entendi. Digite 'ajuda' para ver exemplos.")


def _exibir_resultado_avulso(n: int) -> None:
    if n < 0:
        print("   ❌ Falha ao processar.")
    elif n == 0:
        print("   ℹ️  Já estava indexado.")
    else:
        print(f"   ✅ {n} chunks indexados.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _main(argv: list) -> None:
    args = argv[1:]

    # ── Sem argumentos: modo conversa (minimalista) ───────────────────────
    if not args:
        conversa()
        return

    # ── --ajuda / --help ────────────────────────────────────────────────
    if args[0] in ("-h", "--help", "--ajuda"):
        print(__doc__)
        return

    # ── --menu ───────────────────────────────────────────────────────────
    if args[0] in ("--menu", "-m"):
        menu()
        return

    # ── --status ─────────────────────────────────────────────────────────
    if args[0] == "--status":
        status_indice()
        return

    # ── --pergunta ───────────────────────────────────────────────────────
    if args[0] == "--pergunta":
        pergunta = " ".join(args[1:]).strip()
        if not pergunta:
            sys.exit('  Use: --pergunta "sua consulta clínica aqui"')
        print(f"\n  🩺 Consultando base do Dr. Ajuda...\n  {'─'*55}")
        print(pipeline_perguntar(pergunta))
        return

    # ── --artigo / --artigos / URL direta (somente links informados) ─────
    if args[0] in ("--artigo", "--artigos", "--crawl", "--site") or (args and _eh_url(args[0]) and not _eh_url_youtube(args[0])):
        entradas = args[1:] if args[0] in ("--artigo", "--artigos", "--crawl", "--site") else args
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --artigos "https://site.com/post-1" "https://site.com/post-2"\n'
                "    --artigos lista.txt"
            )

        urls = _resolver_entradas_para_urls_artigos(entradas, run_id="cli-artigos")
        if not urls:
            sys.exit("  Nenhuma URL de artigo válida encontrada (use http(s):// ou arquivo .txt).")

        pipeline_artigos_em_lote(urls)
        return

    # ── --video  (único, vários ou arquivo .txt) ──────────────────────────
    if args[0] in ("--video", "--videos"):
        entradas = args[1:]
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --video "https://youtube.com/watch?v=ID"\n'
                '    --video "url1" "url2" "url3"\n'
                "    --video lista.txt"
            )
        # Se for um único argumento que é um arquivo existente → lê o arquivo
        urls = _resolver_entradas_para_urls_video(entradas, run_id="cli-video")
        if not urls:
            sys.exit("  Nenhuma URL/ID de YouTube válida encontrada.")

        if len(urls) == 1:
            url = _normalizar_url_yt(urls[0])
            processadas = carregar_processadas()
            print(f"\n  🎬 Indexando vídeo: {url}")
            n = processar_video(url, processadas)
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")
        else:
            pipeline_videos_em_lote(urls)
        return

    # ── --local ───────────────────────────────────────────────────────────
    if args[0] in ("--local", "--pasta", "-p"):
        pipeline_processar_pasta()
        return

    # ── --buscar-local ────────────────────────────────────────────────────
    if args[0] in ("--buscar-local", "--indexar-video"):
        pipeline_indexar_video_por_nome()
        return

    # ── --reindexar ───────────────────────────────────────────────────────
    if args[0] == "--reindexar":
        if len(args) < 2:
            sys.exit('  Use: --reindexar "https://..."')
        url = _normalizar_url_yt(args[1])
        processadas = carregar_processadas()
        processadas.discard(url)
        with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
            json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
        print(f"\n  🔄 Reindexando: {url}")
        if _eh_url_youtube(url):
            n = processar_video(url, processadas)
        elif _eh_url(url):
            n = processar_artigo(url, processadas)
        else:
            sys.exit(f"  ❌ URL inválida: {url}")
        _exibir_resultado_avulso(n)
        print(f"  📦 Total de chunks no banco: {colecao.count()}")
        return

    # ── Argumento não reconhecido ─────────────────────────────────────────
    sys.exit(
        f"  ❌ Argumento não reconhecido: '{args[0]}'\n"
        f"  Use --ajuda para ver todas as opções."
    )


if __name__ == "__main__":
    _main(sys.argv)