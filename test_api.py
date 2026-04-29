#!/usr/bin/env python3
"""
test_api.py — Smoke tests do sistema SUP antes do deploy.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO:
  # Com servidor rodando localmente:
  WEB_TOKEN=dev-token-local python test_api.py

  # Contra servidor remoto:
  BASE_URL=https://seu-app.ondigitalocean.app WEB_TOKEN=seu_token python test_api.py
"""

import os
import sys
import time
from typing import Optional

import requests

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
TOKEN_VALIDO = os.getenv("WEB_TOKEN", "").strip()
TIMEOUT_CURTO = 10   # segundos para endpoints rápidos
TIMEOUT_LONGO = 120  # segundos para consulta clínica real

# ── Estado dos testes ─────────────────────────────────────────────────────────
resultados: list[tuple[str, bool, str]] = []  # (nome, passou, detalhe)
t_inicio_total = time.time()


def _ok(nome: str, detalhe: str = "") -> None:
    resultados.append((nome, True, detalhe))
    print(f"  ✅ {nome}" + (f" — {detalhe}" if detalhe else ""))


def _fail(nome: str, detalhe: str = "") -> None:
    resultados.append((nome, False, detalhe))
    print(f"  ❌ {nome}" + (f" — {detalhe}" if detalhe else ""))


def _secao(titulo: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {titulo}")
    print(f"{'─'*55}")


def _get(path: str, token: Optional[str] = None, timeout: int = TIMEOUT_CURTO) -> requests.Response:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(f"{BASE_URL}{path}", headers=headers, timeout=timeout)


def _post(path: str, json_body: dict, token: Optional[str] = None, timeout: int = TIMEOUT_LONGO) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.post(f"{BASE_URL}{path}", json=json_body, headers=headers, timeout=timeout)


# ══════════════════════════════════════════════════════════════════════════════
#  PRÉ-VERIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def verificar_prerequisitos() -> None:
    if not TOKEN_VALIDO:
        print("\n⚠️  WEB_TOKEN não configurado. Defina a variável de ambiente WEB_TOKEN.")
        sys.exit(1)

    print(f"\n🔗 Servidor: {BASE_URL}")
    print(f"🔑 Token configurado: {'sim (******' + TOKEN_VALIDO[-4:] + ')' if TOKEN_VALIDO else 'NÃO'}")


# ══════════════════════════════════════════════════════════════════════════════
#  TESTES
# ══════════════════════════════════════════════════════════════════════════════

def test_health() -> None:
    _secao("1. GET /health — sem autenticação")
    try:
        t0 = time.time()
        r = _get("/health")
        elapsed_ms = (time.time() - t0) * 1000

        if r.status_code == 200:
            _ok("Status HTTP 200", f"{elapsed_ms:.0f}ms")
        else:
            _fail("Status HTTP 200", f"recebido {r.status_code}")

        data = r.json()
        if data.get("status") == "ok":
            _ok("Campo status='ok'")
        else:
            _fail("Campo status='ok'", f"recebido: {data}")

        if elapsed_ms < 500:
            _ok("Tempo de resposta < 500ms", f"{elapsed_ms:.0f}ms")
        else:
            _fail("Tempo de resposta < 500ms", f"{elapsed_ms:.0f}ms (lento para liveness probe)")

    except requests.exceptions.ConnectionError:
        _fail("Conexão com servidor", f"Servidor indisponível em {BASE_URL}")
        print("\n⛔ Servidor não encontrado. Execute o servidor antes dos testes.")
        sys.exit(1)


def test_status_com_token() -> None:
    _secao("2. GET /status — com token válido")
    try:
        r = _get("/status", token=TOKEN_VALIDO)
        if r.status_code == 200:
            _ok("Status HTTP 200")
        else:
            _fail("Status HTTP 200", f"recebido {r.status_code}: {r.text[:200]}")
            return

        data = r.json()
        if "banco" in data:
            _ok("Campo 'banco' presente")
            chunks = data["banco"].get("chunks_totais", -1)
            if chunks > 0:
                _ok(f"Base não vazia", f"{chunks} chunks")
            else:
                _fail("Base não vazia", f"chunks_totais={chunks} — indexe conteúdo primeiro")
        else:
            _fail("Campo 'banco' presente", str(data))

        if "modelos_groq" in data:
            _ok("Campo 'modelos_groq' presente", str(data["modelos_groq"]))
        else:
            _fail("Campo 'modelos_groq' presente")

    except Exception as e:
        _fail("GET /status", str(e))


def test_status_sem_token() -> None:
    _secao("3. GET /status — sem autenticação (deve retornar 401)")
    try:
        r = _get("/status", token=None)
        if r.status_code == 401:
            _ok("Status HTTP 401 (acesso negado sem token)")
        else:
            _fail("Status HTTP 401", f"recebido {r.status_code} — endpoint sem proteção!")
    except Exception as e:
        _fail("GET /status sem token", str(e))


def test_consultar_pergunta_vazia() -> None:
    _secao("4. POST /consultar — pergunta vazia (deve retornar 422)")
    try:
        r = _post("/consultar", {"pergunta": ""}, token=TOKEN_VALIDO, timeout=TIMEOUT_CURTO)
        if r.status_code == 422:
            _ok("Status HTTP 422 (validação correta)")
        else:
            _fail("Status HTTP 422", f"recebido {r.status_code}: {r.text[:200]}")
    except Exception as e:
        _fail("POST /consultar vazia", str(e))


def test_consultar_sem_token() -> None:
    _secao("5. POST /consultar — sem autenticação (deve retornar 401)")
    try:
        r = _post("/consultar", {"pergunta": "teste"}, token=None, timeout=TIMEOUT_CURTO)
        if r.status_code == 401:
            _ok("Status HTTP 401 (acesso negado sem token)")
        else:
            _fail("Status HTTP 401", f"recebido {r.status_code}")
    except Exception as e:
        _fail("POST /consultar sem token", str(e))


def test_consultar_clinica_real() -> None:
    _secao("6. POST /consultar — consulta clínica real (pode levar até 90s)")
    PERGUNTA_TESTE = (
        "Quais os principais critérios diagnósticos para síndrome metabólica "
        "e qual a primeira linha de tratamento farmacológico?"
    )
    print(f"  📋 Pergunta: {PERGUNTA_TESTE[:80]}...")
    try:
        t0 = time.time()
        r = _post("/consultar", {"pergunta": PERGUNTA_TESTE}, token=TOKEN_VALIDO)
        elapsed = time.time() - t0

        if r.status_code == 200:
            _ok("Status HTTP 200", f"em {elapsed:.1f}s")
        else:
            _fail("Status HTTP 200", f"recebido {r.status_code}: {r.text[:300]}")
            return

        data = r.json()

        if "resposta" in data and len(data["resposta"]) > 100:
            _ok("Campo 'resposta' não vazio", f"{len(data['resposta'])} chars")
        else:
            _fail("Campo 'resposta' não vazio", str(data.get("resposta", ""))[:100])

        # Verifica se o aviso médico está presente
        aviso_presente = any(
            termo in data.get("resposta", "").lower()
            for termo in ["aviso de apoio clínico", "não substitui", "julgamento clínico"]
        )
        if aviso_presente:
            _ok("Aviso médico presente na resposta")
        else:
            _fail("Aviso médico presente na resposta", "aviso obrigatório não encontrado")

        if "modelo_usado" in data:
            _ok("Campo 'modelo_usado'", data["modelo_usado"])
        else:
            _fail("Campo 'modelo_usado' ausente")

        if "tempo_resposta_s" in data:
            _ok("Campo 'tempo_resposta_s'", f"{data['tempo_resposta_s']}s")
        else:
            _fail("Campo 'tempo_resposta_s' ausente")

    except requests.exceptions.Timeout:
        _fail("Consulta clínica", "Timeout após 120s — servidor pode estar sobrecarregado")
    except Exception as e:
        _fail("Consulta clínica", str(e))


def test_rate_limit() -> None:
    _secao("7. Rate limiting — 11 requisições rápidas (a 11ª deve retornar 429)")
    print("  ⏳ Enviando 11 requisições rápidas...")
    respostas_429 = 0
    respostas_outras = []

    for i in range(11):
        try:
            r = _post(
                "/consultar",
                {"pergunta": f"teste rate limit {i}"},
                token=TOKEN_VALIDO,
                timeout=5,
            )
            if r.status_code == 429:
                respostas_429 += 1
            elif r.status_code not in (200, 422, 503):
                respostas_outras.append(r.status_code)
        except requests.exceptions.Timeout:
            pass  # timeout esperado em pipeline real durante rate limit test

    if respostas_429 >= 1:
        _ok("Rate limit ativado (HTTP 429)", f"{respostas_429} resposta(s) 429")
    else:
        _fail(
            "Rate limit não ativado",
            "Nenhum HTTP 429 recebido — verifique configuração slowapi"
        )


def test_interface_web() -> None:
    _secao("8. GET / — interface HTML")
    try:
        r = _get("/", timeout=TIMEOUT_CURTO)
        if r.status_code == 200:
            _ok("Status HTTP 200")
        else:
            _fail("Status HTTP 200", f"recebido {r.status_code}")
            return

        if "text/html" in r.headers.get("content-type", ""):
            _ok("Content-Type text/html")
        else:
            _fail("Content-Type text/html", r.headers.get("content-type", ""))

        if "SUP" in r.text or "Dr. Ajuda" in r.text:
            _ok("HTML contém conteúdo da interface")
        else:
            _fail("HTML contém conteúdo esperado", "texto 'SUP' ou 'Dr. Ajuda' não encontrado")

    except Exception as e:
        _fail("GET /", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumo() -> None:
    total = time.time() - t_inicio_total
    passou = sum(1 for _, ok, _ in resultados if ok)
    falhou = sum(1 for _, ok, _ in resultados if not ok)

    print(f"\n{'═'*55}")
    print(f"  📊 RESUMO DOS TESTES")
    print(f"{'═'*55}")
    print(f"  ✅ Passou  : {passou}")
    print(f"  ❌ Falhou  : {falhou}")
    print(f"  ⏱  Total   : {total:.1f}s")
    print(f"{'═'*55}")

    if falhou > 0:
        print("\n  Testes com falha:")
        for nome, ok, detalhe in resultados:
            if not ok:
                print(f"    ❌ {nome}: {detalhe}")
        sys.exit(1)
    else:
        print("\n  🎉 Todos os testes passaram! Pronto para deploy.\n")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  SUP — Smoke Tests pré-deploy")
    print("═"*55)

    verificar_prerequisitos()

    test_health()
    test_status_com_token()
    test_status_sem_token()
    test_consultar_pergunta_vazia()
    test_consultar_sem_token()
    test_consultar_clinica_real()
    test_rate_limit()
    test_interface_web()

    imprimir_resumo()
