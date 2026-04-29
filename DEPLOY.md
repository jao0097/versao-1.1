# DEPLOY.md — Guia de Deploy: SUP na DigitalOcean

> ⚠️ **Sistema de uso restrito**: este sistema deve ser acessado somente por
> profissionais de saúde habilitados. Garanta que a URL nunca seja pública
> e que o `WEB_TOKEN` seja forte e mantido em segredo.

---

## Pré-requisitos

| Requisito | Observação |
|---|---|
| Docker instalado localmente | Para build e teste local antes do deploy |
| Conta DigitalOcean | Ative o GitHub Student Pack em [education.github.com/pack](https://education.github.com/pack) |
| Crédito $200 ativado | Confira em *Billing → Credits* no painel DO |
| Banco ChromaDB populado | `~/Documentos/rag_base/chroma_db` com conteúdo indexado |
| `GROQ_API_KEY` válida | Obtenha em [console.groq.com](https://console.groq.com) |

---

## Estrutura de arquivos esperada

```
seu-repositorio/
├── super.py                # Sistema RAG original (sem modificações)
├── web_api.py              # API FastAPI
├── templates/
│   └── index.html          # Interface web
├── requirements_web.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── test_api.py
└── DEPLOY.md
```

> **Não commite** o arquivo `.env` nem o banco ChromaDB no Git.
> Adicione ambos ao `.gitignore`:
> ```
> .env
> rag_base/
> chroma_db/
> ```

---

## Teste local antes do deploy

```bash
# 1. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env e preencha GROQ_API_KEY e WEB_TOKEN

# 2. Suba o container apontando para seu banco existente
docker compose up --build

# 3. Verifique o health check
curl http://localhost:8000/health

# 4. Execute os smoke tests
WEB_TOKEN=seu_token python test_api.py

# 5. Acesse a interface
open http://localhost:8000
```

---

## Opção A — DigitalOcean App Platform *(recomendado)*

**Custo estimado**: ~$12/mês (plano Basic com 1 GB RAM).
Com $200 de crédito, você tem aproximadamente **16 meses** de uso gratuito.

> ⚠️ **Limitação importante**: o App Platform não suporta volumes persistentes
> de leitura para o banco ChromaDB local. Use a Opção B (Droplet) se o banco
> já estiver populado no seu computador e você quiser enviá-lo para produção.
> A App Platform funciona melhor se o banco estiver em um bucket de Object
> Storage (DigitalOcean Spaces) — o que exige modificar o `super.py` para
> usar `chromadb.HttpClient` em vez de `PersistentClient`.

Se quiser testar mesmo assim com a App Platform:

### Passo a passo

1. **Push do código para um repositório GitHub** (privado recomendado)

2. **Criar app no painel DigitalOcean**
   - Acesse [cloud.digitalocean.com/apps](https://cloud.digitalocean.com/apps)
   - Clique em **Create App**
   - Conecte seu repositório GitHub
   - Selecione a branch principal

3. **Configurar o serviço**
   - Tipo: **Web Service**
   - Build command: *(deixe vazio — usa o Dockerfile automaticamente)*
   - Run command: *(deixe vazio — usa o CMD do Dockerfile)*
   - Porta HTTP: `8000`

4. **Variáveis de ambiente** (em *Settings → Environment Variables*)
   ```
   GROQ_API_KEY = <sua_chave_groq>
   WEB_TOKEN    = <token_seguro_32chars>
   ```

5. **Health check**
   - Path: `/health`
   - Protocolo: HTTP
   - Porta: 8000

6. **Deploy**
   - Clique em **Deploy to Production**
   - O build leva ~10 min (download do modelo embedding ~400 MB)

7. **HTTPS**
   - Automático no App Platform — nenhuma configuração adicional

---

## Opção B — Droplet Ubuntu 22.04 *(mais controle)*

**Custo**: $12/mês (2 GB RAM, 1 vCPU).
Com $200 de crédito: ~16 meses.

> **Por que 2 GB?** O modelo `paraphrase-multilingual-MiniLM-L12-v2`
> carrega ~400 MB em RAM. Com o ChromaDB e o Uvicorn, o sistema usa
> ~900 MB–1,2 GB em operação normal. Com 1 GB de RAM, o container
> pode ser morto pelo OOM killer.

### 1. Criar o Droplet

- SO: **Ubuntu 22.04 LTS**
- Plano: **Basic — $12/mês (2 GB RAM)**
- Região: São Paulo (`nyc1` ou `sfo3` se SP não disponível)
- Autenticação: **Chave SSH** (mais seguro que senha)
- Habilitar: **Monitoring** (gratuito)

### 2. Instalar Docker no servidor

```bash
# Conecte via SSH
ssh root@<IP_DO_DROPLET>

# Instale Docker via snap (mais simples no Ubuntu 22.04)
snap install docker

# Verifique
docker --version
docker compose version
```

### 3. Enviar o banco ChromaDB para o servidor

```bash
# No seu computador local — ajuste os caminhos conforme necessário
rsync -avz --progress \
  ~/Documentos/rag_base/ \
  root@<IP_DO_DROPLET>:/data/rag_base/
```

> ⚠️ O banco pode ter vários GB. Use uma conexão estável.
> Para bancos grandes (>5 GB), prefira compactar antes:
> ```bash
> tar -czf rag_base.tar.gz ~/Documentos/rag_base/
> scp rag_base.tar.gz root@<IP>:/data/
> ssh root@<IP> "cd /data && tar -xzf rag_base.tar.gz"
> ```

### 4. Clonar o código e configurar

```bash
# No servidor
git clone https://github.com/seu-usuario/seu-repositorio.git /app/sup
cd /app/sup

# Copie e edite o .env
cp .env.example .env
nano .env
# → Preencha GROQ_API_KEY, WEB_TOKEN e ajuste os caminhos:
#   RAG_PASTA_SAIDA=/data/rag_base
#   RAG_PASTA_CHROMA=/data/rag_base/chroma_db
```

### 5. Subir o container

```bash
cd /app/sup

# Primeiro build (lento — baixa modelo ~400 MB)
docker compose up -d --build

# Acompanhe os logs de inicialização
docker compose logs -f

# Verifique o health check
curl http://localhost:8000/health
```

### 6. Nginx como reverse proxy com HTTPS

```bash
# Instale Nginx e Certbot
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx

# Crie a configuração do Nginx
cat > /etc/nginx/sites-available/sup << 'EOF'
server {
    listen 80;
    server_name seu-dominio.com;  # substitua pelo seu domínio ou IP

    # Redireciona HTTP → HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name seu-dominio.com;

    # Certificados gerenciados pelo Certbot (gerados no passo seguinte)
    ssl_certificate     /etc/letsencrypt/live/seu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/seu-dominio.com/privkey.pem;

    # Configurações SSL seguras
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Timeout estendido para o pipeline clínico (pode levar até 90s com retry Groq)
    proxy_read_timeout 120s;
    proxy_connect_timeout 10s;

    # Cabeçalhos de segurança
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Referrer-Policy no-referrer;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE: desativa buffer para streaming funcionar
        proxy_buffering off;
        proxy_cache off;
    }
}
EOF

# Ative o site
ln -s /etc/nginx/sites-available/sup /etc/nginx/sites-enabled/
nginx -t  # verifica sintaxe
systemctl reload nginx

# Emita o certificado SSL (substitua pelo seu domínio)
certbot --nginx -d seu-dominio.com --non-interactive --agree-tos -m seu@email.com
```

> Se não tiver domínio, use o IP diretamente (sem HTTPS). Nesse caso,
> **avise os médicos** que o token trafega em claro — risco alto para
> dados clínicos. Domínios `.com.br` custam ~R$40/ano no registro.br.

### 7. Firewall

```bash
# Permite apenas SSH, HTTP e HTTPS. Bloqueia porta 8000 diretamente.
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp   # acesso somente via Nginx, nunca direto
ufw enable
```

### 8. Auto-restart e atualizações

```bash
# O docker-compose.yml já tem restart: unless-stopped
# Para atualizar o código:
cd /app/sup
git pull
docker compose up -d --build
```

---

## ⚠️ HTTPS é obrigatório

Rodar sem HTTPS expõe em texto claro:

- O `WEB_TOKEN` (permite que qualquer um na rede intercepte e acesse o sistema)
- O conteúdo de todas as consultas médicas (violação de privacidade)

**Para a App Platform**: HTTPS é automático.
**Para Droplet**: use o Nginx + Certbot conforme descrito acima.

---

## Limitação crítica: 1 worker Uvicorn

O ChromaDB com `PersistentClient` (banco em disco) usa file-lock exclusivo.
Rodar com `--workers 2` ou mais causa **corrupção silenciosa do banco vetorial**,
pois múltiplos processos tentam escrever no mesmo arquivo simultaneamente.

**Nunca altere** o parâmetro `--workers 1` no CMD do Dockerfile.

Para suportar mais usuários simultâneos no futuro:
- Migre o ChromaDB para modo servidor (`chromadb.HttpClient`) com
  `chroma run --path /data/chroma_db` em container separado
- Aí será possível usar múltiplos workers Uvicorn com segurança

---

## Estimativa de custos com $200 de crédito

| Recurso | Plano | Custo/mês | Meses com $200 |
|---|---|---|---|
| Droplet 2 GB RAM | Basic | $12 | ~16 meses |
| App Platform Basic | 512 MB RAM | $5 | ~40 meses* |
| Bandwidth | 2 TB inclusos | $0 | — |

*App Platform com 512 MB pode ter OOM com o modelo de embedding.
Prefira 1 GB RAM ($10/mês) ou Droplet 2 GB para garantir estabilidade.

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| Container reinicia em loop | OOM (pouca RAM) | Use Droplet 2 GB |
| `/health` demora >5s | ChromaDB travado no startup | Aguarde 90s; veja `docker logs sup-web` |
| `503` em `/consultar` | Rate limit Groq ou timeout | Aguarde e tente novamente |
| `401` mesmo com token correto | `WEB_TOKEN` com espaços ou quebra de linha | `echo -n "$WEB_TOKEN" \| wc -c` deve ser >0 |
| Modelo não encontrado | Banco apontando para pasta errada | Confira `RAG_PASTA_CHROMA` no `.env` |
| SSE não funciona | Nginx com buffering ativo | Verifique `proxy_buffering off` no Nginx |

---

## Contato e suporte

Em caso de problemas técnicos, verifique primeiro:
```bash
docker compose logs --tail=100 sup-web
curl http://localhost:8000/status -H "Authorization: Bearer $WEB_TOKEN"
```
