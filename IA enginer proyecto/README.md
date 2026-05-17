# IA ENGINER PROYECTO

es un chatbot que te ayuda a responder preguntas usando un documento linkedin el cual puedes preguntarle informacion acerca de ese curriculum

## Prerequisites

- **Python 3.11+**
- **API key de los siguientes proveedores:**
    - **OpenAI API key**
    - **GROQ API key**
    - **Gemini API key**


### correr el programa (recommended for development)

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd IA enginer proyecto
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt

4. **copiar el docunento a preguntar a la carpeta /data:**
   ```bash
   copy data/<tu documento>
   ```   ```
5. **correr el programa:**
   ```bash
   python agents/multiagents.py
   ```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | LLM de gemini |
| `LLM_MODEL` | `gemma-3-27b-it` | modelo de carga del LLM |
| `LLM_TEMPERATURE` | `0.2` | introducir el valor de la temperatura para reducir alucionaciones |
| `GEMINI_API_KEY` | *(required)* | Gemini API key |
| `LOG_LEVEL` | `INFO` | nivel de registro de eventos |
| `GEMINI_MODEL` | `gemma-3-27b-it` | modelo de gemini a usar|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | modelo de QroQ a usar |
| `GROQ_API_KEY` | *(required)* | GROQ API key |
| `URL_REMOTE` | `http://<tu.direccion.ip>:1234/v1` | `development` / `staging` / `production` |
| `NAME` | `Ed Donner` | nombre inicial del agente quien respondera las preguntas |


## Project Structure

```
IA enginer proyecto/
├── agents/                # codigo donde se encuentra el multiagente
├── chroma_db/             # lugar donde se indexan los chunks
├── contracts/             # aqui se guardan las validaciones pydantic y los esquemas
├── core/                  # configuracion y logger del cliente LLM
├── data/                  # lugar donde se almacena el CV del integrante
├── evals/                 # evaluaciones de los agentes y del RAG
├── memory/                # lugar donde guarda informacion de la memoria persistente del agente
├── prompting/             # lugar donde se encuentra los templates
├── rag/                   # ingestacion de documentos y chunking
└── tests/                 # protocolo de pruebas para los agentes
```

## Codigo aprendido gracias a Codifofacilito

- **Web** https://codigofacilito.com/
