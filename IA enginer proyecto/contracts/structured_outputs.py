import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"


class DatosContacto(BaseModel):
    nombre: str
    empresa: str
    rol: str
    email: str


# --- Structured Outputs con Pydantic ---
response = client.beta.chat.completions.parse(
    model=MODEL,
    temperature=0.2,
    response_format=DatosContacto,
    messages=[
        {
            "role": "system",
            "content": "Extrae los datos de contacto del texto del usuario.",
        },
        {
            "role": "user",
            "content": (
                "Me llamo Jorge Perez, trabajo como instructor privado"
                "Mi correo es Jorge@gmail.com"
            ),
        },
    ],
)

# Acceder al objeto Pydantic parseado directamente
datos = response.choices[0].message.parsed

print(f"Nombre: {datos.nombre}")
print(f"Empresa: {datos.empresa}")
print(f"Rol: {datos.rol}")
print(f"Email: {datos.email}")
print()

# También podemos serializar a JSON
print("JSON:")
print(datos.model_dump_json(indent=2))
