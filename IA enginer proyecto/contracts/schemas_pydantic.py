import json

from pydantic import BaseModel, Field


class BuscarDatosUsuario(BaseModel):
    query: str = Field(description="Termino de busqueda del dato.")
    categoria: str | None = Field(
        default=None,
        description="Categoría del dato: nombre, email, edad.",
    )


# --- Generar JSON Schema ---
schema = BuscarDatosUsuario.model_json_schema()

print("JSON Schema generado automáticamente:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
print()

# --- Construir definición de tool a partir del schema ---
tool_definition = {
    "type": "function",
    "function": {
        "name": "buscar_datos",
        "description": "Busca los datos en el documento.",
        "parameters": schema,
    },
}

print("Definición de tool completa:")
print(json.dumps(tool_definition, indent=2, ensure_ascii=False))
