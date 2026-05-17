import sys
from pathlib import Path
import json
import time

# Agregar ruta raíz del proyecto
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.metrics import evaluate_answer
from agents.multiagents import invoke_docops

def run_evaluations():
    print("Iniciando evaluación del chatbot...")
    
    dataset_path = _ROOT / "evals" / "eval.dataset.json"
    results_path = _ROOT / "evals" / "eval_results.json"
    
    if not dataset_path.exists():
        print(f"Dataset no encontrado en {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    total_score = 0.0
    
    for i, item in enumerate(dataset):
        question = item["question"]
        golden_answer = item["golden_answer"]
        
        print(f"\n[{i+1}/{len(dataset)}] Evaluando: {question}")
        
        start_time = time.time()
        
        try:
            # Ejecutar el agente para obtener la respuesta
            agent_response = invoke_docops(query=question, verbose=False)
            generated_answer = agent_response["answer"]
            
            # Evaluar la respuesta usando metric.py
            eval_result = evaluate_answer(question, generated_answer, golden_answer)
            
            elapsed_time = time.time() - start_time
            
            result_item = {
                "question": question,
                "golden_answer": golden_answer,
                "generated_answer": generated_answer,
                "score": eval_result.score,
                "justification": eval_result.justification,
                "time_seconds": round(elapsed_time, 2)
            }
            results.append(result_item)
            total_score += eval_result.score
            
            color = "\033[32m" if eval_result.score >= 0.8 else ("\033[33m" if eval_result.score >= 0.5 else "\033[31m")
            reset = "\033[0m"
            print(f"Score: {color}{eval_result.score}{reset} | Tiempo: {round(elapsed_time, 1)}s")
            print(f"Justificación: {eval_result.justification}")
            
        except Exception as e:
            print(f"\033[31mError evaluando la pregunta '{question}': {str(e)}\033[0m")
            results.append({
                "question": question,
                "error": str(e)
            })
            
    # Guardar resultados
    avg_score = total_score / len(dataset) if dataset else 0
    final_output = {
        "summary": {
            "total_questions": len(dataset),
            "average_score": round(avg_score, 2)
        },
        "results": results
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*50)
    print("EVALUACIÓN COMPLETADA")
    print(f"Puntuación media general: {round(avg_score, 2)}")
    print(f"Resultados guardados en: {results_path}")
    print("="*50)

if __name__ == "__main__":
    run_evaluations()
