#!/usr/bin/env python3
import os
import runpod
from dotenv import load_dotenv

load_dotenv()

# Initialize RunPod with the API key from your .env
API_KEY = os.environ.get("RUNPOD_API_KEY")
if not API_KEY:
    print("ERROR: RUNPOD_API_KEY no encontrada en las variables de entorno.")
    exit(1)

runpod.api_key = API_KEY

def main():
    print("Obteniendo la lista de GPUs válidas en RunPod...")
    gpus = runpod.get_gpus()
    
    # Buscar el primer ID que contenga 'H100'
    h100_ids = [g["id"] for g in gpus if "H100" in g["id"]]
    if not h100_ids:
        print("ERROR: La API de RunPod no reporta ninguna GPU H100 existente en su catálogo.")
        exit(1)
        
    gpu_type_id = "NVIDIA H100 80GB HBM3"
    print(f"GPUs H100 disponibles en catálogo: {h100_ids}")
    print(f"Seleccionada estrictamente para el pod: {gpu_type_id} (HBM3 / SXM5)")

    # Set parameters according to the OpenAI Parameter Golf challenge
    image_name = "runpod/parameter-golf:latest"
    gpu_count = 8
    cloud_type = "ALL" # 'ALL' busca tanto en SECURE como COMMUNITY para maximizar chances
    
    try:
        new_pod = runpod.create_pod(
            name="LaPulga-ParameterGolf-8xH100",
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            volume_in_gb=100,
            container_disk_in_gb=100,
            ports="22/tcp",
            cloud_type=cloud_type
        )
        print("\n--- ¡Éxito! Pod Creado ---")
        print(f"ID del nuevo Pod (RUNPOD_POD_ID): {new_pod.get('id')}")
        print(f"Estado inicial: {new_pod.get('desiredStatus')}")
        print("\nPor favor actualiza tu archivo .env con este nuevo RUNPOD_POD_ID.")
        
    except Exception as e:
        print(f"\nOops, ocurrió un error al pedir el Pod: {e}")
        print("Esto generalmente significa que no hay stock de 8xH100 SXM5 en 'SECURE' listos.")
        print("Intenta cambiar el cloud_type a 'COMMUNITY' o 'ALL' en este script, o pide H100 NVL/PCIe.")

if __name__ == "__main__":
    main()
