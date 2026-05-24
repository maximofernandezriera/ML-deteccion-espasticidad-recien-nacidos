#!/usr/bin/env python3
"""
Script de prueba para verificar que el experimento de ablación funciona correctamente.
Ejecuta solo 2 condiciones con un subset pequeño de datos para validar la implementación.
"""

import sys
from pathlib import Path

# Añadir el directorio del proyecto al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importar funciones necesarias
from main_pipeline_29032026_1516 import (
    generate_synthetic_pathological_video,
    ensure_uint8_video,
    extract_features_single,
    load_raw_video_corpus,
    RANDOM_STATE
)
import numpy as np

def test_exclude_functionality():
    """Prueba que el parámetro exclude_set funciona correctamente."""
    print("="*60)
    print("TEST: Verificando funcionalidad de exclude_set")
    print("="*60)
    
    # Cargar un vídeo de prueba
    try:
        videos, _ = load_raw_video_corpus()
        test_video = videos[0]
        print(f"✓ Vídeo de prueba cargado: shape={test_video.shape}")
    except Exception as e:
        print(f"✗ Error cargando vídeo: {e}")
        return False
    
    rng = np.random.RandomState(RANDOM_STATE)
    
    # Test 1: Generar con todas las perturbaciones
    try:
        video_completo = generate_synthetic_pathological_video(test_video, rng, exclude_set=None)
        features_completo = extract_features_single(video_completo)
        print(f"✓ Generación completa exitosa: {features_completo.shape}")
    except Exception as e:
        print(f"✗ Error en generación completa: {e}")
        return False
    
    # Test 2: Generar excluyendo rigidez
    try:
        video_sin_rigidez = generate_synthetic_pathological_video(test_video, rng, exclude_set={'rigidez'})
        features_sin_rigidez = extract_features_single(video_sin_rigidez)
        print(f"✓ Generación sin rigidez exitosa: {features_sin_rigidez.shape}")
    except Exception as e:
        print(f"✗ Error en generación sin rigidez: {e}")
        return False
    
    # Test 3: Generar excluyendo múltiples perturbaciones
    try:
        video_multiple = generate_synthetic_pathological_video(
            test_video, rng, exclude_set={'rigidez', 'temblor', 'ruido'}
        )
        features_multiple = extract_features_single(video_multiple)
        print(f"✓ Generación con múltiples exclusiones exitosa: {features_multiple.shape}")
    except Exception as e:
        print(f"✗ Error en generación con múltiples exclusiones: {e}")
        return False
    
    # Verificar que las features son diferentes
    diff_completo_rigidez = np.mean(np.abs(features_completo - features_sin_rigidez))
    diff_completo_multiple = np.mean(np.abs(features_completo - features_multiple))
    
    print(f"\nDiferencias en features:")
    print(f"  Completo vs Sin rigidez: {diff_completo_rigidez:.4f}")
    print(f"  Completo vs Múltiples exclusiones: {diff_completo_multiple:.4f}")
    
    if diff_completo_rigidez > 0.01 and diff_completo_multiple > 0.01:
        print("✓ Las exclusiones producen diferencias significativas en las features")
        return True
    else:
        print("✗ Las exclusiones no producen diferencias suficientes")
        return False

def main():
    print("\n" + "█"*60)
    print("█  TEST DE EXPERIMENTO B - ABLACIÓN SISTEMÁTICA")
    print("█"*60 + "\n")
    
    success = test_exclude_functionality()
    
    print("\n" + "="*60)
    if success:
        print("✅ TODOS LOS TESTS PASARON")
        print("\nPuedes ejecutar el experimento completo con:")
        print("  python main_pipeline_29032026_1516.py --experiment=ablation")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("Revisa los errores antes de ejecutar el experimento completo")
    print("="*60 + "\n")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
