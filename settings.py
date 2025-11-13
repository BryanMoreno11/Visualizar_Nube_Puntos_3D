"""
Configuración del proyecto de visualización de nubes de puntos LiDAR
Versión con Plotly (visualización en navegador)
"""
import os
from pathlib import Path

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

# Ruta base del proyecto (directorio donde está este archivo)
BASE_DIR = Path(__file__).parent.absolute()

# Ruta al dataset (ajusta según tu estructura)
DATASET_PATH = "dataset"  

# Carpeta con archivos LAS/LAZ
ALS_PATH = os.path.join(DATASET_PATH, 'als')
ORTHO_PATH= os.path.join(DATASET_PATH, 'ortho')

# Archivo LAZ específico a visualizar (nombre del archivo)
# Cambia esto para visualizar diferentes archivos
PLOT_FILE="plot_01"  # Cambia por el nombre real de tu archivo
LAZ_FILE = PLOT_FILE+".las"  # Cambia por el nombre real de tu archivo
ORTHO_FILE = PLOT_FILE+".tif"  
# Ruta completa al archivo
LAZ_FILE_PATH = os.path.join(ALS_PATH, LAZ_FILE)
ORTHO_FILE_PATH= os.path.join(ORTHO_PATH, ORTHO_FILE)

# ============================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================

# Número máximo de puntos a visualizar (para rendimiento en navegador)
MAX_POINTS_VISUALIZATION = 200000  # Plotly funciona bien hasta 200k-500k puntos

# Tamaño de la ventana de visualización (en píxeles)
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Tamaño de los puntos en la visualización
POINT_SIZE = 1  # Ajusta según preferencia (0.5 - 3)

# ============================================================
# CONFIGURACIÓN DE COLORES
# ============================================================

# Rango RGB típico en archivos LAS (0-65535)
RGB_MAX_VALUE = 65535

# Mapa de colores para visualización por altura (si no hay RGB)
# Opciones: 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'
#          'Turbo', 'Rainbow', 'Jet', 'Earth', 'Electric', 'Portland'
HEIGHT_COLORMAP = 'Earth'  # Recomendado para terreno/vegetación

# ============================================================
# CONFIGURACIÓN DE PROCESAMIENTO
# ============================================================

# Clases de clasificación LAS a filtrar (opcional)
# 2: Suelo, 3: Vegetación baja, 4: Vegetación media, 5: Vegetación alta
VEGETATION_CLASSES = [3, 4, 5]
GROUND_CLASS = 2

# Filtrar por clasificación (True/False)
FILTER_BY_CLASSIFICATION = False  # Cambia a True para filtrar solo vegetación

# Si FILTER_BY_CLASSIFICATION = True, qué clases mostrar
CLASSES_TO_SHOW = VEGETATION_CLASSES  # Cambia a [2] para solo suelo, etc.

# ============================================================
# CONFIGURACIÓN DE NORMALIZACIÓN
# ============================================================

# Normalizar altura (restar elevación mínima del suelo)
NORMALIZE_HEIGHT = True

# Centrar coordenadas en el origen (facilita visualización)
CENTER_COORDINATES = True

# ============================================================
# CONFIGURACIÓN DE PLOTLY
# ============================================================

# Abrir automáticamente en el navegador
AUTO_OPEN_BROWSER = True

# Guardar HTML de la visualización
SAVE_HTML = False
HTML_OUTPUT_PATH = os.path.join(BASE_DIR, 'visualizacion_lidar.html')

# Tema de la visualización
PLOT_THEME = 'plotly_dark'  # Opciones: 'plotly', 'plotly_white', 'plotly_dark'

# ============================================================
# MENSAJES Y LOGS
# ============================================================

VERBOSE = True  # Mostrar mensajes detallados

# ============================================================
# VALIDACIÓN DE RUTAS
# ============================================================

def validate_paths():
    """Valida que las rutas configuradas existan"""
    errors = []
    
    if not os.path.exists(DATASET_PATH):
        errors.append(f"❌ Dataset path no existe: {DATASET_PATH}")
    
    if not os.path.exists(ALS_PATH):
        errors.append(f"❌ ALS path no existe: {ALS_PATH}")
    
    if not os.path.exists(LAZ_FILE_PATH):
        errors.append(f"❌ Archivo LAZ no existe: {LAZ_FILE_PATH}")
        # Listar archivos disponibles
        if os.path.exists(ALS_PATH):
            available_files = [f for f in os.listdir(ALS_PATH) if f.endswith(('.laz', '.las'))]
            if available_files:
                errors.append(f"\n📁 Archivos disponibles en {ALS_PATH}:")
                for f in available_files[:10]:  # Mostrar máximo 10
                    errors.append(f"   • {f}")
    
    if errors:
        print("⚠️  ADVERTENCIAS DE CONFIGURACIÓN:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print("✅ Todas las rutas son válidas")
    return True


def print_config_summary():
    """Imprime un resumen de la configuración actual"""
    print(f"\n{'='*60}")
    print("CONFIGURACIÓN ACTUAL")
    print(f"{'='*60}")
    print(f"Archivo a visualizar: {LAZ_FILE}")
    print(f"Puntos máximos: {MAX_POINTS_VISUALIZATION:,}")
    print(f"Tamaño de punto: {POINT_SIZE}")
    print(f"Mapa de colores: {HEIGHT_COLORMAP}")
    print(f"Normalizar altura: {'✓' if NORMALIZE_HEIGHT else '✗'}")
    print(f"Centrar coordenadas: {'✓' if CENTER_COORDINATES else '✗'}")
    print(f"Filtrar por clasificación: {'✓' if FILTER_BY_CLASSIFICATION else '✗'}")
    if FILTER_BY_CLASSIFICATION:
        print(f"Clases a mostrar: {CLASSES_TO_SHOW}")
    print(f"Guardar HTML: {'✓' if SAVE_HTML else '✗'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Ejecutar validación si se ejecuta directamente
    print_config_summary()
    validate_paths()
