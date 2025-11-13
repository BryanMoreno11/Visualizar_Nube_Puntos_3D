
## **Archivo actualizado: `main.py`**


"""
Visualizador 3D de nubes de puntos LiDAR con Plotly
Proyecto: Mapeo de árboles coníferas con Deep Learning
Versión: 3.0 (Fusión multimodal LiDAR + RGB)
"""

import laspy
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Para leer ortofotografías TIFF
try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("⚠️ Advertencia: rasterio no está instalado. Instala con: pip install rasterio")

# Importar configuración
import settings


def load_orthophoto(ortho_path):
    """
    Carga una ortofotografía RGB desde archivo TIFF
    
    Args:
        ortho_path: Ruta al archivo de ortofotografía (.tif)
    
    Returns:
        ortho_data: Diccionario con la imagen RGB y metadatos geoespaciales
    """
    if not RASTERIO_AVAILABLE:
        print("❌ Error: rasterio no está instalado")
        return None
    
    if not os.path.exists(ortho_path):
        print(f"⚠️ Advertencia: Ortofotografía no encontrada en: {ortho_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"CARGANDO ORTOFOTOGRAFÍA RGB")
    print(f"{'='*60}")
    print(f"Archivo: {Path(ortho_path).name}")
    
    try:
        with rasterio.open(ortho_path) as src:
            # Leer bandas RGB (normalmente bandas 1, 2, 3)
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
            
            # Obtener transformación geoespacial
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
            
            ortho_data = {
                'red': red,
                'green': green,
                'blue': blue,
                'transform': transform,
                'bounds': bounds,
                'crs': crs,
                'width': src.width,
                'height': src.height
            }
            
            print(f"\n📸 Información de ortofotografía:")
            print(f"   Dimensiones: {src.width} x {src.height} píxeles")
            print(f"   Bandas: {src.count}")
            print(f"   Sistema de coordenadas: {crs}")
            print(f"   Límites:")
            print(f"     X: {bounds.left:.2f} a {bounds.right:.2f}")
            print(f"     Y: {bounds.bottom:.2f} a {bounds.top:.2f}")
            print(f"   Resolución: {transform.a:.4f} m/píxel")
            
            return ortho_data
            
    except Exception as e:
        print(f"❌ Error al cargar ortofotografía: {e}")
        return None


def sample_rgb_from_orthophoto(x_coords, y_coords, ortho_data):
    """
    Muestrea valores RGB de la ortofotografía para coordenadas XY dadas
    
    Args:
        x_coords: Array de coordenadas X
        y_coords: Array de coordenadas Y
        ortho_data: Diccionario con datos de ortofotografía
    
    Returns:
        rgb_colors: Lista de strings 'rgb(r,g,b)' para cada punto
    """
    print(f"\n{'='*60}")
    print(f"FUSIONANDO RGB DE ORTOFOTOGRAFÍA CON PUNTOS LIDAR")
    print(f"{'='*60}")
    
    red_band = ortho_data['red']
    green_band = ortho_data['green']
    blue_band = ortho_data['blue']
    transform = ortho_data['transform']
    
    rgb_colors = []
    valid_colors = 0
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        try:
            # Convertir coordenadas del mundo real a píxeles de la imagen
            row, col = rowcol(transform, x, y)
            
            # Verificar que esté dentro de los límites de la imagen
            if 0 <= row < ortho_data['height'] and 0 <= col < ortho_data['width']:
                r = int(red_band[row, col])
                g = int(green_band[row, col])
                b = int(blue_band[row, col])
                
                # Verificar que no sea un valor no-data (típicamente 0 o 255)
                if not (r == 0 and g == 0 and b == 0):
                    rgb_colors.append(f'rgb({r},{g},{b})')
                    valid_colors += 1
                else:
                    # Color por defecto si es no-data
                    rgb_colors.append('rgb(128,128,128)')
            else:
                # Punto fuera de la ortofoto - color gris
                rgb_colors.append('rgb(128,128,128)')
                
        except Exception:
            # En caso de error, usar gris
            rgb_colors.append('rgb(128,128,128)')
        
        # Mostrar progreso cada 10%
        if (i + 1) % (len(x_coords) // 10 + 1) == 0:
            progress = ((i + 1) / len(x_coords)) * 100
            print(f"   Procesando: {progress:.0f}% completado", end='\r')
    
    print(f"\n✓ Fusión completada")
    print(f"  Puntos con RGB válido: {valid_colors:,} ({(valid_colors/len(x_coords))*100:.1f}%)")
    print(f"  Puntos sin RGB: {len(x_coords) - valid_colors:,}")
    
    return rgb_colors


def load_las_file(file_path, max_points=None):
    """
    Carga un archivo LAS/LAZ y retorna la información básica
    
    Args:
        file_path: Ruta al archivo LAS/LAZ
        max_points: Número máximo de puntos a cargar (None = todos)
    
    Returns:
        las: Objeto laspy con los datos
        info: Diccionario con información del archivo
    """
    print(f"\n{'='*60}")
    print(f"CARGANDO ARCHIVO LAS/LAZ")
    print(f"{'='*60}")
    print(f"Archivo: {Path(file_path).name}")
    
    # Leer archivo
    las = laspy.read(file_path)
    
    # Información básica
    info = {
        'num_points': len(las.points),
        'version': str(las.header.version),
        'point_format': las.header.point_format.id,
        'has_rgb': hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'),
        'has_classification': hasattr(las, 'classification'),
        'has_intensity': hasattr(las, 'intensity'),
        'x_range': (np.min(las.x), np.max(las.x)),
        'y_range': (np.min(las.y), np.max(las.y)),
        'z_range': (np.min(las.z), np.max(las.z))
    }
    
    # Imprimir información
    print(f"\n📊 Estadísticas del archivo:")
    print(f"   Total de puntos: {info['num_points']:,}")
    print(f"   Versión LAS: {info['version']}")
    print(f"   Formato de punto: {info['point_format']}")
    print(f"   RGB en LAS disponible: {'✓ Sí' if info['has_rgb'] else '✗ No'}")
    print(f"   Clasificación disponible: {'✓ Sí' if info['has_classification'] else '✗ No'}")
    print(f"   Intensidad disponible: {'✓ Sí' if info['has_intensity'] else '✗ No'}")
    
    # Calcular área y densidad
    area = (info['x_range'][1] - info['x_range'][0]) * (info['y_range'][1] - info['y_range'][0])
    density = info['num_points'] / area if area > 0 else 0
    
    print(f"\n📏 Dimensiones:")
    print(f"   Área: {area:.2f} m²")
    print(f"   Ancho (X): {info['x_range'][1] - info['x_range'][0]:.2f} m")
    print(f"   Largo (Y): {info['y_range'][1] - info['y_range'][0]:.2f} m")
    print(f"   Altura (Z): {info['z_range'][1] - info['z_range'][0]:.2f} m")
    print(f"   Densidad: {density:.2f} puntos/m²")
    
    return las, info


def filter_points_by_classification(las, classes_to_keep):
    """
    Filtra puntos según su clasificación
    
    Args:
        las: Objeto laspy
        classes_to_keep: Lista de clases a mantener
    
    Returns:
        indices: Array de índices de puntos filtrados
    """
    if not hasattr(las, 'classification'):
        print("⚠️  Advertencia: No hay información de clasificación")
        return np.arange(len(las.points))
    
    mask = np.isin(las.classification, classes_to_keep)
    indices = np.where(mask)[0]
    
    print(f"\n🔍 Filtrado por clasificación:")
    print(f"   Clases seleccionadas: {classes_to_keep}")
    print(f"   Puntos filtrados: {len(indices):,} ({(len(indices)/len(las.points))*100:.1f}%)")
    
    return indices


def sample_points(total_indices, max_points):
    """
    Toma una muestra aleatoria de puntos si el total excede max_points
    
    Args:
        total_indices: Array de índices disponibles
        max_points: Número máximo de puntos
    
    Returns:
        indices: Array de índices seleccionados
    """
    total_points = len(total_indices)
    
    if max_points is None or total_points <= max_points:
        return total_indices
    
    print(f"\n📉 Muestreo de puntos:")
    print(f"   Total disponible: {total_points:,}")
    print(f"   Muestra seleccionada: {max_points:,} ({(max_points/total_points)*100:.1f}%)")
    
    selected_indices = np.random.choice(total_indices, max_points, replace=False)
    return selected_indices


def prepare_point_cloud_data(las, indices, ortho_data=None):
    """
    Prepara los datos de la nube de puntos para visualización
    
    Args:
        las: Objeto laspy
        indices: Índices de puntos a incluir
        ortho_data: Diccionario con datos de ortofotografía (opcional)
    
    Returns:
        data: Diccionario con coordenadas, colores e información
    """
    print(f"\n{'='*60}")
    print(f"PREPARANDO DATOS PARA VISUALIZACIÓN")
    print(f"{'='*60}")
    
    # Extraer coordenadas ORIGINALES (antes de centrar)
    x_original = las.x[indices]
    y_original = las.y[indices]
    z = las.z[indices]
    
    # Normalizar altura si está configurado
    ground_elevation = 0
    if settings.NORMALIZE_HEIGHT and hasattr(las, 'classification'):
        ground_points = las.classification == settings.GROUND_CLASS
        if np.sum(ground_points) > 0:
            ground_elevation = np.min(las.z[ground_points])
            z = z - ground_elevation
            print(f"✓ Altura normalizada (suelo en Z=0)")
            print(f"  Elevación del suelo: {ground_elevation:.2f} m")
    
    # Preparar colores
    colors = None
    color_array = None
    colorscale = None
    color_source = "altura"
    
    # Prioridad 1: Usar ortofotografía externa si está disponible
    if ortho_data is not None and RASTERIO_AVAILABLE:
        print(f"\n🎨 Aplicando colores desde ortofotografía TIFF...")
        colors = sample_rgb_from_orthophoto(x_original, y_original, ortho_data)
        color_source = "ortofotografía"
        
    # Prioridad 2: Usar RGB del archivo LAS si existe
    elif hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print(f"\n🎨 Usando colores RGB del archivo LAS...")
        r = (las.red[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        g = (las.green[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        b = (las.blue[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        colors = [f'rgb({ri},{gi},{bi})' for ri, gi, bi in zip(r, g, b)]
        color_source = "RGB en LAS"
        print(f"✓ Colores RGB del archivo LAS aplicados")
        
    # Prioridad 3: Colorear por altura
    else:
        print(f"\n🎨 Coloreando por altura...")
        color_array = z
        colorscale = settings.HEIGHT_COLORMAP
        color_source = "altura"
        print(f"✓ Colores por altura aplicados (colormap: {settings.HEIGHT_COLORMAP})")
    
    # Centrar coordenadas DESPUÉS de obtener RGB (para visualización)
    x = x_original
    y = y_original
    x_center = 0
    y_center = 0
    
    if settings.CENTER_COORDINATES:
        x_center = np.mean(x_original)
        y_center = np.mean(y_original)
        x = x_original - x_center
        y = y_original - y_center
        print(f"\n✓ Coordenadas centradas en origen")
        print(f"  Centro original: X={x_center:.2f}, Y={y_center:.2f}")
    
    # Información adicional para hover
    hover_text = []
    for i in range(len(indices)):
        text = f"X: {x[i]:.2f} m<br>Y: {y[i]:.2f} m<br>Z: {z[i]:.2f} m"
        
        if hasattr(las, 'classification'):
            text += f"<br>Clase: {las.classification[indices[i]]}"
        
        if hasattr(las, 'intensity'):
            text += f"<br>Intensidad: {las.intensity[indices[i]]}"
        
        hover_text.append(text)
    
    data = {
        'x': x,
        'y': y,
        'z': z,
        'colors': colors,
        'color_array': color_array,
        'colorscale': colorscale,
        'hover_text': hover_text,
        'num_points': len(indices),
        'color_source': color_source
    }
    
    print(f"\n✓ Datos preparados:")
    print(f"  Puntos procesados: {data['num_points']:,}")
    print(f"  Fuente de color: {color_source}")
    print(f"  Rango X: {np.min(x):.2f} a {np.max(x):.2f} m")
    print(f"  Rango Y: {np.min(y):.2f} a {np.max(y):.2f} m")
    print(f"  Rango Z: {np.min(z):.2f} a {np.max(z):.2f} m")
    
    return data


def create_plotly_figure(data, title):
    """
    Crea una figura de Plotly con la nube de puntos
    
    Args:
        data: Diccionario con datos preparados
        title: Título de la visualización
    
    Returns:
        fig: Figura de Plotly
    """
    print(f"\n{'='*60}")
    print(f"CREANDO VISUALIZACIÓN INTERACTIVA")
    print(f"{'='*60}")
    
    # Crear scatter 3D
    scatter = go.Scatter3d(
        x=data['x'],
        y=data['y'],
        z=data['z'],
        mode='markers',
        marker=dict(
            size=settings.POINT_SIZE,
            color=data['colors'] if data['colors'] is not None else data['color_array'],
            colorscale=data['colorscale'] if data['color_array'] is not None else None,
            showscale=data['color_array'] is not None,
            colorbar=dict(
                title="Altura (m)",
                x=1.02
            ) if data['color_array'] is not None else None,
            opacity=0.8
        ),
        text=data['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        name='Puntos LiDAR'
    )
    
    # Crear figura
    fig = go.Figure(data=[scatter])
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X (metros)',
            yaxis_title='Y (metros)',
            zaxis_title='Z (altura, metros)',
            aspectmode='data',  # Mantener proporciones reales
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        width=settings.WINDOW_WIDTH,
        height=settings.WINDOW_HEIGHT,
        template=settings.PLOT_THEME,
        showlegend=False,
        hovermode='closest'
    )
    
    print(f"✓ Figura creada exitosamente")
    
    return fig


def print_visualization_instructions():
    """Imprime las instrucciones de uso de la visualización"""
    print(f"\n{'='*60}")
    print(f"INSTRUCCIONES DE USO")
    print(f"{'='*60}")
    print("📱 CONTROLES DEL MOUSE:")
    print("   • Clic izquierdo + arrastrar: Rotar vista")
    print("   • Clic derecho + arrastrar: Mover (pan)")
    print("   • Scroll/rueda: Zoom in/out")
    print("   • Doble clic: Resetear vista")
    print("\n🔧 HERRAMIENTAS (barra superior):")
    print("   • 📷 Descargar como PNG")
    print("   • 🔍 Zoom, Pan, Resetear")
    print("   • 🏠 Home: Vista inicial")
    print("\n💡 HOVER:")
    print("   • Pasa el mouse sobre puntos para ver información")
    print(f"{'='*60}\n")


def visualize_point_cloud(fig, file_name):
    """
    Muestra y/o guarda la visualización
    
    Args:
        fig: Figura de Plotly
        file_name: Nombre del archivo original
    """
    print_visualization_instructions()
    
    # Guardar HTML si está configurado
    if settings.SAVE_HTML:
        fig.write_html(settings.HTML_OUTPUT_PATH)
        print(f"💾 Visualización guardada en: {settings.HTML_OUTPUT_PATH}")
    
    # Mostrar en navegador
    print(f"🚀 Abriendo visualización en el navegador...")
    print(f"   (La ventana puede tardar unos segundos en cargar)")
    
    fig.show(config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
    })
    
    print(f"\n✓ Visualización mostrada")


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("VISUALIZADOR DE NUBES DE PUNTOS LIDAR")
    print("Proyecto: Mapeo de árboles coníferas con Deep Learning")
    print("Versión: 3.0 - Fusión multimodal (LiDAR + RGB)")
    print("="*60)
    
    # Mostrar configuración
    settings.print_config_summary()
    
    # Validar configuración
    if not settings.validate_paths():
        print("\n⚠️ Advertencia: Algunas rutas no son válidas")
        print("   Continuando con las rutas disponibles...")
    
    try:
        # 1. Cargar archivo LAS/LAZ
        las, info = load_las_file(settings.LAZ_FILE_PATH)
        
        # 2. Cargar ortofotografía RGB (si existe)
        ortho_data = None
        if os.path.exists(settings.ORTHO_FILE_PATH):
            ortho_data = load_orthophoto(settings.ORTHO_FILE_PATH)
            if ortho_data:
                print(f"\n✅ Ortofotografía cargada exitosamente")
                print(f"   Se usarán colores RGB de la ortofoto TIFF")
        else:
            print(f"\n⚠️ Ortofotografía no encontrada: {settings.ORTHO_FILE_PATH}")
            print(f"   Se usarán colores alternativos (RGB del LAS o por altura)")
        
        # 3. Filtrar por clasificación si está configurado
        if settings.FILTER_BY_CLASSIFICATION and info['has_classification']:
            available_indices = filter_points_by_classification(las, settings.CLASSES_TO_SHOW)
        else:
            available_indices = np.arange(len(las.points))
        
        # 4. Muestrear puntos si es necesario
        indices = sample_points(available_indices, settings.MAX_POINTS_VISUALIZATION)
        
        # 5. Preparar datos (con fusión RGB si hay ortofoto)
        data = prepare_point_cloud_data(las, indices, ortho_data)
        
        # 6. Crear figura
        color_info = f" - Colores: {data['color_source']}"
        title = f"Nube de Puntos LiDAR: {Path(settings.PLOT_FILE).name}<br>" \
                f"<sub>{data['num_points']:,} puntos visualizados{color_info}</sub>"
        fig = create_plotly_figure(data, title)
        
        # 7. Visualizar
        visualize_point_cloud(fig, settings.LAZ_FILE)
        
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Archivo no encontrado")
        print(f"   {e}")
        print(f"\n   Verifica la configuración en settings.py")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
