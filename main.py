"""
Visualizador 3D de nubes de puntos LiDAR con Plotly
Proyecto: Mapeo de árboles coníferas con Deep Learning
Versión: 2.0 (Plotly en navegador)
"""

import laspy
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import webbrowser

# Importar configuración
import settings


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
    print(f"   RGB disponible: {'✓ Sí' if info['has_rgb'] else '✗ No'}")
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


def prepare_point_cloud_data(las, indices):
    """
    Prepara los datos de la nube de puntos para visualización
    
    Args:
        las: Objeto laspy
        indices: Índices de puntos a incluir
    
    Returns:
        data: Diccionario con coordenadas, colores e información
    """
    print(f"\n{'='*60}")
    print(f"PREPARANDO DATOS PARA VISUALIZACIÓN")
    print(f"{'='*60}")
    
    # Extraer coordenadas
    x = las.x[indices]
    y = las.y[indices]
    z = las.z[indices]
    
    # Normalizar altura si está configurado
    if settings.NORMALIZE_HEIGHT and hasattr(las, 'classification'):
        ground_points = las.classification == settings.GROUND_CLASS
        if np.sum(ground_points) > 0:
            ground_elevation = np.min(las.z[ground_points])
            z = z - ground_elevation
            print(f"✓ Altura normalizada (suelo en Z=0)")
            print(f"  Elevación del suelo: {ground_elevation:.2f} m")
    
    # Centrar coordenadas si está configurado
    if settings.CENTER_COORDINATES:
        x_center = np.mean(x)
        y_center = np.mean(y)
        x = x - x_center
        y = y - y_center
        print(f"✓ Coordenadas centradas en origen")
        print(f"  Centro original: X={x_center:.2f}, Y={y_center:.2f}")
    
    # Preparar colores
    colors = None
    color_array = None
    colorscale = None
    
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Usar colores RGB del archivo
        r = (las.red[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        g = (las.green[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        b = (las.blue[indices] / settings.RGB_MAX_VALUE * 255).astype(int)
        colors = [f'rgb({ri},{gi},{bi})' for ri, gi, bi in zip(r, g, b)]
        print(f"✓ Colores RGB aplicados desde ortofotografía")
    else:
        # Colorear por altura
        color_array = z
        colorscale = settings.HEIGHT_COLORMAP
        print(f"✓ Colores aplicados por altura (colormap: {settings.HEIGHT_COLORMAP})")
    
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
        'num_points': len(indices)
    }
    
    print(f"\n✓ Datos preparados:")
    print(f"  Puntos procesados: {data['num_points']:,}")
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
    print("Versión: Plotly (Visualización en navegador)")
    print("="*60)
    
    # Mostrar configuración
    settings.print_config_summary()
    
    # Validar configuración
    if not settings.validate_paths():
        print("\n❌ Error: Verifica las rutas en settings.py")
        return
    
    try:
        # 1. Cargar archivo LAS/LAZ
        las, info = load_las_file(settings.LAZ_FILE_PATH)
        
        # 2. Filtrar por clasificación si está configurado
        if settings.FILTER_BY_CLASSIFICATION and info['has_classification']:
            available_indices = filter_points_by_classification(las, settings.CLASSES_TO_SHOW)
        else:
            available_indices = np.arange(len(las.points))
        
        # 3. Muestrear puntos si es necesario
        indices = sample_points(available_indices, settings.MAX_POINTS_VISUALIZATION)
        
        # 4. Preparar datos
        data = prepare_point_cloud_data(las, indices)
        
        # 5. Crear figura
        title = f"Nube de Puntos LiDAR: {Path(settings.LAZ_FILE).name}<br>" \
                f"<sub>{data['num_points']:,} puntos visualizados</sub>"
        fig = create_plotly_figure(data, title)
        
        # 6. Visualizar
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
