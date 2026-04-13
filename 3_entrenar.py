"""
PASO 3: ENTRENAR MODELO
Entrena YOLOv8 de forma rápida y eficiente

Tiempo estimado: 10-15 minutos
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import time

class EntrenadorRapido:
    def __init__(self, data_yaml, modelo_base='yolov8n.pt'):
        self.data_yaml = Path(data_yaml)
        self.modelo_base = modelo_base
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"❌ No existe: {data_yaml}")
        
        # Detectar GPU
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        print(f"✅ Dataset: {self.data_yaml}")
        print(f"✅ Modelo base: {self.modelo_base}")
        print(f"✅ Device: {'GPU - ' + torch.cuda.get_device_name(0) if self.device == 0 else 'CPU'}")
    
    def entrenar(self, epochs=100, imgsz=640, batch=16, nombre='modelo_concurso'):
        """Entrena el modelo"""
        print("\n" + "="*60)
        print("  ENTRENANDO MODELO YOLOV8")
        print("="*60)
        print(f"\n⚙️  Configuración:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch}")
        print(f"   Image size: {imgsz}")
        print(f"   Device: {self.device}")
        print("\n" + "="*60)
        
        # Cargar modelo
        model = YOLO(self.modelo_base)
        
        # Entrenar
        print("\n🚀 Iniciando entrenamiento...\n")
        inicio = time.time()
        
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=nombre,
            device=self.device,
            
            # Optimizaciones para velocidad
            patience=20,           # Early stopping
            save_period=-1,        # No guardar checkpoints intermedios
            plots=True,
            
            # Augmentations moderados (más rápido)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            
            # Cache para velocidad (si hay RAM suficiente)
            cache=False,  # Cambiar a True si tienes suficiente RAM
            
            # Workers
            workers=8,
        )
        
        tiempo_total = (time.time() - inicio) / 60
        
        # Resumen
        print("\n" + "="*60)
        print("  ✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"\n⏱️  Tiempo: {tiempo_total:.1f} minutos")
        
        # Encontrar modelo
        modelo_path = Path(f"runs/detect/{nombre}/weights/best.pt")
        
        if modelo_path.exists():
            print(f"📦 Modelo guardado: {modelo_path}")
            
            # Evaluar
            print("\n📊 Evaluando modelo...")
            metrics = model.val()
            
            print(f"\n📈 Métricas:")
            print(f"   mAP50:     {metrics.box.map50:.4f}")
            print(f"   mAP50-95:  {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall:    {metrics.box.mr:.4f}")
            
            # Interpretación
            self._interpretar(metrics.box.map50)
            
        else:
            print("⚠️  No se encontró el modelo entrenado")
            modelo_path = None
        
        print("="*60)
        
        return modelo_path
    
    def _interpretar(self, map50):
        """Interpreta resultados"""
        print(f"\n💡 Interpretación:")
        if map50 >= 0.90:
            print("   ✅ EXCELENTE - Modelo listo para competir")
        elif map50 >= 0.75:
            print("   ✅ MUY BUENO - Funcional para la competencia")
        elif map50 >= 0.60:
            print("   ⚠️  ACEPTABLE - Considera entrenar más epochs")
        else:
            print("   ❌ BAJO - Revisa las etiquetas o aumenta epochs")


def main():
    print("\n" + "="*60)
    print("  PASO 3: ENTRENAR MODELO")
    print("="*60)
    
    # Configuración
    data_yaml = input("\n📄 Ruta al data.yaml (default: yolo_dataset/data.yaml): ").strip()
    if not data_yaml:
        data_yaml = "yolo_dataset/data.yaml"
    
    if not Path(data_yaml).exists():
        print(f"❌ No existe: {data_yaml}")
        print("\n💡 Primero ejecuta: python 2_preparar_dataset.py")
        return
    
    print("\n⚙️  Configuración de entrenamiento:")
    print("   • Modelo: yolov8n.pt (nano - más rápido)")
    print("   • Epochs: 100")
    print("   • Batch: 16")
    print("   • Tiempo estimado: 10-15 minutos")
    
    continuar = input("\n¿Continuar? (s/n): ").strip().lower()
    if continuar != 's':
        print("❌ Cancelado")
        return
    
    # Entrenar
    entrenador = EntrenadorRapido(data_yaml)
    modelo_path = entrenador.entrenar()
    
    if modelo_path:
        print("\n✅ Listo para el Paso 4: Probar en Cámara")
        print("   Ejecuta: python 4_probar_camara.py\n")


if __name__ == "__main__":
    main()
