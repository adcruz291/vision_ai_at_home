"""
PASO 4: PROBAR EN CÁMARA EN VIVO
Verifica que el modelo funciona correctamente

CONTROLES:
- Q: Salir
- +/-: Ajustar confianza
- S: Capturar screenshot
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
from pathlib import Path

class ProbadorCamara:
    def __init__(self, modelo_path, confianza=0.5):
        self.modelo_path = Path(modelo_path)
        self.confianza = confianza
        
        if not self.modelo_path.exists():
            raise FileNotFoundError(f"❌ No existe: {modelo_path}")
        
        print(f"✅ Cargando modelo: {modelo_path}")
        self.modelo = YOLO(str(modelo_path))
        
        # Obtener nombres de clases
        self.clases = self.modelo.names
        
        print(f"✅ Clases detectables: {list(self.clases.values())}")
        
        self.screenshots = 0
        self.fps_valores = []
    
    def probar(self, camara_id=2):
        """Prueba el modelo en tiempo real"""
        print("\n" + "="*60)
        print("  PRUEBA EN CÁMARA EN VIVO")
        print("="*60)
        print(f"\n🎯 Confianza inicial: {self.confianza:.0%}")
        print("\n⌨️  CONTROLES:")
        print("   Q: Salir")
        print("   +/-: Ajustar confianza")
        print("   S: Capturar screenshot")
        print("\n" + "="*60)
        
        # Abrir cámara
        cap = cv2.VideoCapture(camara_id, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print("\n❌ Error: No se puede abrir la cámara")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n✅ Cámara activa. Mostrando detecciones...\n")
        
        while True:
            inicio = time.time()
            
            # Leer frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error leyendo frame")
                break
            
            # Detectar
            resultados = self.modelo(frame, conf=self.confianza, verbose=False)
            
            # Dibujar detecciones
            frame_anotado = resultados[0].plot()
            
            # Calcular FPS
            fps = 1.0 / (time.time() - inicio)
            self.fps_valores.append(fps)
            if len(self.fps_valores) > 30:
                self.fps_valores.pop(0)
            fps_promedio = np.mean(self.fps_valores)
            
            # Info en pantalla
            cv2.putText(frame_anotado, f"FPS: {fps_promedio:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_anotado, f"Confianza: {self.confianza:.0%}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Número de detecciones
            num_det = len(resultados[0].boxes)
            if num_det > 0:
                cv2.putText(frame_anotado, f"Detecciones: {num_det}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar
            cv2.imshow('YOLO - Deteccion en Vivo (Q para salir)', frame_anotado)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.confianza = min(0.95, self.confianza + 0.05)
                print(f"🎯 Confianza: {self.confianza:.0%}")
            elif key == ord('-'):
                self.confianza = max(0.05, self.confianza - 0.05)
                print(f"🎯 Confianza: {self.confianza:.0%}")
            elif key == ord('s'):
                screenshot_path = f"screenshot_{self.screenshots:03d}.jpg"
                cv2.imwrite(screenshot_path, frame_anotado)
                print(f"📸 Screenshot: {screenshot_path}")
                self.screenshots += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Prueba finalizada")
        
        return True


def main():
    print("\n" + "="*60)
    print("  PASO 4: PROBAR EN CÁMARA")
    print("="*60)
    
    # Buscar modelo
    modelo_default = "runs/detect/modelo_concurso/weights/best.pt"
    
    modelo_path = input(f"\n📦 Ruta del modelo (default: {modelo_default}): ").strip()
    if not modelo_path:
        modelo_path = modelo_default
    
    if not Path(modelo_path).exists():
        print(f"❌ No existe: {modelo_path}")
        print("\n💡 Primero ejecuta: python 3_entrenar.py")
        return
    
    confianza = input("🎯 Confianza inicial 0-100% (default: 50): ").strip()
    confianza = float(confianza) / 100 if confianza else 0.5
    
    # Probar
    probador = ProbadorCamara(modelo_path, confianza)
    probador.probar()


if __name__ == "__main__":
    main()
