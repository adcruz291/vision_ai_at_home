# 🏆 GUÍA RÁPIDA PARA CONCURSOS

Sistema completo de detección de objetos con YOLOv8 optimizado para competencias.

**Tiempo total estimado: 90 minutos para 6 objetos**

---

## 📋 Instalación (ANTES del concurso)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar instalación
python -c "from ultralytics import YOLO; print('✅ Todo listo')"
```

---

## 🚀 Flujo de Trabajo (DURANTE el concurso)

### **Para cada objeto (15 min):**

#### **Paso 1: Capturar (7 min)**
```bash
python 1_capturar.py
```
- Nombre del objeto: `tornillo`
- 20 lotes × 50 fotos = 1000 fotos
- Etiquetas 1 foto por lote = 20 etiquetados

**Tips:**
- Varía la POSICIÓN entre lotes
- Puedes variar iluminación DENTRO de cada lote
- Cada etiquetado toma ~15 segundos

#### **Paso 2: Preparar Dataset (2 min)**

Después de capturar los 6 objetos:

```bash
python 2_preparar_dataset.py
```
- Carpeta origen: `dataset`
- Carpeta destino: `yolo_dataset`
- Automático: Detecta las 6 clases y organiza todo

#### **Paso 3: Entrenar (10-15 min)**
```bash
python 3_entrenar.py
```
- data.yaml: `yolo_dataset/data.yaml`
- Entrena automáticamente
- Espera a que termine (~10-15 min)

#### **Paso 4: Probar (2 min)**
```bash
python 4_probar_camara.py
```
- Modelo: `runs/detect/modelo_concurso/weights/best.pt`
- Verifica que detecta correctamente
- Ajusta confianza con +/-

---

## ⏱️ Timeline Optimizado para 6 Objetos

```
CAPTURA (paralelo si hay 2 personas):
0:00 - 0:12  Objeto 1: tornillo
0:12 - 0:24  Objeto 2: tuerca
0:24 - 0:36  Objeto 3: arandela
0:36 - 0:48  Objeto 4: perno
0:48 - 1:00  Objeto 5: remache
1:00 - 1:12  Objeto 6: grapa

PREPARAR Y ENTRENAR:
1:12 - 1:14  Preparar dataset (2 min)
1:14 - 1:29  Entrenar modelo (15 min)
1:29 - 1:31  Probar en cámara (2 min)

TOTAL: ~90 minutos
```

---

## 📊 Estructura de Archivos

```
proyecto/
├── 1_capturar.py           # Paso 1
├── 2_preparar_dataset.py   # Paso 2
├── 3_entrenar.py           # Paso 3
├── 4_probar_camara.py      # Paso 4
├── requirements.txt        # Dependencias
│
├── dataset/                # Generado en Paso 1
│   ├── tornillo/
│   │   ├── images/         # 1000 imágenes
│   │   └── labels/         # 1000 etiquetas
│   ├── tuerca/
│   └── ...
│
├── yolo_dataset/           # Generado en Paso 2
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
└── runs/detect/            # Generado en Paso 3
    └── modelo_concurso/
        └── weights/
            └── best.pt     # ← TU MODELO ENTRENADO
```

---

## 💡 Tips para el Concurso

### Antes del Concurso
1. **Practicar el flujo completo** con objetos de prueba
2. **Verificar que la cámara funciona** correctamente
3. **Tener laptop con GPU** (mucho más rápido)
4. **Instalar todas las dependencias** con anticipación

### Durante la Captura
1. **20 posiciones diferentes por objeto:**
   - Esquinas (4): arr-izq, arr-der, abj-izq, abj-der
   - Bordes (4): arriba, abajo, izquierda, derecha
   - Centro (3): cerca, medio, lejos
   - Diagonales (4): 4 diagonales
   - Libres (5): posiciones intermedias

2. **Etiquetar rápido:**
   - Dibuja rectángulo ajustado al objeto
   - No necesita ser perfecto
   - 10-15 segundos por etiqueta

3. **Variar iluminación (opcional):**
   - Enciende/apaga luces entre fotos del mismo lote
   - Mueve ligeramente el objeto
   - Cambios sutiles de ángulo

### Durante el Entrenamiento
- **Mientras entrena:** Captura el siguiente objeto
- **Monitorea el progreso:** Verifica que mAP50 > 0.75
- **Si mAP50 < 0.70:** Revisa las etiquetas

### Durante las Pruebas
- **Prueba con objetos reales** de la competencia
- **Ajusta confianza** según necesites
- **Si no detecta:** Baja confianza con `-`
- **Si detecta de más:** Sube confianza con `+`

---

## 🎯 Métricas Objetivo

Al finalizar el entrenamiento, deberías ver:

```
mAP50:     > 0.75  (Mínimo aceptable)
mAP50:     > 0.85  (Bueno)
mAP50:     > 0.90  (Excelente)
```

Si es menor a 0.75:
1. Verifica que las etiquetas sean correctas
2. Considera capturar más variaciones
3. Entrena más epochs (150-200)

---

## 🚨 Solución de Problemas

### "No se puede abrir la cámara"
```bash
# Verificar cámaras disponibles
ls /dev/video*

# Probar con cámara 1 en vez de 0
# Edita los archivos y cambia: camara_id=0 por camara_id=1
```

### "Error: No existe el modelo"
```bash
# Verifica que el entrenamiento terminó
ls runs/detect/modelo_concurso/weights/best.pt
```

### "Detecciones muy lentas"
- Usa GPU si es posible
- Reduce batch size a 8
- Usa yolov8n (más rápido) en vez de yolov8s

### "No detecta nada"
- Baja el umbral de confianza en Paso 4
- Verifica que el objeto sea uno de los 6 entrenados
- Revisa que las etiquetas estén correctas

---

## 📝 Checklist del Día del Concurso

**2 horas antes:**
- [ ] Laptop cargada
- [ ] Cámara funcionando
- [ ] Dependencias instaladas
- [ ] Scripts probados
- [ ] Espacio en disco > 10GB

**Durante:**
- [ ] Capturar 6 objetos (72 min)
- [ ] Preparar dataset (2 min)
- [ ] Entrenar modelo (15 min)
- [ ] Probar y ajustar (3 min)

**Total: ~92 minutos**

---

## 🎓 Comandos Rápidos

```bash
# Flujo completo
python 1_capturar.py        # Repetir 6 veces
python 2_preparar_dataset.py
python 3_entrenar.py
python 4_probar_camara.py

# Verificar progreso
ls dataset/*/images/*.jpg | wc -l  # Cuenta imágenes capturadas

# Ver modelo entrenado
ls runs/detect/modelo_concurso/weights/

# Exportar modelo (si necesitan formato específico)
from ultralytics import YOLO
model = YOLO('runs/detect/modelo_concurso/weights/best.pt')
model.export(format='onnx')  # o 'tflite', 'torchscript'
```

---

## ⚡ Optimizaciones Avanzadas

### Si tienes MÁS tiempo:
- Captura 2000 fotos por objeto (40 lotes)
- Entrena 200 epochs
- Usa yolov8s en vez de yolov8n (más preciso)

### Si tienes MENOS tiempo:
- Captura 500 fotos por objeto (10 lotes)
- Entrena 50 epochs
- Riesgo: menor precisión

---

**¡Buena suerte en la competencia! 🏆**
