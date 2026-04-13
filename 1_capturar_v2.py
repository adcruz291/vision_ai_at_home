"""
PASO 1: CAPTURA POR LOTES
Sistema optimizado para concursos - Captura 1000 imágenes en ~12 minutos

CÓMO FUNCIONA:
- 20 lotes × 50 fotos = 1000 imágenes
- Colocas objeto en una posición (guía visual en pantalla)
- Capturas 50 fotos (varía iluminación si quieres)
- Etiquetas 1 foto (15 segundos)
- Mueves objeto a OTRA posición
- Repites 20 veces

RESULTADO: 1000 fotos correctamente etiquetadas
"""

import cv2
import time
from pathlib import Path
import numpy as np


# ── Detección y selección de cámara ──────────────────────────────────────────

def detectar_camaras(max_check=10):
    """Detecta índices de cámaras disponibles en el sistema"""
    encontradas = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                encontradas.append(i)
            cap.release()
    return encontradas


def seleccionar_camara():
    """Muestra cámaras detectadas y deja elegir una; retorna el índice."""
    print("\n🔍 Detectando cámaras disponibles...")
    camaras = detectar_camaras()

    if not camaras:
        print("❌ No se detectó ninguna cámara en el sistema")
        return None

    if len(camaras) == 1:
        print(f"✅ Cámara detectada: /dev/video{camaras[0]}")
        return camaras[0]

    print(f"\n📷 Cámaras disponibles ({len(camaras)}):")
    for i, cam_id in enumerate(camaras):
        print(f"   [{i + 1}] /dev/video{cam_id}")

    while True:
        try:
            opcion = input(f"\n🎥 Selecciona cámara (1-{len(camaras)}): ").strip()
            idx = int(opcion) - 1
            if 0 <= idx < len(camaras):
                print(f"✅ Usando /dev/video{camaras[idx]}")
                return camaras[idx]
            print(f"❌ Opción inválida, elige entre 1 y {len(camaras)}")
        except ValueError:
            print("❌ Ingresa un número válido")


# ── Guías visuales por lote ───────────────────────────────────────────────────

_MAPA_POSICIONES = {
    1:  ("Esquina Superior-Izquierda", "esquina_tl"),
    2:  ("Esquina Superior-Derecha",   "esquina_tr"),
    3:  ("Esquina Inferior-Izquierda", "esquina_bl"),
    4:  ("Esquina Inferior-Derecha",   "esquina_br"),
    5:  ("Borde Superior",             "borde_t"),
    6:  ("Borde Inferior",             "borde_b"),
    7:  ("Borde Izquierdo",            "borde_l"),
    8:  ("Borde Derecho",              "borde_r"),
    9:  ("Centro - Objeto cerca",      "centro_cerca"),
    10: ("Centro - Distancia media",   "centro_medio"),
    11: ("Centro - Objeto lejos",      "centro_lejos"),
    12: ("Centro - Ángulo lateral",    "centro_medio"),
    13: ("Diagonal ↘ - primer tercio", "diag_a"),
    14: ("Diagonal ↙ - primer tercio", "diag_b"),
    15: ("Diagonal ↘ - segundo tercio","diag_c"),
    16: ("Diagonal ↙ - segundo tercio","diag_d"),
    17: ("Intermedia Superior",        "inter_t"),
    18: ("Intermedia Inferior",        "inter_b"),
    19: ("Intermedia Izquierda",       "inter_l"),
    20: ("Intermedia Derecha",         "inter_r"),
}


def _info_posicion(num_lote):
    return _MAPA_POSICIONES.get(num_lote, (f"Posición {num_lote}", "centro_medio"))


def _rect_guia(tipo, w, h):
    """Calcula (x1,y1,x2,y2) del rectángulo guía según el tipo de posición."""
    margen = 30
    ow = int(w * 0.22)
    oh = int(h * 0.28)

    anchors = {
        "esquina_tl": (margen, margen),
        "esquina_tr": (w - margen - ow, margen),
        "esquina_bl": (margen, h - margen - oh),
        "esquina_br": (w - margen - ow, h - margen - oh),
        "borde_t":    ((w - ow) // 2, margen),
        "borde_b":    ((w - ow) // 2, h - margen - oh),
        "borde_l":    (margen, (h - oh) // 2),
        "borde_r":    (w - margen - ow, (h - oh) // 2),
        "diag_a":     (int(w * 0.08),  int(h * 0.08)),
        "diag_b":     (int(w * 0.70),  int(h * 0.08)),
        "diag_c":     (int(w * 0.30),  int(h * 0.38)),
        "diag_d":     (int(w * 0.48),  int(h * 0.38)),
        "inter_t":    (int(w * 0.38),  int(h * 0.04)),
        "inter_b":    (int(w * 0.38),  int(h * 0.72)),
        "inter_l":    (int(w * 0.04),  int(h * 0.36)),
        "inter_r":    (int(w * 0.74),  int(h * 0.36)),
    }

    # Variantes de centro con tamaño diferente para simular distancia
    if tipo == "centro_cerca":
        ow, oh = int(ow * 1.5), int(oh * 1.5)
        x1, y1 = (w - ow) // 2, (h - oh) // 2
    elif tipo == "centro_lejos":
        ow, oh = int(ow * 0.6), int(oh * 0.6)
        x1, y1 = (w - ow) // 2, (h - oh) // 2
    elif tipo == "centro_medio":
        x1, y1 = (w - ow) // 2, (h - oh) // 2
    else:
        x1, y1 = anchors.get(tipo, ((w - ow) // 2, (h - oh) // 2))

    return x1, y1, x1 + ow, y1 + oh


def dibujar_guia(frame, num_lote):
    """Superpone guía visual en frame; retorna frame modificado y nombre de pos."""
    h, w = frame.shape[:2]
    nombre, tipo = _info_posicion(num_lote)
    x1, y1, x2, y2 = _rect_guia(tipo, w, h)

    # Fondo semitransparente de la zona objetivo
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Borde del área objetivo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # Esquinas decorativas
    largo = 14
    grosor = 3
    color_esq = (255, 220, 0)
    for px, py, dx, dy in [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x1, y2,  1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * largo, py), color_esq, grosor)
        cv2.line(frame, (px, py), (px, py + dy * largo), color_esq, grosor)

    # Crosshair en el centro del área
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (0, 200, 255), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (0, 200, 255), 1)

    return frame, nombre


def dibujar_hud(frame, num_lote, num_lotes_total,
                nombre_pos="", estado="esperando",
                foto_actual=None, fotos_lote=None, fotos_total=0):
    """Dibuja barra superior e inferior con información del estado."""
    h, w = frame.shape[:2]

    # ── Barra superior ───────────────────────────────────────────────
    bar_h = 58
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Lote / posición
    cv2.putText(frame, f"Lote {num_lote}/{num_lotes_total}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, nombre_pos,
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)

    # Estado (lado derecho)
    if estado == "esperando":
        msg   = "ESPACIO: capturar  |  Q: salir"
        color = (180, 180, 180)
    elif estado == "capturando":
        msg   = f"Foto {foto_actual}/{fotos_lote}"
        color = (60, 255, 60)
    else:
        msg   = estado
        color = (255, 200, 0)

    tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    cv2.putText(frame, msg, (w - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    # ── Barra inferior ───────────────────────────────────────────────
    bot_h = 28
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bot_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)

    # Barra de progreso total
    if fotos_total > 0:
        total_esperado = num_lotes_total * (fotos_lote or 50)
        ratio = min(fotos_total / total_esperado, 1.0)
        bar_w = int((w - 20) * ratio)
        cv2.rectangle(frame, (10, h - bot_h + 6), (10 + bar_w, h - 6),
                      (0, 200, 100), -1)
        cv2.rectangle(frame, (10, h - bot_h + 6), (w - 10, h - 6),
                      (100, 100, 100), 1)
        prog_txt = f"Total capturas: {fotos_total}"
        cv2.putText(frame, prog_txt, (14, h - bot_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)

    return frame


# ── Clase principal ───────────────────────────────────────────────────────────

class CapturadorConcurso:
    def __init__(self, nombre_objeto, fotos_por_lote=50, num_lotes=20):
        self.nombre_objeto  = nombre_objeto
        self.fotos_por_lote = fotos_por_lote
        self.num_lotes      = num_lotes
        self.total_fotos    = fotos_por_lote * num_lotes

        # Crear carpetas
        self.carpeta_base   = Path("dataset") / nombre_objeto
        self.carpeta_images = self.carpeta_base / "images"
        self.carpeta_labels = self.carpeta_base / "labels"

        self.carpeta_images.mkdir(parents=True, exist_ok=True)
        self.carpeta_labels.mkdir(parents=True, exist_ok=True)

        # Estado de etiquetado
        self.bbox    = None
        self.drawing = False
        self.start_x = self.start_y = 0
        self.imagen_actual = None

        print(f"\n✅ Carpetas creadas: {self.carpeta_base}")

    # ── Etiquetado ────────────────────────────────────────────────────

    def mouse_callback(self, event, x, y, flags, param):
        """Dibuja bounding box con el mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.bbox = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.imagen_actual.copy()
            cv2.rectangle(img_copy, (self.start_x, self.start_y), (x, y),
                          (0, 255, 0), 2)
            cv2.imshow('Etiquetar', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self.bbox = (x1, y1, x2, y2)
                cv2.rectangle(self.imagen_actual, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.imshow('Etiquetar', self.imagen_actual)

    def bbox_to_yolo(self, bbox, img_w, img_h):
        """Convierte bbox a formato YOLO"""
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width    = (x2 - x1) / img_w
        height   = (y2 - y1) / img_h
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def etiquetar_lote(self, imagen_path):
        """Etiqueta una imagen del lote; retorna bbox o None si se cancela."""
        self.imagen_actual = cv2.imread(str(imagen_path))
        self.bbox = None

        cv2.namedWindow('Etiquetar')
        cv2.setMouseCallback('Etiquetar', self.mouse_callback)

        print("\n┌─────────────────────────────────────────┐")
        print("│  ETIQUETA EL OBJETO                     │")
        print("├─────────────────────────────────────────┤")
        print("│  • Dibuja rectángulo con el mouse       │")
        print("│  • ENTER: Confirmar                     │")
        print("│  • R: Reintentar                        │")
        print("└─────────────────────────────────────────┘")

        while True:
            cv2.imshow('Etiquetar', self.imagen_actual)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and self.bbox:  # ENTER
                cv2.destroyWindow('Etiquetar')
                return self.bbox
            elif key == ord('r'):  # Reintentar
                self.imagen_actual = cv2.imread(str(imagen_path))
                self.bbox = None

    # ── Espera con cámara viva ────────────────────────────────────────

    def _esperar_con_preview(self, cap, num_lote, fotos_total):
        """
        Muestra feed en vivo con guía hasta que el usuario presione ESPACIO.
        Retorna True para continuar, False para abortar (Q).
        """
        nombre, _ = _info_posicion(num_lote)

        print(f"\n📍 Coloca el objeto en: {nombre}")
        print(f"   (zona indicada en pantalla)")
        print(f"   ESPACIO: iniciar captura  |  Q: salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame, nombre_pos = dibujar_guia(frame, num_lote)
            frame = dibujar_hud(
                frame,
                num_lote      = num_lote,
                num_lotes_total = self.num_lotes,
                nombre_pos    = nombre_pos,
                estado        = "esperando",
                fotos_lote    = self.fotos_por_lote,
                fotos_total   = fotos_total,
            )

            cv2.imshow('Captura - Concurso', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                return True
            elif key == ord('q'):
                return False

    # ── Captura de un lote ────────────────────────────────────────────

    def capturar_lote(self, num_lote, cap, fotos_total_previas):
        """
        Captura un lote completo mostrando cámara en vivo con guías.
        Retorna número de fotos capturadas, o -1 si el usuario abortó.
        """
        print(f"\n{'='*60}")
        print(f"   LOTE {num_lote}/{self.num_lotes}")
        print(f"{'='*60}")

        if num_lote == 1:
            print("\n💡 SUGERENCIAS DE POSICIONES (20 lotes):")
            print("   Lotes 1-4:   Esquinas (arr-izq, arr-der, abj-izq, abj-der)")
            print("   Lotes 5-8:   Bordes (arriba, abajo, izquierda, derecha)")
            print("   Lotes 9-12:  Centro (cerca, medio, lejos)")
            print("   Lotes 13-16: Diagonales y ángulos")
            print("   Lotes 17-20: Posiciones intermedias")

        print(f"\n📸 Se capturarán {self.fotos_por_lote} fotos")
        print("💡 Puedes variar la iluminación entre fotos de este lote")

        # ── Esperar con cámara viva hasta que el usuario esté listo ──
        continuar = self._esperar_con_preview(cap, num_lote, fotos_total_previas)
        if not continuar:
            print("\n🛑 Captura abortada por el usuario")
            return -1

        # ── Capturar fotos ────────────────────────────────────────────
        imagenes_lote = []
        inicio_idx    = (num_lote - 1) * self.fotos_por_lote
        nombre_pos, _ = _info_posicion(num_lote)

        print(f"\n🎬 Capturando {self.fotos_por_lote} fotos...")

        for i in range(self.fotos_por_lote):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Error capturando foto {i + 1}")
                continue

            # Guardar imagen
            idx    = inicio_idx + i
            nombre = f"{self.nombre_objeto}_{idx:04d}.jpg"
            ruta   = self.carpeta_images / nombre
            cv2.imwrite(str(ruta), frame)
            imagenes_lote.append(ruta)

            # Preview con HUD
            preview = frame.copy()
            preview, _ = dibujar_guia(preview, num_lote)
            preview = dibujar_hud(
                preview,
                num_lote        = num_lote,
                num_lotes_total = self.num_lotes,
                nombre_pos      = nombre_pos,
                estado          = "capturando",
                foto_actual     = i + 1,
                fotos_lote      = self.fotos_por_lote,
                fotos_total     = fotos_total_previas + len(imagenes_lote),
            )
            cv2.imshow('Captura - Concurso', preview)
            cv2.waitKey(1)

            if (i + 1) % 10 == 0:
                print(f"   ✓ {i + 1}/{self.fotos_por_lote}")

            time.sleep(0.2)

        print(f"✅ {len(imagenes_lote)} fotos capturadas")

        # ── Etiquetar (foto del medio del lote) ───────────────────────
        img_referencia = imagenes_lote[len(imagenes_lote) // 2]
        print(f"\n📝 Etiquetando: {img_referencia.name}")

        bbox = self.etiquetar_lote(img_referencia)
        if bbox is None:
            print("⚠️  Sin etiqueta — lote guardado sin labels")
            return len(imagenes_lote)

        # Aplicar etiqueta a todo el lote
        img  = cv2.imread(str(imagenes_lote[0]))
        h, w = img.shape[:2]
        yolo_label = self.bbox_to_yolo(bbox, w, h)

        for img_path in imagenes_lote:
            label_path = self.carpeta_labels / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write(yolo_label + '\n')

        print(f"✅ Lote {num_lote} etiquetado: {len(imagenes_lote)} archivos")

        return len(imagenes_lote)

    # ── Ejecución principal ───────────────────────────────────────────

    def ejecutar(self, camara_id):
        """Ejecuta el proceso completo"""
        print("\n" + "="*60)
        print("  CAPTURA POR LOTES - OPTIMIZADO PARA CONCURSOS")
        print("="*60)
        print(f"\n📦 Objeto: {self.nombre_objeto}")
        print(f"📊 Total: {self.total_fotos} fotos en {self.num_lotes} lotes")
        print(f"⏱️  Tiempo estimado: ~12 minutos\n")

        input("▶ Presiona ENTER para iniciar...")

        # Abrir cámara
        cap = cv2.VideoCapture(camara_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("\n❌ Error: No se puede abrir la cámara")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        total         = 0
        tiempo_inicio = time.time()

        for num_lote in range(1, self.num_lotes + 1):
            capturadas = self.capturar_lote(num_lote, cap, fotos_total_previas=total)

            if capturadas == -1:   # usuario abortó
                break

            total += capturadas

            # Estadísticas tras el lote
            tiempo_transcurrido = (time.time() - tiempo_inicio) / 60
            progreso = (num_lote / self.num_lotes) * 100

            print(f"\n📊 PROGRESO: {num_lote}/{self.num_lotes} lotes ({progreso:.1f}%)")
            print(f"📸 Fotos totales: {total}/{self.total_fotos}")
            print(f"⏱️  Tiempo: {tiempo_transcurrido:.1f} min")

            if num_lote < self.num_lotes:
                # La cámara sigue activa; mostramos cuenta regresiva en vivo
                print("\n⏭️  Siguiente lote en 3 segundos...")
                t_espera = time.time()
                while time.time() - t_espera < 3:
                    ret, frame = cap.read()
                    if ret:
                        restante = max(0, 3 - int(time.time() - t_espera))
                        cv2.putText(frame,
                                    f"Siguiente lote en {restante}s...",
                                    (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 200, 255), 2)
                        cv2.imshow('Captura - Concurso', frame)
                    cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        # Resumen final
        tiempo_total = (time.time() - tiempo_inicio) / 60

        print("\n" + "="*60)
        print("  ✅ CAPTURA COMPLETADA")
        print("="*60)
        print(f"\n📁 Ubicación: {self.carpeta_base}")
        print(f"📸 Imágenes: {total}")
        print(f"🏷️  Etiquetas: {total}")
        print(f"📍 Posiciones: {self.num_lotes}")
        print(f"⏱️  Tiempo total: {tiempo_total:.1f} minutos")
        print("\n" + "="*60)

        return True


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  PASO 1: CAPTURA DE IMÁGENES")
    print("="*60)

    nombre = input("\n📦 Nombre del objeto: ").strip()
    if not nombre:
        print("❌ Debes ingresar un nombre")
        return

    print("\n⚙️  Configuración:")
    print("   • Fotos por lote: 50")
    print("   • Número de lotes: 20")
    print("   • Total: 1000 fotos")

    continuar = input("\n¿Continuar? (s/n): ").strip().lower()
    if continuar != 's':
        print("❌ Cancelado")
        return

    camara_id = seleccionar_camara()
    if camara_id is None:
        return

    capturador = CapturadorConcurso(nombre)
    capturador.ejecutar(camara_id)

    print("\n✅ Listo para el Paso 2: Preparar Dataset")
    print("   Ejecuta: python 2_preparar_dataset.py\n")


if __name__ == "__main__":
    main()
