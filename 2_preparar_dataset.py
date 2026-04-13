"""
PASO 2: PREPARAR DATASET
Organiza las imágenes capturadas en estructura YOLOv8

ENTRADA:
dataset/
├── objeto1/
│   ├── images/
│   └── labels/
├── objeto2/
│   ├── images/
│   └── labels/
...

SALIDA:
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
"""

from pathlib import Path
import shutil
import random
import yaml

class PreparadorDataset:
    def __init__(self, carpeta_origen="dataset", carpeta_destino="yolo_dataset", split=0.8):
        self.carpeta_origen = Path(carpeta_origen)
        self.carpeta_destino = Path(carpeta_destino)
        self.split = split
        
        # Detectar clases
        self.clases = self._detectar_clases()
        
        if not self.clases:
            raise ValueError(f"❌ No se encontraron objetos en {carpeta_origen}")
        
        print(f"✅ Detectadas {len(self.clases)} clases: {', '.join(self.clases)}")
        
        # Crear estructura
        self._crear_directorios()
    
    def _detectar_clases(self):
        """Detecta automáticamente las clases"""
        clases = []
        for item in self.carpeta_origen.iterdir():
            if item.is_dir():
                img_dir = item / "images"
                lab_dir = item / "labels"
                if img_dir.exists() and lab_dir.exists():
                    if list(img_dir.glob("*.jpg")) and list(lab_dir.glob("*.txt")):
                        clases.append(item.name)
        return sorted(clases)
    
    def _crear_directorios(self):
        """Crea estructura de directorios"""
        dirs = [
            self.carpeta_destino / "images" / "train",
            self.carpeta_destino / "images" / "val",
            self.carpeta_destino / "labels" / "train",
            self.carpeta_destino / "labels" / "val"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _actualizar_class_id(self, label_path, class_id):
        """Actualiza el class_id en la etiqueta"""
        with open(label_path, 'r') as f:
            lineas = f.readlines()
        
        nuevas = []
        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) == 5:
                partes[0] = str(class_id)
                nuevas.append(' '.join(partes) + '\n')
        
        with open(label_path, 'w') as f:
            f.writelines(nuevas)
    
    def procesar_clase(self, nombre_clase, class_id):
        """Procesa una clase"""
        carpeta_clase = self.carpeta_origen / nombre_clase
        img_dir = carpeta_clase / "images"
        lab_dir = carpeta_clase / "labels"
        
        # Obtener imágenes
        imagenes = sorted(list(img_dir.glob("*.jpg")))
        
        if not imagenes:
            print(f"⚠️  No hay imágenes para {nombre_clase}")
            return 0, 0
        
        # Mezclar y dividir
        random.shuffle(imagenes)
        split_idx = int(len(imagenes) * self.split)
        
        train_imgs = imagenes[:split_idx]
        val_imgs = imagenes[split_idx:]
        
        print(f"\n📦 {nombre_clase} (clase {class_id})")
        print(f"   Total: {len(imagenes)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
        
        # Procesar train
        for img_path in train_imgs:
            # Copiar imagen
            dest_img = self.carpeta_destino / "images" / "train" / f"{nombre_clase}_{img_path.name}"
            shutil.copy2(img_path, dest_img)
            
            # Copiar y actualizar etiqueta
            label_src = lab_dir / f"{img_path.stem}.txt"
            label_dst = self.carpeta_destino / "labels" / "train" / f"{nombre_clase}_{img_path.stem}.txt"
            
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
                self._actualizar_class_id(label_dst, class_id)
        
        # Procesar val
        for img_path in val_imgs:
            dest_img = self.carpeta_destino / "images" / "val" / f"{nombre_clase}_{img_path.name}"
            shutil.copy2(img_path, dest_img)
            
            label_src = lab_dir / f"{img_path.stem}.txt"
            label_dst = self.carpeta_destino / "labels" / "val" / f"{nombre_clase}_{img_path.stem}.txt"
            
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
                self._actualizar_class_id(label_dst, class_id)
        
        return len(train_imgs), len(val_imgs)
    
    def generar_data_yaml(self):
        """Genera archivo data.yaml"""
        config = {
            'path': str(self.carpeta_destino.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.clases),
            'names': self.clases
        }
        
        yaml_path = self.carpeta_destino / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ data.yaml generado: {yaml_path}")
        
        return yaml_path
    
    def preparar(self):
        """Ejecuta preparación completa"""
        print("\n" + "="*60)
        print("  PREPARANDO DATASET PARA YOLOV8")
        print("="*60)
        
        total_train = 0
        total_val = 0
        
        # Procesar cada clase
        for idx, clase in enumerate(self.clases):
            train, val = self.procesar_clase(clase, idx)
            total_train += train
            total_val += val
        
        # Generar data.yaml
        yaml_path = self.generar_data_yaml()
        
        # Resumen
        print("\n" + "="*60)
        print("  ✅ DATASET PREPARADO")
        print("="*60)
        print(f"\n📊 Estadísticas:")
        print(f"   Clases: {len(self.clases)}")
        print(f"   Train: {total_train} imágenes")
        print(f"   Val: {total_val} imágenes")
        print(f"   Total: {total_train + total_val} imágenes")
        print(f"\n📁 Ubicación: {self.carpeta_destino.absolute()}")
        print(f"📄 Config: {yaml_path}")
        print("="*60)
        
        return yaml_path


def main():
    print("\n" + "="*60)
    print("  PASO 2: PREPARAR DATASET")
    print("="*60)
    
    # Configuración
    carpeta_origen = input("\n📁 Carpeta con objetos (default: dataset): ").strip()
    if not carpeta_origen:
        carpeta_origen = "dataset"
    
    if not Path(carpeta_origen).exists():
        print(f"❌ No existe: {carpeta_origen}")
        return
    
    carpeta_destino = input("📁 Carpeta destino (default: yolo_dataset): ").strip()
    if not carpeta_destino:
        carpeta_destino = "yolo_dataset"
    
    # Seed para reproducibilidad
    random.seed(42)
    
    # Preparar
    preparador = PreparadorDataset(carpeta_origen, carpeta_destino)
    yaml_path = preparador.preparar()
    
    print("\n✅ Listo para el Paso 3: Entrenar Modelo")
    print("   Ejecuta: python 3_entrenar.py\n")


if __name__ == "__main__":
    main()
