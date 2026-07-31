# Carpeta de Modelos Offline

Coloca aquí el archivo de pesos base YOLO26n-cls:

**Archivo requerido:** `yolo26n-cls.pt`

## ¿Cómo obtenerlo?

Debes descargar manualmente el modelo una única vez en un PC con internet y luego copiarlo a esta carpeta. DeepSight es 100% offline y NUNCA descargará pesos en tiempo de ejecución (para evitar falsos positivos de antivirus y uso de RAM).

### Método 1: Usando Python/Ultralytics (PC con internet)
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n-cls.pt')"
```
Luego busca `yolov8n-cls.pt` en tu carpeta personal y renómbralo a `yolo26n-cls.pt`.

### Método 2: Desde releases oficiales (opcional)
Si necesitas compatibilidad absoluta, usa `yolov8n-cls.pt` y renombralo. El script `train.py` cargara cualquier clasificador YOLO compatible que pongas con este nombre de archivo.

## Estructura final esperada:
```
python-core/
├── models/
│   └── yolo26n-cls.pt   <-- PEGA AQUÍ EL ARCHIVO
├── train.py
└── infer.py
```
