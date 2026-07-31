# 👁️ DeepSight V2 — Teachable YOLO Desktop

![Tauri v2](https://img.shields.io/badge/Framework-Tauri%20v2-blue?logo=tauri)
![Rust](https://img.shields.io/badge/Backend-Rust%202021-orange?logo=rust)
![Python](https://img.shields.io/badge/AI%20Engine-Python%203.10%2B-yellow?logo=python)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20YOLO11%2Fv8-green)
![License](https://img.shields.io/badge/License-ISC-brightgreen)

> **Documentación Orientada a Agentes de IA y Desarrolladores**  
> Este documento detalla la arquitectura integral, tecnologías, dependencias, flujos de datos e interfaz IPC de **DeepSight**, diseñado para que tanto desarrolladores humanos como **agentes de Inteligencia Artificial** puedan comprender e interactuar rápidamente con el proyecto en todos sus niveles de abstracción.

---

## 📋 Tabla de Contenidos

1. [Resumen del Proyecto](#-resumen-del-proyecto)
2. [Arquitectura General del Sistema](#-arquitectura-general-del-sistema)
3. [Stack Tecnológico y Dependencias](#-stack-tecnológico-y-dependencias)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Módulos Principales y Responsabilidades](#-módulos-principales-y-responsabilidades)
6. [Interfaz IPC (Comandos y Eventos Tauri)](#-interfaz-ipc-comandos-y-eventos-tauri)
7. [Flujo de Trabajo End-to-End (Data Flow)](#-flujo-de-trabajo-end-to-end-data-flow)
8. [Configuración del Entorno y Ejecución](#-configuración-del-entorno-y-ejecución)
9. [Guía y Protocolos para Agentes de IA](#-guía-y-protocolos-para-agentes-de-ia)

---

## 🎯 Resumen del Proyecto

**DeepSight V2** es una aplicación de escritorio nativa multiplataforma que permite a usuarios (incluso sin conocimientos técnicos) construir, entrenar y probar modelos de clasificación de visión por computadora basados en la arquitectura **YOLO** (You Only Look Once / Ultralytics YOLO11/v8) directamente en su máquina local.

### Características Clave

- **Transfer Learning Local**: Entrena clasificadores personalizados con conjuntos pequeños de imágenes en cuestión de segundos/minutos.
- **Sin Dependencias de la Nube**: Funciona 100% offline (sin envío de datos a servidores externos, telemetría ni pesos remotos durante el uso estándar).
- **Ajuste Dinámico de Hiperparámetros**: El motor Python evalúa la cantidad y distribución de imágenes por clase para configurar automáticamente épocas, data augmentation (rotación, brillo, escalado) y capas congeladas (_freeze_).
- **Feedback en Tiempo Real**: Telemetría continua emitida vía subprocesos asíncronos y eventos IPC desde el motor Python al Frontend.

---

## 🏗️ Arquitectura General del Sistema

DeepSight utiliza una arquitectura híbrida de **3 capas desacopladas**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA 1: FRONTEND (UI/UX)                     │
│  - Vanilla JavaScript (ES6+), HTML5, CSS3 Glassmorphism          │
│  - Renderizado de interfaz, control de estados UI y métricas    │
│  - Comunicación IPC vía @tauri-apps/api (window.__TAURI__)      │
└────────────────────────────────────────┬────────────────────────┘
                                         │ Comandos IPC / Eventos
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              CAPA 2: MOTOR DE ESCRITORIO (RUST)                 │
│  - Tauri v2 + Tokio Async Runtime                               │
│  - Gestión de estado del sistema (AppState + Mutex)             │
│  - Sanitización de rutas/nombres y staging de dataset temporal  │
│  - Orquestación de subprocesos Python (BufReader + MPSC)        │
│  - Telemetría del sistema (sysinfo)                             │
└────────────────────────────────────────┬────────────────────────┘
                                         │ Subproceso async (std/tokio::process)
                                         │ JSON streaming vía STDOUT
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               CAPA 3: NÚCLEO IA (PYTHON + YOLO)                 │
│  - python-core/train.py & infer.py                              │
│  - PyTorch + Ultralytics YOLO (yolo11n-cls.pt / yolov8n-cls.pt)  │
│  - Algoritmo de Hiperparámetros Dinámicos (Fast Augment)         │
│  - Inferencia y exportación de pesos (.pt)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnológico y Dependencias

### 1. Frontend (Capa de Presentación)

- **Lenguaje**: JavaScript (ES6 Vanilla), HTML5, CSS3.
- **Librerías**: `@tauri-apps/api` (v2.11.1) para puente IPC nativo.
- **Estilos**: Vanilla CSS con diseño Glassmorphism, animaciones CSS avanzadas, tipografía moderna y tema oscuro dinámico.

### 2. Backend Nativo (Capa de Control y Orquestación)

- **Lenguaje**: Rust 2021 (Rustc 1.70+).
- **Framework**: Tauri v2 (`tauri` v2.x).
- **Plugins de Tauri**: `tauri-plugin-dialog`, `tauri-plugin-shell`, `tauri-plugin-fs`.
- **Crates Clave**:
  - `tokio`: Runtime asíncrono para concurrencia no bloqueante.
  - `sysinfo`: Recolección de telemetría de hardware (RAM, CPU, S.O.).
  - `serde` / `serde_json`: Serialización y deserialización de estructuras IPC y JSON.
  - `regex` / `unicode-normalization`: Sanitización estricta de cadenas UTF-8 y nombres de archivos/clases.
  - `uuid`, `tempfile`, `dirs`, `anyhow`, `thiserror`.

### 3. Núcleo de Inteligencia Artificial (Engine Python)

- **Lenguaje**: Python 3.10+.
- **Librerías**:
  - `ultralytics`: Framework YOLO (versiones 11 / 8 para clasificación `n-cls`).
  - `torch` / `torchvision`: Framework de Deep Learning e inferencia en GPU/CPU.
  - `numpy`, `Pillow` (PIL): Procesamiento y transformación de tensores/imágenes.

---

## 📁 Estructura del Proyecto

```
DeepSight_Rust/
├── src/                        # Capa Frontend (UI Web App)
│   ├── index.html              # Estructura principal y layout SPA
│   ├── styles.css              # Sistema de diseño CSS Glassmorphism y temas
│   └── app.js                  # Lógica UI, manejo de estado cliente e invocación IPC
├── src-tauri/                  # Capa Backend Nativo (Rust + Tauri v2)
│   ├── Cargo.toml              # Manifiesto y dependencias de Rust
│   ├── tauri.conf.json         # Configuración de Tauri (ventanas, permisos, plugins)
│   ├── build.rs                # Script de compilación Tauri
│   └── src/
│       ├── main.rs             # Entrypoint Rust, estado global (AppState) y comandos IPC
│       ├── process.rs          # Preparación de dataset, ejecución Python y stream de eventos
│       └── system_info.rs      # Diagnóstico de hardware (CPU, RAM, S.O.)
├── python-core/                # Capa Núcleo IA (Scripts de Python)
│   ├── train.py                # Script de entrenamiento YOLO, hiperparámetros y emisor JSON
│   ├── infer.py                # Script de inferencia y análisis de probabilidades tensor
│   └── models/                 # Modelos base preliminares (.pt)
├── package.json                # Configuración de paquetes Node y CLI Tauri
└── pnpm-lock.yaml              # Lockfile de gestor de paquetes pnpm
```

---

## ⚙️ Módulos Principales y Responsabilidades

### Backend (Rust)

- **`src-tauri/src/main.rs`**:
  - Administra la estructura de estado mutable de forma segura entre hilos (`AppState` con `Mutex`):
    - `latest_best_pt`: Ruta absoluta al último modelo `.pt` generado exitosamente.
    - `latest_class_names`: Lista ordenada de nombres de clases entrenadas.
    - `is_training`: Booleano atómico que bloquea inferencias durante el entrenamiento.
    - `training_temp_dir`: Ruta del directorio temporal activo.
  - Registra los handlers de comandos accesibles desde el frontend.
- **`src-tauri/src/process.rs`**:
  - `find_python_interpreter()`: Localiza el ejecutable de Python en el PATH del sistema.
  - `prepare_temp_dataset()`: Copia y normaliza la estructura de carpetas necesaria para PyTorch/YOLO.
  - `run_training()` / `run_inference()`: Lanza procesos hijo `tokio::process::Command` y canaliza la salida JSON línea a línea con `BufReader`.
- **`src-tauri/src/system_info.rs`**:
  - Consulta métricas de hardware mediante `sysinfo::System`.

### IA Core (Python)

- **`python-core/train.py`**:
  - Desactiva explícitamente telemetría remota (`YOLO_OFFLINE=1`, `ULTRALYTICS_DISABLE_TELEMETRY=1`, `WANDB_DISABLED=true`).
  - `determine_hyperparameters()`: Calcula épocas, tamaño de lote (_batch_), aumentos y capas congeladas según el volumen de imágenes por clase.
  - Captura callbacks de Ultralytics e imprime cadenas JSON formateadas en `stdout` (`type: progress`, `type: log`, `type: complete`).
- **`python-core/infer.py`**:
  - `flatten_probs_tensor()`: Normaliza tensores PyTorch/NumPy unidimensionales o multidimensionales.
  - Procesa la imagen solicitada y retorna un top de predicciones con porcentaje de confianza.

---

## 🔌 Interfaz IPC (Comandos y Eventos Tauri)

### Comandos Invocables (`tauri::command`)

| Comando               | Parámetros                 | Retorno                 | Descripción                                                          |
| --------------------- | -------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `get_system_info`     | Ninguno                    | `SystemInfo`            | Obtiene uso de CPU, núcleos, RAM total/libre y S.O.                  |
| `check_python`        | Ninguno                    | `SanitizeResult`        | Verifica disponibilidad de intérprete Python en el sistema.          |
| `sanitize_name`       | `name: String`             | `SanitizeResult`        | Sanitiza cadenas eliminando caracteres inválidos/acentos.            |
| `has_trained_model`   | Ninguno                    | `bool`                  | Retorna `true` si existe un modelo `best.pt` listo en memoria/disco. |
| `start_training`      | `classes: Vec<ClassInput>` | `StartTrainingResponse` | Prepara el dataset y lanza la tarea asíncrona de entrenamiento.      |
| `run_test_inference`  | `image_path: String`       | `InferenceResult`       | Corre inferencia de una imagen contra el modelo actual.              |
| `clear_trained_model` | Ninguno                    | `()`                    | Limpia el modelo cargado del estado.                                 |
| `copy_best_pt`        | `from: String, to: String` | `()`                    | Copia el archivo de pesos `.pt` a la ruta elegida por el usuario.    |

### Eventos Emitidos (Rust ➔ Frontend)

| Evento                     | Payload                                   | Descripción                                                       |
| -------------------------- | ----------------------------------------- | ----------------------------------------------------------------- |
| `training:progress`        | `{ stage, epoch, total_epochs, percent }` | Progreso por época durante el entrenamiento.                      |
| `training:log`             | `{ level, message }`                      | Logs en tiempo real emitidos por la ejecución de Python.          |
| `training:hyperparameters` | Objeto JSON con hiperparámetros           | Configuración calculada para la sesión de entrenamiento.          |
| `training:complete`        | `TrainingResult`                          | Emitido al finalizar exitosamente la generación del modelo.       |
| `training:error`           | `{ message, raw }`                        | Notificación de error crítico durante la preparación o ejecución. |

---

## 🔄 Flujo de Trabajo End-to-End (Data Flow)

1. **Configuración en Frontend**: El usuario agrega al menos 2 clases con sus respectivas imágenes desde la UI.
2. **Solicitud de Entrenamiento**:
   - `app.js` llama a `invoke('start_training', { classes })`.
   - Rust valida que `is_training == false` y sanitiza los nombres de clases y archivos.
3. **Staging de Dataset**:
   - Rust genera un directorio temporal (`temp_dataset/`) organizando las imágenes en carpetas por clase (`temp_dataset/class_a/`, `temp_dataset/class_b/`).
4. **Ejecución del Motor Python**:
   - Rust ejecuta `python python-core/train.py --dataset_dir <ruta_temp> --class_names <clases>`.
   - Python evalúa el dataset y configura los hiperparámetros óptimos.
   - Ultralytics entrena el modelo `yolo11n-cls.pt`.
5. **Streaming de Métricas**:
   - Cada época genera una línea JSON en `stdout`.
   - El hilo asíncrono en `process.rs` lee cada línea y la reemite al Frontend con `app_clone.emit(...)`.
   - `app.js` actualiza las barras de progreso, velocímetros y consolas de logs.
6. **Inferencia y Prueba**:
   - Al finalizar, la ruta de `best.pt` se guarda en `AppState.latest_best_pt`.
   - El usuario selecciona una imagen de prueba ➔ llama a `invoke('run_test_inference', { image_path })`.
   - `infer.py` clasifica la imagen y devuelve las probabilidades ordenadas.

---

## 🚀 Configuración del Entorno y Ejecución

### Requisitos Previos

1. **Node.js** (v18+) y gestor de paquetes **pnpm** (`npm install -g pnpm`).
2. **Rust** (edición 2021) instalado vía [rustup](https://rustup.rs/).
3. **Python 3.10+** agregado al PATH del sistema.
4. Paquetes Python necesarios:
   ```bash
   pip install ultralytics torch torchvision pillow numpy
   ```

### Comandos de Desarrollo

- **Instalar dependencias frontend/desktop**:
  ```bash
  pnpm install
  ```
- **Iniciar aplicación en modo desarrollo (Tauri Dev)**:
  ```bash
  pnpm tauri dev
  ```
- **Compilar ejecutable de producción**:
  ```bash
  pnpm tauri build
  ```

---

## 🤖 Guía y Protocolos para Agentes de IA

Si eres un agente de Inteligencia Artificial que trabaja en este repositorio, **sigue estrictamente estas reglas y patrones de diseño**:

### 1. Principio de Separación de Capas (Decoupling)

- **NO** intentes ejecutar código de PyTorch o manipulaciones pesadas de imágenes directamente dentro de Rust ni JS. Mantén toda la lógica de IA en `python-core/`.
- La comunicación entre Rust y Python se realiza **exclusivamente** mediante procesos secundarios y streams JSON formateados por `stdout`.

### 2. Formato Mandatorio de Salida en Python (`print_json`)

- Cada mensaje producido por los scripts de Python que requiera interpretación en el backend/frontend debe ser emitido con `print_json()` en una única línea de `stdout`.
- Ejemplo:
  ```python
  print(json.dumps({"type": "progress", "epoch": 1, "total_epochs": 15}), flush=True)
  ```

### 3. Manejo de Rutas Multiplataforma y Sanitización

- En Rust, utiliza siempre `resolve_project_root()` y `resolve_python_core_dir()` de `process.rs` para asegurar compatibilidad tanto en `tauri dev` como en binarios compilados en producción.
- Aplica `sanitize_class_name()` y `sanitize_filename()` a todas las entradas de usuario para prevenir errores de caracteres UTF-8 o inyecciones en sistemas de archivos Windows/Linux/macOS.

### 4. Seguridad de Estado y Concurrencia

- La variable `AppState.is_training` actúa como guardián de concurrencia. **Nunca** permitas que se inicie una inferencia o un nuevo entrenamiento si `is_training` se encuentra en `true`.
- Al modificar variables dentro de `AppState`, asegura siempre el bloqueo del `Mutex` y libera el guardián de forma segura.

### 5. Incorporación de Nuevos Comandos IPC

Para agregar una nueva funcionalidad expuesta al Frontend:

1. Define la función en Rust con el atributo `#[tauri::command]`.
2. Incluye el comando en la lista `invoke_handler![...]` dentro de `src-tauri/src/main.rs`.
3. Invoca la función desde `app.js` usando `getTauriCore().invoke('nombre_comando', { args })`.

---

_DeepSight V2 — Mantenido para desarrollo ágil y colaboración asistida por IA._
