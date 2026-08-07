import json
import os
import sys
import shutil
import re
import tempfile
import uuid
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_DISABLE_TELEMETRY"] = "1"
os.environ["ULTRALYTICS_NO_UPDATE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["SYNC_WANDB"] = "0"

CLASS_SEPARATOR = "\x1f"
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def print_json(data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def print_progress(epoch, total_epochs, stage="training"):
    print_json({
        "type": "progress",
        "stage": stage,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "percent": round((epoch / total_epochs) * 100, 1) if total_epochs > 0 else 0
    })


def print_log(message, level="info"):
    print_json({
        "type": "log",
        "level": level,
        "message": str(message)
    })


def count_images_per_class(dataset_path):
    class_counts = {}
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        return class_counts
    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            count = 0
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                count += len(list(class_dir.glob(f"*{ext}")))
                count += len(list(class_dir.glob(f"*{ext.upper()}")))
            class_counts[class_dir.name] = count
    return class_counts


def determine_hyperparameters(class_counts):
    defaults = {
        "epochs": 15,
        "freeze": 4,
        "imgsz": 224,
        "batch": 4,
        "workers": 2,
        "augmentation_mode": "moderate",
        "degrees": 10.0,
        "fliplr": 0.5,
        "scale": 0.5,
        "brightness": 0.2,
        "mode_name": "standard"
    }
    if not class_counts:
        return defaults

    min_count = min(class_counts.values())
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)

    if min_count < 30:
        hp = {
            "epochs": 20,
            "freeze": 10,
            "imgsz": 224,
            "batch": 4,
            "workers": 2,
            "augmentation_mode": "aggressive",
            "degrees": 15.0,
            "fliplr": 0.5,
            "scale": 0.7,
            "brightness": 0.3,
            "mode_name": "fast_augment"
        }
    else:
        hp = {
            "epochs": 15,
            "freeze": 4,
            "imgsz": 224,
            "batch": 4,
            "workers": 2,
            "augmentation_mode": "moderate",
            "degrees": 10.0,
            "fliplr": 0.5,
            "scale": 0.5,
            "brightness": 0.2,
            "mode_name": "precision"
        }

    hp["min_images_per_class"] = min_count
    hp["total_images"] = total_images
    hp["num_classes"] = num_classes
    return hp


def sanitize_label(name):
    safe = re.sub(r"[^\w\.\- ]", "_", str(name), flags=re.UNICODE).strip()
    safe = safe.replace(" ", "_").replace("-", "_")
    return safe or "class"


def is_image_file(path, log_fn=None):
    """Valida que el archivo sea una imagen legible, no esté vacío y no esté corrupto.

    Realiza 4 comprobaciones en orden:
      1. Existencia y tipo de archivo
      2. Extensión en VALID_IMAGE_EXT
      3. Tamaño mínimo > 0 (un PNG válido mínimo son ~67 bytes)
      4. Apertura con Pillow + Image.verify()

    NOTA IMPORTANTE sobre Image.verify():
      El método verify() deja el objeto Image en un estado no reutilizable.
      Por diseño de Pillow, después de llamar verify() el handle interno queda
      cerrado/agotado. Si la imagen supera esta validación, quien la necesite
      para lectura o copia DEBE abrirla de nuevo con Image.open().
      Aquí solo validamos; la copia se hace vía shutil.copy2() sobre el path.

    Args:
        path: pathlib.Path al archivo a validar.
        log_fn: callable(message, level) para registrar el motivo de rechazo.
                Si es None, los rechazos se descartan silenciosamente.

    Returns:
        True si la imagen pasa todas las comprobaciones, False en caso contrario.
    """
    def _warn(msg):
        if log_fn is not None:
            log_fn(msg, "warning")

    # Capa 1: existencia y tipo
    if not path.exists():
        _warn(f"Imagen ignorada — no existe en disco: {path}. Acción: descartando.")
        return False
    if not path.is_file():
        _warn(f"Imagen ignorada — no es un archivo regular (puede ser directorio): {path}. Acción: descartando.")
        return False

    # Capa 2: extensión
    ext = path.suffix.lower()
    if ext not in VALID_IMAGE_EXT:
        _warn(
            f"Imagen ignorada — extensión '{ext}' no soportada en '{path.name}'. "
            f"Extensiones válidas: {sorted(VALID_IMAGE_EXT)}. Acción: descartando."
        )
        return False

    # Capa 3: tamaño mínimo
    try:
        size = path.stat().st_size
    except OSError as e:
        _warn(f"Imagen ignorada — no se pudo leer tamaño de '{path.name}': {type(e).__name__}: {e}. Acción: descartando.")
        return False
    if size == 0:
        _warn(f"Imagen ignorada — archivo vacío (0 bytes): '{path.name}'. Acción: descartando.")
        return False
    if size < 64:
        _warn(
            f"Imagen ignorada — archivo demasiado pequeño ({size} bytes) para ser una imagen válida: '{path.name}'. "
            f"Acción: descartando."
        )
        return False

    # Capa 4: apertura y verificación con Pillow
    # IMPORTANTE: verify() invalida el handle; no reutilizar el objeto Image.
    # La copia posterior usa shutil.copy2() sobre el Path, no sobre este handle.
    try:
        from PIL import Image, UnidentifiedImageError
        with Image.open(path) as img:
            img.verify()
        return True
    except UnidentifiedImageError as e:
        _warn(
            f"Imagen ignorada — Pillow no reconoce el formato de '{path.name}': "
            f"UnidentifiedImageError: {e}. Acción: descartando."
        )
        return False
    except Exception as e:
        _warn(
            f"Imagen ignorada — fallo en verify() para '{path.name}': "
            f"{type(e).__name__}: {e}. El archivo puede estar corrupto o truncado. Acción: descartando."
        )
        return False


def extract_class_index(class_dir):
    for image_path in sorted(class_dir.iterdir()):
        if not is_image_file(image_path):
            continue
        match = re.search(r"_(\d+)\.[^.]+$", image_path.name)
        if match:
            return int(match.group(1))
    return None


def map_class_dirs(dataset_dir, class_names):
    by_index = {}
    remaining = []

    for class_dir in sorted(dataset_dir.iterdir(), key=lambda p: p.name.lower()):
        if not class_dir.is_dir():
            continue
        class_idx = extract_class_index(class_dir)
        if class_idx is not None and 0 <= class_idx < len(class_names):
            by_index[class_idx] = class_dir
        else:
            remaining.append(class_dir)

    if remaining:
        for idx in range(len(class_names)):
            if idx not in by_index and remaining:
                by_index[idx] = remaining.pop(0)

    missing = [idx for idx in range(len(class_names)) if idx not in by_index]
    if missing:
        raise RuntimeError(f"No se pudieron mapear las clases del dataset. Faltan indices: {missing}")

    return by_index


def split_counts(total_images):
    if total_images <= 1:
        return 1, 0
    val_count = max(1, int(round(total_images * 0.2)))
    if val_count >= total_images:
        val_count = total_images - 1
    train_count = total_images - val_count
    return train_count, val_count


def prepare_classification_dataset(dataset_path, class_names):
    """Prepara el dataset de clasificación dividido en train/val.

    Tolerancia a imágenes inválidas:
      - Cada imagen se valida con is_image_file() antes de ser usada.
      - Las imágenes que no pasan la validación se ignoran individualmente
        con un warning; NO se cancela el entrenamiento por una sola imagen.
      - El entrenamiento solo se cancela si, tras el filtrado, una clase
        queda con CERO imágenes válidas.

    Manejo de errores de copia:
      - PermissionError y OSError se capturan con mensajes que identifican
        el archivo fuente y el destino exactos.
    """
    dataset_dir = Path(dataset_path).resolve()
    split_root = dataset_dir.parent / f"{dataset_dir.name}_cls"

    if split_root.exists():
        shutil.rmtree(split_root, ignore_errors=True)

    train_root = split_root / "train"
    val_root = split_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    class_dir_map = map_class_dirs(dataset_dir, class_names)

    for idx, class_name in enumerate(class_names):
        source_dir = class_dir_map[idx]

        # Recolectar todos los candidatos antes de filtrar
        try:
            candidates = sorted(source_dir.iterdir(), key=lambda p: p.name.lower())
        except PermissionError as e:
            raise RuntimeError(
                f"Sin permisos para leer la carpeta de clase '{class_name}' "
                f"en '{source_dir}': {e}"
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Error de sistema al leer carpeta de clase '{class_name}' "
                f"en '{source_dir}': {type(e).__name__}: {e}"
            ) from e

        # Filtrar con validación individual — una imagen inválida no cancela la clase.
        # Usamos un log_fn capturador: registra el warning normalmente Y almacena
        # el motivo asociado al archivo actual para el resumen final de clase.
        valid_images = []
        rejected_details = []   # lista de (nombre_archivo, motivo_breve)

        # _last_reason actúa como buffer de un solo elemento: is_image_file()
        # llama a log_fn exactamente una vez cuando rechaza un archivo.
        _last_reason: list = []

        def _capturing_log(msg, level):
            """Registra el mensaje normalmente y captura la razón de rechazo."""
            print_log(msg, level)
            if level == "warning":
                _last_reason.clear()
                _last_reason.append(msg)

        for candidate in candidates:
            _last_reason.clear()
            if is_image_file(candidate, log_fn=_capturing_log):
                valid_images.append(candidate)
            elif candidate.is_file():
                # Extraer motivo breve: quitar prefijo "Imagen ignorada — " para el resumen
                raw_reason = _last_reason[0] if _last_reason else "Razón no especificada"
                brief = raw_reason.replace("Imagen ignorada — ", "").split(". Acción:")[0]
                rejected_details.append((candidate.name, brief))

        # ── Resumen por clase ─────────────────────────────────────────────────
        total_candidates = len(valid_images) + len(rejected_details)
        print_log(
            f"=== Resumen de clase '{class_name}' ===",
            "info"
        )
        print_log(
            f"  Encontradas : {total_candidates} archivo(s)",
            "info"
        )
        print_log(
            f"  Aceptadas   : {len(valid_images)} imagen(es) válida(s)",
            "info"
        )
        print_log(
            f"  Descartadas : {len(rejected_details)} archivo(s)",
            "warning" if rejected_details else "info"
        )
        if rejected_details:
            print_log("  Detalle de descartadas:", "warning")
            for name, reason in rejected_details:
                print_log(f"    - {name} → {reason}", "warning")
        # ── Fin resumen ───────────────────────────────────────────────────────

        # Cancelar solo si la clase queda completamente sin imágenes válidas
        if not valid_images:
            raise RuntimeError(
                f"La clase '{class_name}' no contiene imágenes válidas tras el filtrado. "
                f"Se descartaron {len(rejected_details)} archivo(s). "
                f"Verifica que las imágenes no estén corruptas y tengan extensión válida: "
                f"{sorted(VALID_IMAGE_EXT)}"
            )

        train_count, _ = split_counts(len(valid_images))
        folder_name = f"{idx:04d}_{sanitize_label(class_name)}"
        train_class_dir = train_root / folder_name
        val_class_dir = val_root / folder_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for image_idx, image_path in enumerate(valid_images):
            target_root = train_class_dir if image_idx < train_count else val_class_dir
            dest_path = target_root / image_path.name
            # NOTA: No reutilizamos el objeto Image de verify() porque verify()
            # invalida el handle. shutil.copy2() trabaja sobre el Path original,
            # que sigue siendo completamente válido para lectura.
            try:
                shutil.copy2(str(image_path), str(dest_path))
            except PermissionError as e:
                raise RuntimeError(
                    f"Sin permisos para copiar '{image_path.name}' "
                    f"(clase '{class_name}') "
                    f"→ '{dest_path}': PermissionError: {e}. "
                    f"Acción: abortando preparación del dataset."
                ) from e
            except OSError as e:
                raise RuntimeError(
                    f"Error de sistema al copiar '{image_path.name}' "
                    f"(clase '{class_name}') "
                    f"→ '{dest_path}': {type(e).__name__}: {e}. "
                    f"Acción: abortando preparación del dataset."
                ) from e

    return split_root



def to_float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def extract_metrics(results):
    metrics_dict = {}
    if results is None:
        return metrics_dict

    if isinstance(results, dict):
        for k, v in results.items():
            value = to_float_or_none(v)
            metrics_dict[str(k)] = value if value is not None else str(v)
        return metrics_dict

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            value = to_float_or_none(v)
            metrics_dict[str(k)] = value if value is not None else str(v)

    top1 = to_float_or_none(getattr(results, "top1", None))
    top5 = to_float_or_none(getattr(results, "top5", None))
    if top1 is not None:
        metrics_dict["top1_acc"] = top1
    if top5 is not None:
        metrics_dict["top5_acc"] = top5

    return metrics_dict


def main():
    prepared_dataset_path = None
    try:
        if len(sys.argv) < 3:
            print_json({
                "type": "error",
                "message": "Uso: train.py <dataset_path> <project_root> [class_names_delimited]"
            })
            sys.exit(1)

        dataset_path = sys.argv[1]
        project_root = sys.argv[2]  # Solo para logging; NO se usa como destino de escritura.

        class_names = []
        if len(sys.argv) >= 4 and sys.argv[3].strip():
            class_names = [c.strip() for c in sys.argv[3].split(CLASS_SEPARATOR) if c.strip()]

        # ── Directorio de salida con permisos garantizados ──────────────────────
        # En produccion, project_root apunta a C:\Program Files (x86)\DeepSight,
        # que es de solo lectura para usuarios sin privilegios. Usamos siempre
        # la carpeta temporal del SO (%TEMP%\DeepSight\train_<uuid>) para que
        # YOLO pueda crear train_output y guardar los pesos sin PermissionError.
        run_id = str(uuid.uuid4().hex[:12])
        safe_output_dir = Path(tempfile.gettempdir()) / "DeepSight" / f"train_{run_id}"
        safe_output_dir.mkdir(parents=True, exist_ok=True)

        print_log(f"Dataset path: {dataset_path}", "info")
        print_log(f"Project root (referencia): {project_root}", "info")
        print_log(f"Output dir seguro: {safe_output_dir}", "info")

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            print_json({"type": "error", "message": f"Dataset no encontrado: {dataset_path}"})
            sys.exit(1)

        if not class_names:
            class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])

        if not class_names:
            print_json({"type": "error", "message": "No se encontraron clases (carpetas) en el dataset"})
            sys.exit(1)

        print_log(f"Clases detectadas ({len(class_names)}): {', '.join(class_names)}", "info")

        class_counts = count_images_per_class(dataset_path)
        print_log(f"Conteo por clase: {class_counts}", "info")

        hp = determine_hyperparameters(class_counts)
        print_log(
            f"Modo: {hp['mode_name']} | Imagenes min/clase: {hp.get('min_images_per_class', '?')}",
            "info"
        )
        print_log(
            f"Hiperparametros: epochs={hp['epochs']}, freeze={hp['freeze']}, "
            f"imgsz={hp['imgsz']}, batch={hp['batch']}, augmentation={hp['augmentation_mode']}",
            "info"
        )

        print_json({"type": "hyperparameters", "data": hp, "class_counts": class_counts})

        script_dir = Path(__file__).parent.resolve()
        model_path = script_dir / "models" / "yolo26n-cls.pt"
        if not model_path.exists():
            print_json({
                "type": "error",
                "message": f"Modelo base no encontrado: {model_path}. "
                           f"Consulta python-core/models/README.md para descargarlo manualmente."
            })
            sys.exit(1)

        print_log(f"Modelo base cargado (offline): {model_path}", "info")

        prepared_dataset_path = prepare_classification_dataset(dataset_path, class_names)
        print_log(f"Dataset de clasificacion preparado: {prepared_dataset_path}", "info")

        print_progress(0, hp["epochs"], "preparing")
        print_log("Iniciando transfer learning con YOLO26n-cls (modo offline)...", "info")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            print_json({
                "type": "error",
                "message": f"Falta dependencia ultralytics: {e}. "
                           f"Instala: pip install ultralytics"
            })
            sys.exit(1)

        print_log("YOLO importado correctamente", "info")

        model = YOLO(str(model_path))
        print_log("Modelo base cargado en memoria", "info")

        # YOLO escribe sus artefactos en safe_output_dir/train_output/
        # Esta ruta siempre tiene permisos de escritura independientemente
        # de donde este instalada la aplicacion.
        runs_output_dir = safe_output_dir

        results = model.train(
            data=str(prepared_dataset_path),
            epochs=hp["epochs"],
            imgsz=hp["imgsz"],
            batch=hp["batch"],
            workers=hp["workers"],
            freeze=hp["freeze"],
            degrees=hp["degrees"],
            fliplr=hp["fliplr"],
            scale=hp["scale"],
            hsv_v=hp["brightness"],
            pretrained=True,
            device="cpu",
            project=str(runs_output_dir),
            name="train_output",
            exist_ok=True
        )

        print_log("Entrenamiento completado. Procesando resultados...", "success")
        print_progress(hp["epochs"], hp["epochs"], "finalizing")

        final_output_path = (safe_output_dir / "best.pt").resolve()

        # Buscar explícitamente el archivo best.pt / last.pt en las subcarpetas de pesos (weights/best.pt)
        candidate_sources = []

        # 1. Atributo save_dir del resultado de Ultralytics
        if hasattr(results, "save_dir") and results.save_dir:
            res_dir = Path(results.save_dir).resolve()
            candidate_sources.extend([
                res_dir / "weights" / "best.pt",
                res_dir / "weights" / "last.pt",
                res_dir / "best.pt",
            ])

        # 2. Atributo trainer.save_dir del modelo
        if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
            trainer_dir = Path(model.trainer.save_dir).resolve()
            candidate_sources.extend([
                trainer_dir / "weights" / "best.pt",
                trainer_dir / "weights" / "last.pt",
                trainer_dir / "best.pt",
            ])

        # 3. Rutas estándar con safe_output_dir / train_output / weights
        save_dir = runs_output_dir / "train_output"
        candidate_sources.extend([
            save_dir / "weights" / "best.pt",
            save_dir / "weights" / "last.pt",
            safe_output_dir / "weights" / "best.pt",
            safe_output_dir / "weights" / "last.pt",
        ])

        # 4. Búsqueda por rglob en safe_output_dir (por si Ultralytics creó carpetas con sufijo)
        for p in safe_output_dir.rglob("best.pt"):
            if p.resolve() != final_output_path:
                candidate_sources.append(p.resolve())
        for p in safe_output_dir.rglob("last.pt"):
            if p.resolve() != final_output_path:
                candidate_sources.append(p.resolve())

        # 5. Búsqueda en el directorio de trabajo (runs/...)
        for p in Path.cwd().rglob("best.pt"):
            if p.resolve() != final_output_path:
                candidate_sources.append(p.resolve())
        for p in Path.cwd().rglob("last.pt"):
            if p.resolve() != final_output_path:
                candidate_sources.append(p.resolve())

        best_pt_source = None
        for cand in candidate_sources:
            if cand and cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                best_pt_source = cand
                break

        if best_pt_source:
            shutil.copy2(str(best_pt_source), str(final_output_path))
            print_log(f"Modelo encontrado en '{best_pt_source}' y copiado a: {final_output_path}", "success")
        else:
            print_json({
                "type": "error",
                "message": f"No se encontro best.pt ni last.pt en la carpeta de pesos despué del entrenamiento en {safe_output_dir}"
            })
            sys.exit(1)

        try:
            metrics_dict = extract_metrics(results)
        except Exception as e:
            metrics_dict = {}
            print_log(f"Aviso extrayendo metricas: {e}", "warning")

        print_json({
            "type": "complete",
            "best_pt_path": str(final_output_path),
            "class_names": class_names,
            "hyperparameters": hp,
            "metrics": metrics_dict
        })
    except Exception as e:
        exc_type = type(e).__name__
        print_json({
            "type": "error",
            "message": f"{exc_type}: {str(e)}"
        })
        try:
            import traceback
            tb_lines = traceback.format_exc().strip().split("\n")
            for line in tb_lines[-5:]:
                print_log(line, "error")
        except Exception:
            pass
        sys.exit(1)
    finally:
        if prepared_dataset_path:
            try:
                shutil.rmtree(prepared_dataset_path, ignore_errors=True)
                print_log(f"Dataset temporal eliminado: {prepared_dataset_path}", "info")
            except Exception:
                pass


if __name__ == "__main__":
    main()
