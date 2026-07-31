import json
import sys
import os
import traceback
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
    try:
        serialized = json.dumps(data, ensure_ascii=False)
        print(serialized, flush=True)
    except (TypeError, ValueError):
        safe_data = {
            "type": data.get("type", "unknown") if isinstance(data, dict) else "unknown",
            "message": "Fallo al serializar JSON"
        }
        try:
            print(json.dumps(safe_data, ensure_ascii=False), flush=True)
        except Exception:
            pass
    except OSError:
        try:
            sys.exit(1)
        except SystemExit:
            raise


def print_log(message, level="info"):
    print_json({
        "type": "log",
        "level": level,
        "message": str(message)
    })


def emit_error(message, include_traceback=False):
    payload = {
        "type": "error",
        "message": str(message)
    }
    if include_traceback:
        payload["traceback"] = traceback.format_exc(limit=6)
    print_json(payload)


def safe_class_name(class_names, idx, fallback_prefix="Clase"):
    if idx < 0 or idx >= len(class_names):
        return f"{fallback_prefix}_{idx}"
    return class_names[idx]


def flatten_probs_tensor(raw):
    """Asegura 1D numpy array (casos (1,N), listas, tensores)."""
    arr = None
    if hasattr(raw, "detach"):
        try:
            raw = raw.detach()
        except Exception:
            pass
    if hasattr(raw, "cpu"):
        try:
            raw = raw.cpu()
        except Exception:
            pass
    if hasattr(raw, "numpy"):
        try:
            arr = raw.numpy()
        except Exception:
            arr = None
    else:
        arr = raw

    if arr is None:
        try:
            import numpy as np
            arr = np.array(raw, dtype=float)
        except Exception:
            return None

    try:
        if hasattr(arr, "flatten"):
            arr = arr.flatten()
        if hasattr(arr, "reshape"):
            arr = arr.reshape(-1)
    except Exception:
        pass
    return arr


def dedup_probs(all_probs):
    """Elimina entradas duplicadas por class_index, manteniendo la mayor confianza."""
    if not all_probs:
        return all_probs
    best_by_idx = {}
    order = []
    for p in all_probs:
        i = int(p["class_index"])
        if i not in best_by_idx:
            best_by_idx[i] = p
            order.append(i)
        elif p["confidence"] > best_by_idx[i]["confidence"]:
            best_by_idx[i] = {
                "class_index": i,
                "class_name": p["class_name"],
                "confidence": float(p["confidence"]),
                "confidence_raw": float(p["confidence_raw"])
            }
    return [best_by_idx[i] for i in order]


def extract_probs_universal(probs_obj, class_names):
    """Multi-estrategia: YOLOv8, YOLO11, versiones antiguas/nuevas."""
    all_probs = []

    # ESTRATEGIA 1: probs.data / probs.probs / probs.conf / probs.scores (tensor raw)
    candidate_attrs = ["data", "probs", "conf", "scores"]
    tensor_source = None
    for attr in candidate_attrs:
        val = getattr(probs_obj, attr, None)
        if val is not None:
            try:
                if hasattr(val, "__len__") and len(val) > 0:
                    tensor_source = val
                    break
            except Exception:
                continue

    if tensor_source is not None:
        arr = flatten_probs_tensor(tensor_source)
        if arr is not None and len(arr) > 0:
            for i, p in enumerate(arr):
                try:
                    conf_val = float(p)
                except (TypeError, ValueError):
                    continue
                all_probs.append({
                    "class_index": int(i),
                    "class_name": safe_class_name(class_names, i),
                    "confidence": round(conf_val * 100.0, 2),
                    "confidence_raw": round(conf_val, 4)
                })

    # ESTRATEGIA 2: top1/top5 atributos directos (YOLO11+ o si E1 falló)
    if not all_probs:
        top1_idx = None
        top1_conf = None
        try:
            if hasattr(probs_obj, "top1"):
                v = probs_obj.top1
                if hasattr(v, "item"):
                    v = v.item()
                top1_idx = int(v)
            if hasattr(probs_obj, "top1conf"):
                v = probs_obj.top1conf
                if hasattr(v, "item"):
                    v = v.item()
                top1_conf = float(v)
        except Exception:
            top1_idx = None
            top1_conf = None

        top5_idxs = []
        top5_confs = []
        try:
            if hasattr(probs_obj, "top5"):
                raw = probs_obj.top5
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                top5_idxs = [int(x) for x in raw]
            if hasattr(probs_obj, "top5conf"):
                raw = probs_obj.top5conf
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                top5_confs = [float(x) for x in raw]
        except Exception:
            top5_idxs = []
            top5_confs = []

        if top5_idxs:
            for pos, idx in enumerate(top5_idxs):
                c = top5_confs[pos] if pos < len(top5_confs) else (top1_conf if idx == top1_idx else 0.0)
                if c is None:
                    c = 0.0
                all_probs.append({
                    "class_index": int(idx),
                    "class_name": safe_class_name(class_names, idx),
                    "confidence": round(c * 100.0, 2),
                    "confidence_raw": round(c, 4)
                })
        elif top1_idx is not None:
            c = top1_conf if top1_conf is not None else 0.0
            all_probs.append({
                "class_index": int(top1_idx),
                "class_name": safe_class_name(class_names, top1_idx),
                "confidence": round(c * 100.0, 2),
                "confidence_raw": round(c, 4)
            })

    # ESTRATEGIA 3: iterar probs_obj como diccionario / ultralytics new API
    if not all_probs:
        try:
            if hasattr(probs_obj, "to_dict"):
                d = probs_obj.to_dict()
                for k, v in d.items():
                    try:
                        i = int(k)
                        c = float(v)
                    except (TypeError, ValueError):
                        continue
                    all_probs.append({
                        "class_index": i,
                        "class_name": safe_class_name(class_names, i),
                        "confidence": round(c * 100.0, 2),
                        "confidence_raw": round(c, 4)
                    })
        except Exception:
            pass

    return dedup_probs(all_probs)


def handle_classification(result, class_names):
    probs_obj = getattr(result, "probs", None)
    if probs_obj is None:
        emit_error(
            "Modelo no produjo probabilidades (probs). Verifica que sea un modelo de CLASIFICACION (-cls.pt), no de deteccion/segmentacion."
        )
        return False

    all_probs = extract_probs_universal(probs_obj, class_names)
    if not all_probs:
        emit_error(
            "No se pudo extraer probabilidades con ninguna estrategia de extraccion (3 intentos)."
            " El formato del objeto probs es incompatible con esta version de Ultralytics."
        )
        return False

    all_probs.sort(key=lambda x: x["confidence"], reverse=True)
    top_pred = all_probs[0]
    top5 = all_probs[:5]

    print_log("Inferencia completada", "success")
    print_json({
        "type": "result",
        "class_index": int(top_pred["class_index"]),
        "class_name": top_pred["class_name"],
        "confidence": float(top_pred["confidence"]),
        "confidence_raw": float(top_pred["confidence_raw"]),
        "top_predictions": top5
    })
    return True


def handle_detection_fallback(result, class_names):
    """Solo si user cargó detector (error de uso); devolvemos 1ra detección amigablemente."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return False
    n_boxes = 0
    try:
        n_boxes = len(boxes)
    except Exception:
        n_boxes = 0
    if n_boxes <= 0:
        emit_error(
            "El modelo pareciera ser de DETECCION pero no detecto objetos en la imagen."
            " Asegurate de entrenar/exportar un modelo de CLASIFICACION (-cls.pt)."
        )
        return False

    cls_tensor = getattr(boxes, "cls", None)
    conf_tensor = getattr(boxes, "conf", None)
    if cls_tensor is None:
        emit_error("Boxes sin atributo cls (version incompatible de Ultralytics).")
        return False

    try:
        arr_cls = flatten_probs_tensor(cls_tensor)
        arr_conf = flatten_probs_tensor(conf_tensor) if conf_tensor is not None else None
        if arr_cls is None or len(arr_cls) < 1:
            raise ValueError("cls tensor vacio despues de flatten")
        cls_idx = int(arr_cls[0])
        conf = float(arr_conf[0]) if arr_conf is not None and len(arr_conf) > 0 else 0.0
    except Exception as e:
        emit_error(f"Fallo al extraer caja de deteccion: {e}", include_traceback=True)
        return False

    class_name = safe_class_name(class_names, cls_idx)
    print_log(
        "Advertencia: este es un modelo de DETECCION, no de CLASIFICACION."
        " La precision de clasificacion sera aproximada (toma la 1ra caja detectada).",
        "warning"
    )
    print_json({
        "type": "result",
        "class_index": cls_idx,
        "class_name": class_name,
        "confidence": round(conf * 100.0, 2),
        "confidence_raw": round(conf, 4),
        "note": "resultado desde modelo de deteccion (fallback)",
        "top_predictions": [
            {
                "class_index": cls_idx,
                "class_name": class_name,
                "confidence": round(conf * 100.0, 2),
                "confidence_raw": round(conf, 4)
            }
        ]
    })
    return True


def main():
    exit_code = 1
    model = None
    results = None
    result = None
    try:
        n_args = len(sys.argv)
        if n_args < 4:
            emit_error(
                "Uso: infer.py <model_path> <image_path> <class_names_delimited_by_ASCII_U001F>"
                f" (Unit Separator 0x1F). Se recibieron {n_args - 1} argumentos."
            )
            exit_code = 1
            return

        model_path = sys.argv[1]
        image_path = sys.argv[2]
        class_names_raw = sys.argv[3]

        model_file = Path(model_path)
        if not model_file.is_file():
            emit_error(f"Archivo de modelo NO EXISTE o es un directorio: {model_path}")
            exit_code = 1
            return
        if model_file.suffix.lower() not in (".pt", ".onnx", ".engine"):
            print_log(
                f"Extension inesperada del modelo ({model_file.suffix.lower()})."
                " Se esperaba .pt (PyTorch). Continuando bajo riesgo del usuario.",
                "warning"
            )

        image_file = Path(image_path)
        if not image_file.is_file():
            emit_error(f"Archivo de imagen NO EXISTE: {image_path}")
            exit_code = 1
            return
        if image_file.suffix.lower() not in VALID_IMAGE_EXT:
            emit_error(
                f"Extension de imagen NO VALIDA: {image_file.suffix.lower()}."
                f" Permitidas: {sorted(VALID_IMAGE_EXT)}"
            )
            exit_code = 1
            return

        class_names = [c.strip() for c in class_names_raw.split(CLASS_SEPARATOR) if c.strip()]
        if len(class_names) < 2:
            emit_error(
                f"Se requieren al menos 2 nombres de clases (se recibieron {len(class_names)})."
                " El delimitador entre clases debe ser el ASCII U+001F (Unit Separator)."
            )
            exit_code = 1
            return

        print_log(f"Cargando modelo: {model_file.name}", "info")
        print_log(f"Analizando imagen: {image_file.name}", "info")
        print_log(f"Clases registradas: {len(class_names)}", "info")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            emit_error(
                "Falta dependencia ultralytics. Instala primero con:"
                f" pip install ultralytics. Detalle: {e}"
            )
            exit_code = 1
            return

        try:
            model = YOLO(str(model_file))
        except Exception as e:
            emit_error(
                f"No se pudo cargar el archivo del modelo (corrupto o formato incorrecto): {e}",
                include_traceback=True
            )
            exit_code = 1
            return

        print_log("Modelo cargado OK. Ejecutando inferencia...", "info")

        try:
            results = model.predict(
                source=str(image_file),
                imgsz=224,
                verbose=False,
                device="cpu",
                stream=False
            )
        except Exception as e:
            emit_error(f"model.predict() lanzo excepcion: {e}", include_traceback=True)
            exit_code = 1
            return

        if not results:
            emit_error("model.predict() devolvio lista VACIA.")
            exit_code = 1
            return

        result = results[0]
        ok = False

        probs_here = getattr(result, "probs", None)
        if probs_here is not None:
            ok = handle_classification(result, class_names)
            if not ok:
                exit_code = 1
                return
        else:
            ok = handle_detection_fallback(result, class_names)
            if not ok:
                emit_error(
                    "Formato de resultados desconocido: ni probs (clasificacion) ni boxes (deteccion)."
                    " Asegurate de entrenar un modelo YOLO de CLASIFICACION (sufijo -cls)."
                )
                exit_code = 1
                return

        exit_code = 0

    except SystemExit:
        raise
    except MemoryError:
        emit_error("Memoria RAM insuficiente durante la inferencia. Cierra otras aplicaciones.")
        exit_code = 2
    except Exception as e:
        exc_type = type(e).__name__
        emit_error(f"{exc_type}: {str(e)}", include_traceback=True)
        exit_code = 1
    finally:
        refs_to_clean = [result, results, model]
        refs_to_clean.clear()
        refs_to_clean = None
        try:
            del result
        except Exception:
            pass
        try:
            del results
        except Exception:
            pass
        try:
            if model is not None:
                del model
        except Exception:
            pass
        # PARCHE B-01: Liberar VRAM explícitamente si se usó CUDA.
        # Aunque el device actual es "cpu", esto es defensivo para futuros cambios.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    sys.exit(exit_code)



if __name__ == "__main__":
    main()
