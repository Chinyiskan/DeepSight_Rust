use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, LazyLock, Mutex};
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;
use tokio::sync::mpsc;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

const CLASS_SEPARATOR: &str = "\x1f";
static RE_ACCENTS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\u{0300}-\u{036f}]").expect("static regex valida"));
static RE_ILLEGAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\w\.\- ]").expect("static regex valida"));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessOutput {
    pub line_type: String,
    pub content: serde_json::Value,
    pub raw: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub success: bool,
    pub best_pt_path: Option<String>,
    pub class_names: Vec<String>,
    pub error_message: Option<String>,
    pub hyperparameters: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub success: bool,
    pub class_name: Option<String>,
    pub confidence: Option<f64>,
    pub class_index: Option<i64>,
    pub top_predictions: Option<Vec<serde_json::Value>>,
    pub error_message: Option<String>,
}

pub const VALID_IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

pub fn is_valid_image_extension(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| VALID_IMAGE_EXT.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn find_python_interpreter() -> Result<String> {
    let candidates = if cfg!(windows) {
        vec![
            "python".to_string(),
            "python3".to_string(),
            "py".to_string(),
        ]
    } else {
        vec!["python3".to_string(), "python".to_string()]
    };

    for cmd in &candidates {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                return Ok(cmd.clone());
            }
        }
    }

    Err(anyhow::anyhow!(
        "Python no detectado. Instala Python 3.10+ desde https://python.org y agregalo al PATH."
    ))
}

/// Resuelve la carpeta `python-core` usando la API de recursos de Tauri v2.
///
/// Orden de búsqueda:
///   1. `resource_dir()` de Tauri → correcto en el bundle de producción
///      (`C:\Program Files (x86)\DeepSight\resources\python-core\`)
///   2. Fallback de desarrollo: sube desde el exe/cwd buscando `python-core/train.py`
///      (útil con `pnpm tauri dev` donde los recursos no están copiados aún).
pub fn resolve_python_core_dir_with_app(app: &AppHandle) -> PathBuf {
    // ── 1. Fuente primaria: resource_dir de Tauri ──────────────────────────────
    if let Ok(res_dir) = app.path().resource_dir() {
        let candidate = res_dir.join("python-core");
        if candidate.join("train.py").exists() {
            return std::fs::canonicalize(&candidate).unwrap_or(candidate);
        }
        // Algunos empaquetadores aplanan resources/ directamente
        if res_dir.join("train.py").exists() {
            return std::fs::canonicalize(&res_dir).unwrap_or(res_dir);
        }
    }

    // ── 2. Fallback de desarrollo: subir desde exe o cwd ──────────────────────
    let mut seen = HashSet::new();
    let mut starts: Vec<PathBuf> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            starts.push(dir.to_path_buf());
        }
    }

    for start in starts {
        for ancestor in start.ancestors() {
            let buf = ancestor.to_path_buf();
            if !seen.insert(buf.clone()) {
                continue;
            }
            let candidate = buf.join("python-core");
            if candidate.join("train.py").exists() {
                return std::fs::canonicalize(&candidate).unwrap_or(candidate);
            }
        }
    }

    // ── 3. Último recurso: junto al ejecutable ────────────────────────────────
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("python-core")))
        .unwrap_or_else(|| PathBuf::from("python-core"))
}

/// Mantiene compatibilidad con código que no tiene AppHandle disponible.
/// Usar `resolve_python_core_dir_with_app` siempre que sea posible.
#[allow(dead_code)]
pub fn resolve_project_root() -> Result<PathBuf> {
    let mut seen = HashSet::new();
    let mut candidate_roots = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        candidate_roots.push(cwd);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidate_roots.push(exe_dir.to_path_buf());
        }
    }

    for start in candidate_roots {
        for ancestor in start.ancestors() {
            let ancestor_buf = ancestor.to_path_buf();
            if !seen.insert(ancestor_buf.clone()) {
                continue;
            }

            if ancestor.join("python-core").join("train.py").exists() {
                return Ok(std::fs::canonicalize(&ancestor_buf).unwrap_or(ancestor_buf));
            }
        }
    }

    let fallback = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    Ok(std::fs::canonicalize(&fallback).unwrap_or(fallback))
}

#[allow(dead_code)]
pub fn resolve_python_core_dir(project_root: &Path) -> PathBuf {
    let candidates = vec![
        project_root.join("python-core"),
        project_root.join("..").join("python-core"),
        project_root.join("resources").join("python-core"),
    ];
    for c in candidates {
        let train_script = c.join("train.py");
        if train_script.exists() {
            return std::fs::canonicalize(&c).unwrap_or(c);
        }
    }
    project_root.join("python-core")
}

pub fn sanitize_filename(name: &str) -> String {
    // SEGURIDAD: Extraer solo el componente basename para prevenir Path Traversal.
    // Un nombre como "../../passwd" queda reducido a "passwd" antes de sanitizar.
    let basename = Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(name);

    // Eliminar separadores de directorio que puedan sobrevivir (Windows '\', Unix '/')
    let no_separators: String = basename
        .chars()
        .filter(|&c| c != '/' && c != '\\')
        .collect();

    // Normalización Unicode y eliminación de acentos
    let nfkd: String = no_separators.nfkd().collect();
    let without_accents = RE_ACCENTS.replace_all(&nfkd, "").to_string();

    // Eliminar caracteres ilegales
    let cleaned = RE_ILLEGAL.replace_all(&without_accents, "_").to_string();

    // Trim de espacios y puntos al inicio/fin, y eliminar componentes ".." residuales
    let trimmed = cleaned
        .trim()
        .trim_matches('.')
        .replace("..", "_")
        .to_string();

    // Limitar longitud para prevenir DoS por nombres extremadamente largos
    let limited = if trimmed.len() > 64 {
        trimmed.chars().take(64).collect::<String>()
    } else {
        trimmed
    };

    if limited.trim().is_empty() {
        format!("file_{}", Uuid::new_v4().simple())
    } else {
        limited
    }
}

pub fn sanitize_class_name(name: &str) -> String {
    let base = sanitize_filename(name);
    base.replace(" ", "_").replace("-", "_")
}

pub fn join_class_names_for_arg(class_names: &[String]) -> String {
    class_names.join(CLASS_SEPARATOR)
}

fn prepare_temp_dataset_sync(
    _project_root: &Path,
    classes: &[(String, Vec<String>)],
    tx: Option<&mpsc::UnboundedSender<ProcessOutput>>,
) -> Result<(PathBuf, Vec<String>)> {
    let temp_id = Uuid::new_v4().simple().to_string();
    // Usar la carpeta temporal del SO en lugar de project_root para evitar errores
    // de permisos cuando la app está instalada en C:\Program Files (x86) (bug producción).
    let temp_root = std::env::temp_dir().join("DeepSight").join(&temp_id);
    std::fs::create_dir_all(&temp_root)
        .with_context(|| format!("No se pudo crear carpeta temporal: {}", temp_root.display()))?;

    if let Some(sender) = tx {
        let _ = sender.send(ProcessOutput {
            line_type: "log".to_string(),
            content: serde_json::json!({
                "level": "info",
                "message": format!("Preparando dataset temporal en: {}", temp_root.display())
            }),
            raw: format!("[SYS] Preparando dataset temporal: {}", temp_root.display()),
        });
    }

    let mut class_dir_order: Vec<String> = Vec::new();

    for (idx, (class_name, file_paths)) in classes.iter().enumerate() {
        let safe_class_name = sanitize_class_name(class_name);
        let class_dir_name = if safe_class_name.is_empty() {
            format!("class_{}", idx)
        } else {
            safe_class_name.clone()
        };
        let class_dir = temp_root.join(&class_dir_name);
        std::fs::create_dir_all(&class_dir).with_context(|| {
            format!("No se pudo crear carpeta de clase: {}", class_dir.display())
        })?;
        class_dir_order.push(class_name.clone());

        for (file_idx, src_path_str) in file_paths.iter().enumerate() {
            let src_path = Path::new(src_path_str);
            if !src_path.exists() {
                if let Some(sender) = tx {
                    let _ = sender.send(ProcessOutput {
                        line_type: "log".to_string(),
                        content: serde_json::json!({
                            "level": "warning",
                            "message": format!("Archivo no encontrado, saltando: {}", src_path.display())
                        }),
                        raw: format!("[WARN] No existe: {}", src_path.display()),
                    });
                }
                continue;
            }

            let ext = src_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("jpg")
                .to_lowercase();
            let dest_name = format!("img_{:04}_{}.{}", file_idx, idx, ext);
            let dest_path = class_dir.join(&dest_name);

            if let Some(sender) = tx {
                let _ = sender.send(ProcessOutput {
                    line_type: "log".to_string(),
                    content: serde_json::json!({
                        "level": "info",
                        "message": format!(
                            "[{}] Copiando {} -> {}",
                            class_name,
                            src_path.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
                            dest_name
                        )
                    }),
                    raw: format!("[COPY] {} -> {}", src_path.display(), dest_name),
                });
            }

            if src_path.is_file() {
                std::fs::copy(src_path, &dest_path).with_context(|| {
                    format!(
                        "Error copiando: {} -> {}",
                        src_path.display(),
                        dest_path.display()
                    )
                })?;
            }
        }
    }

    if let Some(sender) = tx {
        let _ = sender.send(ProcessOutput {
            line_type: "log".to_string(),
            content: serde_json::json!({
                "level": "success",
                "message": format!(
                    "Dataset temporal preparado: {} clases en {}",
                    class_dir_order.len(),
                    temp_root.display()
                )
            }),
            raw: format!(
                "[OK] Dataset temporal listo ({} clases)",
                class_dir_order.len()
            ),
        });
    }

    Ok((temp_root, class_dir_order))
}

pub async fn prepare_temp_dataset(
    project_root: PathBuf,
    classes: Vec<(String, Vec<String>)>,
    tx: Option<mpsc::UnboundedSender<ProcessOutput>>,
) -> Result<(PathBuf, Vec<String>)> {
    tokio::task::spawn_blocking(move || -> Result<(PathBuf, Vec<String>)> {
        let tx_ref = tx.as_ref();
        prepare_temp_dataset_sync(&project_root, &classes, tx_ref)
    })
    .await
    .unwrap_or_else(|join_err| Err(anyhow::anyhow!("Task join error: {}", join_err)))
}

pub async fn run_training(
    app: AppHandle,
    project_root: PathBuf,
    temp_dataset: PathBuf,
    class_names: Vec<String>,
    tx: mpsc::UnboundedSender<ProcessOutput>,
    // pid_tx: Canal para notificar el PID del proceso hijo a main.rs (parche A-01).
    // Se usa para registrar el PID en AppState y poder hacer kill() en CloseRequested.
    pid_tx: Option<tokio::sync::oneshot::Sender<u32>>,
) -> Result<TrainingResult> {
    let python_cmd = find_python_interpreter()?;
    // Usar la API de Tauri v2 para resolver recursos correctamente tanto en dev
    // como en el bundle de producción instalado en Program Files.
    let python_core = resolve_python_core_dir_with_app(&app);
    let train_script = python_core.join("train.py");

    if !train_script.exists() {
        return Err(anyhow::anyhow!(
            "Script train.py no encontrado: {}\n[resource_dir={:?}]",
            train_script.display(),
            app.path().resource_dir().ok()
        ));
    }

    let classes_arg = join_class_names_for_arg(&class_names);

    let _ = tx.send(ProcessOutput {
        line_type: "log".to_string(),
        content: serde_json::json!({
            "level": "info",
            "message": format!("Ejecutando: {} train.py ...", python_cmd)
        }),
        raw: format!("[RUN] {} train.py", python_cmd),
    });

    let mut child = TokioCommand::new(&python_cmd)
        .arg(&train_script)
        .arg(temp_dataset.to_string_lossy().to_string())
        .arg(project_root.to_string_lossy().to_string())
        .arg(&classes_arg)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| "No se pudo iniciar proceso Python")?;

    let child_id = child.id().unwrap_or(0);

    // PARCHE A-01: Notificar PID al llamador para registrarlo en AppState.
    if let Some(sender) = pid_tx {
        let _ = sender.send(child_id);
    }

    let _ = tx.send(ProcessOutput {
        line_type: "log".to_string(),
        content: serde_json::json!({
            "level": "info",
            "message": format!("Proceso Python iniciado (PID: {})", child_id)
        }),
        raw: format!("[PID] {} iniciado", child_id),
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("No se pudo capturar stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("No se pudo capturar stderr"))?;

    let tx_stdout = tx.clone();
    let tx_stderr = tx.clone();

    // Captura compartida: el task de stdout guarda best_pt_path cuando
    // llega el evento JSON "complete" del proceso Python.
    // Esto evita depender de project_root (que puede ser C:\Program Files, de solo lectura).
    let captured_best_pt: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured_best_pt_writer = Arc::clone(&captured_best_pt);

    let stdout_task = tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let output = parse_python_line(&line);
            // Capturar best_pt_path desde el evento "complete" antes de enviarlo al canal.
            if output.line_type == "complete" {
                if let Some(path) = output.content.get("best_pt_path").and_then(|v| v.as_str()) {
                    if let Ok(mut g) = captured_best_pt_writer.lock() {
                        *g = Some(path.to_string());
                    }
                }
            }
            let _ = tx_stdout.send(output);
        }
    });

    let stderr_task = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            let _ = tx_stderr.send(ProcessOutput {
                line_type: "log".to_string(),
                content: serde_json::json!({
                    "level": "error",
                    "message": line
                }),
                raw: format!("[STDERR] {}", line),
            });
        }
    });

    let status = child.wait().await.unwrap_or_else(|e| {
        let _ = tx.send(ProcessOutput {
            line_type: "log".to_string(),
            content: serde_json::json!({
                "level": "error",
                "message": format!("Error esperando proceso: {}", e)
            }),
            raw: format!("[ERR] wait: {}", e),
        });
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            ExitStatusExt::from_raw(1)
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::process::ExitStatusExt;
            ExitStatusExt::from_raw(1)
        }
    });

    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let _ = tx.send(ProcessOutput {
        line_type: "log".to_string(),
        content: serde_json::json!({
            "level": if status.success() { "success" } else { "error" },
            "message": format!("Proceso Python finalizado. Codigo: {:?}", status.code())
        }),
        raw: format!("[END] Codigo {:?}", status.code()),
    });

    // La ruta de best.pt viene del mensaje JSON "complete" del proceso Python.
    // Python la guarda en %TEMP%\DeepSight\train_<id>\best.pt, que siempre
    // tiene permisos de escritura (evita PermissionError en Program Files).
    let best_pt_from_python = captured_best_pt.lock().ok().and_then(|g| g.clone());

    if status.success() {
        if let Some(path) = best_pt_from_python {
            let best_pt = PathBuf::from(&path);
            if best_pt.exists() {
                Ok(TrainingResult {
                    success: true,
                    best_pt_path: Some(path),
                    class_names,
                    error_message: None,
                    hyperparameters: None,
                    metrics: None,
                })
            } else {
                Ok(TrainingResult {
                    success: false,
                    best_pt_path: None,
                    class_names,
                    error_message: Some(format!(
                        "Entrenamiento completado pero best.pt no se encontro en: {}",
                        path
                    )),
                    hyperparameters: None,
                    metrics: None,
                })
            }
        } else {
            Ok(TrainingResult {
                success: false,
                best_pt_path: None,
                class_names,
                error_message: Some(
                    "Entrenamiento termino pero no se recibio la ruta de best.pt".to_string(),
                ),
                hyperparameters: None,
                metrics: None,
            })
        }
    } else {
        Ok(TrainingResult {
            success: false,
            best_pt_path: None,
            class_names,
            error_message: Some(format!(
                "El proceso Python devolvio codigo de error: {:?}",
                status.code()
            )),
            hyperparameters: None,
            metrics: None,
        })
    }
}

pub async fn run_inference(
    app: AppHandle,
    _project_root: PathBuf,
    model_path: PathBuf,
    image_path: PathBuf,
    class_names: Vec<String>,
) -> Result<InferenceResult> {
    let python_cmd = find_python_interpreter()?;
    // Usar la API de Tauri v2 para resolver recursos correctamente tanto en dev
    // como en el bundle de producción instalado en Program Files.
    let python_core = resolve_python_core_dir_with_app(&app);
    let infer_script = python_core.join("infer.py");

    if !infer_script.exists() {
        return Err(anyhow::anyhow!(
            "Script infer.py no encontrado: {}\n[resource_dir={:?}]",
            infer_script.display(),
            app.path().resource_dir().ok()
        ));
    }

    let classes_arg = join_class_names_for_arg(&class_names);

    let child = TokioCommand::new(&python_cmd)
        .arg(&infer_script)
        .arg(model_path.to_string_lossy().to_string())
        .arg(image_path.to_string_lossy().to_string())
        .arg(&classes_arg)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| "No se pudo iniciar inferencia Python")?;

    let output = child
        .wait_with_output()
        .await
        .with_context(|| "No se pudo obtener salida de la inferencia Python")?;

    let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Ok(InferenceResult {
            success: false,
            class_name: None,
            confidence: None,
            class_index: None,
            top_predictions: None,
            error_message: Some(if stderr_str.is_empty() {
                format!(
                    "Error ejecutando inferencia (codigo {:?})",
                    output.status.code()
                )
            } else {
                stderr_str
                    .lines()
                    .last()
                    .unwrap_or("Error desconocido")
                    .to_string()
            }),
        });
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
    let mut last_result: Option<InferenceResult> = None;
    let mut last_error: Option<String> = None;

    for line in stdout_str.lines() {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = parsed.get("type").and_then(|v| v.as_str()) {
                if t == "result" {
                    last_result = Some(InferenceResult {
                        success: true,
                        class_name: parsed
                            .get("class_name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        confidence: parsed.get("confidence").and_then(|v| v.as_f64()),
                        class_index: parsed.get("class_index").and_then(|v| v.as_i64()),
                        top_predictions: parsed
                            .get("top_predictions")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().cloned().collect()),
                        error_message: None,
                    });
                } else if t == "error" {
                    last_error = parsed
                        .get("message")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
            }
        }
    }

    if let Some(r) = last_result {
        Ok(r)
    } else if let Some(e) = last_error {
        Ok(InferenceResult {
            success: false,
            class_name: None,
            confidence: None,
            class_index: None,
            top_predictions: None,
            error_message: Some(e),
        })
    } else {
        Ok(InferenceResult {
            success: false,
            class_name: None,
            confidence: None,
            class_index: None,
            top_predictions: None,
            error_message: Some("No se pudo parsear la salida de inferencia".to_string()),
        })
    }
}

fn parse_python_line(line: &str) -> ProcessOutput {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ProcessOutput {
            line_type: "raw".to_string(),
            content: serde_json::Value::Null,
            raw: line.to_string(),
        };
    }

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let dtype = json
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("json")
            .to_string();
        ProcessOutput {
            line_type: dtype,
            content: json,
            raw: line.to_string(),
        }
    } else {
        ProcessOutput {
            line_type: "raw".to_string(),
            content: serde_json::json!({ "message": trimmed }),
            raw: line.to_string(),
        }
    }
}

pub fn cleanup_temp_files(project_root: &Path, temp_dataset: &Path) {
    if temp_dataset.exists() && temp_dataset.as_os_str().len() > 0 {
        if let Err(e) = std::fs::remove_dir_all(temp_dataset) {
            eprintln!(
                "[cleanup] No se pudo borrar dataset temporal {}: {}",
                temp_dataset.display(),
                e
            );
        }
    }

    let runs_dir = project_root.join("train_output");
    if runs_dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(&runs_dir) {
            eprintln!(
                "[cleanup] No se pudo borrar runs {}: {}",
                runs_dir.display(),
                e
            );
        }
    }

    if let Ok(entries) = std::fs::read_dir(project_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("runs") {
                        if let Err(e) = std::fs::remove_dir_all(&path) {
                            eprintln!(
                                "[cleanup] No se pudo borrar carpeta runs {}: {}",
                                path.display(),
                                e
                            );
                        }
                    }
                }
            }
        }
    }
}

pub fn cleanup_legacy_temp_at_startup(project_root: &Path) {
    // Limpiar sesiones huérfanas de DeepSight en la carpeta temporal del SO.
    let sys_temp_root = std::env::temp_dir().join("DeepSight");
    if sys_temp_root.exists() {
        match std::fs::read_dir(&sys_temp_root) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let _ = std::fs::remove_dir_all(&path);
                    } else {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            Err(_) => {}
        }
    }

    // Limpiar carpetas de runs/train_output heredadas junto al binario.
    if let Ok(entries) = std::fs::read_dir(project_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("runs") || name == "train_output" {
                        let _ = std::fs::remove_dir_all(&path);
                    }
                }
            }
        }
    }
}
