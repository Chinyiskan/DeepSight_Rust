#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod process;
mod system_info;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, DragDropEvent, Emitter, Manager, State, WindowEvent};
use tokio::sync::mpsc;

use crate::process::{
    cleanup_legacy_temp_at_startup, cleanup_temp_files, find_python_interpreter,
    is_valid_image_extension, prepare_temp_dataset, resolve_project_root, run_inference,
    run_training, sanitize_filename, InferenceResult, ProcessOutput, TrainingResult,
};
use crate::system_info::SystemInfo;

#[derive(Default)]
struct AppState {
    latest_best_pt: Mutex<Option<String>>,
    latest_class_names: Mutex<Vec<String>>,
    is_training: Mutex<bool>,
    training_temp_dir: Mutex<Option<PathBuf>>,
    /// PID del proceso Python activo. Se usa para kill() explícito en CloseRequested
    /// y evitar procesos huérfanos consumiendo GPU/RAM (parche A-01).
    active_python_pid: Mutex<Option<u32>>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassInput {
    name: String,
    images: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StartTrainingResponse {
    success: bool,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SanitizeResult {
    original: String,
    sanitized: String,
}

#[tauri::command]
async fn get_system_info() -> Result<SystemInfo, String> {
    tokio::task::spawn_blocking(move || crate::system_info::gather_system_info())
        .await
        .map_err(|e| format!("Error al recolectar info del sistema: {}", e))
}

#[tauri::command]
async fn check_python() -> Result<SanitizeResult, String> {
    let cmd = tokio::task::spawn_blocking(move || find_python_interpreter())
        .await
        .map_err(|e| format!("Error interno: {}", e))?
        .map_err(|e| e.to_string())?;
    Ok(SanitizeResult {
        original: cmd.clone(),
        sanitized: cmd,
    })
}

#[tauri::command]
fn sanitize_name(name: String) -> Result<SanitizeResult, String> {
    Ok(SanitizeResult {
        original: name.clone(),
        sanitized: sanitize_filename(&name),
    })
}

#[tauri::command]
fn has_trained_model(state: State<'_, AppState>) -> Result<bool, String> {
    let guard = state
        .latest_best_pt
        .lock()
        .map_err(|e| format!("Estado bloqueado: {}", e))?;
    if let Some(path) = guard.as_ref() {
        return Ok(std::path::Path::new(path).exists());
    }
    Ok(false)
}

#[tauri::command]
async fn start_training(
    app: AppHandle,
    state: State<'_, AppState>,
    classes: Vec<ClassInput>,
) -> Result<StartTrainingResponse, String> {
    {
        let training_guard = state
            .is_training
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        if *training_guard {
            return Ok(StartTrainingResponse {
                success: false,
                message: "Ya hay un entrenamiento en curso".to_string(),
            });
        }
    }

    if classes.is_empty() {
        return Err("No hay clases definidas. Crea al menos 2 clases con imagenes.".to_string());
    }

    if classes.len() < 2 {
        return Err("Se requieren MINIMO 2 clases para entrenar un clasificador.".to_string());
    }

    let total_images: usize = classes.iter().map(|c| c.images.len()).sum();
    if total_images == 0 {
        return Err("No hay imagenes. Agrega al menos una por clase.".to_string());
    }

    for c in &classes {
        let trimmed_name = c.name.trim();
        if trimmed_name.is_empty() {
            return Err("Hay una clase con nombre vacio.".to_string());
        }
        if c.images.is_empty() {
            return Err(format!(
                "La clase '{}' no tiene ninguna imagen. Agrega al menos 1.",
                trimmed_name
            ));
        }
        for p in &c.images {
            if !is_valid_image_extension(p) {
                return Err(format!(
                    "Formato de imagen invalido en clase '{}': {}",
                    trimmed_name, p
                ));
            }
        }
    }

    {
        let mut training_guard = state
            .is_training
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        *training_guard = true;
    }

    let class_tuples: Vec<(String, Vec<String>)> = classes
        .iter()
        .map(|c| (c.name.trim().to_string(), c.images.clone()))
        .collect();
    let class_names: Vec<String> = class_tuples.iter().map(|(n, _)| n.clone()).collect();

    {
        let mut names_guard = state
            .latest_class_names
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        *names_guard = class_names.clone();
    }

    let project_root = resolve_project_root().map_err(|e| {
        format!(
            "No se pudo resolver la raiz del proyecto para entrenamiento: {}",
            e
        )
    })?;

    let (tx, mut rx) = mpsc::unbounded_channel::<ProcessOutput>();

    let app_clone = app.clone();
    let class_names_clone = class_names.clone();
    let project_root_clone = project_root.clone();
    let class_tuples_clone = class_tuples.clone();

    // SEGURIDAD (C-03): Capturamos el JoinHandle para poder detectar pánicos.
    // Si el spawn entra en pánico, el JoinHandle retorna Err(JoinError::panicked),
    // lo que garantiza que el watcher exterior pueda resetear is_training = false.
    let training_handle = tokio::spawn(async move {
        let tx_for_prep = tx.clone();
        let prep_result = prepare_temp_dataset(
            project_root_clone.clone(),
            class_tuples_clone,
            Some(tx_for_prep),
        )
        .await;

        let st = app_clone.state::<AppState>();
        let (temp_dir, _order) = match prep_result {
            Ok(v) => v,
            Err(e) => {
                let err_msg = format!("Error preparando dataset: {}", e);
                let _ = tx.send(ProcessOutput {
                    line_type: "log".to_string(),
                    content: serde_json::json!({
                        "level": "error",
                        "message": &err_msg
                    }),
                    raw: format!("[FATAL] {}", e),
                });
                let _ = app_clone.emit::<TrainingResult>(
                    "training:complete",
                    TrainingResult {
                        success: false,
                        best_pt_path: None,
                        class_names: class_names_clone.clone(),
                        error_message: Some(err_msg),
                        hyperparameters: None,
                        metrics: None,
                    },
                );
                if let Ok(mut g) = st.is_training.lock() {
                    *g = false;
                }
                let cleanup_root = project_root_clone.clone();
                tokio::task::spawn_blocking(move || {
                    let dummy = PathBuf::new();
                    cleanup_temp_files(&cleanup_root, &dummy);
                });
                return;
            }
        };

        {
            if let Ok(mut g) = st.training_temp_dir.lock() {
                *g = Some(temp_dir.clone());
            }
        }

        let (pid_tx, pid_rx) = tokio::sync::oneshot::channel::<u32>();
        let app_for_pid = app_clone.clone();
        tokio::spawn(async move {
            if let Ok(pid) = pid_rx.await {
                if let Ok(mut g) = app_for_pid.state::<AppState>().active_python_pid.lock() {
                    *g = Some(pid);
                }
            }
        });



        let training_result = run_training(
            project_root_clone.clone(),
            temp_dir.clone(),
            class_names_clone.clone(),
            tx.clone(),
            Some(pid_tx),
        )
        .await
        .unwrap_or_else(|e| TrainingResult {
            success: false,
            best_pt_path: None,
            class_names: class_names_clone.clone(),
            error_message: Some(e.to_string()),
            hyperparameters: None,
            metrics: None,
        });

        // Limpiar PID al finalizar (proceso ya no existe)
        if let Ok(mut g) = st.active_python_pid.lock() {
            *g = None;
        }

        if let Some(ref best) = training_result.best_pt_path {
            if let Ok(mut g) = st.latest_best_pt.lock() {
                *g = Some(best.clone());
            }
        }

        let cleanup_root = project_root_clone.clone();
        let cleanup_temp = temp_dir.clone();
        tokio::task::spawn_blocking(move || {
            cleanup_temp_files(&cleanup_root, &cleanup_temp);
        });

        {
            if let Ok(mut g) = st.training_temp_dir.lock() {
                *g = None;
            }
        }

        {
            if let Ok(mut g) = st.is_training.lock() {
                *g = false;
            }
        }

        let _ = app_clone.emit::<TrainingResult>("training:complete", training_result);
    });

    // Watcher de seguridad: si el spawn paniquea (JoinError::panicked),
    // garantizamos que is_training se resetea a false (parche C-03).
    let app_panic_guard = app.clone();
    tokio::spawn(async move {
        if let Err(join_err) = training_handle.await {
            eprintln!("[DeepSight] PANIC en training spawn: {:?}", join_err);
            let st = app_panic_guard.state::<AppState>();
            if let Ok(mut g) = st.is_training.lock() {
                *g = false;
            }
            if let Ok(mut g) = st.training_temp_dir.lock() {
                *g = None;
            }
            let _ = app_panic_guard.emit::<TrainingResult>(
                "training:complete",
                TrainingResult {
                    success: false,
                    best_pt_path: None,
                    class_names: vec![],
                    error_message: Some(
                        "Error interno inesperado en el motor de entrenamiento. Reinicia la aplicacion."
                            .to_string(),
                    ),
                    hyperparameters: None,
                    metrics: None,
                },
            );
        }
    });


    let app_clone_rx = app.clone();
    tokio::spawn(async move {
        // PARCHE A-03: Rate-limiting del IPC bridge.
        // Los eventos 'log' y 'raw' se acumulan en un buffer y se emiten en batch
        // cada 150ms para evitar saturar el WebView con decenas de mensajes/segundo.
        // Los eventos críticos (progress, hyperparameters, complete, error) pasan sin throttling.
        use tokio::time::{interval, Duration};
        let mut flush_ticker = interval(Duration::from_millis(150));
        let mut log_buffer: Vec<ProcessOutput> = Vec::new();

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        None => {
                            // Canal cerrado: vaciar buffer pendiente y salir
                            if !log_buffer.is_empty() {
                                let batch = std::mem::take(&mut log_buffer);
                                let _ = app_clone_rx.emit("training:log_batch", &batch);
                            }
                            break;
                        }
                        Some(output) => {
                            let is_critical = matches!(
                                output.line_type.as_str(),
                                "progress" | "hyperparameters" | "complete" | "error"
                            );

                            if is_critical {
                                // Vaciar buffer acumulado antes de emitir el crítico
                                if !log_buffer.is_empty() {
                                    let batch = std::mem::take(&mut log_buffer);
                                    let _ = app_clone_rx.emit("training:log_batch", &batch);
                                }
                                let emit_type = match output.line_type.as_str() {
                                    "progress" => "training:progress",
                                    "hyperparameters" => "training:hyperparameters",
                                    "complete" => "training:json_complete",
                                    "error" => "training:error",
                                    _ => "training:log",
                                };
                                let _ = app_clone_rx.emit(emit_type, output.clone());

                                if output.line_type == "complete" {
                                    if let Some(best) = output.content.get("best_pt_path").and_then(|v| v.as_str()) {
                                        if let Ok(mut g) = app_clone_rx.state::<AppState>().latest_best_pt.lock() {
                                            *g = Some(best.to_string());
                                        }
                                    }
                                }

                            } else {
                                // Acumular en buffer para el próximo flush
                                log_buffer.push(output);
                                // Límite de buffer para no acumular memoria indefinidamente
                                if log_buffer.len() >= 50 {
                                    let batch = std::mem::take(&mut log_buffer);
                                    let _ = app_clone_rx.emit("training:log_batch", &batch);
                                }
                            }
                        }
                    }
                }
                _ = flush_ticker.tick() => {
                    if !log_buffer.is_empty() {
                        let batch = std::mem::take(&mut log_buffer);
                        let _ = app_clone_rx.emit("training:log_batch", &batch);
                    }
                }
            }
        }
    });


    Ok(StartTrainingResponse {
        success: true,
        message: "Entrenamiento iniciado en segundo plano".to_string(),
    })
}

#[tauri::command]
async fn run_test_inference(
    state: State<'_, AppState>,
    image_path: String,
) -> Result<InferenceResult, String> {
    {
        let training_guard = state
            .is_training
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        if *training_guard {
            return Err(
                "No se puede ejecutar inferencia mientras se entrena el modelo.".to_string(),
            );
        }
    }

    let model_path = {
        let guard = state
            .latest_best_pt
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        guard.clone().ok_or_else(|| {
            "No hay modelo entrenado. Entrena primero antes de probar.".to_string()
        })?
    };

    let model_pb = PathBuf::from(&model_path);
    if !model_pb.exists() {
        return Err(format!("Archivo best.pt no encontrado en: {}", model_path));
    }

    let image_pb = PathBuf::from(&image_path);
    if !image_pb.exists() {
        return Err(format!("Imagen no encontrada: {}", image_path));
    }

    if !is_valid_image_extension(&image_path) {
        return Err(format!("Formato de imagen invalido: {}", image_path));
    }

    let class_names = {
        let guard = state
            .latest_class_names
            .lock()
            .map_err(|e| format!("Estado bloqueado: {}", e))?;
        guard.clone()
    };

    if class_names.is_empty() {
        return Err("No hay nombres de clases guardados. Re-entrena el modelo.".to_string());
    }

    let project_root = resolve_project_root().map_err(|e| {
        format!(
            "No se pudo resolver la raiz del proyecto para inferencia: {}",
            e
        )
    })?;

    run_inference(project_root, model_pb, image_pb, class_names)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn clear_trained_model(state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state
        .latest_best_pt
        .lock()
        .map_err(|e| format!("Estado bloqueado: {}", e))?;
    *guard = None;
    Ok(())
}

fn main() {
    let startup_root = resolve_project_root()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    cleanup_legacy_temp_at_startup(&startup_root);

    tauri::Builder::default()
        .on_window_event(|window, event| {
            // PARCHE A-01: Matar proceso Python huérfano en cierre de ventana.
            // kill_on_drop(true) no es suficiente si el runtime Tokio se destruye
            // antes de que el Drop del Child se ejecute. El kill() explícito garantiza
            // que el proceso no sobreviva consumiendo GPU/RAM.
            if let WindowEvent::CloseRequested { .. } = event {
                if let Ok(pid_guard) = window.app_handle().state::<AppState>().active_python_pid.lock() {
                    if let Some(pid) = *pid_guard {
                        #[cfg(windows)]
                        {
                            // En Windows usamos taskkill para matar el árbol de procesos completo
                            let _ = std::process::Command::new("taskkill")
                                .args(["/F", "/T", "/PID", &pid.to_string()])
                                .output();
                        }
                        #[cfg(not(windows))]
                        {
                            let _ = std::process::Command::new("kill")
                                .args(["-9", &pid.to_string()])
                                .output();
                        }
                        eprintln!("[DeepSight] Proceso Python (PID {}) terminado en cierre de ventana.", pid);
                    }
                }
            }

            if let WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event {
                let path_strings: Vec<String> = paths
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                let _ = window.emit("file-drop-paths", path_strings);
            }
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            check_python,
            sanitize_name,
            has_trained_model,
            start_training,
            run_test_inference,
            clear_trained_model,
            copy_best_pt,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("DeepSight - Error fatal al iniciar: {}", e);
            std::process::exit(1);
        });
}

#[tauri::command]
fn copy_best_pt(from: String, to: String) -> Result<(), String> {
    let src = PathBuf::from(&from);
    let dst = PathBuf::from(&to);
    if !src.exists() {
        return Err(format!("Archivo origen no existe: {}", from));
    }
    if let Some(parent) = dst.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    std::fs::copy(&src, &dst)
        .map(|_| ())
        .map_err(|e| format!("No se pudo copiar: {}", e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    main();
}
