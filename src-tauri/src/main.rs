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

    tokio::spawn(async move {
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

        let training_result = run_training(
            project_root_clone.clone(),
            temp_dir.clone(),
            class_names_clone.clone(),
            tx.clone(),
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

    let app_clone_rx = app.clone();
    tokio::spawn(async move {
        while let Some(output) = rx.recv().await {
            let emit_type = match output.line_type.as_str() {
                "progress" => "training:progress",
                "log" => "training:log",
                "hyperparameters" => "training:hyperparameters",
                "complete" => "training:json_complete",
                "error" => "training:error",
                _ => "training:raw",
            };
            let _ = app_clone_rx.emit(emit_type, output.clone());

            if output.line_type == "complete" {
                if let Some(best) = output.content.get("best_pt_path").and_then(|v| v.as_str()) {
                    let st = app_clone_rx.state::<AppState>();
                    if let Ok(mut g) = st.latest_best_pt.lock() {
                        *g = Some(best.to_string());
                    };
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
