use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os_name: String,
    pub os_version: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub total_disk_gb: u64,
    pub available_disk_gb: u64,
    pub is_low_spec: bool,
    pub python_available: bool,
    pub python_version: String,
}

pub fn gather_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let total_memory_bytes = sys.total_memory();
    let available_memory_bytes = sys.available_memory();
    let total_memory_mb = total_memory_bytes / 1024 / 1024;
    let available_memory_mb = available_memory_bytes / 1024 / 1024;

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "CPU Desconocido".to_string());

    let cpu_cores = sys.cpus().len();

    let os_name = System::name().unwrap_or_else(|| "Desconocido".to_string());
    let os_version = System::os_version().unwrap_or_else(|| "N/A".to_string());

    let mut total_disk_gb: u64 = 0;
    let mut available_disk_gb: u64 = 0;
    let disks = Disks::new_with_refreshed_list();
    for disk in disks.list() {
        total_disk_gb += disk.total_space() / 1024 / 1024 / 1024;
        available_disk_gb += disk.available_space() / 1024 / 1024 / 1024;
    }

    let is_low_spec = total_memory_mb < 4096 || cpu_cores <= 2;

    let (python_available, python_version) = check_python();

    SystemInfo {
        os_name,
        os_version,
        cpu_brand,
        cpu_cores,
        total_memory_mb,
        available_memory_mb,
        total_disk_gb,
        available_disk_gb,
        is_low_spec,
        python_available,
        python_version,
    }
}

fn check_python() -> (bool, String) {
    use std::process::Command;

    let candidates = if cfg!(windows) {
        vec!["python", "python3", "py"]
    } else {
        vec!["python3", "python"]
    };

    for cmd in &candidates {
        let mut command = Command::new(cmd);
        command.arg("--version");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        if let Ok(output) = command.output() {
            if output.status.success() {
                let stdout_ver = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .to_string()
                    .trim_start_matches("Python ")
                    .to_string();
                let stderr_ver = String::from_utf8_lossy(&output.stderr)
                    .trim()
                    .to_string()
                    .trim_start_matches("Python ")
                    .to_string();
                let ver = if !stdout_ver.is_empty() {
                    stdout_ver
                } else if !stderr_ver.is_empty() {
                    stderr_ver
                } else {
                    "Desconocida".to_string()
                };
                return (true, ver);
            }
        }
    }

    (false, "No detectado".to_string())
}

#[allow(dead_code)]
pub fn check_ram_sufficient(required_mb: u64) -> bool {
    let mut sys = System::new_all();
    sys.refresh_memory();
    let available = sys.available_memory() / 1024 / 1024;
    available >= required_mb
}
