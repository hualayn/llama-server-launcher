use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct LoraScaledItem {
    pub file: PathBuf,
    pub scale: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoraConfig {
    #[serde(default)]
    pub file: Option<PathBuf>,
    #[serde(default)]
    pub scaled: Option<Vec<LoraScaledItem>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum GpuLayers {
    Int(i32),
    String(String),
}

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub llama_server_path: Option<PathBuf>,
    pub model: PathBuf,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub ctx_size: Option<i32>,
    pub n_predict: Option<i32>,
    pub batch_size: Option<i32>,
    pub ubatch_size: Option<i32>,
    pub n_gpu_layers: Option<GpuLayers>,
    pub device: Option<String>,
    pub split_mode: Option<String>,
    pub tensor_split: Option<String>,
    pub main_gpu: Option<i32>,
    pub seed: Option<i32>,
    pub temp: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub verbose: Option<bool>,
    pub log_verbosity: Option<i32>,
    pub webui: Option<bool>,
    pub embeddings: Option<bool>,
    pub continuous_batching: Option<bool>,
    pub cpu_moe: Option<bool>,
    pub n_cpu_moe: Option<i32>,
    pub mmproj: Option<PathBuf>,
    pub lora: Option<LoraConfig>,
    pub chat_template_kwargs: Option<String>,
    pub parallel: Option<i32>,
    pub reasoning: Option<String>,
}

#[derive(Debug)]
pub struct Config {
    pub llama_server_path: Option<PathBuf>,
    pub model: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub ctx_size: Option<i32>,
    pub n_predict: Option<i32>,
    pub batch_size: Option<i32>,
    pub ubatch_size: Option<i32>,
    pub n_gpu_layers: Option<String>,
    pub device: Option<String>,
    pub split_mode: Option<String>,
    pub tensor_split: Option<String>,
    pub main_gpu: Option<i32>,
    pub seed: Option<i32>,
    pub temp: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub verbose: Option<bool>,
    pub log_verbosity: Option<i32>,
    pub webui: Option<bool>,
    pub embeddings: Option<bool>,
    pub continuous_batching: Option<bool>,
    pub cpu_moe: Option<bool>,
    pub n_cpu_moe: Option<i32>,
    pub mmproj: Option<PathBuf>,
    pub lora_file: Option<PathBuf>,
    pub lora_scaled: Vec<(PathBuf, f64)>,
    pub chat_template_kwargs: Option<String>,
    pub parallel: Option<i32>,
    pub reasoning: Option<String>,
}

impl From<RawConfig> for Config {
    fn from(raw: RawConfig) -> Self {
        let lora_file = raw.lora.as_ref().and_then(|l| l.file.clone());

        let lora_scaled_vec: Vec<(PathBuf, f64)> = raw
            .lora
            .as_ref()
            .and_then(|l| {
                l.scaled.as_ref().map(|items| {
                    items
                        .iter()
                        .map(|item| (item.file.clone(), item.scale))
                        .collect()
                })
            })
            .unwrap_or_default();

        let n_gpu_layers = raw.n_gpu_layers.map(|gl| match gl {
            GpuLayers::Int(i) => i.to_string(),
            GpuLayers::String(s) => s,
        });

        Config {
            llama_server_path: raw.llama_server_path,
            model: Some(raw.model),
            host: raw.host,
            port: raw.port,
            threads: raw.threads,
            threads_batch: raw.threads_batch,
            ctx_size: raw.ctx_size,
            n_predict: raw.n_predict,
            batch_size: raw.batch_size,
            ubatch_size: raw.ubatch_size,
            n_gpu_layers,
            device: raw.device,
            split_mode: raw.split_mode,
            tensor_split: raw.tensor_split,
            main_gpu: raw.main_gpu,
            seed: raw.seed,
            temp: raw.temp,
            top_k: raw.top_k,
            top_p: raw.top_p,
            min_p: raw.min_p,
            presence_penalty: raw.presence_penalty,
            repeat_penalty: raw.repeat_penalty,
            verbose: raw.verbose,
            log_verbosity: raw.log_verbosity,
            webui: raw.webui,
            embeddings: raw.embeddings,
            continuous_batching: raw.continuous_batching,
            cpu_moe: raw.cpu_moe,
            n_cpu_moe: raw.n_cpu_moe,
            mmproj: raw.mmproj,
            lora_file,
            lora_scaled: lora_scaled_vec,
            chat_template_kwargs: raw.chat_template_kwargs,
            parallel: raw.parallel,
            reasoning: raw.reasoning,
        }
    }
}

pub fn load_config(path: &str) -> Result<Config, String> {
    let raw_config_str =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read config file: {}", e))?;

    let raw_config: RawConfig =
        toml::from_str(&raw_config_str).map_err(|e| format!("Failed to parse TOML: {}", e))?;

    Ok(Config::from(raw_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_config_basic() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
model = "./models/llama.bin"
host = "127.0.0.1"
port = 8080
ctx_size = 4096
temp = 0.7
top_p = 0.9
presence_penalty = 1.5
"#
        )
        .unwrap();

        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.model.unwrap(), PathBuf::from("./models/llama.bin"));
        assert_eq!(config.host.unwrap(), "127.0.0.1");
        assert_eq!(config.port.unwrap(), 8080);
        assert_eq!(config.ctx_size.unwrap(), 4096);
        assert!((config.temp.unwrap() - 0.7).abs() < 1e-6);
        assert!((config.top_p.unwrap() - 0.9).abs() < 1e-6);
        assert!((config.presence_penalty.unwrap() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_load_config_missing_model() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "host = \"127.0.0.1\"").unwrap();

        let result = load_config(temp_file.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_gpu_layers_int() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
model = "./model.bin"
n_gpu_layers = 32
"#
        )
        .unwrap();

        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.n_gpu_layers.unwrap(), "32");
    }

    #[test]
    fn test_load_config_gpu_layers_string() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
model = "./model.bin"
n_gpu_layers = "all"
"#
        )
        .unwrap();

        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.n_gpu_layers.unwrap(), "all");
    }

    #[test]
    fn test_load_config_lora() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
model = "./model.bin"

[lora]
file = "./lora.bin"

[[lora.scaled]]
file = "./lora_scaled.bin"
scale = 0.5
"#
        )
        .unwrap();

        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.lora_file.unwrap(), PathBuf::from("./lora.bin"));
        assert_eq!(config.lora_scaled.len(), 1);
        assert_eq!(config.lora_scaled[0].0, PathBuf::from("./lora_scaled.bin"));
        assert!((config.lora_scaled[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_load_config_file_not_found() {
        let result = load_config("nonexistent_config.toml");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read config file"));
    }
}
