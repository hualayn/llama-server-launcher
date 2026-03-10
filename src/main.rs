use serde::Deserialize;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;

// ==========================================
// 1. 配置结构体定义
// ==========================================

/// LoRA 缩放项配置
/// 包含 LoRA 适配器文件路径和缩放系数
#[derive(Debug, Deserialize, Clone)]
struct LoraScaledItem {
    file: PathBuf,
    scale: f64,
}

/// LoRA 配置
/// 支持普通 LoRA 和带缩放的 LoRA 适配器
#[derive(Debug, Deserialize, Clone)]
struct LoraConfig {
    #[serde(default)]
    file: Option<PathBuf>,
    #[serde(default)]
    scaled: Option<Vec<LoraScaledItem>>,
}

/// GPU 层数配置
/// 支持整型 (指定层数) 或字符串 ("auto", "all")
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum GpuLayers {
    Int(i32),
    String(String),
}

/// 原始配置结构体
/// 直接从 TOML 文件解析，字段类型与配置文件一致
#[derive(Debug, Deserialize)]
struct RawConfig {
    llama_server_path: Option<PathBuf>,
    model: PathBuf,
    host: Option<String>,
    port: Option<u16>,
    threads: Option<i32>,
    threads_batch: Option<i32>,
    ctx_size: Option<i32>,
    n_predict: Option<i32>,
    batch_size: Option<i32>,
    ubatch_size: Option<i32>,
    n_gpu_layers: Option<GpuLayers>,
    device: Option<String>,
    split_mode: Option<String>,
    tensor_split: Option<String>,
    main_gpu: Option<i32>,
    seed: Option<i32>,
    temp: Option<f64>,
    top_k: Option<i32>,
    top_p: Option<f64>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    verbose: Option<bool>,
    log_verbosity: Option<i32>,
    webui: Option<bool>,
    embeddings: Option<bool>,
    continuous_batching: Option<bool>,
    cpu_moe: Option<bool>,
    n_cpu_moe: Option<i32>,
    mmproj: Option<PathBuf>,
    lora: Option<LoraConfig>,
    chat_template_kwargs: Option<String>,
}

/// 解析后的配置结构体
/// 只包含被用户配置的字段，用于构建命令行参数
/// 如果用户在 TOML 中没写，这个字段就是 None，构建命令时会跳过
#[derive(Debug)]
struct Config {
    llama_server_path: Option<PathBuf>,
    model: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    threads: Option<i32>,
    threads_batch: Option<i32>,
    ctx_size: Option<i32>,
    n_predict: Option<i32>,
    batch_size: Option<i32>,
    ubatch_size: Option<i32>,
    n_gpu_layers: Option<String>,
    device: Option<String>,
    split_mode: Option<String>,
    tensor_split: Option<String>,
    main_gpu: Option<i32>,
    seed: Option<i32>,
    temp: Option<f64>,
    top_k: Option<i32>,
    top_p: Option<f64>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    verbose: Option<bool>,
    log_verbosity: Option<i32>,
    webui: Option<bool>,
    embeddings: Option<bool>,
    continuous_batching: Option<bool>,
    cpu_moe: Option<bool>,
    n_cpu_moe: Option<i32>,
    mmproj: Option<PathBuf>,
    lora_file: Option<PathBuf>,
    lora_scaled: Vec<(PathBuf, f64)>,
    chat_template_kwargs: Option<String>,
}

/// 将 RawConfig 转换为 Config
/// 处理类型转换和默认值设置
impl From<RawConfig> for Config {
    fn from(raw: RawConfig) -> Self {
        // 提取 LoRA 数据
        let lora_file = raw.lora.as_ref().and_then(|l| l.file.clone());
        
        // 提取带缩放的 LoRA 列表
        let lora_scaled_vec: Vec<(PathBuf, f64)> = raw.lora
            .as_ref()
            .and_then(|l| l.scaled.as_ref().map(|items| {
                items.iter()
                    .map(|item| (item.file.clone(), item.scale))
                    .collect()
            }))
            .unwrap_or_default();

        // 将 GPU 层数统一转换为字符串
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
        }
    }
}

// ==========================================
// 2. 主逻辑
// ==========================================

/// 主函数入口
/// 1. 读取并解析配置文件
/// 2. 构建 llama-server 命令行参数
/// 3. 启动 llama-server 进程
/// 4. 实时输出服务器日志
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 获取配置文件路径（默认 config.toml）
    let config_path = env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    
    println!("Loading configuration from: {}", config_path);
    
    // 读取配置文件内容
    let raw_config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config file: {}", e))?;
    
    // 解析 TOML 格式配置
    let raw_config: RawConfig = toml::from_str(&raw_config_str)
        .map_err(|e| format!("Failed to parse TOML: {}", e))?;
    
    // 转换为内部配置结构
    let config = Config::from(raw_config);

    // 1. 获取模型路径 (必填)
    let model_path = config.model
        .as_ref()
        .ok_or_else(|| "Error: 'model' path is required in config.toml")?;

    // 2. 确定 llama-server 路径
    let server_path = if let Some(path) = config.llama_server_path {
        path
    } else {
        println!("Warning: No server path specified in config. Trying to find 'llama-server' in PATH...");
        PathBuf::from("llama-server")
    };

    // 创建命令对象
    let mut cmd = Command::new(&server_path);
    cmd.arg("-m").arg(model_path.to_str().unwrap_or(""));
    
    // ==========================================
    // 构建命令行参数：只添加 Config 中非 None 的字段
    // ==========================================

    // 通用参数
    if let Some(h) = config.host { cmd.arg("--host").arg(h); }
    if let Some(p) = config.port { cmd.arg("--port").arg(p.to_string()); }
    if let Some(t) = config.threads { cmd.arg("-t").arg(t.to_string()); }
    if let Some(tb) = config.threads_batch { cmd.arg("--threads-batch").arg(tb.to_string()); }
    if let Some(c) = config.ctx_size { cmd.arg("-c").arg(c.to_string()); }
    if let Some(np) = config.n_predict { cmd.arg("-n").arg(np.to_string()); }
    if let Some(bs) = config.batch_size { cmd.arg("-b").arg(bs.to_string()); }
    if let Some(ubs) = config.ubatch_size { cmd.arg("--ubatch-size").arg(ubs.to_string()); }
    
    // GPU 参数
    if let Some(gpu_layers) = config.n_gpu_layers { cmd.arg("--gpu-layers").arg(gpu_layers); }
    if let Some(dev) = config.device { cmd.arg("--device").arg(dev); }
    if let Some(sm) = config.split_mode { cmd.arg("--split-mode").arg(sm); }
    if let Some(ts) = config.tensor_split { cmd.arg("--tensor-split").arg(ts); }
    if let Some(mg) = config.main_gpu { cmd.arg("--main-gpu").arg(mg.to_string()); }
    
    // 采样参数
    if let Some(seed) = config.seed { cmd.arg("-s").arg(seed.to_string()); }
    if let Some(temp) = config.temp { cmd.arg("--temp").arg(temp.to_string()); }
    if let Some(tk) = config.top_k { cmd.arg("--top-k").arg(tk.to_string()); }
    if let Some(tp) = config.top_p { cmd.arg("--top-p").arg(tp.to_string()); }
    if let Some(mp) = config.min_p { cmd.arg("--min-p").arg(mp.to_string()); }
    if let Some(rp) = config.repeat_penalty { cmd.arg("--repeat-penalty").arg(rp.to_string()); }

    // 对话模板参数
    if let Some(ct) = config.chat_template_kwargs {
        cmd.arg("--chat-template-kwargs").arg(ct);
    }
    
    // 其他参数 (显式布尔值处理)
    if let Some(v) = config.verbose { 
        if v { cmd.arg("-v"); } 
    }
    if let Some(lv) = config.log_verbosity { cmd.arg("--log-verbosity").arg(lv.to_string()); }
    
    // WebUI 处理：如果配置了 false，显式传 --no-webui；如果未配置，默认开启
    if let Some(w) = config.webui {
        if w {
            // 显式开启也可以，但通常不加参数默认就是开启
        } else {
            cmd.arg("--no-webui");
        }
    }

    // Embeddings
    if let Some(e) = config.embeddings {
        if e { cmd.arg("--embedding"); }
    }

    // Continuous Batching
    if let Some(cb) = config.continuous_batching {
        if cb {
            // 默认开启，不加参数
        } else {
            cmd.arg("--no-cont-batching");
        }
    }

    // MoE 参数
    if let Some(val) = config.cpu_moe {
        if val {
            cmd.arg("--cpu-moe");
        }
    }
    if let Some(n) = config.n_cpu_moe {
        cmd.arg("--n-cpu-moe").arg(n.to_string());
    }

    // 多模态参数
    if let Some(mmproj_path) = config.mmproj {
        cmd.arg("--mmproj").arg(mmproj_path.to_str().unwrap_or(""));
    }

    // LoRA 参数
    if let Some(lora_file) = config.lora_file {
        cmd.arg("--lora").arg(lora_file.to_str().unwrap_or(""));
    }
    
    // 带缩放的 LoRA
    if !config.lora_scaled.is_empty() {
        let lora_args: Vec<String> = config.lora_scaled
            .iter()
            .map(|(path, scale)| format!("{}:{:.2}", path.display(), scale))
            .collect();
        cmd.arg("--lora-scaled").arg(&lora_args.join(","));
    }

    println!("Starting command: {:?}", cmd);
    println!("--------------------------------------------------");

    // 启动 llama-server 进程
    let mut child = TokioCommand::from(cmd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start server at {:?}: {}", server_path, e))?;

    // 获取标准输出和错误输出流
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let stderr = child.stderr.take().expect("Failed to get stderr");
    
    let stdout_reader = BufReader::new(stdout);
    let stderr_reader = BufReader::new(stderr);

    // 异步任务：实时输出标准输出
    let stdout_task = tokio::task::spawn(async move {
        let mut lines = stdout_reader.lines();
        while let Some(line) = lines.next_line().await.unwrap_or_else(|_| None) {
            println!("{}", line);
        }
    });

    // 异步任务：实时输出错误输出
    let stderr_task = tokio::task::spawn(async move {
        let mut lines = stderr_reader.lines();
        while let Some(line) = lines.next_line().await.unwrap_or_else(|_| None) {
            eprintln!("{}", line);
        }
    });

    // 等待进程退出
    let status = child.wait().await?;
    
    // 等待所有输出任务完成
    let _ = tokio::join!(stdout_task, stderr_task);

    println!("--------------------------------------------------");
    println!("Server exited with status: {:?}", status);

    Ok(())
}
