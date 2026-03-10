# Llama Server Launcher

一个用于启动和配置 llama.cpp `llama-server` 的 Rust 工具。

## 功能特性

- 通过 TOML 配置文件灵活配置 llama-server 启动参数
- 支持 GPU layers、LoRA 适配器、多模态投影等高级配置
- 实时输出服务器日志
- 自动处理命令行参数构建

## 快速开始

### 1. 配置文件

编辑 `config.toml` 文件，配置您的模型路径和参数：

```toml
# llama-server 路径 (必填)
llama_server_path = "path/to/llama-server.exe"

# 模型路径 (必填)
model = "path/to/model.gguf"

# 服务器设置
host = "0.0.0.0"
port = 8080

# GPU 设置
n_gpu_layers = 99

# 更多配置项见 config.toml
```

### 2. 运行

```bash
# 使用默认配置文件 (config.toml)
cargo run

# 指定其他配置文件
cargo run -- --config my-config.toml
```

### 3. 构建发布版本

```bash
cargo build --release
```

构建产物位于 `target/release/llama-server-launcher.exe`

## 配置选项

### 服务器设置

| 参数 | 类型 | 描述 |
|------|------|------|
| `llama_server_path` | PathBuf | llama-server 可执行文件路径 |
| `model` | PathBuf | 模型文件路径 (必填) |
| `host` | String | 监听地址 |
| `port` | u16 | 监听端口 |

### 通用参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `threads` | i32 | CPU 线程数 |
| `threads_batch` | i32 | 批处理线程数 |
| `ctx_size` | i32 | 上下文大小 |
| `n_predict` | i32 | 最大生成 token 数 (-1 无限) |
| `batch_size` | i32 | 批处理大小 |
| `ubatch_size` | i32 | 微批处理大小 |

### GPU 参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_gpu_layers` | i32/String | GPU 加载层数 ("auto" 或数字) |
| `device` | String | 设备名称 |
| `split_mode` | String | 分割模式 ("layer", "row", "column") |
| `tensor_split` | String | GPU 间张量分割 (如 "5,2") |
| `main_gpu` | i32 | 主 GPU 索引 |

### 采样参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `seed` | i32 | 随机种子 |
| `temp` | f64 | 温度 |
| `top_k` | i32 | Top-K 采样 |
| `top_p` | f64 | Top-P 采样 |
| `min_p` | f64 | Min-P 采样 |
| `repeat_penalty` | f64 | 重复惩罚 |

### 其他参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `verbose` | bool | 详细输出 |
| `log_verbosity` | i32 | 日志级别 |
| `webui` | bool | 启用 Web UI |
| `embeddings` | bool | 启用 Embedding 模式 |
| `continuous_batching` | bool | 启用连续批处理 |
| `cpu_moe` | bool | 将 MoE 权重保留在 CPU |
| `n_cpu_moe` | i32 | CPU 上保留的 MoE 层数 |
| `mmproj` | PathBuf | 多模态投影文件路径 |
| `chat_template_kwargs` | String | 对话模板参数 |

### LoRA 配置

```toml
[lora]
file = "path/to/lora.gguf"

[[lora.scaled]]
file = "path/to/lora1.gguf"
scale = 1.5
```

## 项目结构

```
.
├── Cargo.toml          # 项目配置和依赖
├── config.toml         # 配置文件示例
├── src/
│   └── main.rs         # 主程序源码
└── target/             # 构建输出
```

## 依赖

- Rust 2021+
- tokio (异步运行时)
- serde (配置解析)
- toml (TOML 解析)

## 构建优化

项目已配置发布模式优化：

- 大小优化 (`opt-level = "z"`)
- LTO 链接时优化
- 符号表剥离

## 许可证

MIT
