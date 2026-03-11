//! Llama Server Launcher
//!
//! 一个用于启动 llama-server 的配置管理工具
//! 通过 TOML 配置文件轻松配置和启动 llama-server
//!
//! 作者: b站我爱吃娃娃雪糕
//! 日期: 2026年03月11日

mod config;
mod command;
mod process;

use std::env;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    
    println!("Loading configuration from: {}", config_path);
    
    let config = config::load_config(&config_path)?;

    let model_path = config.model
        .as_ref()
        .ok_or_else(|| "Error: 'model' path is required in config.toml")?;

    let server_path = if let Some(path) = &config.llama_server_path {
        path.clone()
    } else {
        println!("Warning: No server path specified in config. Trying to find 'llama-server' in PATH...");
        PathBuf::from("llama-server")
    };

    let cmd = command::build_command(server_path, model_path, &config);

    println!("Starting_path, model_path command: {:?}", cmd);
    println!("--------------------------------------------------");

    process::run_server(cmd).await?;

    Ok(())
}
