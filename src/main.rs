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
use colored::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{} 🦙 Llama Server Launcher", "🚀".cyan());
    println!("{} {}", "─".repeat(30).cyan(), "─".repeat(20).yellow());
    
    let config_path = env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    
    println!("\n{} Loading config from: {}", "📂".blue(), config_path.yellow());
    
    let config = config::load_config(&config_path)?;

    let model_path = config.model
        .as_ref()
        .ok_or_else(|| "Error: 'model' path is required in config.toml")?;

    let server_path = if let Some(path) = &config.llama_server_path {
        path.clone()
    } else {
        println!("\n{} No server path in config, searching in PATH...", "⚠️".yellow());
        PathBuf::from("llama-server")
    };

    let cmd = command::build_command(server_path, model_path, &config);

    let cmd_str = format!("{:?}", cmd);
    let cmd_clean = cmd_str.replace('"', "");
    println!("\n{} Command:\n  {}", "🔧".green(), cmd_clean.yellow());
    println!("{}", "─".repeat(50).white());
    println!();

    process::run_server(cmd).await?;

    Ok(())
}
