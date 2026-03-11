use std::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;

pub async fn run_server(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    let mut child = TokioCommand::from(cmd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start server: {}", e))?;

    let stdout = child.stdout.take().expect("Failed to get stdout");
    let stderr = child.stderr.take().expect("Failed to get stderr");
    
    let stdout_reader = BufReader::new(stdout);
    let stderr_reader = BufReader::new(stderr);

    let stdout_task = tokio::task::spawn(async move {
        let mut lines = stdout_reader.lines();
        while let Some(line) = lines.next_line().await.unwrap_or_else(|_| None) {
            println!("{}", line);
        }
    });

    let stderr_task = tokio::task::spawn(async move {
        let mut lines = stderr_reader.lines();
        while let Some(line) = lines.next_line().await.unwrap_or_else(|_| None) {
            eprintln!("{}", line);
        }
    });

    let status = child.wait().await?;
    
    let _ = tokio::join!(stdout_task, stderr_task);

    println!("--------------------------------------------------");
    println!("Server exited with status: {:?}", status);

    Ok(())
}
