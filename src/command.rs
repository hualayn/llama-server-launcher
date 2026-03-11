use crate::config::Config;
use std::path::PathBuf;
use std::process::Command;

pub fn build_command(server_path: PathBuf, model_path: &PathBuf, config: &Config) -> Command {
    let mut cmd = Command::new(&server_path);
    cmd.arg("-m").arg(model_path.to_str().unwrap_or(""));

    if let Some(h) = &config.host {
        cmd.arg("--host").arg(h.clone());
    }
    if let Some(p) = config.port {
        cmd.arg("--port").arg(p.to_string());
    }
    if let Some(t) = config.threads {
        cmd.arg("-t").arg(t.to_string());
    }
    if let Some(tb) = config.threads_batch {
        cmd.arg("--threads-batch").arg(tb.to_string());
    }
    if let Some(c) = config.ctx_size {
        cmd.arg("-c").arg(c.to_string());
    }
    if let Some(np) = config.n_predict {
        cmd.arg("-n").arg(np.to_string());
    }
    if let Some(bs) = config.batch_size {
        cmd.arg("-b").arg(bs.to_string());
    }
    if let Some(ubs) = config.ubatch_size {
        cmd.arg("--ubatch-size").arg(ubs.to_string());
    }

    if let Some(gpu_layers) = &config.n_gpu_layers {
        cmd.arg("--gpu-layers").arg(gpu_layers.clone());
    }
    if let Some(dev) = &config.device {
        cmd.arg("--device").arg(dev.clone());
    }
    if let Some(sm) = &config.split_mode {
        cmd.arg("--split-mode").arg(sm.clone());
    }
    if let Some(ts) = &config.tensor_split {
        cmd.arg("--tensor-split").arg(ts.clone());
    }
    if let Some(mg) = config.main_gpu {
        cmd.arg("--main-gpu").arg(mg.to_string());
    }

    if let Some(seed) = config.seed {
        cmd.arg("-s").arg(seed.to_string());
    }
    if let Some(temp) = config.temp {
        cmd.arg("--temp").arg(temp.to_string());
    }
    if let Some(tk) = config.top_k {
        cmd.arg("--top-k").arg(tk.to_string());
    }
    if let Some(tp) = config.top_p {
        cmd.arg("--top-p").arg(tp.to_string());
    }
    if let Some(mp) = config.min_p {
        cmd.arg("--min-p").arg(mp.to_string());
    }
    if let Some(pp) = config.presence_penalty {
        cmd.arg("--presence-penalty").arg(pp.to_string());
    }
    if let Some(rp) = config.repeat_penalty {
        cmd.arg("--repeat-penalty").arg(rp.to_string());
    }

    if let Some(ct) = &config.chat_template_kwargs {
        cmd.arg("--chat-template-kwargs").arg(ct.clone());
    }

    if let Some(v) = config.verbose {
        if v {
            cmd.arg("-v");
        }
    }
    if let Some(lv) = config.log_verbosity {
        cmd.arg("--log-verbosity").arg(lv.to_string());
    }

    if let Some(w) = config.webui {
        if !w {
            cmd.arg("--no-webui");
        }
    }

    if let Some(e) = config.embeddings {
        if e {
            cmd.arg("--embedding");
        }
    }

    if let Some(cb) = config.continuous_batching {
        if !cb {
            cmd.arg("--no-cont-batching");
        }
    }

    if let Some(val) = config.cpu_moe {
        if val {
            cmd.arg("--cpu-moe");
        }
    }
    if let Some(n) = config.n_cpu_moe {
        cmd.arg("--n-cpu-moe").arg(n.to_string());
    }

    if let Some(mmproj_path) = &config.mmproj {
        cmd.arg("--mmproj").arg(mmproj_path.to_str().unwrap_or(""));
    }

    if let Some(lora_file) = &config.lora_file {
        cmd.arg("--lora").arg(lora_file.to_str().unwrap_or(""));
    }

    if !config.lora_scaled.is_empty() {
        let lora_args: Vec<String> = config
            .lora_scaled
            .iter()
            .map(|(path, scale)| format!("{}:{:.2}", path.display(), scale))
            .collect();
        cmd.arg("--lora-scaled").arg(&lora_args.join(","));
    }

    cmd
}
