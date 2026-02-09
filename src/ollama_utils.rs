use anyhow::Result;
use reqwest;
use serde::Deserialize;
use tracing::{debug, error, info, warn};

/// Universal structure for Ollama API response.
/// Supports all formats: success responses, errors, chat format.
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    // Fields for successful response
    pub model: Option<String>,
    pub created_at: Option<String>,
    pub response: Option<String>,     // /api/generate
    pub message: Option<ChatMessage>, // /api/chat
    pub done: Option<bool>,

    // Field for errors
    pub error: Option<String>,

    // Additional performance fields
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<i32>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<i32>,
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

impl OllamaResponse {
    /// Checks if the response contains an error
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }

    /// Returns error text if present
    pub fn get_error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Extracts generated content from any format
    pub fn get_content(&self) -> Result<String, String> {
        // Check for error first
        if let Some(error) = &self.error {
            return Err(error.clone());
        }

        // Check /api/generate format
        if let Some(response) = &self.response {
            return Ok(response.clone());
        }

        // Check /api/chat format
        if let Some(message) = &self.message {
            return Ok(message.content.clone());
        }

        // If nothing found
        Err("No content found in Ollama response".to_string())
    }

    /// Checks if generation is complete
    pub fn is_done(&self) -> bool {
        self.done.unwrap_or(false)
    }

    /// Returns performance information
    pub fn get_performance_info(&self) -> Option<String> {
        if let (Some(total), Some(eval_count)) = (self.total_duration, self.eval_count) {
            if total == 0 {
                return None;
            }
            let total_seconds = total as f64 / 1_000_000_000.0;
            let tokens_per_second = eval_count as f64 / total_seconds;
            Some(format!(
                "Generated {} tokens in {:.2}s ({:.1} tokens/s)",
                eval_count, total_seconds, tokens_per_second
            ))
        } else {
            None
        }
    }
}

/// Checks model availability in Ollama
pub async fn check_model_availability(model_name: &str, ollama_host: &str) -> Result<bool> {
    let url = format!("{}/api/tags", ollama_host);

    match reqwest::get(&url).await {
        Ok(response) => {
            let tags: OllamaTagsResponse = response.json().await?;
            Ok(tags.models.iter().any(|model| model.name == model_name))
        }
        Err(e) => {
            error!("Failed to check available models: {}", e);
            Ok(false)
        }
    }
}

/// Gets a list of all available models
pub async fn get_available_models(ollama_host: &str) -> Result<Vec<String>> {
    let url = format!("{}/api/tags", ollama_host);

    match reqwest::get(&url).await {
        Ok(response) => {
            let tags: OllamaTagsResponse = response.json().await?;
            Ok(tags.models.into_iter().map(|model| model.name).collect())
        }
        Err(e) => {
            error!("Failed to get available models: {}", e);
            Ok(Vec::new())
        }
    }
}

/// Checks availability of Ollama server
pub async fn check_ollama_health(ollama_host: &str) -> Result<bool> {
    let health_url = format!("{}/api/tags", ollama_host);

    match reqwest::get(&health_url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// Validates environment before starting generation
pub async fn validate_environment(model_name: &str, ollama_host: &str) -> Result<()> {
    info!("🔍 Checking environment...");

    // Check Ollama availability
    if !check_ollama_health(ollama_host).await? {
        return Err(anyhow::anyhow!(
            "❌ Ollama server unavailable at: {}\n\
            Ensure Ollama is running: `ollama serve`",
            ollama_host
        ));
    }
    info!("✅ Ollama server available");

    // Check model
    if check_model_availability(model_name, ollama_host).await? {
        info!("✅ Model '{}' available", model_name);
    } else {
        let available = get_available_models(ollama_host).await?;
        if available.is_empty() {
            return Err(anyhow::anyhow!(
                "❌ No models installed.\n\
                Install model using command: ollama pull {}",
                model_name
            ));
        } else {
            return Err(anyhow::anyhow!(
                "❌ Model '{}' not found.\n\
                Available models:\n  • {}\n\n\
                Install required model: ollama pull {}",
                model_name,
                available.join("\n  • "),
                model_name
            ));
        }
    }

    Ok(())
}

/// Automatically installs model if missing
pub async fn auto_install_model(model_name: &str) -> Result<()> {
    info!("🔄 Automatically installing model {}...", model_name);

    let output = tokio::process::Command::new("ollama")
        .args(["pull", model_name])
        .output()
        .await?;

    if output.status.success() {
        info!("✅ Model {} successfully installed", model_name);
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(anyhow::anyhow!("❌ Error installing model: {}", error))
    }
}

/// Generation with fallback to popular models
pub async fn generate_with_fallback<F, Fut>(
    prompt: &str,
    primary_model: &str,
    ollama_host: &str,
    generate_fn: F,
) -> Result<String>
where
    F: Fn(&str, &str) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    // Try primary model first
    match generate_fn(prompt, primary_model).await {
        Ok(response) => Ok(response),
        Err(e) if e.to_string().contains("not found") => {
            warn!(
                "Primary model '{}' not found, trying backup models",
                primary_model
            );

            // Popular small models for fallback
            let fallback_models = [
                "llama3.2:3b",
                "llama3.1:8b",
                "qwen2.5:7b",
                "phi3:mini",
                "gemma2:2b",
            ];

            for model in &fallback_models {
                if check_model_availability(model, ollama_host).await? {
                    match generate_fn(prompt, model).await {
                        Ok(response) => {
                            info!("✅ Successfully used backup model: {}", model);
                            return Ok(response);
                        }
                        Err(e) => {
                            warn!("Backup model {} failed: {}", model, e);
                            continue;
                        }
                    }
                }
            }

            Err(anyhow::anyhow!(
                "❌ No models available.\n\
                Install model using command: ollama pull llama3.2:3b"
            ))
        }
        Err(e) => Err(e),
    }
}

/// Parses Ollama response with error handling
pub fn parse_ollama_response(response_text: &str) -> Result<String> {
    // Log raw response for debugging at debug level to avoid sensitive data leak
    debug!("Raw Ollama response: {}", response_text);

    // Parse universal structure
    let ollama_response: OllamaResponse = serde_json::from_str(response_text)
        .map_err(|e| anyhow::anyhow!("Failed to parse Ollama response: {}", e))?;

    // Check errors and extract content
    match ollama_response.get_content() {
        Ok(content) => {
            info!("✅ Successfully generated {} characters", content.len());

            // Show performance info if available
            if let Some(perf_info) = ollama_response.get_performance_info() {
                info!("📊 Performance: {}", perf_info);
            }

            Ok(content)
        }
        Err(error_msg) => {
            error!("❌ Ollama API error: {}", error_msg);

            // Specific handling for common errors
            if error_msg.contains("not found") {
                return Err(anyhow::anyhow!(
                    "Model not found. Check if model is installed: `ollama pull <model_name>`"
                ));
            } else if error_msg.contains("not loaded") {
                return Err(anyhow::anyhow!(
                    "Model not loaded. Wait for model loading or restart Ollama"
                ));
            } else if error_msg.contains("connection") || error_msg.contains("network") {
                return Err(anyhow::anyhow!(
                    "Network error connecting to Ollama. Check if server is running"
                ));
            } else {
                return Err(anyhow::anyhow!("Ollama error: {}", error_msg));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_success_response() {
        let json = r#"{"model":"test","response":"Hello world","done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Hello world");
        assert!(!response.has_error());
    }

    #[test]
    fn test_parse_error_response() {
        let json = r#"{"error":"model not found"}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.has_error());
        assert_eq!(response.get_error().unwrap(), "model not found");
    }

    #[test]
    fn test_parse_chat_response() {
        let json =
            r#"{"model":"test","message":{"role":"assistant","content":"Hello"},"done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Hello");
    }

    #[test]
    fn test_ollama_response_is_done() {
        let json = r#"{"model":"test","response":"content","done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.is_done());

        let json2 = r#"{"model":"test","response":"content","done":false}"#;
        let response2: OllamaResponse = serde_json::from_str(json2).unwrap();
        assert!(!response2.is_done());
    }

    #[test]
    fn test_ollama_response_with_performance_info() {
        let json = r#"{
            "model":"test",
            "response":"content",
            "done":true,
            "total_duration":1000000000,
            "eval_count":10
        }"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        let perf_info = response.get_performance_info();
        assert!(perf_info.is_some());
        assert!(perf_info.unwrap().contains("tokens/s"));
    }

    #[test]
    fn test_ollama_response_without_performance_info() {
        let json = r#"{"model":"test","response":"content","done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.get_performance_info().is_none());
    }

    #[test]
    fn test_get_content_error_priority() {
        // Error should take priority over response field
        let json = r#"{"model":"test","response":"content","error":"something failed"}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.get_content().is_err());
        assert_eq!(response.get_content().unwrap_err(), "something failed");
    }

    #[test]
    fn test_get_content_no_content() {
        let json = r#"{"model":"test","done":true}"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert!(response.get_content().is_err());
        assert!(response
            .get_content()
            .unwrap_err()
            .contains("No content found"));
    }

    #[test]
    fn test_parse_ollama_response_success() {
        let response_text = r#"{"model":"test","response":"Generated text","done":true}"#;
        let result = parse_ollama_response(response_text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Generated text");
    }

    #[test]
    fn test_parse_ollama_response_error() {
        let response_text = r#"{"error":"model not found"}"#;
        let result = parse_ollama_response(response_text);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_parse_ollama_response_invalid_json() {
        let response_text = "invalid json{";
        let result = parse_ollama_response(response_text);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[test]
    fn test_parse_ollama_response_with_performance() {
        let response_text = r#"{
            "model":"test",
            "response":"content",
            "done":true,
            "total_duration":2000000000,
            "eval_count":20
        }"#;
        let result = parse_ollama_response(response_text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_info_deserialization() {
        let json = r#"{
            "name":"llama2:7b",
            "modified_at":"2024-01-01T00:00:00Z",
            "size":3825819519
        }"#;
        let model_info: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(model_info.name, "llama2:7b");
        assert_eq!(model_info.size, 3825819519);
    }

    #[test]
    fn test_ollama_tags_response_deserialization() {
        let json = r#"{
            "models": [
                {"name":"model1","modified_at":"2024-01-01T00:00:00Z","size":1000},
                {"name":"model2","modified_at":"2024-01-01T00:00:00Z","size":2000}
            ]
        }"#;
        let tags: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(tags.models.len(), 2);
        assert_eq!(tags.models[0].name, "model1");
        assert_eq!(tags.models[1].name, "model2");
    }

    #[test]
    fn test_chat_message_deserialization() {
        let json = r#"{"role":"assistant","content":"Hello there"}"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, "assistant");
        assert_eq!(message.content, "Hello there");
    }

    #[test]
    fn test_ollama_response_chat_format() {
        let json = r#"{
            "model":"test",
            "message":{"role":"assistant","content":"Chat response"},
            "done":true
        }"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Chat response");
        assert!(response.message.is_some());
        assert_eq!(response.message.as_ref().unwrap().role, "assistant");
    }

    #[test]
    fn test_ollama_response_generate_format() {
        let json = r#"{
            "model":"test",
            "response":"Generate response",
            "done":true
        }"#;
        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.get_content().unwrap(), "Generate response");
        assert!(response.response.is_some());
    }
}
