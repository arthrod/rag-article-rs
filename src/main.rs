use anyhow::Result;
use enhanced_rag_article_generator::cli::run_cli;
use tracing_subscriber;

// Application constants
const APP_NAME: &str = "Enhanced RAG Article Generator";
const VERSION: &str = "1.0.0";

/// Main entry point for the application.
///
/// Initializes logging, runtime environment, and hands off control
/// to the CLI module. Handles top-level errors gracefully.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging system
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Run CLI interface with error handling
    match run_cli().await {
        Ok(_) => {
            println!(
                "Application {} v{} finished successfully",
                APP_NAME, VERSION
            );
            Ok(())
        }
        Err(e) => {
            eprintln!("Critical application error: {}", e);

            // Print error cause chain for debugging
            let mut source = e.source();
            while let Some(err) = source {
                eprintln!("  Cause: {}", err);
                source = err.source();
            }

            std::process::exit(1);
        }
    }
}
