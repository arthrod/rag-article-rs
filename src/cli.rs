use crate::ollama_utils::validate_environment;
use crate::persistent::*;
use anyhow::Result;
use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{error, info};

/// Constructs the command-line interface for the application.
///
/// This function defines all available arguments, flags, and options
/// using the `clap` library.
pub fn cli() -> Command {
    Command::new("enhanced-rag-generator")
        .about(
            "Enhanced RAG Article Generator - AI-powered article generation with advanced caching",
        )
        .version("2.0.0")
        .arg(
            Arg::new("query")
                .help("Query for article generation")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("searx-host")
                .long("searx-host")
                .help("SearXNG server address")
                .default_value("http://127.0.0.1:8080"),
        )
        .arg(
            Arg::new("ollama-host")
                .long("ollama-host")
                .help("Ollama server address")
                .default_value("http://localhost:11434"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .short('m')
                .help("Ollama model name for text generation")
                .default_value("qwen3:30b"),
        )
        .arg(
            Arg::new("embedding-model")
                .long("embedding-model")
                .help("Model name for creating embeddings")
                .default_value("nomic-embed-text:latest"),
        )
        .arg(
            Arg::new("max-docs")
                .long("max-docs")
                .help("Maximum number of documents to search")
                .default_value("15")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("File to save the result")
                .default_value("enhanced_article.md"),
        )
        .arg(
            Arg::new("database")
                .long("database")
                .short('d')
                .help("Path to database for persistent storage (optional)")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("cache-days")
                .long("cache-days")
                .help("Maximum age of documents in cache (days)")
                .default_value("7")
                .value_parser(clap::value_parser!(i64)),
        )
        .arg(
            Arg::new("similarity-threshold")
                .long("similarity-threshold")
                .help("Minimum similarity for using cached queries (0.0-1.0)")
                .default_value("0.7")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("quality-threshold")
                .long("quality-threshold")
                .help("Minimum source quality threshold (0.0-1.0)")
                .default_value("0.3")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("enable-semantic")
                .long("enable-semantic")
                .help("Enable semantic search with embeddings")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-personalization")
                .long("enable-personalization")
                .help("Enable personalization (experimental)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("show-cache-stats")
                .long("show-cache-stats")
                .help("Show cache statistics")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("show-quality-stats")
                .long("show-quality-stats")
                .help("Show source quality statistics")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cleanup-cache")
                .long("cleanup-cache")
                .help("Clean outdated entries from cache")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("expertise-level")
                .long("expertise-level")
                .help("User expertise level")
                .value_parser(["beginner", "intermediate", "advanced", "expert"])
                .default_value("intermediate"),
        )
        .arg(
            Arg::new("validate-env")
                .long("validate-env")
                .help("Check environment before running")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("auto-install")
                .long("auto-install")
                .help("Automatically install model if missing")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("concurrent-downloads")
                .long("concurrent-downloads")
                .help("Maximum number of concurrent downloads")
                .default_value("8")
                .value_parser(clap::value_parser!(usize)),
        )
}

/// Main entry point for the CLI application logic.
///
/// Parses command line arguments, initializes the generator,
/// and executes the requested actions (stats, cleanup, or generation).
pub async fn run_cli() -> Result<()> {
    let matches = cli().get_matches();

    let query = matches.get_one::<String>("query").unwrap();
    let searx_host = matches.get_one::<String>("searx-host").unwrap().clone();
    let ollama_host = matches.get_one::<String>("ollama-host").unwrap().clone();
    let model = matches.get_one::<String>("model").unwrap().clone();
    let embedding_model = matches
        .get_one::<String>("embedding-model")
        .unwrap()
        .clone();
    let max_docs = *matches.get_one::<usize>("max-docs").unwrap();
    let output = matches.get_one::<String>("output").unwrap();

    // Extended parameters
    let database_path = matches.get_one::<PathBuf>("database");
    let cache_days = *matches.get_one::<i64>("cache-days").unwrap();
    let similarity_threshold = *matches.get_one::<f32>("similarity-threshold").unwrap();
    let quality_threshold = *matches.get_one::<f32>("quality-threshold").unwrap();
    let enable_semantic = matches.get_flag("enable-semantic");
    let enable_personalization = matches.get_flag("enable-personalization");
    let show_stats = matches.get_flag("show-cache-stats");
    let show_quality_stats = matches.get_flag("show-quality-stats");
    let cleanup_cache = matches.get_flag("cleanup-cache");
    let validate_env = matches.get_flag("validate-env");
    let auto_install = matches.get_flag("auto-install");
    let concurrent_downloads = *matches.get_one::<usize>("concurrent-downloads").unwrap();
    let expertise_level = matches.get_one::<String>("expertise-level").unwrap();

    info!("🚀 Enhanced RAG Article Generator v2.0 - AI-Powered Edition");
    info!("Startup parameters:");
    info!("  📝 Query: {}", query);
    info!("  🔍 SearXNG: {}", searx_host);
    info!("  🤖 Ollama: {}", ollama_host);
    info!("  🧠 LLM Model: {}", model);
    info!("  🎯 Embedding Model: {}", embedding_model);
    info!("  📊 Max Docs: {}", max_docs);
    info!(
        "  🔄 Concurrent Downloads: {} threads",
        concurrent_downloads
    );
    info!("  💾 Output File: {}", output);

    if let Some(db_path) = database_path {
        info!("  🗄️ Database: {:?}", db_path);
        info!("  ⏰ Cache Duration: {} days", cache_days);
        info!("  🎯 Similarity Threshold: {:.2}", similarity_threshold);
        info!("  ⭐ Quality Threshold: {:.2}", quality_threshold);
        info!(
            "  🧠 Semantic Search: {}",
            if enable_semantic {
                "enabled"
            } else {
                "disabled"
            }
        );
        info!(
            "  👤 Personalization: {}",
            if enable_personalization {
                "enabled"
            } else {
                "disabled"
            }
        );
        info!("  🎓 Expertise Level: {}", expertise_level);
    } else {
        info!("  💭 Mode: Memory-only (no persistent storage)");
    }

    // NEW: Environment validation before starting work
    if validate_env || auto_install {
        println!("\n{}", "=".repeat(60));
        println!("🔍 ENVIRONMENT CHECK");
        println!("{}", "=".repeat(60));

        match validate_environment(&model, &ollama_host).await {
            Ok(()) => {
                info!("✅ Environment ready to work");
            }
            Err(e) => {
                // If auto-install is enabled and error is about missing model
                if auto_install && e.to_string().contains("not found") {
                    // Translated error check string might need adjustment in ollama_utils too
                    println!("🔄 Attempting automatic model installation...");

                    match crate::ollama_utils::auto_install_model(&model).await {
                        Ok(()) => {
                            info!("✅ Model installed successfully, re-checking...");
                            validate_environment(&model, &ollama_host).await?;
                        }
                        Err(install_error) => {
                            error!(
                                "❌ Failed to automatically install model: {}",
                                install_error
                            );
                            return Err(e);
                        }
                    }
                } else {
                    error!("❌ {}", e);
                    return Err(e);
                }
            }
        }
    }

    // Settings for the extended cache
    let cache_settings = CacheSettings {
        max_document_age_days: cache_days,
        min_query_similarity: similarity_threshold,
        max_cached_docs: max_docs,
        embedding_dim: None,
        enable_semantic_search: enable_semantic,
        min_quality_score: quality_threshold,
        enable_personalization,
        auto_reindex_interval_hours: 24,
        max_vector_cache_size: 10000,
        max_concurrent_downloads: concurrent_downloads, // NEW FIELD
    };

    // User context for personalization
    let user_context = if enable_personalization {
        let expertise = match expertise_level.as_str() {
            "beginner" => ExpertiseLevel::Beginner,
            "intermediate" => ExpertiseLevel::Intermediate,
            "advanced" => ExpertiseLevel::Advanced,
            "expert" => ExpertiseLevel::Expert,
            _ => ExpertiseLevel::Intermediate,
        };

        Some(UserContext {
            expertise_level: expertise,
            preferred_languages: vec!["en".to_string(), "ru".to_string()],
            frequent_topics: vec![], // Will be populated from history
            interaction_history: vec![chrono::Utc::now()],
        })
    } else {
        None
    };

    // Create the enhanced generator, cloning values as needed
    let mut generator = if let Some(db_path) = database_path {
        info!("🔧 Initializing enhanced persistent storage...");
        PersistentEnhancedRAG::new_with_persistent_storage(
            db_path,
            searx_host.clone(),
            model.clone(),
            embedding_model.clone(),
            Some(ollama_host.clone()),
            Some(cache_settings),
        )?
    } else {
        info!("🧠 Initializing in-memory mode with AI capabilities...");
        PersistentEnhancedRAG::new_in_memory(
            searx_host.clone(),
            model.clone(),
            embedding_model.clone(),
            Some(ollama_host.clone()),
        )?
    };

    // Show cache statistics
    if show_stats {
        let stats = generator.cache_stats().await?;
        println!("\n{}", "=".repeat(60));
        println!("📊 CACHE STATISTICS");
        println!("{}", "=".repeat(60));
        println!("📄 Total Documents: {}", stats.total_documents);
        println!("🆕 Fresh Documents: {}", stats.fresh_documents);
        println!("🔍 Total Queries: {}", stats.total_queries);
        println!("💾 DB Size: {:.2} MB", stats.database_size_mb);

        if database_path.is_some() {
            let cache_efficiency = if stats.total_documents > 0 {
                (stats.fresh_documents as f32 / stats.total_documents as f32) * 100.0
            } else {
                0.0
            };
            println!("⚡ Cache Efficiency: {:.1}%", cache_efficiency);
        }

        println!("{}", "=".repeat(60));

        if show_stats && !show_quality_stats && !cleanup_cache {
            return Ok(());
        }
    }

    // Show source quality statistics
    if show_quality_stats {
        let quality_stats = generator.get_quality_stats().await?;
        println!("\n{}", "=".repeat(60));
        println!("⭐ SOURCE QUALITY STATISTICS");
        println!("{}", "=".repeat(60));
        println!("📊 Total Sources: {}", quality_stats.total_sources);
        println!("🏆 Very High Quality: {}", quality_stats.very_high_quality);
        println!("✨ High Quality: {}", quality_stats.high_quality);
        println!("👍 Medium Quality: {}", quality_stats.medium_quality);
        println!("⚠️ Low Quality: {}", quality_stats.low_quality);
        println!("❌ Very Low Quality: {}", quality_stats.very_low_quality);
        println!(
            "📈 Average Quality Score: {:.3}",
            quality_stats.average_quality_score
        );
        println!("{}", "=".repeat(60));

        if show_quality_stats && !cleanup_cache {
            return Ok(());
        }
    }

    // Clean up cache
    if cleanup_cache {
        println!("\n🧹 Performing cache cleanup...");
        let cleanup_stats = generator.cleanup_cache().await?;
        println!("✅ Documents Removed: {}", cleanup_stats.deleted_documents);
        println!("✅ Queries Removed: {}", cleanup_stats.deleted_queries);

        if cleanup_cache && !show_stats && !show_quality_stats {
            return Ok(());
        }
    }

    // Generate article with extended capabilities
    println!("\n{}", "=".repeat(80));
    println!("🚀 GENERATING AI-ENHANCED ARTICLE");
    println!("{}", "=".repeat(80));

    let start_time = std::time::Instant::now();

    match generator
        .generate_article_with_enhanced_cache(query, max_docs, user_context)
        .await
    {
        Ok(article) => {
            let generation_time = start_time.elapsed();

            println!("\n{}", "=".repeat(80));
            println!("✨ GENERATED AI-ENHANCED ARTICLE");
            println!("{}", "=".repeat(80));
            println!("\n{}", article);

            // Save to file
            tokio::fs::write(output, &article).await?;

            println!("\n{}", "=".repeat(60));
            println!("📊 GENERATION RESULTS");
            println!("{}", "=".repeat(60));
            println!(
                "⏱️ Generation Time: {:.2} seconds",
                generation_time.as_secs_f32()
            );
            println!("📄 Article Length: {} characters", article.len());
            println!("📝 Saved to: {}", output);

            // Show final statistics
            if database_path.is_some() {
                let final_stats = generator.cache_stats().await?;
                println!("\n🔄 UPDATED CACHE STATISTICS:");
                println!(
                    "  📄 Documents: {} (fresh: {})",
                    final_stats.total_documents, final_stats.fresh_documents
                );
                println!("  🔍 Queries: {}", final_stats.total_queries);
                println!("  💾 DB Size: {:.2} MB", final_stats.database_size_mb);

                let quality_stats = generator.get_quality_stats().await?;
                if quality_stats.total_sources > 0 {
                    println!(
                        "  ⭐ Average Quality Score: {:.3}",
                        quality_stats.average_quality_score
                    );
                }
            }

            Ok(())
        }
        Err(e) => {
            let generation_time = start_time.elapsed();

            error!(
                "❌ Error generating article (after {:.2}s): {}",
                generation_time.as_secs_f32(),
                e
            );

            // Show detailed error information
            let mut source = e.source();
            let mut error_chain = 1;
            while let Some(err) = source {
                error!("  📍 Cause {}: {}", error_chain, err);
                source = err.source();
                error_chain += 1;
            }

            // Diagnostic information
            error!("🔍 DIAGNOSTICS:");
            error!("  🌐 SearXNG Available: {}", searx_host);
            error!("  🤖 Ollama Available: {}", ollama_host);
            error!("  🧠 Model: {}", model);
            error!("  🔄 Parallel Threads: {}", concurrent_downloads);

            if let Some(db_path) = database_path {
                error!("  🗄️ Database Path: {:?}", db_path);
            }

            // Suggestions for fixing
            error!("💡 POSSIBLE SOLUTIONS:");
            if e.to_string().contains("not found") {
                error!("  • Install model: ollama pull {}", model);
                error!("  • Or use another model with --model");
            }
            if e.to_string().contains("connection") || e.to_string().contains("network") {
                error!("  • Check if Ollama is running: ollama serve");
                error!("  • Check Ollama address: {}", ollama_host);
            }
            error!("  • Run with --validate-env for diagnostics");
            error!("  • Run with --auto-install for automatic installation");
            error!("  • Reduce number of threads: --concurrent-downloads 4");

            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_construction() {
        let app = cli();
        assert_eq!(app.get_name(), "enhanced-rag-generator");
        assert_eq!(app.get_version().unwrap(), "2.0.0");
    }

    #[test]
    fn test_cli_required_query_argument() {
        let app = cli();
        let result = app.try_get_matches_from(vec!["prog"]);
        // Should fail because query is required
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_with_query() {
        let app = cli();
        let result = app.try_get_matches_from(vec!["prog", "test query"]);
        assert!(result.is_ok());

        let matches = result.unwrap();
        assert_eq!(matches.get_one::<String>("query").unwrap(), "test query");
    }

    #[test]
    fn test_cli_default_values() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "test query"])
            .unwrap();

        assert_eq!(
            matches.get_one::<String>("searx-host").unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            matches.get_one::<String>("ollama-host").unwrap(),
            "http://localhost:11434"
        );
        assert_eq!(matches.get_one::<String>("model").unwrap(), "qwen3:30b");
        assert_eq!(
            matches.get_one::<String>("embedding-model").unwrap(),
            "nomic-embed-text:latest"
        );
        assert_eq!(*matches.get_one::<usize>("max-docs").unwrap(), 15);
        assert_eq!(
            matches.get_one::<String>("output").unwrap(),
            "enhanced_article.md"
        );
        assert_eq!(*matches.get_one::<i64>("cache-days").unwrap(), 7);
        assert_eq!(
            *matches.get_one::<f32>("similarity-threshold").unwrap(),
            0.7
        );
        assert_eq!(*matches.get_one::<f32>("quality-threshold").unwrap(), 0.3);
        assert_eq!(
            matches.get_one::<String>("expertise-level").unwrap(),
            "intermediate"
        );
        assert_eq!(
            *matches.get_one::<usize>("concurrent-downloads").unwrap(),
            8
        );
    }

    #[test]
    fn test_cli_custom_values() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec![
                "prog",
                "test query",
                "--searx-host",
                "http://custom:9090",
                "--ollama-host",
                "http://custom:11434",
                "--model",
                "custom-model",
                "--embedding-model",
                "custom-embed",
                "--max-docs",
                "25",
                "--output",
                "custom.md",
                "--cache-days",
                "14",
                "--similarity-threshold",
                "0.8",
                "--quality-threshold",
                "0.5",
                "--concurrent-downloads",
                "16",
            ])
            .unwrap();

        assert_eq!(
            matches.get_one::<String>("searx-host").unwrap(),
            "http://custom:9090"
        );
        assert_eq!(
            matches.get_one::<String>("ollama-host").unwrap(),
            "http://custom:11434"
        );
        assert_eq!(
            matches.get_one::<String>("model").unwrap(),
            "custom-model"
        );
        assert_eq!(
            matches.get_one::<String>("embedding-model").unwrap(),
            "custom-embed"
        );
        assert_eq!(*matches.get_one::<usize>("max-docs").unwrap(), 25);
        assert_eq!(matches.get_one::<String>("output").unwrap(), "custom.md");
        assert_eq!(*matches.get_one::<i64>("cache-days").unwrap(), 14);
        assert_eq!(
            *matches.get_one::<f32>("similarity-threshold").unwrap(),
            0.8
        );
        assert_eq!(*matches.get_one::<f32>("quality-threshold").unwrap(), 0.5);
        assert_eq!(
            *matches.get_one::<usize>("concurrent-downloads").unwrap(),
            16
        );
    }

    #[test]
    fn test_cli_short_flags() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "test query", "-m", "short-model", "-o", "out.md"])
            .unwrap();

        assert_eq!(
            matches.get_one::<String>("model").unwrap(),
            "short-model"
        );
        assert_eq!(matches.get_one::<String>("output").unwrap(), "out.md");
    }

    #[test]
    fn test_cli_boolean_flags() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec![
                "prog",
                "test query",
                "--enable-semantic",
                "--enable-personalization",
                "--show-cache-stats",
                "--show-quality-stats",
                "--cleanup-cache",
                "--validate-env",
                "--auto-install",
            ])
            .unwrap();

        assert!(matches.get_flag("enable-semantic"));
        assert!(matches.get_flag("enable-personalization"));
        assert!(matches.get_flag("show-cache-stats"));
        assert!(matches.get_flag("show-quality-stats"));
        assert!(matches.get_flag("cleanup-cache"));
        assert!(matches.get_flag("validate-env"));
        assert!(matches.get_flag("auto-install"));
    }

    #[test]
    fn test_cli_boolean_flags_default_false() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "test query"])
            .unwrap();

        assert!(!matches.get_flag("enable-semantic"));
        assert!(!matches.get_flag("enable-personalization"));
        assert!(!matches.get_flag("show-cache-stats"));
        assert!(!matches.get_flag("show-quality-stats"));
        assert!(!matches.get_flag("cleanup-cache"));
        assert!(!matches.get_flag("validate-env"));
        assert!(!matches.get_flag("auto-install"));
    }

    #[test]
    fn test_cli_expertise_level_values() {
        let app = cli();

        // Test valid expertise levels
        let levels = vec!["beginner", "intermediate", "advanced", "expert"];
        for level in levels {
            let matches = app
                .clone()
                .try_get_matches_from(vec!["prog", "test query", "--expertise-level", level])
                .unwrap();
            assert_eq!(matches.get_one::<String>("expertise-level").unwrap(), level);
        }
    }

    #[test]
    fn test_cli_expertise_level_invalid() {
        let app = cli();
        let result = app.try_get_matches_from(vec!["prog", "test query", "--expertise-level", "invalid"]);
        // Should fail with invalid expertise level
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_database_path() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "test query", "--database", "/path/to/db"])
            .unwrap();

        let db_path = matches.get_one::<PathBuf>("database").unwrap();
        assert_eq!(db_path.to_str().unwrap(), "/path/to/db");
    }

    #[test]
    fn test_cli_database_path_short() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "test query", "-d", "/path/to/db"])
            .unwrap();

        let db_path = matches.get_one::<PathBuf>("database").unwrap();
        assert_eq!(db_path.to_str().unwrap(), "/path/to/db");
    }

    #[test]
    fn test_cli_numeric_validation() {
        let app = cli();

        // Valid numeric values
        let matches = app
            .clone()
            .try_get_matches_from(vec!["prog", "test query", "--max-docs", "20"])
            .unwrap();
        assert_eq!(*matches.get_one::<usize>("max-docs").unwrap(), 20);

        // Invalid numeric value should fail
        let result = app
            .clone()
            .try_get_matches_from(vec!["prog", "test query", "--max-docs", "invalid"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_float_validation() {
        let app = cli();

        // Valid float values
        let matches = app
            .clone()
            .try_get_matches_from(vec![
                "prog",
                "test query",
                "--similarity-threshold",
                "0.85",
                "--quality-threshold",
                "0.45",
            ])
            .unwrap();

        assert_eq!(
            *matches.get_one::<f32>("similarity-threshold").unwrap(),
            0.85
        );
        assert_eq!(*matches.get_one::<f32>("quality-threshold").unwrap(), 0.45);
    }

    #[test]
    fn test_cli_help_contains_description() {
        let app = cli();
        let help = format!("{}", app.render_help());

        assert!(help.contains("Enhanced RAG Article Generator"));
        assert!(help.contains("AI-powered article generation"));
    }

    #[test]
    fn test_cli_all_arguments_present() {
        let app = cli();

        // Verify all expected arguments are present
        let arg_names = vec![
            "query",
            "searx-host",
            "ollama-host",
            "model",
            "embedding-model",
            "max-docs",
            "output",
            "database",
            "cache-days",
            "similarity-threshold",
            "quality-threshold",
            "enable-semantic",
            "enable-personalization",
            "show-cache-stats",
            "show-quality-stats",
            "cleanup-cache",
            "expertise-level",
            "validate-env",
            "auto-install",
            "concurrent-downloads",
        ];

        for arg_name in arg_names {
            let arg = app.get_arguments().find(|a| a.get_id() == arg_name);
            assert!(arg.is_some(), "Argument '{}' should be present", arg_name);
        }
    }

    #[test]
    fn test_cli_query_is_positional() {
        let app = cli();
        let matches = app
            .try_get_matches_from(vec!["prog", "my test query"])
            .unwrap();

        assert_eq!(
            matches.get_one::<String>("query").unwrap(),
            "my test query"
        );
    }

    #[test]
    fn test_cli_concurrent_downloads_range() {
        let app = cli();

        // Test various valid values
        for val in [1, 4, 8, 16, 32] {
            let matches = app
                .clone()
                .try_get_matches_from(vec![
                    "prog",
                    "test query",
                    "--concurrent-downloads",
                    &val.to_string(),
                ])
                .unwrap();
            assert_eq!(
                *matches.get_one::<usize>("concurrent-downloads").unwrap(),
                val
            );
        }
    }
}