// Integration tests for the enhanced RAG article generator
use enhanced_rag_article_generator::cli::cli;
use enhanced_rag_article_generator::persistent::{CacheSettings, PersistentEnhancedRAG};
use enhanced_rag_article_generator::{
    cosine_similarity, Document, EnhancedRAGArticleGenerator, OllamaEmbeddings,
    OllamaLLM, RecursiveUrlLoader, SearchResultItem, SearxSearchWrapper, SimpleVectorStore,
    SourceMetadata,
};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_cli_integration() {
    let app = cli();

    // Test that the CLI can be constructed and has expected properties
    assert_eq!(app.get_name(), "enhanced-rag-generator");
    assert_eq!(app.get_version().unwrap(), "2.0.0");

    // Test valid command line
    let result = app.try_get_matches_from(vec![
        "prog",
        "test query",
        "--model",
        "test-model",
        "--output",
        "test.md",
    ]);
    assert!(result.is_ok());
}

#[test]
fn test_document_workflow() {
    // Create a document
    let mut metadata = HashMap::new();
    metadata.insert("source_url".to_string(), "https://example.com".to_string());
    metadata.insert("source_title".to_string(), "Example".to_string());

    let doc = Document {
        page_content: "This is test content about Rust programming".to_string(),
        metadata,
    };

    // Verify document properties
    assert!(doc.page_content.contains("Rust"));
    assert_eq!(
        doc.metadata.get("source_url").unwrap(),
        "https://example.com"
    );
}

#[test]
fn test_source_metadata_workflow() {
    let metadata = SourceMetadata {
        url: "https://example.com/article".to_string(),
        title: "Example Article".to_string(),
        snippet: "A snippet of the article".to_string(),
        domain: "example.com".to_string(),
        content_summary: "Summary of the article content".to_string(),
        topics_covered: vec!["rust".to_string(), "programming".to_string()],
    };

    assert_eq!(metadata.domain, "example.com");
    assert_eq!(metadata.topics_covered.len(), 2);
}

#[test]
fn test_enhanced_rag_generator_initialization() {
    let generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        Some("http://localhost:11434".to_string()),
    );

    assert_eq!(generator.source_counter, 1);
    assert!(generator.sources_metadata().is_empty());
}

#[test]
fn test_text_ranking_integration() {
    let generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        None,
    );

    let mut metadata1 = HashMap::new();
    metadata1.insert("source_title".to_string(), "Rust Guide".to_string());

    let mut metadata2 = HashMap::new();
    metadata2.insert("source_title".to_string(), "Python Guide".to_string());

    let docs = vec![
        Document {
            page_content: "A comprehensive guide to Rust programming language with examples"
                .to_string(),
            metadata: metadata1,
        },
        Document {
            page_content: "Python programming basics for beginners".to_string(),
            metadata: metadata2,
        },
    ];

    let ranked = generator.simple_text_ranking(&docs, "Rust programming", 1);

    assert_eq!(ranked.len(), 1);
    assert!(ranked[0].page_content.contains("Rust"));
}

#[test]
fn test_cosine_similarity_integration() {
    // Test with various vector combinations
    let identical = vec![1.0, 2.0, 3.0];
    let similarity = cosine_similarity(&identical, &identical);
    assert!((similarity - 1.0).abs() < 0.001);

    let orthogonal_a = vec![1.0, 0.0, 0.0];
    let orthogonal_b = vec![0.0, 1.0, 0.0];
    let similarity = cosine_similarity(&orthogonal_a, &orthogonal_b);
    assert!(similarity.abs() < 0.001);
}

#[tokio::test]
async fn test_persistent_rag_in_memory_integration() {
    let rag = PersistentEnhancedRAG::new_in_memory(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        Some("http://localhost:11434".to_string()),
    );

    assert!(rag.is_ok());

    let rag = rag.unwrap();
    let stats = rag.cache_stats().await;
    assert!(stats.is_ok());

    let stats = stats.unwrap();
    assert_eq!(stats.total_documents, 0);
    assert_eq!(stats.total_queries, 0);
}

#[tokio::test]
async fn test_persistent_rag_with_storage_integration() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("integration_test.db");

    let result = PersistentEnhancedRAG::new_with_persistent_storage(
        &db_path,
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        Some("http://localhost:11434".to_string()),
        None,
    );

    assert!(result.is_ok());

    let rag = result.unwrap();

    // Test cache operations
    let stats = rag.cache_stats().await.unwrap();
    assert_eq!(stats.total_documents, 0);

    // Test quality stats
    let quality_stats = rag.get_quality_stats().await.unwrap();
    assert_eq!(quality_stats.total_sources, 0);
}

#[tokio::test]
async fn test_persistent_rag_cleanup_integration() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("cleanup_test.db");

    let mut rag = PersistentEnhancedRAG::new_with_persistent_storage(
        &db_path,
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        None,
        None,
    )
    .unwrap();

    let cleanup_stats = rag.cleanup_cache().await.unwrap();
    assert_eq!(cleanup_stats.deleted_documents, 0);
    assert_eq!(cleanup_stats.deleted_queries, 0);
}

#[test]
fn test_cache_settings_integration() {
    let settings = CacheSettings::default();

    assert_eq!(settings.max_document_age_days, 7);
    assert_eq!(settings.min_query_similarity, 0.7);
    assert_eq!(settings.max_cached_docs, 10);
    assert_eq!(settings.max_concurrent_downloads, 8);
    assert!(settings.enable_semantic_search);

    // Test custom settings
    let custom_settings = CacheSettings {
        max_document_age_days: 14,
        min_query_similarity: 0.8,
        max_cached_docs: 20,
        embedding_dim: Some(768),
        enable_semantic_search: true,
        min_quality_score: 0.5,
        enable_personalization: true,
        auto_reindex_interval_hours: 48,
        max_vector_cache_size: 20000,
        max_concurrent_downloads: 16,
    };

    assert_eq!(custom_settings.max_document_age_days, 14);
    assert_eq!(custom_settings.max_concurrent_downloads, 16);
}

#[test]
fn test_search_result_item_integration() {
    let json = r#"{
        "url": "https://example.com",
        "title": "Example",
        "content": "content text"
    }"#;

    let item: SearchResultItem = serde_json::from_str(json).unwrap();
    assert_eq!(item.url, "https://example.com");
    assert_eq!(item.title, "Example");
    assert!(item.content.is_some());
}

#[test]
fn test_component_initialization() {
    // Test that all major components can be initialized
    let _searx = SearxSearchWrapper::new("http://localhost:8080".to_string());
    let _loader = RecursiveUrlLoader::new(2, 30);
    let _llm = OllamaLLM::new("test-model".to_string(), None);
    let _embeddings = OllamaEmbeddings::new("test-embed".to_string(), None);

    // If we get here without panic, initialization works
    assert!(true);
}

#[tokio::test]
async fn test_vector_store_initialization() {
    let embedding_model = Box::new(OllamaEmbeddings::new("test-model".to_string(), None));
    let store = SimpleVectorStore::new(embedding_model);

    assert_eq!(store.documents().len(), 0);
    assert_eq!(store.embeddings().nrows(), 0);
}

#[test]
fn test_context_preparation_integration() {
    let generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        None,
    );

    let mut metadata = HashMap::new();
    metadata.insert("source_number".to_string(), "1".to_string());
    metadata.insert("source_title".to_string(), "Test Source".to_string());
    metadata.insert("source_domain".to_string(), "test.com".to_string());

    let docs = vec![
        Document {
            page_content: "First document content".to_string(),
            metadata: metadata.clone(),
        },
        Document {
            page_content: "Second document content".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("source_number".to_string(), "2".to_string());
                m.insert("source_title".to_string(), "Another Source".to_string());
                m.insert("source_domain".to_string(), "other.com".to_string());
                m
            },
        },
    ];

    let context = generator.prepare_context_with_sources(&docs);

    assert!(context.contains("SOURCE 1"));
    assert!(context.contains("SOURCE 2"));
    assert!(context.contains("Test Source"));
    assert!(context.contains("Another Source"));
    assert!(context.contains("First document content"));
    assert!(context.contains("Second document content"));
}

#[test]
fn test_enhanced_sources_list_integration() {
    let mut generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        None,
    );

    // Add multiple sources
    for i in 1..=3 {
        generator.sources_metadata.insert(
            i,
            SourceMetadata {
                url: format!("https://example{}.com", i),
                title: format!("Source {}", i),
                snippet: "snippet".to_string(),
                domain: format!("example{}.com", i),
                content_summary: format!("Summary {}", i),
                topics_covered: vec![format!("topic{}", i)],
            },
        );
    }

    let article = "# Test Article\n\nContent with citations [1] and [2] and [3].";
    let with_sources = generator.add_enhanced_sources_list(article);

    assert!(with_sources.contains("# Test Article"));
    assert!(with_sources.contains("## Sources"));
    assert!(with_sources.contains("Source 1"));
    assert!(with_sources.contains("Source 2"));
    assert!(with_sources.contains("Source 3"));
    assert!(with_sources.contains("example1.com"));
    assert!(with_sources.contains("example2.com"));
    assert!(with_sources.contains("example3.com"));
}

#[tokio::test]
async fn test_full_workflow_simulation() {
    // Simulate a simplified end-to-end workflow

    // 1. Initialize generator
    let mut generator = EnhancedRAGArticleGenerator::new(
        "http://localhost:8080".to_string(),
        "test-model".to_string(),
        "test-embed".to_string(),
        None,
    );

    // 2. Create mock documents
    let mut metadata1 = HashMap::new();
    metadata1.insert("source_url".to_string(), "https://example.com/1".to_string());
    metadata1.insert("source_title".to_string(), "Rust Basics".to_string());
    metadata1.insert("source_domain".to_string(), "example.com".to_string());
    metadata1.insert("source_number".to_string(), "1".to_string());

    let doc1 = Document {
        page_content: "Rust is a systems programming language focused on safety".to_string(),
        metadata: metadata1,
    };

    // 3. Test ranking
    let ranked = generator.simple_text_ranking(&[doc1.clone()], "Rust programming", 1);
    assert_eq!(ranked.len(), 1);

    // 4. Test context preparation
    let context = generator.prepare_context_with_sources(&ranked);
    assert!(context.contains("Rust"));

    // 5. Add source metadata
    generator.sources_metadata.insert(
        1,
        SourceMetadata {
            url: "https://example.com/1".to_string(),
            title: "Rust Basics".to_string(),
            snippet: "snippet".to_string(),
            domain: "example.com".to_string(),
            content_summary: "Rust programming summary".to_string(),
            topics_covered: vec!["rust".to_string(), "safety".to_string()],
        },
    );

    // 6. Generate sources list
    let article = "# Rust Guide\n\nRust is great [1].";
    let final_article = generator.add_enhanced_sources_list(article);

    assert!(final_article.contains("# Rust Guide"));
    assert!(final_article.contains("## Sources"));
    assert!(final_article.contains("Rust Basics"));
}