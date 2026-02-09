use anyhow::Result;
use futures::{stream, StreamExt};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::{Document, EnhancedRAGArticleGenerator};

/// Constant for limiting concurrent downloads
const MAX_CONCURRENT_DOWNLOADS: usize = 8;

/// Document download statistics
#[derive(Debug)]
pub struct DownloadStats {
    pub total_urls: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_bytes: usize,
    pub elapsed_time: Duration,
    pub throughput: f64, // documents per second
}

impl EnhancedRAGArticleGenerator {
    /// Parallel document download with concurrency limit.
    /// Uses futures::stream::buffer_unordered for efficient management.
    pub async fn load_and_process_documents_parallel(
        &self,
        urls: Vec<String>,
    ) -> Result<Vec<Document>> {
        let (documents, stats) = self
            .load_documents_with_stats(urls, MAX_CONCURRENT_DOWNLOADS)
            .await?;

        info!("📊 Download statistics:");
        info!("  ✅ Successful: {} of {}", stats.successful, stats.total_urls);
        info!("  ❌ Failed: {}", stats.failed);
        info!("  ⏱️ Time: {:.2}s", stats.elapsed_time.as_secs_f32());
        info!("  🚀 Speed: {:.1} docs/sec", stats.throughput);
        info!(
            "  💾 Data: {:.2} MB",
            stats.total_bytes as f64 / 1_048_576.0
        );

        Ok(documents)
    }

    /// Parallel download with configurable concurrency limit
    pub async fn load_documents_with_concurrency_limit(
        &self,
        urls: Vec<String>,
        concurrent_limit: usize,
    ) -> Result<Vec<Document>> {
        let (documents, _) = self
            .load_documents_with_stats(urls, concurrent_limit)
            .await?;
        Ok(documents)
    }

    /// Load documents with detailed statistics
    pub async fn load_documents_with_stats(
        &self,
        urls: Vec<String>,
        concurrent_limit: usize,
    ) -> Result<(Vec<Document>, DownloadStats)> {
        if urls.is_empty() {
            return Ok((Vec::new(), DownloadStats::default()));
        }

        // Validate concurrent_limit
        if concurrent_limit == 0 {
            return Err(anyhow::anyhow!("concurrent_limit must be >= 1"));
        }

        info!(
            "🚀 Parallel download of {} documents (concurrency: {})",
            urls.len(),
            concurrent_limit
        );

        let start_time = Instant::now();

        // Create HTTP client with optimized settings
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .user_agent("Enhanced-RAG-Generator/2.0")
            .build()?;

        // Stats counters
        let mut successful = 0;
        let mut failed = 0;
        let mut total_bytes = 0;

        // Create futures stream for parallel download
        let download_stream = stream::iter(urls.iter().enumerate())
            .map(|(index, url)| {
                let client = client.clone();
                let url = url.clone();

                async move {
                    info!(
                        "📥 Downloading document {} from {}",
                        index + 1,
                        Self::truncate_url(&url, 50)
                    );

                    match self.download_and_process_document(&client, &url).await {
                        Ok(doc) => {
                            info!(
                                "✅ Document {} downloaded successfully ({} chars)",
                                index + 1,
                                doc.page_content.len()
                            );
                            Ok((doc.clone(), doc.page_content.len()))
                        }
                        Err(e) => {
                            warn!(
                                "⚠️ Error downloading document {} ({}): {}",
                                index + 1,
                                Self::truncate_url(&url, 30),
                                e
                            );
                            Err(e)
                        }
                    }
                }
            })
            // ⭐ KEY LINE: buffer_unordered limits concurrency
            .buffer_unordered(concurrent_limit);

        // Collect results separating successful and failed
        let mut documents = Vec::new();
        let mut results = download_stream.collect::<Vec<_>>().await;

        for result in results.drain(..) {
            match result {
                Ok((doc, bytes)) => {
                    successful += 1;
                    total_bytes += bytes;
                    documents.push(doc);
                }
                Err(_) => {
                    failed += 1;
                    // Error already logged above
                }
            }
        }

        let elapsed_time = start_time.elapsed();
        let elapsed_secs = elapsed_time.as_secs_f64();
        let throughput = if elapsed_secs > 0.0 {
            successful as f64 / elapsed_secs
        } else {
            0.0
        };

        let stats = DownloadStats {
            total_urls: urls.len(),
            successful,
            failed,
            total_bytes,
            elapsed_time,
            throughput,
        };

        info!(
            "🎉 Parallel download completed in {:.2}s",
            elapsed_time.as_secs_f32()
        );

        Ok((documents, stats))
    }

    /// Downloads and processes a single document
    async fn download_and_process_document(
        &self,
        client: &reqwest::Client,
        url: &str,
    ) -> Result<Document> {
        // HTTP request with error handling
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("HTTP request error: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HTTP error {}: {}",
                response.status(),
                url
            ));
        }

        let content = response
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("Content reading error: {}", e))?;

        // Content validation
        if content.len() < 100 {
            return Err(anyhow::anyhow!(
                "Content too short: {} chars (minimum 100)",
                content.len()
            ));
        }

        if content.len() > 1_000_000 {
            // 1MB limit
            return Err(anyhow::anyhow!(
                "Content too large: {} chars (maximum 1M)",
                content.len()
            ));
        }

        // Create document with extended metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source_url".to_string(), url.to_string());
        metadata.insert("content_length".to_string(), content.len().to_string());
        metadata.insert("download_time".to_string(), chrono::Utc::now().to_rfc3339());

        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        metadata.insert(
            "content_hash".to_string(),
            format!("{:x}", hasher.finalize()),
        );

        // Extract domain for metadata
        if let Ok(parsed_url) = url::Url::parse(url) {
            if let Some(host) = parsed_url.host_str() {
                metadata.insert("domain".to_string(), host.to_string());
            }
        }

        Ok(Document {
            page_content: content,
            metadata,
        })
    }

    /// Download with retry mechanism
    pub async fn download_with_retry(
        &self,
        client: &reqwest::Client,
        url: &str,
        max_retries: u32,
    ) -> Result<Document> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= max_retries {
            match self.download_and_process_document(client, url).await {
                Ok(doc) => return Ok(doc),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;

                    if attempts <= max_retries {
                        let delay = Duration::from_secs(2u64.pow(attempts.min(5))); // Cap at 32s
                        warn!(
                            "⚠️ Attempt {} failed for {}, retrying in {:?}",
                            attempts,
                            Self::truncate_url(url, 40),
                            delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown retry error")))
    }

    /// Utility to truncate long URLs in logs
    fn truncate_url(url: &str, max_len: usize) -> String {
        if max_len == 0 {
            return String::new();
        }
        if url.chars().count() <= max_len {
            return url.to_string();
        }
        if max_len <= 3 {
            return url.chars().take(max_len).collect();
        }
        let truncated: String = url.chars().take(max_len - 3).collect();
        format!("{truncated}...")
    }
}

impl Default for DownloadStats {
    fn default() -> Self {
        Self {
            total_urls: 0,
            successful: 0,
            failed: 0,
            total_bytes: 0,
            elapsed_time: Duration::from_secs(0),
            throughput: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_url() {
        let long_url = "https://example.com/very/long/path/that/exceeds/limit";
        let truncated = EnhancedRAGArticleGenerator::truncate_url(long_url, 20);
        assert!(truncated.len() <= 20);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_url_short_string() {
        let short_url = "https://test.com";
        let truncated = EnhancedRAGArticleGenerator::truncate_url(short_url, 50);
        assert_eq!(truncated, short_url);
        assert!(!truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_url_exact_length() {
        let url = "https://example.com";
        let truncated = EnhancedRAGArticleGenerator::truncate_url(url, url.len());
        assert_eq!(truncated, url);
    }

    #[test]
    fn test_truncate_url_edge_cases() {
        // Test with very small max_len
        let url = "https://example.com/path";
        let truncated = EnhancedRAGArticleGenerator::truncate_url(url, 5);
        assert!(truncated.len() <= 5);
        assert!(truncated.ends_with("..."));

        // Test with zero
        let truncated_zero = EnhancedRAGArticleGenerator::truncate_url(url, 0);
        assert_eq!(truncated_zero, "...");
    }

    #[tokio::test]
    async fn test_parallel_download_empty_urls() {
        // Test with empty URL list
        let generator = EnhancedRAGArticleGenerator::new(
            "http://test".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let result = generator.load_documents_with_stats(vec![], 5).await;
        assert!(result.is_ok());

        let (docs, stats) = result.unwrap();
        assert_eq!(docs.len(), 0);
        assert_eq!(stats.total_urls, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_parallel_download_with_concurrency_limit() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://test".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        // Test with different concurrency limits
        let result = generator.load_documents_with_stats(vec![], 1).await;
        assert!(result.is_ok());

        let result = generator.load_documents_with_stats(vec![], 16).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_download_stats_default() {
        let stats = DownloadStats::default();
        assert_eq!(stats.total_urls, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.throughput, 0.0);
        assert_eq!(stats.elapsed_time.as_secs(), 0);
    }

    #[test]
    fn test_download_stats_debug() {
        let stats = DownloadStats {
            total_urls: 10,
            successful: 8,
            failed: 2,
            total_bytes: 50000,
            elapsed_time: Duration::from_secs(5),
            throughput: 1.6,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("8"));
        assert!(debug_str.contains("2"));
    }

    #[test]
    fn test_max_concurrent_downloads_constant() {
        assert_eq!(MAX_CONCURRENT_DOWNLOADS, 8);
    }

    // Additional edge case tests

    #[tokio::test]
    async fn test_load_documents_with_concurrency_limit_wrapper() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://test".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        // This should use load_documents_with_stats internally
        let result = generator
            .load_documents_with_concurrency_limit(vec![], 4)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_parallel_download_uses_max_concurrent() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://test".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        // This should internally use MAX_CONCURRENT_DOWNLOADS
        let result = generator.load_and_process_documents_parallel(vec![]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}