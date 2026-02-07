use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};
use url::Url;

pub mod cli;
pub mod ollama_utils;
pub mod parallel_downloader;
pub mod persistent;
pub use ollama_utils::*;

/// Structure for source metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub url: String,
    pub title: String,
    pub snippet: String,
    pub domain: String,
    pub content_summary: String,
    pub topics_covered: Vec<String>,
}

/// Structure for SearXNG search result
#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub results: Vec<SearchResultItem>,
}

#[derive(Debug, Deserialize)]
pub struct SearchResultItem {
    pub url: String,
    pub title: String,
    pub content: Option<String>,
}

/// Structure for document with content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, String>,
}

/// Trait for working with search engines
#[async_trait]
pub trait SearchWrapper {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>>;
}

/// SearXNG search implementation
pub struct SearxSearchWrapper {
    client: Client,
    host: String,
}

impl SearxSearchWrapper {
    pub fn new(host: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Enhanced RAG Article Generator/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self { client, host }
    }
}

#[async_trait]
impl SearchWrapper for SearxSearchWrapper {
    async fn search(&self, query: &str, num_results: u32) -> Result<Vec<SearchResultItem>> {
        let search_url = format!(
            "{}/search?q={}&format=json&safesearch=0&pageno=1&time_range=year",
            self.host,
            urlencoding::encode(query)
        );

        info!("Executing search: {}", query);

        let response = self.client.get(&search_url).send().await?;

        let search_result: SearchResult = response.json().await?;

        let mut results = search_result.results;
        results.truncate(num_results as usize);

        info!("Found {} results", results.len());
        Ok(results)
    }
}

/// Trait for loading documents
#[async_trait]
pub trait DocumentLoader {
    async fn load(&self, url: &str) -> Result<Vec<Document>>;
}

/// Recursive document loader implementation
pub struct RecursiveUrlLoader {
    client: Client,
}

impl RecursiveUrlLoader {
    pub fn new(_max_depth: u32, timeout_secs: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("Mozilla/5.0 (compatible; Enhanced-RAG-Bot/1.0)")
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Convert HTML to Markdown
    fn convert_html_to_markdown(&self, html_content: &str) -> String {
        // Use html2text for conversion
        let markdown = html2text::from_read(html_content.as_bytes(), 80);

        // Clean up excessive newlines
        let re_newlines = Regex::new(r"\n{3,}").unwrap();
        let cleaned = re_newlines.replace_all(&markdown, "\n\n");

        // Remove navigation elements
        let re_nav =
            Regex::new(r"(?m)^\s*\*\s*(Home|Menu|Navigation|Skip to|Back to top).*$").unwrap();
        let cleaned = re_nav.replace_all(&cleaned, "");

        // Filter short lines
        let lines: Vec<&str> = cleaned.lines().collect();
        let filtered_lines: Vec<&str> = lines
            .into_iter()
            .filter(|line| {
                let stripped = line.trim();
                stripped.len() > 10 || stripped.starts_with('#')
            })
            .collect();

        filtered_lines.join("\n")
    }
}

#[async_trait]
impl DocumentLoader for RecursiveUrlLoader {
    async fn load(&self, url: &str) -> Result<Vec<Document>> {
        info!("Loading document: {}", url);

        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
        }

        let html_content = response.text().await?;

        // Convert HTML to Markdown
        let markdown_content = self.convert_html_to_markdown(&html_content);

        if markdown_content.trim().len() < 100 {
            warn!("Document too short: {}", url);
            return Ok(vec![]);
        }

        let mut metadata = HashMap::new();
        metadata.insert("source_url".to_string(), url.to_string());

        // Extract page title
        let document = Html::parse_document(&html_content);
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|elem| elem.text().collect::<String>())
            .unwrap_or_else(|| "Untitled".to_string());

        metadata.insert("source_title".to_string(), title);

        // Extract domain
        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(domain) = parsed_url.domain() {
                metadata.insert("source_domain".to_string(), domain.to_string());
            }
        }

        let document = Document {
            page_content: markdown_content,
            metadata,
        };

        Ok(vec![document])
    }
}

/// Trait for vector embeddings
#[async_trait]
pub trait EmbeddingModel {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// Simple embedding implementation via Ollama API
pub struct OllamaEmbeddings {
    client: Client,
    model_name: String,
    ollama_host: String,
}

impl OllamaEmbeddings {
    pub fn new(model_name: String, ollama_host: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            model_name,
            ollama_host: ollama_host.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

#[async_trait]
impl EmbeddingModel for OllamaEmbeddings {
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let embedding = self.embed_query(text).await?;
            embeddings.push(embedding);

            // Small pause between requests
            sleep(Duration::from_millis(100)).await;
        }

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.ollama_host);

        let request_body = serde_json::json!({
            "model": self.model_name,
            "prompt": text
        });

        let response = self.client.post(&url).json(&request_body).send().await?;

        #[derive(Deserialize)]
        struct EmbeddingResponse {
            embedding: Vec<f32>,
        }

        let embedding_response: EmbeddingResponse = response.json().await?;

        Ok(embedding_response.embedding)
    }
}

/// Simple in-memory vector store with basic mathematics
pub struct SimpleVectorStore {
    documents: Vec<Document>,
    embeddings: Array2<f32>,
    embedding_model: Box<dyn EmbeddingModel + Send + Sync>,
    embedding_dim: usize,
}

impl SimpleVectorStore {
    pub fn new(embedding_model: Box<dyn EmbeddingModel + Send + Sync>) -> Self {
        Self {
            documents: Vec::new(),
            embeddings: Array2::zeros((0, 0)),
            embedding_model,
            embedding_dim: 0,
        }
    }

    pub async fn add_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        info!("Adding {} documents to vector store", documents.len());

        let texts: Vec<String> = documents
            .iter()
            .map(|doc| doc.page_content.clone())
            .collect();

        let new_embeddings = self.embedding_model.embed_documents(&texts).await?;

        if new_embeddings.is_empty() {
            return Ok(());
        }

        // Determine embedding dimension
        if self.embedding_dim == 0 {
            self.embedding_dim = new_embeddings[0].len();
            self.embeddings = Array2::zeros((0, self.embedding_dim));
        }

        // Convert Vec<Vec<f32>> to Array2<f32>
        let new_embeddings_array = Array2::from_shape_vec(
            (new_embeddings.len(), self.embedding_dim),
            new_embeddings.into_iter().flatten().collect(),
        )?;

        // Combine old and new embeddings
        if self.embeddings.nrows() == 0 {
            self.embeddings = new_embeddings_array;
        } else {
            let old_embeddings = self.embeddings.clone();
            self.embeddings = Array2::zeros((
                old_embeddings.nrows() + new_embeddings_array.nrows(),
                self.embedding_dim,
            ));

            // Copy old embeddings
            self.embeddings
                .slice_mut(ndarray::s![..old_embeddings.nrows(), ..])
                .assign(&old_embeddings);

            // Copy new embeddings
            self.embeddings
                .slice_mut(ndarray::s![old_embeddings.nrows().., ..])
                .assign(&new_embeddings_array);
        }

        self.documents.extend(documents);

        info!("Documents successfully added to vector store");
        Ok(())
    }

    pub async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = self.embedding_model.embed_query(query).await?;
        let query_array = Array1::from(query_embedding);

        // Calculate cosine similarity for each document
        let mut similarities = Vec::new();

        for (idx, doc_embedding) in self.embeddings.outer_iter().enumerate() {
            let similarity = cosine_similarity_arrays(&query_array, &doc_embedding.to_owned());
            similarities.push((idx, similarity));
        }

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top-k documents
        let results = similarities
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.documents[idx].clone())
            .collect();

        Ok(results)
    }

    /// Getter for documents (for testing)
    pub fn documents(&self) -> &Vec<Document> {
        &self.documents
    }

    /// Getter for embeddings (for testing)
    pub fn embeddings(&self) -> &Array2<f32> {
        &self.embeddings
    }
}

/// Calculate cosine similarity for Array1
pub fn cosine_similarity_arrays(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Calculate cosine similarity for Vec (for backward compatibility)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let array_a = Array1::from(a.to_vec());
    let array_b = Array1::from(b.to_vec());
    cosine_similarity_arrays(&array_a, &array_b)
}

/// Trait for language models
#[async_trait]
pub trait LanguageModel {
    async fn generate(&self, prompt: &str) -> Result<String>;
}

/// Implementation for Ollama LLM
pub struct OllamaLLM {
    client: Client,
    model_name: String,
    ollama_host: String,
}

impl OllamaLLM {
    pub fn new(model_name: String, ollama_host: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(3600)) // 1 hour for generation
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            model_name,
            ollama_host: ollama_host.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

#[async_trait]
impl LanguageModel for OllamaLLM {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.ollama_host);

        let request_body = serde_json::json!({
            "model": self.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        });

        // dbg!(request_body.clone());
        info!("Sending request to Ollama for text generation");

        let response = self.client.post(&url).json(&request_body).send().await?;

        // #[derive(Deserialize)]
        // struct GenerationResponse {
        //     response: String,
        // }

        // let generation_response: GenerationResponse = response.json().await?;
        let content = parse_ollama_response(&response.text().await?)?;

        // Ok(generation_response.response)
        Ok(content)
    }
}

/// Main article generator
pub struct EnhancedRAGArticleGenerator {
    search_wrapper: Box<dyn SearchWrapper + Send + Sync>,
    document_loader: Box<dyn DocumentLoader + Send + Sync>,
    language_model: Box<dyn LanguageModel + Send + Sync>,
    sources_metadata: HashMap<u32, SourceMetadata>,
    source_counter: u32,
}

impl EnhancedRAGArticleGenerator {
    pub fn new(
        searx_host: String,
        model_name: String,
        _embedding_model_name: String, // Not used in simplified version
        ollama_host: Option<String>,
    ) -> Self {
        let search_wrapper = Box::new(SearxSearchWrapper::new(searx_host));
        let document_loader = Box::new(RecursiveUrlLoader::new(2, 20));
        let language_model = Box::new(OllamaLLM::new(model_name, ollama_host));

        Self {
            search_wrapper,
            document_loader,
            language_model,
            sources_metadata: HashMap::new(),
            source_counter: 1,
        }
    }

    pub async fn search_and_collect_urls(
        &mut self,
        query: &str,
        num_results: u32,
    ) -> Result<Vec<String>> {
        info!("Searching URLs for query: {}", query);

        let search_results = self.search_wrapper.search(query, num_results).await?;

        let mut urls = Vec::new();

        for result in search_results {
            if !urls.contains(&result.url) {
                urls.push(result.url.clone());

                // Extract domain
                let domain = Url::parse(&result.url)
                    .ok()
                    .and_then(|url| url.domain().map(|d| d.to_string()))
                    .unwrap_or_else(|| "unknown".to_string());

                // Save metadata
                let metadata = SourceMetadata {
                    url: result.url,
                    title: result.title,
                    snippet: result.content.unwrap_or_default(),
                    domain,
                    content_summary: String::new(), // Will be filled later
                    topics_covered: Vec::new(),     // Will be filled later
                };

                self.sources_metadata.insert(self.source_counter, metadata);
                self.source_counter += 1;
            }
        }

        info!("Found {} unique URLs", urls.len());
        Ok(urls)
    }

    pub async fn load_and_process_documents(&mut self, urls: Vec<String>) -> Result<Vec<Document>> {
        info!("Loading and processing {} documents", urls.len());

        let mut all_documents = Vec::new();

        for (i, url) in urls.iter().enumerate() {
            let source_number = (i + 1) as u32;

            match self.document_loader.load(url).await {
                Ok(mut documents) => {
                    if !documents.is_empty() {
                        // Add source metadata
                        for doc in &mut documents {
                            doc.metadata
                                .insert("source_number".to_string(), source_number.to_string());
                        }

                        // Update source metadata with content
                        let combined_content: String = documents
                            .iter()
                            .map(|doc| doc.page_content.clone())
                            .collect::<Vec<_>>()
                            .join("\n");

                        let (summary, topics) = self.extract_content_summary(&combined_content);

                        if let Some(metadata) = self.sources_metadata.get_mut(&source_number) {
                            metadata.content_summary = summary;
                            metadata.topics_covered = topics;
                        }

                        all_documents.extend(documents);
                        info!("Loaded document {}/{}: {}", i + 1, urls.len(), url);
                    }
                }
                Err(e) => {
                    warn!("Error loading {}: {}", url, e);
                }
            }
        }

        info!("Total loaded {} documents", all_documents.len());
        Ok(all_documents)
    }

    pub fn extract_content_summary(&self, content: &str) -> (String, Vec<String>) {
        // Extract headers
        let re_headers = Regex::new(r"(?m)^#+\s+(.+)$").unwrap();
        let topics: Vec<String> = re_headers
            .captures_iter(content)
            .map(|cap| cap[1].trim().to_string())
            .take(5)
            .collect();

        // Brief summary
        let summary = if content.len() > 300 {
            format!("{}...", &content[..300].replace('\n', " ").trim())
        } else {
            content.replace('\n', " ").trim().to_string()
        };

        (summary, topics)
    }

    pub async fn generate_article_simple(
        &mut self,
        query: &str,
        max_retrieved_docs: usize,
    ) -> Result<String> {
        info!("Starting article generation for query: {}", query);

        // 1. Search URLs
        let urls = self.search_and_collect_urls(query, 15).await?;

        if urls.is_empty() {
            return Ok("No sources found to create article.".to_string());
        }

        // 2. Load documents
        // let documents = self.load_and_process_documents(urls).await?;
        let documents = self.load_and_process_documents_parallel(urls).await?;

        if documents.is_empty() {
            return Ok("Failed to load documents from found sources.".to_string());
        }

        // 3. Simple relevance ranking without vector search
        let retrieved_docs = self.simple_text_ranking(&documents, query, max_retrieved_docs);

        // 4. Prepare context
        let context_with_sources = self.prepare_context_with_sources(&retrieved_docs);

        // 5. Create prompt
        let article_prompt = format!(
            "You are an expert academic writer creating a comprehensive research article based on provided context documents.\n\n\
            AVAILABLE SOURCE DOCUMENTS:\n{}\n\n\
            TASK: Write a detailed, well-structured article about: {}\n\n\
            CRITICAL CITATION REQUIREMENTS:\n\
            1. When referencing specific information, data, quotes, or concepts from a source, immediately follow with a citation [X] where X is the source number\n\
            2. Citations must be semantically meaningful - each citation should guide readers to sources containing MORE DETAILED information about that specific topic\n\
            3. Place citations at the end of sentences or paragraphs that contain information from that source\n\
            4. Use multiple citations [1][2] when information is supported by multiple sources\n\
            5. Ensure each citation logically connects the content to the source that elaborates on that topic\n\n\
            WRITING STYLE:\n\
            - Academic but accessible tone\n\
            - Clear section headings (use ##, ###)\n\
            - Logical progression of ideas\n\
            - Incorporate specific technical details, examples, and methodologies from sources\n\
            - Each paragraph should develop a distinct aspect of the topic\n\n\
            STRUCTURE REQUIREMENTS:\n\
            - Engaging introduction with context and importance\n\
            - Multiple main sections covering different aspects\n\
            - Technical implementation details where relevant\n\
            - Real-world applications and use cases\n\
            - Critical analysis and limitations\n\
            - Future directions and conclusions\n\
            - DO NOT add a references section (will be added automatically)\n\n\
            CONTENT DEPTH:\n\
            - Minimum 1500 words\n\
            - Include technical specifications, code examples, and implementation details from sources\n\
            - Discuss advantages, limitations, and comparison with alternatives\n\
            - Provide practical insights and recommendations\n\n\
            Begin writing the comprehensive article now:",
            context_with_sources, query
        );

        // 6. Generate article
        info!("Generating article with LLM...");
        let article_text = self.language_model.generate(&article_prompt).await?;

        // 7. Add sources list
        let article_with_sources = self.add_enhanced_sources_list(&article_text);

        info!("Article successfully generated");
        Ok(article_with_sources)
    }

    /// Simple text relevance ranking without vector calculations
    pub fn simple_text_ranking(
        &self,
        documents: &[Document],
        query: &str,
        max_docs: usize,
    ) -> Vec<Document> {
        // Fix issue with temporary values
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored_docs: Vec<(Document, f32)> = documents
            .iter()
            .map(|doc| {
                let content_lower = doc.page_content.to_lowercase();
                let title_lower = doc
                    .metadata
                    .get("source_title")
                    .unwrap_or(&String::new())
                    .to_lowercase();

                let mut score = 0.0;

                for word in &query_words {
                    // Count occurrences in content
                    let content_matches = content_lower.matches(word).count() as f32;
                    score += content_matches * 1.0;

                    // Bonus for occurrence in title
                    let title_matches = title_lower.matches(word).count() as f32;
                    score += title_matches * 3.0;
                }

                // Normalize by document length
                let normalized_score = score / (doc.page_content.len() as f32 + 1.0) * 1000.0;

                (doc.clone(), normalized_score)
            })
            .collect();

        // Sort by relevance descending
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored_docs
            .into_iter()
            .take(max_docs)
            .map(|(doc, _)| doc)
            .collect()
    }

    pub async fn generate_article(
        &mut self,
        query: &str,
        max_retrieved_docs: usize,
    ) -> Result<String> {
        // Use simplified version without vector calculations
        self.generate_article_simple(query, max_retrieved_docs)
            .await
    }

    pub fn prepare_context_with_sources(&self, retrieved_docs: &[Document]) -> String {
        let mut context_parts = Vec::new();

        for doc in retrieved_docs {
            let source_number = doc
                .metadata
                .get("source_number")
                .unwrap_or(&"Unknown".to_string())
                .clone();

            let source_title = doc
                .metadata
                .get("source_title")
                .unwrap_or(&"Untitled".to_string())
                .clone();

            let source_domain = doc
                .metadata
                .get("source_domain")
                .unwrap_or(&"".to_string())
                .clone();

            // Get topics from metadata
            let topics = if let Ok(source_num) = source_number.parse::<u32>() {
                self.sources_metadata
                    .get(&source_num)
                    .map(|metadata| metadata.topics_covered.join(", "))
                    .unwrap_or_else(|| "General information".to_string())
            } else {
                "General information".to_string()
            };

            let context_part = format!(
                "\n=== SOURCE {}: {} ===\nDomain: {}\nKey Topics: {}\nContent:\n{}\n",
                source_number, source_title, source_domain, topics, doc.page_content
            );

            context_parts.push(context_part);
        }

        context_parts.join("\n")
    }

    pub fn add_enhanced_sources_list(&self, article_text: &str) -> String {
        let mut sources_list = String::from("\n\n## Sources\n\n");

        for (source_num, metadata) in &self.sources_metadata {
            let topics_str = metadata
                .topics_covered
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");

            sources_list.push_str(&format!(
                "{}. **{}**\n   - URL: {}\n   - Domain: {}\n",
                source_num, metadata.title, metadata.url, metadata.domain
            ));

            if !topics_str.is_empty() {
                sources_list.push_str(&format!("   - Key Topics: {}\n", topics_str));
            }

            if !metadata.content_summary.is_empty() {
                sources_list.push_str(&format!(
                    "   - Brief Summary: {}\n",
                    metadata.content_summary
                ));
            }

            sources_list.push('\n');
        }

        format!("{}{}", article_text, sources_list)
    }

    /// Getter for sources metadata (for testing)
    pub fn sources_metadata(&self) -> &HashMap<u32, SourceMetadata> {
        &self.sources_metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test SourceMetadata
    #[test]
    fn test_source_metadata_creation() {
        let metadata = SourceMetadata {
            url: "https://example.com".to_string(),
            title: "Test Article".to_string(),
            snippet: "A test snippet".to_string(),
            domain: "example.com".to_string(),
            content_summary: "Summary of content".to_string(),
            topics_covered: vec!["topic1".to_string(), "topic2".to_string()],
        };

        assert_eq!(metadata.url, "https://example.com");
        assert_eq!(metadata.title, "Test Article");
        assert_eq!(metadata.topics_covered.len(), 2);
    }

    // Test Document structure
    #[test]
    fn test_document_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("source_url".to_string(), "https://test.com".to_string());

        let doc = Document {
            page_content: "Test content".to_string(),
            metadata,
        };

        assert_eq!(doc.page_content, "Test content");
        assert_eq!(
            doc.metadata.get("source_url").unwrap(),
            "https://test.com"
        );
    }

    // Test SearchResultItem deserialization
    #[test]
    fn test_search_result_item_deserialization() {
        let json = r#"{
            "url": "https://example.com",
            "title": "Example Title",
            "content": "Example content"
        }"#;

        let item: SearchResultItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.url, "https://example.com");
        assert_eq!(item.title, "Example Title");
        assert_eq!(item.content, Some("Example content".to_string()));
    }

    #[test]
    fn test_search_result_item_without_content() {
        let json = r#"{
            "url": "https://example.com",
            "title": "Example Title"
        }"#;

        let item: SearchResultItem = serde_json::from_str(json).unwrap();
        assert!(item.content.is_none());
    }

    // Test cosine similarity
    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let vec_a = vec![1.0, 0.0];
        let vec_b = vec![0.0, 1.0];
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert!((similarity - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let vec_a = vec![1.0, 2.0];
        let vec_b = vec![-1.0, -2.0];
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert!(similarity < 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let vec_a = vec![0.0, 0.0, 0.0];
        let vec_b = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_arrays() {
        let arr_a = Array1::from(vec![1.0, 2.0, 3.0]);
        let arr_b = Array1::from(vec![1.0, 2.0, 3.0]);
        let similarity = cosine_similarity_arrays(&arr_a, &arr_b);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_arrays_zero_norm() {
        let arr_a = Array1::from(vec![0.0, 0.0]);
        let arr_b = Array1::from(vec![1.0, 1.0]);
        let similarity = cosine_similarity_arrays(&arr_a, &arr_b);
        assert_eq!(similarity, 0.0);
    }

    // Test SearxSearchWrapper
    #[test]
    fn test_searx_search_wrapper_creation() {
        let wrapper = SearxSearchWrapper::new("http://localhost:8080".to_string());
        assert_eq!(wrapper.host, "http://localhost:8080");
    }

    // Test RecursiveUrlLoader
    #[test]
    fn test_recursive_url_loader_creation() {
        let loader = RecursiveUrlLoader::new(2, 30);
        // Just verify it constructs without panic
        assert!(std::mem::size_of_val(&loader) > 0);
    }

    #[test]
    fn test_html_to_markdown_conversion() {
        let loader = RecursiveUrlLoader::new(2, 30);
        let html = r#"<html><body><h1>Test Header</h1><p>This is a paragraph with some content.</p></body></html>"#;
        let markdown = loader.convert_html_to_markdown(html);

        assert!(markdown.contains("Test Header"));
        assert!(markdown.len() > 0);
    }

    #[test]
    fn test_html_to_markdown_cleans_navigation() {
        let loader = RecursiveUrlLoader::new(2, 30);
        let html = r#"<html><body><nav>Skip to content</nav><p>Real content here</p></body></html>"#;
        let markdown = loader.convert_html_to_markdown(html);

        // Navigation elements should be removed or minimized
        assert!(markdown.contains("Real content"));
    }

    // Test OllamaEmbeddings
    #[test]
    fn test_ollama_embeddings_creation() {
        let embeddings = OllamaEmbeddings::new(
            "test-model".to_string(),
            Some("http://localhost:11434".to_string()),
        );
        assert_eq!(embeddings.model_name, "test-model");
        assert_eq!(embeddings.ollama_host, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_embeddings_default_host() {
        let embeddings = OllamaEmbeddings::new("test-model".to_string(), None);
        assert_eq!(embeddings.ollama_host, "http://localhost:11434");
    }

    // Test OllamaLLM
    #[test]
    fn test_ollama_llm_creation() {
        let llm = OllamaLLM::new(
            "test-model".to_string(),
            Some("http://localhost:11434".to_string()),
        );
        assert_eq!(llm.model_name, "test-model");
        assert_eq!(llm.ollama_host, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_llm_default_host() {
        let llm = OllamaLLM::new("test-model".to_string(), None);
        assert_eq!(llm.ollama_host, "http://localhost:11434");
    }

    // Test SimpleVectorStore
    #[tokio::test]
    async fn test_simple_vector_store_creation() {
        let embedding_model =
            Box::new(OllamaEmbeddings::new("test-model".to_string(), None))
                as Box<dyn EmbeddingModel + Send + Sync>;
        let store = SimpleVectorStore::new(embedding_model);

        assert_eq!(store.documents().len(), 0);
        assert_eq!(store.embeddings().nrows(), 0);
        assert_eq!(store.embedding_dim, 0);
    }

    // Test EnhancedRAGArticleGenerator
    #[test]
    fn test_enhanced_rag_generator_creation() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            Some("http://localhost:11434".to_string()),
        );

        assert_eq!(generator.source_counter, 1);
        assert_eq!(generator.sources_metadata().len(), 0);
    }

    #[test]
    fn test_extract_content_summary() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let content = "# Topic 1\nSome content\n## Topic 2\nMore content";
        let (summary, topics) = generator.extract_content_summary(content);

        assert!(summary.len() > 0);
        assert!(topics.len() > 0);
        assert!(topics.contains(&"Topic 1".to_string()) || topics.contains(&"Topic 2".to_string()));
    }

    #[test]
    fn test_extract_content_summary_long_content() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let long_content = "a".repeat(500);
        let (summary, _topics) = generator.extract_content_summary(&long_content);

        assert!(summary.len() <= 303); // 300 + "..."
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn test_simple_text_ranking() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let mut metadata1 = HashMap::new();
        metadata1.insert("source_title".to_string(), "Rust Programming".to_string());

        let mut metadata2 = HashMap::new();
        metadata2.insert("source_title".to_string(), "Python Guide".to_string());

        let mut metadata3 = HashMap::new();
        metadata3.insert("source_title".to_string(), "Rust Advanced Topics".to_string());

        let docs = vec![
            Document {
                page_content: "This is about Python programming".to_string(),
                metadata: metadata1,
            },
            Document {
                page_content: "This is about Rust programming and Rust features".to_string(),
                metadata: metadata2,
            },
            Document {
                page_content: "Advanced Rust topics and Rust patterns".to_string(),
                metadata: metadata3,
            },
        ];

        let ranked = generator.simple_text_ranking(&docs, "Rust programming", 2);

        assert_eq!(ranked.len(), 2);
        // The documents with "Rust" should rank higher
        assert!(ranked[0].page_content.contains("Rust"));
    }

    #[test]
    fn test_simple_text_ranking_empty_query() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let docs = vec![Document {
            page_content: "Test content".to_string(),
            metadata: HashMap::new(),
        }];

        let ranked = generator.simple_text_ranking(&docs, "", 10);
        assert_eq!(ranked.len(), 1);
    }

    #[test]
    fn test_simple_text_ranking_no_documents() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let ranked = generator.simple_text_ranking(&[], "test query", 10);
        assert_eq!(ranked.len(), 0);
    }

    #[test]
    fn test_prepare_context_with_sources() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let mut metadata = HashMap::new();
        metadata.insert("source_number".to_string(), "1".to_string());
        metadata.insert("source_title".to_string(), "Test Article".to_string());
        metadata.insert("source_domain".to_string(), "example.com".to_string());

        let docs = vec![Document {
            page_content: "Test content".to_string(),
            metadata,
        }];

        let context = generator.prepare_context_with_sources(&docs);

        assert!(context.contains("SOURCE 1"));
        assert!(context.contains("Test Article"));
        assert!(context.contains("example.com"));
        assert!(context.contains("Test content"));
    }

    #[test]
    fn test_prepare_context_empty_documents() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let context = generator.prepare_context_with_sources(&[]);
        assert_eq!(context, "");
    }

    #[test]
    fn test_add_enhanced_sources_list() {
        let mut generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        // Add some source metadata
        generator.sources_metadata.insert(
            1,
            SourceMetadata {
                url: "https://example.com".to_string(),
                title: "Test Source".to_string(),
                snippet: "snippet".to_string(),
                domain: "example.com".to_string(),
                content_summary: "A test summary".to_string(),
                topics_covered: vec!["topic1".to_string(), "topic2".to_string()],
            },
        );

        let article = "# My Article\nContent here";
        let with_sources = generator.add_enhanced_sources_list(article);

        assert!(with_sources.contains("# My Article"));
        assert!(with_sources.contains("## Sources"));
        assert!(with_sources.contains("Test Source"));
        assert!(with_sources.contains("https://example.com"));
        assert!(with_sources.contains("example.com"));
        assert!(with_sources.contains("topic1, topic2"));
    }

    #[test]
    fn test_add_enhanced_sources_list_no_sources() {
        let generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        let article = "# My Article\nContent here";
        let with_sources = generator.add_enhanced_sources_list(article);

        assert!(with_sources.contains("# My Article"));
        assert!(with_sources.contains("## Sources"));
    }

    #[test]
    fn test_sources_metadata_getter() {
        let mut generator = EnhancedRAGArticleGenerator::new(
            "http://localhost:8080".to_string(),
            "test-model".to_string(),
            "test-embed".to_string(),
            None,
        );

        assert_eq!(generator.sources_metadata().len(), 0);

        generator.sources_metadata.insert(
            1,
            SourceMetadata {
                url: "test".to_string(),
                title: "test".to_string(),
                snippet: "test".to_string(),
                domain: "test".to_string(),
                content_summary: "test".to_string(),
                topics_covered: vec![],
            },
        );

        assert_eq!(generator.sources_metadata().len(), 1);
    }
}