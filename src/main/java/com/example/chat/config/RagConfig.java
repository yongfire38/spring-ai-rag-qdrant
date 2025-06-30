package com.example.chat.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.generation.augmentation.ContextualQueryAugmenter;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.qdrant.QdrantVectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.example.chat.service.DocumentService;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Configuration
public class RagConfig {

    private final DocumentService documentService;
    private final QdrantVectorStore qdrantVectorStore;

    public RagConfig(DocumentService documentService, QdrantVectorStore qdrantVectorStore) {
        this.documentService = documentService;
        this.qdrantVectorStore = qdrantVectorStore;
    }

    /**
     * 애플리케이션 시작 시 문서 로드 및 임베딩 저장
     */
    @Bean
    public CommandLineRunner loadDocuments() {
        return args -> {
            log.info("RAG 시스템 초기화 시작");
            int count = documentService.loadDocuments();
            log.info("RAG 시스템 초기화 완료 - {} 문서 처리됨", count);
        };
    }

    /**
     * RAG Advisor 설정
     */
    @Bean
    public Advisor retrievalAugmentationAdvisor() {
        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(VectorStoreDocumentRetriever.builder()
                        .similarityThreshold((double) 0.60f)
                        .vectorStore(qdrantVectorStore)
                        .build())
                .queryAugmenter(ContextualQueryAugmenter.builder()
                        .allowEmptyContext(true) // 빈 컨텍스트 허용
                        .build())
                .build();
    }

    /**
     * ChatClient Bean 설정
     */
    @Bean
    public ChatClient ollamaChatClient(OllamaChatModel chatModel) {
        return ChatClient.create(chatModel);
    }
}