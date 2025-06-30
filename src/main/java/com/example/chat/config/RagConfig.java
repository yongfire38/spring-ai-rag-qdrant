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

    /**
     * 애플리케이션 시작 시 문서 로드 및 임베딩 저장
     */
    @Bean
    public CommandLineRunner loadDocuments(DocumentService documentService) {
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
    public Advisor retrievalAugmentationAdvisor(QdrantVectorStore qdrantVectorStore) {
        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(VectorStoreDocumentRetriever.builder()
                        .similarityThreshold(0.60)
                        .vectorStore(qdrantVectorStore)
                        .build())
                .queryAugmenter(ContextualQueryAugmenter.builder()
                        .allowEmptyContext(true)
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