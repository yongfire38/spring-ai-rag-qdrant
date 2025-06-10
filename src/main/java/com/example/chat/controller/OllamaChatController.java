package com.example.chat.controller;

import java.util.Map;

import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.chat.service.ChatService;
import com.example.chat.service.DocumentService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;

@Slf4j
@RestController
@RequiredArgsConstructor
public class OllamaChatController {

    private final OllamaChatModel chatModel;
    private final ChatService chatService;
    private final DocumentService documentService;

    /**
     * 일반 응답 생성 (테스트용)
     */
    @GetMapping("/ai/generate")
    public Map<String, String> generate(
            @RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
        return Map.of("generation", this.chatModel.call(message));
    }

    /**
     * 스트리밍 응답 생성 (테스트용)
     */
    @GetMapping("/ai/generateStream")
    public Flux<ChatResponse> generateStream(
            @RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
        Prompt prompt = new Prompt(new UserMessage(message));
        return this.chatModel.stream(prompt);
    }

    /**
     * RAG 기반 스트리밍 응답 생성
     */
    @GetMapping("/ai/rag/stream")
    public Flux<ChatResponse> streamRagResponse(
            @RequestParam(value = "message", defaultValue = "Tell me about this document") String message) {
        log.info("RAG 기반 스트리밍 질의 수신: {}", message);
        return chatService.streamRagResponse(message);
    }

    /**
     * 일반 스트리밍 응답 생성 (ChatService 사용)
     */
    @GetMapping("/ai/simple/stream")
    public Flux<ChatResponse> streamSimpleResponse(
            @RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
        log.info("일반 스트리밍 질의 수신: {}", message);
        return chatService.streamSimpleResponse(message);
    }

    /**
     * 문서 로드 수동 트리거 : 마크다운 파일을 전부 새로 읽어서 임베딩 후 저장
     */
    @PostMapping("/api/documents/load")
    public Map<String, Object> loadDocuments() {
        log.info("문서 로드 요청 수신");
        int count = documentService.loadDocuments();
        return Map.of("success", true, "count", count);
    }

}
