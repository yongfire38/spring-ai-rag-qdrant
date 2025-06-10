package com.example.chat.config;

import java.util.Map;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.ResourceLoader;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 로컬 ONNX 모델을 사용하는 임베딩 모델 빈 구성 클래스
 */
@Slf4j
@Configuration
@RequiredArgsConstructor
public class EmbeddingConfig {

    // 하드코딩 하지 말고 설정 파일에서 맞출 경우를 위해 남겨 둠
    private final ResourceLoader resourceLoader;

    /**
     * 로컬 ONNX 모델을 사용하는 임베딩 모델 빈 구성
     * 
     * @return 로컬 ONNX 모델을 사용하는 TransformersEmbeddingModel
     */
    @Bean
    @Primary
    public EmbeddingModel embeddingModel() throws Exception {

        log.info("로컬 ONNX 모델을 사용하는 임베딩 모델 초기화");

        // 모델 및 토크나이저 경로 설정
        String modelPath = "classpath:model/model.onnx";
        String tokenizerPath = "classpath:model/tokenizer.json";

        // TransformersEmbeddingModel 생성 및 설정
        TransformersEmbeddingModel model = new TransformersEmbeddingModel();

        // 로컬 모델 및 토크나이저 설정
        model.setModelResource(modelPath);
        model.setTokenizerResource(tokenizerPath);

        // 토크나이저 옵션 설정 (오류 방지)
        model.setTokenizerOptions(Map.of("padding", "true"));

        // 캐시 디렉토리 설정 (선택사항)
        // model.setResourceCacheDirectory("c:/temp/onnx-models");

        // 모델 초기화
        if (model instanceof InitializingBean) {
            ((InitializingBean) model).afterPropertiesSet();
        }

        log.info("로컬 ONNX 임베딩 모델 초기화 완료: {}", modelPath);
        return model;
    }
}
