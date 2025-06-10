package com.example.chat.service;

/**
 * 문서 처리 서비스 인터페이스
 * 마크다운 문서를 로드하고 벡터 저장소에 저장하는 기능 제공
 */
public interface DocumentService {

    /**
     * 문서를 로드하고 벡터 저장소에 저장
     * 
     * @return 처리된 문서 수
     */
    int loadDocuments();
}
