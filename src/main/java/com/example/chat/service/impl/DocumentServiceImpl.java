package com.example.chat.service.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.vectorstore.qdrant.QdrantVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import com.example.chat.service.DocumentService;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.CollectionInfo;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class DocumentServiceImpl implements DocumentService {

    // Spring AI 자동 설정을 통해 주입되는 QdrantVectorStore 사용
    private final QdrantVectorStore qdrantVectorStore;

    private final TextSplitter textSplitter;

    @Value("${spring.ai.document.path}")
    private String documentPath;

    @Value("${spring.ai.vectorstore.qdrant.host}")
    private String qdrantHost;

    @Value("${spring.ai.vectorstore.qdrant.port}")
    private int qdrantPort;

    @Value("${spring.ai.vectorstore.qdrant.collection-name}")
    private String collectionName;

    @Value("${spring.ai.vectorstore.qdrant.dimension}")
    private int vectorSize;

    @Override
    public int loadDocuments() {
        try {
            // 컬렉션 존재 여부 확인 및 필요시 초기화
            if (!checkAndInitializeCollection()) {
                log.info("컬렉션이 이미 존재하고 데이터가 있습니다. 문서 로드를 건너뜁니다.");
                return 0;
            }

            // step 1. 문서 로드
            List<Document> documents = loadMarkdownDocuments();

            if (documents.isEmpty()) {
                log.info("로드할 문서가 없습니다.");
                return 0;
            }

            log.info("총 {}개의 문서를 처리합니다.", documents.size());

            // step 2. 문서 분할
            List<Document> splitDocuments = textSplitter.split(documents);
            log.info("{}개의 원본 문서를 {}개의 청크로 분할했습니다.", documents.size(), splitDocuments.size());

            List<Document> processedDocuments = new ArrayList<>();
            Map<String, Integer> chunkCountByOriginal = new HashMap<>();

            // step 3. 문서 임베딩 및 벡터 저장소에 추가
            for (Document chunk : splitDocuments) {
                // 파일명 기반으로 안정적인 ID 생성
                String source = (String) chunk.getMetadata().get("source");
                String stableId = "";

                if (source != null && !source.isEmpty()) {
                    // 파일명에서 확장자를 제거하고 특수문자 대체
                    stableId = source.replaceAll("\\.md$", "").replaceAll("[^a-zA-Z0-9가-힣]", "_");
                } else {
                    // source가 없는 경우 임의의 고정 ID 사용
                    stableId = "unknown_document";
                }

                // 해당 원본 문서의 청크 카운트 증가
                int chunkIndex = chunkCountByOriginal.getOrDefault(stableId, 0) + 1;
                chunkCountByOriginal.put(stableId, chunkIndex);

                // 결정론적 UUID 생성 (동일 입력에 대해 항상 동일 UUID 생성)
                String seedForUuid = stableId + "_" + chunkIndex;
                UUID uuid = UUID.nameUUIDFromBytes(seedForUuid.getBytes(StandardCharsets.UTF_8));
                String newId = uuid.toString();

                // 메타데이터 복사 및 추가 정보 설정
                Map<String, Object> metadata = new HashMap<>(chunk.getMetadata());
                metadata.put("original_document_id", stableId);
                metadata.put("chunk_index", chunkIndex);
                metadata.put("deterministic_id", seedForUuid); // 디버깅 및 추적용 원본 ID 저장

                // 새 문서 생성 - UUID 형식의 ID 사용
                Document newChunk = new Document(newId, chunk.getText(), metadata);
                processedDocuments.add(newChunk);
            }

            try {
                qdrantVectorStore.add(processedDocuments);
                log.info("총 {}개 청크 처리 완료", processedDocuments.size());
                return processedDocuments.size();
            } catch (Exception e) {
                log.error("문서 처리 중 오류 발생", e);
                return 0;
            }
        } catch (Exception e) {
            log.error("문서 로드 과정에서 예상치 못한 오류 발생", e);
            return 0;
        }
    }

    /**
     * Qdrant 컬렉션 존재 여부 확인 및 필요시 초기화
     * 
     * @return 문서 로드가 필요하면 true, 아니면 false
     */
    private boolean checkAndInitializeCollection() {
        QdrantClient client = null;
        try {
            // Qdrant 클라이언트 생성 - gRPC 포트 사용
            client = new QdrantClient(
                    QdrantGrpcClient.newBuilder(qdrantHost, qdrantPort, false).build());

            return checkCollectionExists(client) ? createOrUseCollection(client) : false;
        } catch (Exception e) {
            log.error("Qdrant 컬렉션 확인 중 오류 발생", e);
            throw new RuntimeException("Qdrant 컬렉션 확인 및 초기화 실패", e);
        } finally {
            if (client != null) {
                client.close();
            }
        }
    }

    /**
     * 컬렉션 존재 여부 확인
     */
    private boolean checkCollectionExists(QdrantClient client) {
        try {
            // 컬렉션 정보 조회
            CollectionInfo info = client.getCollectionInfoAsync(collectionName).get();

            // 컬렉션이 존재하고 벡터가 있는지 확인
            long vectorCount = info.getPointsCount();
            log.info("컬렉션 '{}' 존재함. 현재 벡터 수: {}", collectionName, vectorCount);

            if (vectorCount > 0) {
                // 이미 데이터가 있으면 로드 건너뛰기
                log.info("컬렉션에 이미 데이터가 있으므로 문서 로드를 건너뜁니다.");
                return false;
            } else {
                // 컬렉션은 있지만 비어있으면 문서 로드 진행
                log.info("컬렉션이 비어 있습니다. 문서를 로드합니다.");
                return true;
            }
        } catch (InterruptedException | ExecutionException e) {
            // 컬렉션이 없는 경우
            log.info("컬렉션 '{}' 없음. 새로 생성해야 합니다.", collectionName);
            return true;
        }
    }

    /**
     * 필요시 컬렉션 생성
     */
    private boolean createOrUseCollection(QdrantClient client) {
        try {
            // 컬렉션 정보 조회 시도
            client.getCollectionInfoAsync(collectionName).get();
            // 컬렉션이 이미 존재하면 문서 로드 진행
            return true;
        } catch (InterruptedException | ExecutionException e) {
            // 컬렉션이 없는 경우 - 새로 생성
            return createCollection(client);
        }
    }

    /**
     * 새 컬렉션 생성
     */
    private boolean createCollection(QdrantClient client) {
        try {
            // 컬렉션 생성
            log.info("벡터 크기 {}로 컬렉션 '{}' 생성 시도", vectorSize, collectionName);

            // 컬렉션 생성
            client.createCollectionAsync(
                    collectionName,
                    VectorParams.newBuilder()
                            .setSize(vectorSize)
                            .setDistance(Distance.Cosine)
                            .build())
                    .get();

            log.info("컬렉션 '{}' 생성 완료", collectionName);
            return true;
        } catch (Exception createEx) {
            log.error("컬렉션 생성 중 오류 발생", createEx);
            throw new RuntimeException("컬렉션 생성 실패", createEx);
        }
    }

    /**
     * 리소스 디렉토리에서 모든 마크다운 파일을 로드
     */
    private List<Document> loadMarkdownDocuments() {
        List<Document> documents = new ArrayList<>();
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();

        try {
            Resource[] resources = resolver.getResources(documentPath);
            log.info("{}개의 마크다운 파일을 찾았습니다.", resources.length);

            for (Resource resource : resources) {
                try {
                    String content = readResourceContent(resource);
                    String filename = resource.getFilename();

                    // 메타데이터 생성
                    Map<String, Object> metadata = new HashMap<>();
                    metadata.put("source", filename);
                    metadata.put("type", "markdown");

                    // 결정론적 UUID 생성 (파일명 기반)
                    String seedForUuid = "doc-" + filename;
                    UUID uuid = UUID.nameUUIDFromBytes(seedForUuid.getBytes(StandardCharsets.UTF_8));
                    String docId = uuid.toString();

                    Document doc = new Document(docId, content, metadata);
                    documents.add(doc);
                    log.info("문서 로드 완료: {}, ID: {}, 크기: {}바이트", filename, docId, content.length());
                } catch (IOException e) {
                    log.error("파일 읽기 오류: {}", resource.getFilename(), e);
                }
            }
        } catch (IOException e) {
            log.error("리소스 검색 중 오류 발생", e);
        }

        return documents;
    }

    /**
     * 리소스 파일의 내용을 문자열로 읽기
     */
    private String readResourceContent(Resource resource) throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            return reader.lines().collect(Collectors.joining("\n"));
        }
    }
}