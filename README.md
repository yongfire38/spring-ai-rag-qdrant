# Spring AI와 Qdrant를 사용한 RAG(Retrieval-Augmented Generation) 샘플

## 환경 설정

| 프로그램 명 | 버전 명 |
| :--------- | :------ |
| java | 17 이상 |
| maven | 3.8.4 |
| spring boot | 3.5.0 |
| Spring AI | 1.0.0 |

## 사용 기술

1. Java
2. Spring Boot (Maven)
3. Langchain4j
4. Qdrant DB
5. Ollama

## 사전 준비

1. [Ollama](https://ollama.com/download) 설치
2. [Huggingface](https://huggingface.co/) 에서 PC 스펙에 적합한 GGUF 모델을 준비. 완료된 모델 명은 `application.properties` 파일의 `ollama.model-name`에 설정한다.
3. `\src\main\resources\data` 경로에 디폴트 문서가 존재하나, 다른 md 파일로 대체도 가능
4. `docker-compose.yml` 을 사용해 `docker compose up -d`로 docker container 기반의 Qdrant DB 설정을 해 둔다.

## Onnx 모델 익스포트

- `Onnx (Open Neural Network Exchange)`는 기계학습 모델을 다른 딥러닝 프레임워크 환경 (ex. Tensorflow, PyTorch, etc..)에서 서로 호환되게 사용할 수 있도록 만들어진 공유 플랫폼이다.
- 다양한 프레임워크 간의 호환성 문제를 해결하고, 모델 배포 및 활용에 유연성을 제공할 수 있다.
- 로컬 환경에서 Embedding 작업을 수행하기 위해서는 Embedding 모델을 onnx로 익스포트하여 사용하여야 할 필요가 있으므로 그 과정을 소개한다.

### 사용 모델 소개

- Huggingface 에 배포되는 여러 모델 중 적합한 모델을 취사 선택하여 `Onnx (Open Neural Network Exchange)`로 변환 후, 로컬에서 사용하는 것이 가능하다.
- 본 프로젝트에서는 [ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask) 모델을 사용하는 방법을 소개한다.
- Python에서 가상 환경을 설정하고 필요한 패키지를 설치하며, 모델을 `Onnx` 포맷으로 익스포트한다.

```bash
# 패키지 버전 충돌을 방지하기 위해 venv 라는 이름으로 가상 환경을 생성한다
python3 -m venv venv

# 윈도우 환경에서 가상환경을 활성화 한다
.\venv\Scripts\activate

# 리눅스, macOS 환경에서 가상환경을 활성화 한다
source ./venv/bin/activate //

# 가상 환경이 활성화 된 상태에서 pip를 최신 버전으로 업그레이드한다
python -m pip install --upgrade pip

# 필요한 패키지들을 설치한다
pip install optimum onnx onnxruntime sentence-transformers

# 현재 경로에 jhgan/ko-sroberta-multitask 모델을 onnx 포맷으로 내보낸다
optimum-cli export onnx -m jhgan/ko-sroberta-multitask .
```

- 익스포트가 완료되면 Embedding에 사용되는 `tokenizer.json` 및 `model.onnx` 파일이 생성된다.
- 해당 파일들은 `\src\main\resources\model` 내에 설정하도록 한다.
- 상세 정보는 Spring에서 제공하는 [공식 문서](https://docs.spring.io/spring-ai/reference/api/embeddings/onnx.html#_prerequisites) 를 참조 가능하다.

## 실행

1. 애플리케이션 실행 후 도큐먼트 생성 및 임베딩, 적재가 실행된다. 수동으로 실행하려면 메인 화면의 `문서 로드` 버튼을 클릭한다.
2. `http://localhost:8001/` 에서 데이터 확인이 가능하다.
3. 메인 화면의 `RAG 채팅 모드`, `일반 채팅 모드` 버튼으로 RAG가 적용된 질의 답변, 일반적인 질의 답변을 받을 수 있다.


