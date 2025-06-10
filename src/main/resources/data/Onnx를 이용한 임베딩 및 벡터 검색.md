---
제목: Onnx를 이용한 임베딩 및 벡터 검색
날짜: 2024-06-28 17:49
태그(키워드):
  - opensearch
  - java
  - springboot
  - embedding
  - vector-search
  - python
---
- [[OpenSearch 검색 in Java (Spring Boot 연동 + 한국어 텍스트 검색)]]에서 수행한 방법은 Hugging face의 API를 호출하는 방식이라 많은 자료를 처리하려고 하면 한도에 걸려서 과금이 필요하게 된다.
- 월 9달러라고는 하는데 정확한 한도를 알 수 없어 로컬에서 처리할 수 있는 방법을 찾던 중, [Spring AI Transformers (ONNX) Embeddings](https://docs.spring.io/spring-ai/reference/api/embeddings/onnx.html)에서 예전에 Onnx형식의 모델 익스포트를 해 본 적이 있기 때문에 해당 방법을 이용해 로컬에서 처리해 보기로 함.

## 준비 과정

- 먼저 사용할 데이터를 텍스트화해서 준비해야 한다.
- [네이버 영화 리뷰](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/nsmc.html)를 이용함. python을 이용한 방법을 표준으로 제공하므로 코드 만들고 실행해서 받아옴. [[nsmc_download.py]] 참고

## 모델을 ONNX로 익스포트

- Python에서 가상 환경을 설정하고 필요한 패키지를 설치하며, 특정 모델을 ONNX 포맷으로 내보내야 하는데 예시는 다음과 같다.


```
python3 -m venv venv
source ./venv/bin/activate
(venv) pip install --upgrade pip
(venv) pip install optimum onnx onnxruntime
(venv) optimum-cli export onnx --generative sentence-transformers/all-MiniLM-L6-v2 onnx-output-folder
```

- 윈도우에서는 다음과 같이 처리한다.
- 다음 과정 수행으로 나중에 임베딩 모델에 사용되는 tokenizer.json 및 model.onnx 파일이 익스포트된다.

```
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install optimum onnx onnxruntime
(마지막에 오류나면 sentence-transformers도 추가)
pip install optimum onnx onnxruntime sentence-transformers

(모델 변경해서 현재 폴더에 내보내기)
optimum-cli export onnx -m snunlp/KR-SBERT-V40K-klueNLI-augSTS .
```

- 이제 기존과 마찬가지로 뽑아낸 텍스트 파일을 가지고 json으로 만들고, 이걸 오픈서치 인덱스에 집어넣으면 됨.
- 다른 점이라면 임베딩 모델 선언하는 법이랑 결과값의 자료형이 다르다는 것.
- 허깅페이스 호출의 경우

```java
Response<Embedding> response = embeddingModel.embed(query);

float[] vector = response.content().vector();
```

- Onnx의 경우는 그냥 Embedding 이다.

```java
Embedding response = embeddingModel.embed(line).content();

float[] vector = response.vector();
```

- 그리고 모델 선언시에 파일 경로를 넣어 줘야 하는데 왜인지는 모르겠지만 application.properties에 넣는 게 안 되더라... 그래서 일단 상대경로로 잡음.


```java
public void toJsonAltConverter() throws IOException {
		
		EmbeddingModel embeddingModel = new OnnxEmbeddingModel(
				"./model/model.onnx",
				"./model/tokenizer.json",
                PoolingMode.MEAN);
		
		
		try (FileWriter writer = new FileWriter("./example/output.json"); BufferedReader reader = new BufferedReader(new FileReader(txtFilePath))) {
			
			String line;
            JSONArray jsonArray = new JSONArray();
            
            int id = 1; // ID가 1부터 시작하다고 가정
            
            while ((line = reader.readLine())!= null) {
            	
            	JSONObject jsonObject = new JSONObject();
                jsonObject.put("id", id);
                jsonObject.put("text", line);
                
                Embedding response = embeddingModel.embed(line).content();
            	
                float[] embeddings =  response.vector();
                JSONArray embeddingArray = new JSONArray(embeddings);
                jsonObject.put("embedding", embeddingArray);

                jsonArray.put(jsonObject);
                
            	id++; 
            }
            
            writer.write(jsonArray.toString());
            writer.flush();
            
            log.debug(":::::변환 완료되었습니다.");
            
			
		} catch (IOException e) {
            e.printStackTrace();
        }
		
	}

//open search 인덱스에 넣는 건 기존과 동일
public void insertAltEmbeddingData(String indexName) throws IOException {

long beforeTime = System.currentTimeMillis();

//문자열 파일을 읽어서 json 파일로 변환, 이 과정에서 각 문자열을 임베딩한 결과도 추가한다
//toJsonAltConverter 호출...
toJsonAltConverter();

//만들어진 json 파일을 읽어서 벌크 인덱싱 처리
List<Map<String, Object>> parsedJsonList = JsonParser.parseJsonList(jsonFilePath);
Builder bulkRequestBuilder = new BulkRequest.Builder();
for (Map<String, Object> jsonMap : parsedJsonList) {
	Map<String, Object> data = jsonMap;
	bulkRequestBuilder.operations(ops -> ops
		.index(IndexOperation.of(io -> io.index(indexName).id(String.valueOf(index)).document(data)))
);

index++;

}

BulkRequest bulkRequest = bulkRequestBuilder.build();
client.bulk(bulkRequest);

long afterTime = System.currentTimeMillis();
long secDiffTime = (afterTime - beforeTime)/1000;

log.debug("시간차이(m) : " + secDiffTime + "초");

}

```


---

## 참고 문서

- [Spring AI Transformers (ONNX) Embeddings](https://docs.spring.io/spring-ai/reference/api/embeddings/onnx.html)
- [[OpenSearch 검색 in Java (Spring Boot 연동 + 한국어 텍스트 검색)]]
- [[nsmc_download.py]]
