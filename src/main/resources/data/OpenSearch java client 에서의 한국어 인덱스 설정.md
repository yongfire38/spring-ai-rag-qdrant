---
제목: OpenSearch java client 에서의 한국어 인덱스 설정
날짜: 2024-05-29 14:13
태그(키워드):
  - opensearch
  - java
  - springboot
---
# OpenSearch java client 에서의 한국어 인덱스 설정

- OpenSearch Java Client를 이용하여 한국어 데이터를 검색 시 설정할 필요성이 있는 index 설정을 기술한다.

## 기본 개념

1. Analyzer : 텍스트 분석을 포괄하는 추상화. Analyzer는 문자 필터(Character Filters), 토크나이저(Tokenizers), 그리고 토큰 필터(Token Filters)를 순차적으로 적용
2. Tokenizer : 문자열을 토큰(일반적으로 단어)으로 분리하는 역할. 디폴트는 공백을 기준으로 분리.
3. TokenFilter : Tokenizer : 토크나이저로부터 받은 토큰 스트림을 추가, 제거, 수정하는 역할. 예를 들면 모든 토큰을 소문자로 변환하거나, 불용어(stop words)를 제거하거나, 동의어(synonyms)를 추가하는 등의 작업을 수행 가능하다.
4. Normalizer : Analyzer와 유사하지만 Tokenizer 없이 작동하며 문자 레벨의 작업만 가능하다. Analyzer를 사용하지 않는 term query등에 쓰인다.

-> Analyzer는 정확히 하나의 토크나이저를 포함해야 하며, 선택적으로 여러 문자 필터와 토큰 필터를 포함

![](https://i.imgur.com/pLIbKLF.png)

## java client 에서의 구현 방법

- [[openSearch 쿼리의 종류와 Java Client에서의 구현 방법]] 문서를 참조하여 기본적인 한국어 인덱스를 만들 경우, Analyzer 설정이 되어 있지 않게 된다.
- 한국어를 포함한 기본적인 인덱스 설정 방법은 다음과 같이 진행한다.


```java
@Override

public void createIndex(String indexName) throws IOException {

//적용할 내용들을 Map 자료형으로 선언한다.
Map<String, Tokenizer> tokenizerMap = new HashMap<>();
Map<String, Analyzer> analyzerMap = new HashMap<>();
Map<String, TokenFilter> tokenFilterMap = new HashMap<>();
//Map<String, Normalizer> normalizerMap = new HashMap<>(); //normalizer 이용 시에만 추가
Map<String, CharFilter> charFilterMap = new HashMap<>();

//문자 필터 
// html 태그를 제거하는 HtmlStripCharFilter랑 구두점을 제거하는 PatternReplaceCharFilter를 charFilterMap에 넣는다. 나중에 인덱스 생성할 때 이걸 적용한다.

// char filter : html 태그를 제거한다
HtmlStripCharFilter htmlStripFilter = new HtmlStripCharFilter.Builder().build();
CharFilter chrFilter = new CharFilter.Builder().definition(htmlStripFilter._toCharFilterDefinition()).build();
charFilterMap.put("htmlfilter", chrFilter);

// remove punctuation chars : 구두점을 제거한다

PatternReplaceCharFilter patternCharFilter = new PatternReplaceCharFilter.Builder().pattern("\\p{Punct}").replacement("").flags("CASE_INSENSITIVE|MULTILINE").build();
CharFilter chrPatternFilter = new CharFilter.Builder().definition(patternCharFilter._toCharFilterDefinition()).build();
charFilterMap.put("patternfilter", chrPatternFilter);

// 인덱스 생성할 때 Analyzer가 참조할 문자 필터들 리스트이다
// 적용과 참조 둘 다 걸려 있어야 되는 것으로 보인다
List<String> charFilterList = new ArrayList<>();
charFilterList.add("htmlfilter");
charFilterList.add("patternfilter");

//토큰 필터
//소문자 변환 / 비ASCII 문자를 ASCII 문자로 변환 / 한국어의 특정 품사를 제거 / 동의저 처리의 4가지를 해 본다.

// 제거할 품사를 열거한다 : NR - 수사 (테스트를 위해서 이렇게 한 거고, 실제로는 조사나 형용사 같은 걸 제거하는 게 정석)
List<String> stopTags = Arrays.asList("NR");

//토큰 필터 적용 : 소문자 변환하는 LowercaseTokenFilter, 아스키 변환하는 AsciiFoldingTokenFilter, 품사 제거하는 NoriPartOfSpeechTokenFilter, 복합어의 동의어 처리용인 SynonymGraphTokenFilter를 tokenFilterMap에 넣는다
LowercaseTokenFilter lowerFilter = new LowercaseTokenFilter.Builder().build();

AsciiFoldingTokenFilter asciiFilter = new AsciiFoldingTokenFilter.Builder().preserveOriginal(false).build();

NoriPartOfSpeechTokenFilter noriPartOfSpeechFilter = new NoriPartOfSpeechTokenFilter.Builder().stoptags(stopTags).build();

tokenFilterMap.put("lowercase", new TokenFilter.Builder().definition(lowerFilter._toTokenFilterDefinition()).build());

tokenFilterMap.put("asciifolding", new TokenFilter.Builder().definition(asciiFilter._toTokenFilterDefinition()).build());

tokenFilterMap.put("nori_part_of_speech", new TokenFilter.Builder().definition(noriPartOfSpeechFilter._toTokenFilterDefinition()).build());

List<String> synonym = Arrays.asList("amazon, aws", "풋사과, 햇사과, 사과");

SynonymGraphTokenFilter synonymFilter = new SynonymGraphTokenFilter.Builder().synonyms(synonym).expand(true).build();

tokenFilterMap.put("synonym_graph", new TokenFilter.Builder().definition(synonymFilter._toTokenFilterDefinition()).build());

// 인덱스 생성할 때 Analyzer가 참조할 토큰 필터 리스트를 만든다
// 이 중 nori_number와 nori_readingform은 적용부에 작성 안하고 참조부에만 작성해도 동작하더라... nori 분석기에서 제공하는 기능이라 그런 거 같음
List<String> tokenFilterList = new ArrayList<>();

tokenFilterList.add("lowercase");
tokenFilterList.add("asciifolding");
tokenFilterList.add("synonym_graph");
tokenFilterList.add("nori_number"); // 한국어 숫자의 검색을 가능하게 함
tokenFilterList.add("nori_readingform"); // 한자의 한국어 검색을 가능하게 함
tokenFilterList.add("nori_part_of_speech");

//토크나이저 설정 전, 분해하지 말아야 할 단어의 List를 만든다
List<String> userDictionaryRules = Arrays.asList("낮말", "밤말");

//토크나이저
//1개만 설정 가능하며, 여기서는 당연히 한국어 토크나이저인 NoriTokenizer를 설정
//Nori 플러그인이 미리 설치되어 있어야 함
NoriTokenizer noriTokenizer = new NoriTokenizer.Builder()

// .decompoundMode(NoriDecompoundMode.Mixed)로 한국어 동의어 설정하면 오류난다
.decompoundMode(NoriDecompoundMode.Discard)

.discardPunctuation(true)

.userDictionaryRules(userDictionaryRules)

.build();

Tokenizer tokenizer = new Tokenizer.Builder().definition(noriTokenizer._toTokenizerDefinition()).build();

tokenizerMap.put("nori-tokenizer", tokenizer);

//커스텀 Analyzer 구성 : char_filter ==> tokenizer ==> token filter
//참조용으로 만들어 놨던 List들을 넣는다
CustomAnalyzer noriAnalyzer = new CustomAnalyzer.Builder()

.charFilter(charFilterList)

.tokenizer("nori-tokenizer")

.filter(tokenFilterList).build();

Analyzer analyzer = new Analyzer.Builder().custom(noriAnalyzer).build();

analyzerMap.put("nori-analyzer", analyzer);

/* normalizer 설정 : term query와 같은 분석기를 사용하지 않는 질의에 적용된다. 문자 레벨의 작업만 가능하므로 lowercase, asciifolding 정도가 한계이며 실제로 tokenFilterList를 넣으면 에러 난다.

normalizerMap.put("keyword_normalizer", new Normalizer.Builder()

.custom(new CustomNormalizer.Builder().charFilter("patternfilter").filter(tokenFilterList).build())

.build());

*/

//analysis에다가는 설정 때 만들었던 Map들을 걸어 주고, analyzer 이름은 analyzerMap의 키 값을 넣어 주면 됨
CreateIndexRequest createIndexRequest = new CreateIndexRequest.Builder()

.index(indexName)

.settings(s -> s

.analysis(a -> a

.charFilter(charFilterMap)

//.normalizer(normalizerMap)

.tokenizer(tokenizerMap)

.filter(tokenFilterMap)

.analyzer(analyzerMap)

)

)

.mappings(m -> m

.properties("director", p -> p

.text(f -> f

.index(true)

.analyzer("nori-analyzer")

)

)

.properties("title", p -> p

.text(f -> f

.index(true)

.analyzer("nori-analyzer")

)

)

.properties("year", p -> p

.long_(f -> f

.index(true)

)

)

)

.build();

//요청 끝나고 실제 인덱스 만드는 부분
try {

CreateIndexResponse createIndexResponse = client.indices().create(createIndexRequest);

log.debug(String.format("Index %s.", createIndexResponse.index().toString().toLowerCase()));

} catch (OpenSearchException ex) {

final String errorType = Objects.requireNonNull(ex.response().error().type());

if (! errorType.equals("resource_already_exists_exception")) {

throw ex;

}

}

}
```


---

## 동의어나 분해 금지 같은 거 하드 코딩으로 안 하려면

- 외부 파일에서 읽는 메서드 만들어 둠

```java
private static List<String> readWordsFromFile(String filePath) {
    List<String> words = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
        String line;
        while ((line = br.readLine()) != null) {
            words.add(line.trim());
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
    return words;
}

//application.properties에다가 명시해 놓고 가져다 쓰면 됨
@Value("${synonyms.path}")
public String synonymsPath;

//List<String> synonym = Arrays.asList("amazon, aws", "풋사과, 햇사과, 사과");
List<String> synonym = readWordsFromFile(synonymsPath);
```

---
## 적용한 내용

- Character Filters : `HtmlStripCharFilter`(html 태그를 제거) / `PatternReplaceCharFilter`(구두점을 제거)
- Tokenizer : 한국어 분석 토크나이저인 `NoriTokenizer`를 사용. `decompoundMode` 설정 (복합 명사를 어떻게 다룰지), `discardPunctuation`(구두점 및 문장부호의 제거) 및 `userDictionaryRules`의 설정이 가능
- TokenFilter : 소문자 변환하는 `LowercaseTokenFilter`, 아스키 변환하는 `AsciiFoldingTokenFilter`, 품사 제거하는 `NoriPartOfSpeechTokenFilter`, 복합어의 동의어 처리용인 `SynonymGraphTokenFilter` 등의 설정이 가능

- CreateIndexRequest 에서 설정을 하게 되며, 설정 및 적용에는 Map을 사용하였다.
- analyzerMap에다가는 charFilter, tokenizer, filter를 참조하도록 List를 사용
- 둘 중 하나가 누락되어도 작동되지 않았었음.
- 단, nori_number 와 nori_readingform는 설정을 직접 하지 않고(애초에 클래스도 없다) tokenFilterList에다가 이름만 넣어도 되었음.
- decompoundMode(NoriDecompoundMode.Mixed)로 한국어 동의어 설정하면 오류난다.

---
## 참고 문서

- https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/
- https://realkoy.tistory.com/entry/elasticsearch-8x-Java-api-client%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-IndexTemplate-%EC%83%9D%EC%84%B1
- https://opster.com/guides/elasticsearch/data-architecture/elasticsearch-text-analyzers/#elasticsearch-text-analysis-tokenization-normalization
- https://github.com/elastic/elasticsearch/issues/37751
