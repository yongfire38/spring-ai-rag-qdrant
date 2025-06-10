---
제목: Term-level Queries의 종류
날짜: 2024-05-29 14:35
태그(키워드):
  - opensearch
  - java
  - springboot
---
# Term-level Queries의 종류

- Open Search의 Term-level Queries 의 종류를 기술한다.

1. term query: 지정된 필드에서 정확히 일치하는 용어를 검색. 텍스트를 분석하지 않음.
- term query를 사용할 경우에는 검색어에 애초에 문장을 넣지 말 것.
- case_insensitive 파라미터를 true로 주면 대소문자를 구분 안 하게 할 수 있다.

예 ) shakespeare라는 인덱스에서 speaker가 HAMLET이고, 대소문자의 구별은 없이 검색


```json
* dashboard에서의 GET 요청

GET shakespeare/_search 
{
  "query": {
    "term": {
      "speaker": {
        "value": "HAMLET",
        "case_insensitive": true
      }
    }
  }
}
```

```java
* java client에서의 SearchRequest

SearchRequest termSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
	.query(q -> q.term(t-> t.field("speaker").value(FieldValue.of("HAMLET")).caseInsensitive(true)))
	.build();
```

2. terms query: 여러 용어를 포함하는 문서를 검색. 지정된 용어 중 하나라도 일치하면 결과에 포함.

예) shakespeare라는 인덱스에서 line_id가 "61809" 또는 "61810"인 문서를 검색

```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "terms": {
      "line_id": [
        "61809",
        "61810"
      ]
    }
  }
}
```

```java
//java client에서의 SearchRequest

List<String> lineIds = List.of("61809", "61810");

TermsQueryField idTerms = new TermsQueryField.Builder()
	.value(lineIds.stream().map(FieldValue::of).collect(Collectors.toList()))
	.build();
				
SearchRequest termsSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
	.query(q -> q
		.bool(b -> b
			.should(shd -> shd
				.terms(t -> t
					.field("line_id")
					.terms(idTerms))
						)))
.build();

//또는 이렇게도 가능하다.
SearchRequest termsSearchRequest = SearchRequest.of(s -> s
	.index("shakespeare")
	.query(q -> q
	     .bool(b -> b
	        .should(shd -> shd
	            .terms(t -> t
	                .field("line_id")
	                .terms(idTerms)
	                    )
	                )
	          )
	    )
);
```

3. range query: 지정된 범위 내의 값을 가진 문서를 검색합니다. 주로 숫자나 날짜 필드에 사용

예) line_id 값이 >= 10, <= 20인 문서를 검색

```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "range": {
      "line_id": {
        "gte": 10,
        "lte": 20
      }
    }
  }
}
```

```java
//java client에서의 SearchRequest

SearchRequest rangeSearchRequest = new SearchRequest.Builder()
	.index(indexName)
		.query(q -> q
			.range(r -> r
				.field("line_id")
					.gte(JsonData.of(10))
					.lte(JsonData.of(20))
				)
			)
.build();
```
- 연산자의 뜻은 다음과 같다.
- gte : 크거나 같은 / gt : 큰
- lte : 작거나 같은 / lt : 작은

- 포맷을 지정할 수도 있다.

```json
* dashboard에서의 GET 요청

GET /products/_search
{
  "query": {
    "range": {
      "created": {
        "gte": "01/01/2022",
        "lte": "31/12/2022",
        "format":"dd/MM/yyyy"
      }
    }
  }
}
```

```java
//java client에서의 SearchRequest

SearchRequest rangeSearchRequest = new SearchRequest.Builder()
	.index("products")
		.query(q -> q
			.range(r -> r
				.field("graduation_year")
					.gte(JsonData.of("01/01/2022"))
					.lte(JsonData.of("31/12/2022"))
						.format("dd/MM/yyyy")
					)
				)
.build();
```

4. prefix query: 지정된 접두사로 시작하는 용어를 포함하는 문서를 검색

예) speaker 필드에 KING H로 시작하는 용어가 포함된 문서 검색

```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "prefix": {
      "speaker": {
        "value": "KING H"
      }
    }
  }
}
```

```java
//java client에서의 SearchRequest

SearchRequest prefixSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
	.query(q -> q
		.prefix(p -> p
			.field("speaker")
			.value("KING H")
				)
			)
.build();
```

5. wildcard query: 와일드카드 문자 (* 및 ?)를 사용하여 일치하는 패턴을 검색. 연산자는 다음과 같다.

- * : 0개 이상 문자와 일치
- ? : 단일 문자 일치
- case_insensitive : 대소문자 구분 여부. true면 구분하지 않으며 기본은 false(대소문자 구분)

```
'aabbcd' 라는 용어가 있다고 하면

* 의 경우에는

*   :  매칭성공
aabbbcd* : 매칭성공
b* : 매칭실패
a*e : 매칭실패

? 의 경우에는

a??bb???  :  매칭성공
a????cd : 매칭성공
a? : 매칭실패
aabbbcd? : 매칭실패
```

- 남용하면 속도 저하 발생. 특히, 검색어 맨 앞에는 사용하지 않는 것이 권장됨

예 ) H로 시작하고 Y로 끝나는 용어에 대한 대소문자 구분 검색

```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "wildcard": {
      "speaker": {
        "value": "H*Y",
        "case_insensitive": false
      }
    }
  }
}
```

```java
//java client에서의 SearchRequest

SearchRequest wildcardSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
		.query(q -> q
			.wildcard(w -> w
				.field("speaker")
				.value("H*Y")
				.caseInsensitive(false)
					)
				)
.build();
```

6. regexp query: 정규 표현식을 사용하여 일치하는 패턴을 검색

- Java와 동일한 java.util.regex 라이브러리의 정규식 문법을 기반으로 함.
- 단, 정규식의 최대 길이는 1,000 문자. 이 제한을 변경하려면 index.max_regex_length 설정을 업데이트할 필요가 있음.
- 와일드카드와 마찬가지로 성능 저하에 주의

예) 첫 글자는 아무 문자여도 상관 없고, 'amlet'이 이어지는 문자열이 있는 도큐먼트를 검색


```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "regexp": {
      "play_name": "[a-zA-Z]amlet"
    }
  }
}
```

```java
//java client에서의 SearchRequest

SearchRequest regexpSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
	.query(q -> q
		.regexp(r -> r
			.field("play_name")
			.value("[a-zA-Z]amlet")
				)
			)
.build();
```


---
## 참고 문서

- [[openSearch 쿼리의 종류와 Java Client에서의 구현 방법]]
- [[Full-text queries 의 종류]]