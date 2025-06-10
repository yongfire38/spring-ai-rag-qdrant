---
제목: openSearch 쿼리의 종류와 Java Client에서의 구현 방법
날짜: 2024-05-29 14:04
태그(키워드):
  - opensearch
  - java
  - springboot
  - embedding
---
# openSearch 쿼리의 종류와 Java Client에서의 구현 방법

- openSearch에서 사용되는 쿼리의 종류와 Java Client 및 dashboard에서의 구현, 실행 방법을 기술한다.
- 많이 쓰일 만한 것만 정리하였으므로 더 자세한 사항은 공식 문서를 참조

## 분류

- 필터 컨텍스트와 쿼리 컨텍스트로 분류

## 필터 컨텍스트

> "문서가 쿼리 절과 일치하는지" -> 일치하는 문서를 반환
> 문서를 포함하거나 제외하는 데 사용되며, 점수 계산을 하지 않음 (따라서 성능이 더 빠름) / 캐싱을 통해 추가적인 성능 향상 가능
> 주로 결과를 제한하거나 전처리된 데이터를 필터링하는 데 사용

예 ) 2020-2022년에 우등으로 졸업한 학생을 검색 -> '학생의 우등 상태가 true로 설정되어 있는가? & 학생의 졸업_연도가 2020-2022년 범위인가?'


```json
* dashboard에서의 GET 요청

GET students/_search
{
  "query": { 
    "bool": { 
      "filter": [ 
        { "term":  { "honors": true }},
        { "range": { "graduation_year": { "gte": 2020, "lte": 2022 }}}
      ]
    }
  }
}
```


```java
//java client에서의 SearchRequest

TermQuery termQuery = TermQuery.of(t -> t
        .field("honors")
        .value(FieldValue.of(true))
    );

RangeQuery rangeQuery = RangeQuery.of(r -> r
		.field("graduation_year")
		.gte(JsonData.of(2020))
		.lte(JsonData.of(2022))
	);

BoolQuery boolQuery = BoolQuery.of(b -> b
		.filter(Query.of(q -> q.term(termQuery)))
		.filter(Query.of(q -> q.range(rangeQuery)))
	);

SearchRequest filterSearchRequest = SearchRequest.of(s -> s
        .index("students")
        .query(Query.of(q -> q.bool(boolQuery)))
    );
```

## 쿼리 컨텍스트

> "문서가 쿼리 절과 얼마나 잘 일치하는지" -> 각 문서의 관련성을 관련성 점수 형태로 제공

예 ) text_entry 필드에서 "long live king"이라는 단어와 일치하는 문서 검색


```json
* dashboard에서의 GET 요청

GET shakespeare/_search
{
  "query": {
    "match": {
      "text_entry": "long live king"
    }
  }
}
```


```java
//java client에서의 SearchRequest

SearchRequest textSearchRequest = new SearchRequest.Builder()
	.index("shakespeare")
	.query(q -> q.match(m -> m.field("text_entry").query(FieldValue.of("long live king"))))
	.build();
```

> 쿼리 컨텍스트는 다시 Term-level Queries와 Full-text queries로 나뉜다.
> 간단히 설명하면, Term-level Queries는 단어 레벨로 검색 / Full-text queries는 문장 레벨로 검색한다는 것

| 분류      | Term-level Queries                                | Full-text queries                      |
| :------ | :------------------------------------------------ | :------------------------------------- |
| 설명      | 쿼리와 일치하는 문서                                       | 쿼리와 얼마나 잘 일치하는지                        |
| 분석      | 검색어를 분석하지 않음(주어진 단어 그대로 검색)                       | 색인될 당시 특정 문서 필드에 사용된 것과 동일한 분석기로 분석됨   |
| 관련성     | 관련성 점수 기준으로 정렬하지 않음 (정확히는 모두 같은 점수로 나옴)           | 각 일치 항목에 대한 관련성 점수를 계산하고 결과를 정렬        |
| 사용되는 경우 | 숫자, 날짜 또는 태그와 같은 정확한 값을 일치시키고 관련성별로 정렬할 필요가 없는 경우 | 대소문자 및 어간 변형과 같은 요소를 고려한 후 정렬을 해야 할 경우 |

예를 들어, 셰익스피어 전체 작품이 인덱싱된 인덱스가 있다고 가정할 경우

Term-level 의 다음 질의 요청 결과는 0건이다.

```json
GET shakespeare/_search
{
  "query": {
    "term": {
      "text_entry": "To be, or not to be"
    }
  }
}

*질의 요청 결과는 0건*
{
  "took" : 3,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 0,
      "relation" : "eq"
    },
    "max_score" : null,
    "hits" : [ ]
  }
}
```

같은 단어를 Full-text queries인 match로 실행한 경우


```json
GET shakespeare/_search
{
  "query": {
    "match": {
      "text_entry": "To be, or not to be"
    }
  }
}

*매칭된 결과를 얻을 수 있다*
{
  "took" : 19,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 10000,
      "relation" : "gte"
    },
    "max_score" : 17.419369,
    "hits" : [
      {
        "_index" : "shakespeare",
        "_id" : "34229",
        "_score" : 17.419369,
        "_source" : {
          "type" : "line",
          "line_id" : 34230,
          "play_name" : "Hamlet",
          "speech_number" : 19,
          "line_number" : "3.1.64",
          "speaker" : "HAMLET",
          "text_entry" : "To be, or not to be: that is the question:"
        }
      },
      {
        "_index" : "shakespeare",
        "_id" : "109930",
        "_score" : 14.883024,
        "_source" : {
          "type" : "line",
          "line_id" : 109931,
          "play_name" : "A Winters Tale",
          "speech_number" : 23,
          "line_number" : "4.4.153",
          "speaker" : "PERDITA",
          "text_entry" : "Not like a corse; or if, not to be buried,"
        }
      },
      {
        "_index" : "shakespeare",
        "_id" : "103117",
        "_score" : 14.782743,
        "_source" : {
          "type" : "line",
          "line_id" : 103118,
          "play_name" : "Twelfth Night",
          "speech_number" : 53,
          "line_number" : "1.3.95",
          "speaker" : "SIR ANDREW",
          "text_entry" : "will not be seen; or if she be, its four to one"
        }
      }
    ]
  }
}
```

-> 그럼 Term-level Queries는 언제 쓰냐?
-> 다음과 같이 단순 일치 조회에 사용 (speaker가 HAMLET인 문서를 검색)

```json
GET shakespeare/_search
{
  "query": {
    "term": {
      "speaker": "HAMLET"
    }
  }
}
```

## Term-level Queries의 종류

[[Term-level Queries의 종류]]

## Full-text queries 의 종류

[[Full-text queries 의 종류]]


---

## 참조 문서
