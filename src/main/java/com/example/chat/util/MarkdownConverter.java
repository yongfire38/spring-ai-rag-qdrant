package com.example.chat.util;

import com.vladsch.flexmark.html.HtmlRenderer;
import com.vladsch.flexmark.parser.Parser;
import com.vladsch.flexmark.util.ast.Node;
import com.vladsch.flexmark.util.data.MutableDataSet;

import lombok.extern.slf4j.Slf4j;

/**
 * 마크다운을 HTML로 변환하는 유틸리티 클래스
 */
@Slf4j
public class MarkdownConverter {

    private static final Parser PARSER;
    private static final HtmlRenderer RENDERER;

    static {
        MutableDataSet options = new MutableDataSet();
        // 필요한 옵션 설정
        options.set(Parser.EXTENSIONS, Parser.EXTENSIONS.get(options));

        PARSER = Parser.builder(options).build();
        RENDERER = HtmlRenderer.builder(options).build();
    }

    /**
     * 마크다운 텍스트를 HTML로 변환합니다.
     * 
     * @param markdown 마크다운 텍스트
     * @return 변환된 HTML
     */
    public static String convertToHtml(String markdown) {
        if (markdown == null || markdown.isEmpty()) {
            return "";
        }

        // 디버깅 로그
        log.info("원본 응답 길이: " + markdown.length());
        log.info("원본 응답 시작: " + markdown.substring(0, Math.min(200, markdown.length())));
        log.info("원본 응답 끝: " + markdown.substring(Math.max(0, markdown.length() - 200)));

        // 응답에서 HTML 태그 제거 - 직접 마크다운으로 처리
        if (markdown.contains("<p>") || markdown.contains("</p>") || markdown.contains("<h3>")) {
            markdown = removeHtmlTags(markdown);
            log.info("HTML 태그 제거 후: " + markdown.substring(0, Math.min(100, markdown.length())));
        }

        // Generation[assistantMessage=AssistantMessage [...] 형태 처리
        if (markdown.contains("Generation[") && markdown.contains("AssistantMessage")) {
            log.info("Generation 형태 발견");

            // 실제 컨텐츠 추출 - textContent= 부분 찾기
            int contentStart = markdown.indexOf("textContent=");
            if (contentStart != -1) {
                contentStart += 12; // "textContent=" 길이
                log.info("textContent 발견 위치: " + contentStart);

                // 마지막 부분 찾기 (metadata= 또는 ]], 또는 ]]</p>)
                int contentEnd = -1;
                int metadataIndex = markdown.lastIndexOf(", metadata=");
                int closeBracketIndex = markdown.lastIndexOf("]]</p>");
                int simpleBracketIndex = markdown.lastIndexOf("]]" + '"');
                int finishReasonIndex = markdown.lastIndexOf("finishReason");

                if (metadataIndex != -1) {
                    contentEnd = metadataIndex;
                    log.info("metadata= 마커 발견");
                } else if (closeBracketIndex != -1) {
                    contentEnd = closeBracketIndex;
                    log.info("]]</p> 마커 발견");
                } else if (simpleBracketIndex != -1) {
                    contentEnd = simpleBracketIndex;
                    log.info("]] 마커 발견");
                } else if (finishReasonIndex != -1) {
                    contentEnd = finishReasonIndex - 10; // 안전하게 앞쪽으로 약간 여유 두기
                    log.info("finishReason 마커 발견");
                }

                if (contentEnd != -1) {
                    markdown = markdown.substring(contentStart, contentEnd);
                    log.info("추출된 컨텐츠 길이: " + markdown.length());
                    log.info("추출된 컨텐츠 시작: " + markdown.substring(0, Math.min(100, markdown.length())));
                }
            }
        }

        // HTML 태그 제거 (응답에 있는 <p>, <h3> 등의 태그)
        markdown = removeHtmlTags(markdown);

        // <think> 태그 제거 (모델의 내부 사고 과정 제거)
        markdown = removeThinkTag(markdown);

        // 마크다운 파서 적용
        Node document = PARSER.parse(markdown);
        return RENDERER.render(document);
    }

    /**
     * <think> 태그와 그 내용을 제거합니다.
     * 
     * @param text 원본 텍스트
     * @return <think> 태그가 제거된 텍스트
     */
    private static String removeThinkTag(String text) {
        // <think>로 시작하고 </think>로 끝나는 부분 제거 (여러 줄에 걸쳐 있는 경우도 처리)
        return text.replaceAll("(?s)<think>.*?</think>", "").trim();
    }

    /**
     * HTML 태그를 마크다운 형식으로 변환합니다.
     * 
     * @param text HTML 태그가 포함된 텍스트
     * @return HTML 태그가 마크다운으로 변환된 텍스트
     */
    private static String removeHtmlTags(String text) {
        // HTML 태그를 마크다운 형식으로 변환
        return text.replaceAll("<p>", "\n\n")
                .replaceAll("</p>", "")
                .replaceAll("<hr ?/>", "---\n")
                .replaceAll("<br ?/>", "\n")
                .replaceAll("<h3>", "### ")
                .replaceAll("</h3>", "\n")
                .replaceAll("<ul>", "")
                .replaceAll("</ul>", "")
                .replaceAll("<li>", "* ")
                .replaceAll("</li>", "\n")
                .replaceAll("<strong>", "**")
                .replaceAll("</strong>", "**")
                .replaceAll("<code>", "```")
                .replaceAll("</code>", "```")
                .replaceAll("<pre>", "")
                .replaceAll("</pre>", "")
                .trim();
    }
}
