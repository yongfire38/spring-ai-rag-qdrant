package com.example.chat.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class WebController {

    /**
     * 채팅 페이지 제공
     */
    @GetMapping("/")
    public String chatPage() {
        return "chat";
    }
}
