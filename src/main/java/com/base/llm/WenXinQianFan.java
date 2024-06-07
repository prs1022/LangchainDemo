package com.base.llm;

import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.TypeReference;
import com.hw.langchain.chains.query.constructor.JsonUtils;
import com.hw.langchain.llms.base.BaseLLM;
import com.hw.langchain.requests.TextRequestsWrapper;
import com.hw.langchain.schema.GenerationChunk;
import com.hw.langchain.schema.LLMResult;
import lombok.Builder;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static java.util.Objects.requireNonNull;

/**
 * @author rensong.pu
 * @date 2024/6/7
 */

@SuperBuilder
public class WenXinQianFan extends BaseLLM {
    private static final Logger LOG = LoggerFactory.getLogger(WenXinQianFan.class);

    /**
     * Endpoint URL to use.
     */
    @Builder.Default
    private String endpointUrl = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=";

    /**
     * Max token allowed to pass to the model.
     */
    @Builder.Default
    private int maxToken = 4096;

    /**
     * LLM model temperature from 0 to 10.
     */
    @Builder.Default
    private float temperature = 0.95f;

    /**
     * History of the conversation
     */
    @Builder.Default
    private List<JSONObject> messages = new ArrayList<>();

    /**
     * Top P for nucleus sampling from 0 to 1
     */
    @Builder.Default
    private float topP = 1.0f;

    @Builder.Default
    private float penaltyScore=1.5f;

    @Builder.Default
    private boolean stream = false;

    @Builder.Default
    private String accessToken="";




    /**
     * Whether to use history or not
     */
    private boolean withHistory;

    private TextRequestsWrapper requestsWrapper;

    public WenXinQianFan init() {
        Map<String, String> headers = Map.of("Content-Type", "application/json");
        this.requestsWrapper = new TextRequestsWrapper(headers);
        return this;
    }


    @Override
    public String llmType() {
        return "wenxin";
    }


    public List<String> createStream(String prompt, List<String> stop) {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("content",prompt);
        jsonObject.put("role","user");
        messages.add(jsonObject);
        Map<String, Object> payload = Map.of(
                "temperature", temperature,
                "messages", messages,
                "max_length", maxToken,
                "stream",stream,
                "top_p", topP);
        LOG.debug("WenXin payload: {}", payload);
        String response = requestsWrapper.post(endpointUrl+this.accessToken, payload);
        LOG.debug("WenXin response: {}", response);
        return response.lines().toList();
    }
    @Override
    protected LLMResult innerGenerate(List<String> prompts, List<String> stop) {
        List<List<GenerationChunk>> generations = new ArrayList<>();

        for (String prompt : prompts) {
            GenerationChunk finalChunk = null;

            for (String streamResp : createStream(prompt, stop)) {
                if (StringUtils.isNotEmpty(streamResp)) {
                    GenerationChunk chunk = streamResponseToGenerationChunk(streamResp);
                    if (finalChunk == null) {
                        finalChunk = chunk;
                    } else {
                        finalChunk = finalChunk.add(chunk);
                    }
                }
            }
            generations.add(List.of(requireNonNull(finalChunk)));
        }
        return new LLMResult(generations);
    }

    public static GenerationChunk streamResponseToGenerationChunk(String streamResponse) {
        String replace = streamResponse.replace("data:", "");
        Map<String, Object> parsedResponse = JsonUtils.convertFromJsonStr(replace, new TypeReference<>() {});

        String text = (String) parsedResponse.getOrDefault("result", "");
        return new GenerationChunk(text, null);
    }
}