use futures::StreamExt;
use tokio::sync::broadcast;

use crate::domain::events::ServerEvent;
use crate::services::tool::ToolDefinition;

use super::helpers::cache_real_input_tokens;
use super::types::{ContentBlock, ImageSource, Message, ToolUseBlock, StreamOnceResult};

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Responses API
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an Anthropic-format image source to an OpenAI `image_url` content part.
fn image_source_to_openai(source: &ImageSource) -> serde_json::Value {
    let url = match source {
        ImageSource::Base64 { media_type, data } => {
            format!("data:{media_type};base64,{data}")
        }
        ImageSource::Url { url } => url.clone(),
    };
    serde_json::json!({"type": "image_url", "image_url": {"url": url}})
}

/// Convert Anthropic tool_result content (may contain image blocks) to a plain string for OpenAI.
fn tool_result_to_string(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut parts = Vec::new();
        for block in arr {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block["text"].as_str() {
                        parts.push(t.to_string());
                    }
                }
                _ => {}
            }
        }
        return parts.join("\n");
    }
    content.to_string()
}

/// Convert our internal Message format to OpenAI Responses API input items.
pub(crate) fn messages_to_openai(system: &[&str], messages: &[Message]) -> (String, Vec<serde_json::Value>) {
    let instructions = system.join("\n\n");
    let mut input = Vec::new();

    for msg in messages {
        match msg {
            Message::User { content } => {
                for block in content {
                    match block {
                        ContentBlock::ToolResult { tool_use_id, content, .. } => {
                            // Function call output item
                            input.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": tool_result_to_string(content),
                            }));
                        }
                        ContentBlock::Text { text } => {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "user",
                                "content": text,
                            }));
                        }
                        ContentBlock::Image { source } => {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "user",
                                "content": [image_source_to_openai(source)],
                            }));
                        }
                        ContentBlock::Compaction { .. } => {
                            // Compaction blocks are opaque — pass through as-is
                            // (they were returned by the API in a previous response)
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant { content } => {
                // Collect text into a message item
                let mut text = String::new();
                for block in content {
                    match block {
                        ContentBlock::Text { text: t } => text.push_str(t),
                        ContentBlock::ToolUse { id, name, input: args } => {
                            // Flush any text before tool calls
                            if !text.is_empty() {
                                input.push(serde_json::json!({
                                    "type": "message",
                                    "role": "assistant",
                                    "content": text,
                                }));
                                text.clear();
                            }
                            // Function call item
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(args).unwrap_or_default(),
                            }));
                        }
                        ContentBlock::Compaction { .. } => {
                            // Compaction blocks passed through
                        }
                        _ => {}
                    }
                }
                if !text.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text,
                    }));
                }
            }
        }
    }

    (instructions, input)
}

/// Convert tool definitions to OpenAI Responses API format.
pub(crate) fn tools_to_openai(tool_defs: &[ToolDefinition], stream: bool) -> Vec<serde_json::Value> {
    let mut tools: Vec<serde_json::Value> = tool_defs.iter().map(|t| {
        serde_json::json!({
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "strict": false,
        })
    }).collect();

    // Add web search for streaming (interactive) requests
    if stream {
        tools.push(serde_json::json!({"type": "web_search"}));
    }

    tools
}

/// Non-streaming OpenAI Responses API call.
pub(crate) async fn openai_complete(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    system: &[&str],
    tool_defs: &[ToolDefinition],
    messages: &[Message],
    max_tokens: u64,
    base_url: &str,
) -> anyhow::Result<(String, Vec<ToolUseBlock>, String, u64)> {
    let (instructions, input) = messages_to_openai(system, messages);
    let tools = tools_to_openai(tool_defs, false);

    let mut body = serde_json::json!({
        "model": model,
        "max_output_tokens": max_tokens,
        "input": input,
        "store": false,
        "stream": false,
    });
    if !instructions.is_empty() {
        body["instructions"] = serde_json::Value::String(instructions);
    }
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(tools);
    }

    let resp = http
        .post(&format!("{base_url}/v1/responses"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let resp_text = resp.text().await?;
    if !status.is_success() {
        return Err(anyhow::anyhow!("OpenAI API error {status}: {resp_text}"));
    }

    let resp_json: serde_json::Value = serde_json::from_str(&resp_text)?;

    // Parse status → stop_reason
    let api_status = resp_json["status"].as_str().unwrap_or("completed");
    let stop_reason = match api_status {
        "incomplete" => {
            let reason = resp_json["incomplete_details"]["reason"].as_str().unwrap_or("");
            match reason {
                "max_output_tokens" => "max_tokens",
                _ => "end_turn",
            }.to_string()
        }
        _ => "end_turn".to_string(),
    };

    // Parse output items
    let mut text = String::new();
    let mut tool_uses = Vec::new();
    let mut has_function_calls = false;

    if let Some(output) = resp_json["output"].as_array() {
        for item in output {
            match item["type"].as_str() {
                Some("message") => {
                    if let Some(content) = item["content"].as_array() {
                        for part in content {
                            if part["type"].as_str() == Some("output_text") {
                                if let Some(t) = part["text"].as_str() {
                                    text.push_str(t);
                                }
                            }
                        }
                    }
                }
                Some("function_call") => {
                    has_function_calls = true;
                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let args_str = item["arguments"].as_str().unwrap_or("{}");
                    let input = serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                    tool_uses.push(ToolUseBlock { id: call_id, name, input });
                }
                _ => {}
            }
        }
    }

    // If there are function calls, set stop_reason to tool_use
    let stop_reason = if has_function_calls { "tool_use".to_string() } else { stop_reason };

    let input_tokens = resp_json["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = resp_json["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let cached_tokens = resp_json["usage"]["input_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let uncached_tokens = input_tokens.saturating_sub(cached_tokens);
    let tokens_used = (output_tokens as f64
        + uncached_tokens as f64 * 0.2
        + cached_tokens as f64 * 0.1) as u64;

    Ok((text, tool_uses, stop_reason, tokens_used))
}

/// Streaming OpenAI Responses API call.
pub(crate) async fn openai_stream(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    system: &[&str],
    tool_defs: &[ToolDefinition],
    messages: &[Message],
    max_tokens: u64,
    events: &broadcast::Sender<ServerEvent>,
    instance_slug: &str,
    chat_id: &str,
    message_id: &str,
    base_url: &str,
) -> anyhow::Result<StreamOnceResult> {
    let (instructions, input) = messages_to_openai(system, messages);
    let tools = tools_to_openai(tool_defs, true);

    let mut body = serde_json::json!({
        "model": model,
        "max_output_tokens": max_tokens,
        "input": input,
        "store": false,
        "stream": true,
        "context_management": [{"type": "compaction", "compact_threshold": 100000}],
    });
    if !instructions.is_empty() {
        body["instructions"] = serde_json::Value::String(instructions);
    }
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(tools);
    }

    let resp = http
        .post(&format!("{base_url}/v1/responses"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("OpenAI stream error {status}: {text}"));
    }

    let mut text = String::new();
    // Track function calls by output_index
    let mut fn_calls: std::collections::HashMap<usize, (String, String, String)> = std::collections::HashMap::new(); // idx -> (call_id, name, args)
    let mut stop_reason = String::new();
    let mut ordered_content: Vec<ContentBlock> = Vec::new();
    let mut current_fn_index: usize = 0;
    let mut tokens_used: u64 = 0;

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    const STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(480);

    loop {
        let chunk = tokio::time::timeout(STREAM_TIMEOUT, stream.next()).await;
        let chunk = match chunk {
            Ok(Some(Ok(c))) => c,
            Ok(Some(Err(e))) => return Err(e.into()),
            Ok(None) => break,
            Err(_) => break,
        };

        buf.extend_from_slice(&chunk);

        while let Some(newline_pos) = buf.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&buf[..newline_pos]).to_string();
            buf = buf[newline_pos + 1..].to_vec();

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Responses API uses "event: <type>" + "data: <json>" format
            // We need to track event types
            if line.starts_with("event:") {
                continue; // We parse from the data payload which has "type" field
            }

            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data == "[DONE]" { break; }
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(data) else { continue };

            let event_type = ev["type"].as_str().unwrap_or("");

            match event_type {
                // Text content streaming
                "response.output_text.delta" => {
                    if let Some(delta) = ev["delta"].as_str() {
                        text.push_str(delta);
                        let _ = events.send(ServerEvent::ChatStreamDelta {
                            instance_slug: instance_slug.to_string(),
                            chat_id: chat_id.to_string(),
                            message_id: message_id.to_string(),
                            delta: delta.to_string(),
                        });
                    }
                }

                // Function call arguments streaming
                "response.function_call_arguments.delta" => {
                    if let Some(delta) = ev["delta"].as_str() {
                        let idx = ev["output_index"].as_u64().unwrap_or(current_fn_index as u64) as usize;
                        current_fn_index = idx;
                        let entry = fn_calls.entry(idx).or_insert_with(|| (String::new(), String::new(), String::new()));
                        entry.2.push_str(delta);
                    }
                }

                // New output item added — capture function call metadata
                "response.output_item.added" => {
                    if let Some(item) = ev.get("item") {
                        let idx = ev["output_index"].as_u64().unwrap_or(0) as usize;
                        if item["type"].as_str() == Some("function_call") {
                            let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                            let name = item["name"].as_str().unwrap_or("").to_string();
                            fn_calls.insert(idx, (call_id, name, String::new()));

                            // Broadcast tool call start
                            let msg = crate::domain::chat::ChatMessage {
                                id: format!("fn_{}",
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap().as_millis()),
                                role: crate::domain::chat::ChatRole::Assistant,
                                content: format!("calling {}", item["name"].as_str().unwrap_or("function")),
                                created_at: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap().as_millis().to_string(),
                                kind: crate::domain::chat::MessageKind::ToolCall,
                                tool_name: item["name"].as_str().map(|s| s.to_string()),
                                mcp_app_html: None, mcp_app_input: None, model: None,
                            };
                            let _ = events.send(ServerEvent::ChatMessageCreated {
                                instance_slug: instance_slug.to_string(),
                                chat_id: chat_id.to_string(),
                                message: msg,
                            });
                        }
                        // Web search activity
                        if item["type"].as_str() == Some("web_search_call") {
                            let msg = crate::domain::chat::ChatMessage {
                                id: format!("ws_{}",
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap().as_millis()),
                                role: crate::domain::chat::ChatRole::Assistant,
                                content: "searching the web".to_string(),
                                created_at: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap().as_millis().to_string(),
                                kind: crate::domain::chat::MessageKind::ToolCall,
                                tool_name: Some("web_search".to_string()),
                                mcp_app_html: None, mcp_app_input: None, model: None,
                            };
                            let _ = events.send(ServerEvent::ChatMessageCreated {
                                instance_slug: instance_slug.to_string(),
                                chat_id: chat_id.to_string(),
                                message: msg,
                            });
                        }
                    }
                }

                // Response completed — extract final status and usage
                "response.completed" => {
                    if let Some(response) = ev.get("response") {
                        let api_status = response["status"].as_str().unwrap_or("completed");
                        stop_reason = match api_status {
                            "incomplete" => {
                                let reason = response["incomplete_details"]["reason"].as_str().unwrap_or("");
                                match reason {
                                    "max_output_tokens" => "max_tokens",
                                    _ => "end_turn",
                                }.to_string()
                            }
                            _ => "end_turn".to_string(),
                        };

                        // Check if output has function calls
                        if let Some(output) = response["output"].as_array() {
                            if output.iter().any(|item| item["type"].as_str() == Some("function_call")) {
                                stop_reason = "tool_use".to_string();
                            }
                        }

                        // Usage — includes prompt caching details
                        if let Some(usage) = response.get("usage") {
                            let input_t = usage["input_tokens"].as_u64().unwrap_or(0);
                            let output_t = usage["output_tokens"].as_u64().unwrap_or(0);
                            let cached_t = usage["input_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
                            let uncached_t = input_t.saturating_sub(cached_t);
                            log::info!(
                                "openai usage: input={} (cached={} uncached={}) output={}",
                                input_t, cached_t, uncached_t, output_t,
                            );
                            cache_real_input_tokens(instance_slug, chat_id, input_t);
                            // Normalize: cached input is ~50% cheaper on OpenAI
                            tokens_used = (output_t as f64
                                + uncached_t as f64 * 0.2
                                + cached_t as f64 * 0.1) as u64;
                        }
                    }
                }

                _ => {}
            }
        }
    }

    // Build tool use blocks from collected function calls
    let mut tool_uses: Vec<ToolUseBlock> = Vec::new();
    let mut indices: Vec<usize> = fn_calls.keys().copied().collect();
    indices.sort();
    for idx in indices {
        let (call_id, name, args_str) = &fn_calls[&idx];
        let input = serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
        tool_uses.push(ToolUseBlock { id: call_id.clone(), name: name.clone(), input });
    }

    if !text.is_empty() {
        ordered_content.push(ContentBlock::Text { text: text.clone() });
    }

    if stop_reason.is_empty() {
        stop_reason = if tool_uses.is_empty() { "end_turn" } else { "tool_use" }.to_string();
    }

    Ok(StreamOnceResult { text, tool_uses, stop_reason, tokens_used, ordered_content })
}
