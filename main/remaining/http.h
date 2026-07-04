// serving/nova_server_http.h
#ifndef NOVA_SERVER_HTTP_H
#define NOVA_SERVER_HTTP_H

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>

namespace nova {

struct CompletionRequest {
    std::string model;
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream = false;
    float repetition_penalty = 1.0f;
    std::vector<std::string> stop;
    int seed = 42;
    bool echo = false;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::vector<std::string> logit_bias;
};

struct ChatRequest {
    std::string model;
    std::vector<std::map<std::string, std::string>> messages;
    int max_tokens = 100;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream = false;
    std::vector<std::string> stop;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::vector<std::map<std::string, std::string>> tools;
    std::string tool_choice = "auto";
};

struct CompletionResponse {
    std::string id;
    std::string object;
    int created;
    std::string model;
    std::vector<std::map<std::string, std::string>> choices;
    std::map<std::string, int> usage;
};

class HTTPServer {
public:
    HTTPServer();
    ~HTTPServer();

    void init(int port = 8080, int workers = 4);
    void start();
    void stop();

    void set_completion_handler(std::function<CompletionResponse(const CompletionRequest&)> handler);
    void set_chat_handler(std::function<CompletionResponse(const ChatRequest&)> handler);

private:
    int port_;
    int workers_;
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;

    std::function<CompletionResponse(const CompletionRequest&)> completion_handler_;
    std::function<CompletionResponse(const ChatRequest&)> chat_handler_;

    void worker_loop(int worker_id);
    void handle_client(int client_fd);
    std::string handle_request(const std::string& request);
    CompletionResponse handle_completion(const CompletionRequest& req);
    CompletionResponse handle_chat(const ChatRequest& req);

    std::string parse_request(const std::string& raw);
    CompletionRequest parse_completion_request(const std::string& body);
    ChatRequest parse_chat_request(const std::string& body);
    std::string serialize_response(const CompletionResponse& resp);
};

} // namespace nova