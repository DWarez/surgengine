#include <string>

struct ModelConfig {
  int hidden_size;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  float attention_dropout;
  bool attention_bias;
  std::string _attn_implementation;

  ModelConfig(int hidden_size = 4096, int num_attention_heads = 32,
              int num_key_value_heads = 8, float attention_dropout = 0.0f,
              bool attention_bias = false,
              const std::string &attn_impl = "eager")
      : hidden_size(hidden_size), num_attention_heads(num_attention_heads),
        num_key_value_heads(num_key_value_heads),
        head_dim(hidden_size / num_attention_heads),
        attention_dropout(attention_dropout), attention_bias(attention_bias),
        _attn_implementation(attn_impl) {}
};