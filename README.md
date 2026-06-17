# Stage 1
prompt string
  -> tokenize
  -> mock model produces logits
  -> sampler picks next token
  -> append token
  -> repeat until EOS or max_new_tokens

### Naive autoregressive loop:
$$x_{1:t} → model → logits_{s_t} → sampler → x_{t+1}$$

