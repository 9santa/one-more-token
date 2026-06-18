# Stage 1
prompt string
  - tokenize
  - mock model produces logits
  - sampler picks next token
  - append token
  - repeat until EOS or max_new_tokens

### Naive autoregressive loop:
$$x_{1:t} → model → logits_{s_t} → sampler → x_{t+1}$$


# Stage 2
single request
  -> multiple request states

### What changed
Before:
for step in maxNewTokens:
    generate one token for one request

Now:
while there are active requests:
    for each active request:
        generate one token
        maybe finish it


# Stage 3
batched decode
we want:
Batch batch = build from active sequences
allLogits = model.forwardNextTokenBatch(batch.contexts)

for each row in batch:
    sample token
    update corresponding sequence


# Stage 4
scheduler + v1 continuous/in-flight batching


# Stage 5
right now every steps ends the full context: prompt + generated_tokens to the model

but we want serving like this:
Prefill:
  process the whole prompt once
  produce logits for the first generated token
  create/fill KV cache

Decode:
  process only the last generated token
  reuse KV cache
  produce logits for the next token




