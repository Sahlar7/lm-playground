import torch


def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0, top_k=None, top_p=None):
    model.eval()  # turn off dropout etc if you add them later
    device = next(model.parameters()).device

    # encode the prompt
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    with torch.no_grad():  # no gradients needed for inference
        for _ in range(max_new_tokens):
            # crop to max_seq_len if needed
            ids_cropped = ids[:, -model.config.max_seq_len:]

            # forward pass
            logits = model(ids_cropped)  # (1, seq_len, vocab_size)

            # take only the last position's logits — that's the next token prediction
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # greedy — just pick the highest scoring token
            #next_id = torch.argmax(next_logits).item()
            #temperature sampling:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            if top_k is not None:
                top_k = min(top_k, probs.size(-1))
                top_k_probs, top_k_indices = torch.topk(probs, top_k) #torch topk returns top k (second arg) values from the provided tensor (first arg)
                probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs) # scatter_ fills the tensor with values from top_k_probs at the indices specified by top_k_indices, and sets all other positions to zero
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0

            next_id = torch.multinomial(probs, num_samples=1).item()

            # append to sequence and loop
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    # decode the full sequence
    return tokenizer.decode(ids[0].tolist())
