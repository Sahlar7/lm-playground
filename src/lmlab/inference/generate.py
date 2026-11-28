import torch


def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0):
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
            next_id = torch.multinomial(probs, num_samples=1).item()

            # append to sequence and loop
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    # decode the full sequence
    return tokenizer.decode(ids[0].tolist())
