"""ToyGPT attention: a single causal self-attention head, and multi-head attention that runs several
heads in parallel, concatenates them and mixes them with an output projection (W_O).
Expects a config with d_model, n_heads, max_seq_len (e.g. GPTConfig)."""
import torch
from torch import nn


class AttentionHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    #since our q, k, v vectors are just basically different versions of our x vectors storing different info about the tokens,
    #we just make the weights that get applied to them nxn where n is our embedding dimension. Lets also say m is our d_head, the dimension
    #of each attention head. For single head attention, m = n, but in multiheaded attention we essentially split the information encoded
    #in each q, k, v across multiple heads. We can evenly distribute the dimension too and say m = n/num_heads.
    #Since x can be thought of as a matrix of seq_len*n dimension, where each vector x_i is an embedding for the ith token in our sequence,
    #we use the learned weights of n*m to get new vectors q, k, and v of seq_len*m, where qi, ki, and vi are the m*1 attention head
    #space q/k/v vectors of the ith token and just represent different information about these vectors.
    #The whole idea of learning our q k v weights is to better encode information in them. And remember, if we just have one attention head,
    #m=n, so our q k v weights are n*n, and our q k v matrices are seq_len*n with n*1 qi ki vi individual vectors

    self.dk = config.d_model // config.n_heads #m in the explanation above, we'll also need it as a scaling factor for attention calculation
    #to ensure values aren't too large

    self.wq = nn.Linear(config.d_model, self.dk, bias=False) #Just a cleaner way of doing W = torch.randn(d_model, d_head)
    #which also lets us call w(x) which outputs output = x*W.T + b where b is an optional bias. Most modern implementations don't use
    #a bias. GPT 2 specifically does, but for this we'll just turn it off.
    self.wk = nn.Linear(config.d_model, self.dk, bias=False)
    self.wv = nn.Linear(config.d_model, self.dk, bias=False)

    self.register_buffer('mask', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))) #mask init, will set to 0's
    self.dropout = nn.Dropout(0.1)#prevent memorizing train set
    #(created here rather than in forward() so model.eval() can actually switch it off)


  def forward(self, x):
    B, T, C = x.shape # Batch size, Sequence length, Embedding dimension
    q = self.wq(x) #sequence (query) vectors - looking at tokens in sequence
    k = self.wk(x) #context (key) vectors - how much our sequence tokens look at every other token
    v = self.wv(x) #value vectors - what parts of our words semantically do we need to emphasize, considering mixing with attention

    #compute attention
    pre_scores = q @ k.transpose(-2, -1) / self.dk**0.5
    #@ operator is dot product. This is our raw attention score matrix, which is how much a token qi from our sequence vector
    #attends to token kj from our context vector
    #this is one way of thinking about q and k: they are both matrices for each token in our sequence representing some amount of semantic information in a vector for each word,
    #but we can think of q as containing vectors for each token we're looking at, and k as vectors for each token as context to the q vectors
    #when training, we pass batches of sequences at a time, so each of our input x is actually going to have dimension batch_size*seq_len*d_model
    #so, we do transpose (-2,-1) to essentially just swap the last two dimensions, which will always be our seq_len and d_model (or d_head for q k v)
    #Therefore, raw_attn_i = vector for token qi, and raw_attn_ij = how much token qi attends to token kj
    #dk, our head dimension, is just a scaling factor

    pre_scores = pre_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
    #here we mask all the future tokens in the sequence that have 0 attention with '-inf' so that when we next take the softmax, our raw scores
    #actually give 0 for future tokens. This makes sense because a given token in a sequence should only depend on/attend to previous tokens

    raw_scores = nn.functional.softmax(pre_scores, dim=-1) #now we softmax to get the actual raw scores, which are our raw attention weights as a probability distribution
    #for each vector a_i in our attention weight matrix, the attentions scores should add up to 1, representing the distribution of
    #attention token i gives to each preceeding token j.

    #now that we know how much each token qi attends to tokens kj, and have some raw weights to assign that, we now can store
    #the actual token vector's information that we should be paying attention to. We will store this in what we call our value, or v vectors.
    #Similarly to q and k, v is also just another way of storing information for each token in our sequence. The difference is that this time
    #we know how much each token depends on others as indicated by our raw scores

    #when we computed the prescores, we took our q matrix (seq_len*d_k) and multiplied it by the transpose of our k matrix (d_k*seq_len).
    #After masking out future tokens and softmaxing to get a probability distribution, this left us with an attention matrix of
    #seq_len*seq_len, which makes sense because each a_i is an attention vector for each token i, and each a_ij represents how much that token
    #depends on token j. Now that we have these attention weights in our vocabulary/sequence space, we need to transform back into our embedding space.
    #This is because we need to actually "embed" the attentions that we learned for each of our seq_len tokens into the original embedding space for our
    #attention heads, d_k, so that these vectors actually represent the semantic information of each token.
    #So, generally speaking, when we have vectors in our embedding dimension d_model or our attention head's proportion of it, d_k,
    #it means we are looking at the semantics of the word as reflected in our embedding model, which captures token relationships in terms
    #of distance in a space. On the other hand, when we are looking at our vectors in terms of the vocabulary space or our sequence length,
    #it means we are looking at specifically relationships between tokens in the context of our sequence. This is the part that lets us
    #capture dependencies.
    raw_scores = self.dropout(raw_scores)

    attention_out = raw_scores @ v

    #when we get our final attention scores by multiplying our raw scores by our value vectors, we go back from the
    #vocabulary/sequence length space to our embedding dimension/attention head dimension (which is a proportion of our total embedding space).
    #So, two tokens who are normally not super similar in our embedding space based on the raw vectors from our embedding model
    #effectively become more similar after going through attention. As in two far away tokens in our embedding space become closer
    #after being transformed in attention and learning that they actually depend on each other in a sequence.
    #Basically the output attention matrix (seq_len*d_k) which contains vectors (d_k*1) for each token gives us
    #vectors for each token that remap it in almost a new version of our original embedding space
    #based on their dependencies on other words.
    #So, words that are dependent on each other become closer in our new "value" space which, represents semantic similarity, and
    #focuses on important learned semantic information that we actually need to convey
    #Remember, the attention output is really just a weighted sum on our value vector, ao attention_out_i = w_i0*v_0+w_i1*v_1+...
    #Where w is our raw_scores. So two words with similar raw scores are going to approach the same final attention vectors in attention_out.
    #And in this case similar raw scores means that the two tokens attend similarly to each other token in the sequence or strongly to each other
    #A good example is the sentence "the animal didn't cross the street because it was too tired".
    #The word it needs to resolve to animal, because "it" refers to "animal".
    #After attention, the representation of it gets heavily mixed with the representation of animal, pulling them close together in
    #the space even though their raw embeddings were probably quite different.
    #One important nuance though is that this new space that our final attention_out lives in, while it is a seq_len*d_k (or d_model for single head)
    #dimension matrix, the actual positions in this space don't map directly with our original embedding model. Rather, this new space,
    #which we call the value space, is a projection of our original embedding space via w_v, our value weight matrix, which learns to control
    #what information is important to extract from our attention matrix and random initialized values. So, when we mutliply x and w_v to get
    #v our value vectors, that w_v throughout training learns what parts to weigh more heavily from x so that after attention we are correctly
    #capturing the information we need to feed forward. So, when we have a trained w_v, we are able to select semantic meaning from our words
    #dependent on context capture by attention, which depends on our ability to train w_k and w_q to produce good q and k vectors that effectively
    #capture how much words in the sequence depend on each other. Eventaully, for multiheaded attention, we need to go back from d_k to d_model,
    #and we do a concat later to do that.

    return attention_out




class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_heads)])
    self.proj = nn.Linear(config.d_model, config.d_model) #parameters: W_O. Our output attention weight that
    #learns how to mix the outputs of each attention head. this is also a learned weight. We could technically not use it, but then our model
    #would be a lot less expressive. This helps combine the outputs of each attention head better. By placing a parameter here, we know
    #that any loss gradients that hit this weight will be directly in the process of combining attention heads, so it just optimizes on the
    #combination step.

  def forward(self, x):
    head_outputs = [head.forward(x) for head in self.heads]
    concat_out = torch.cat(head_outputs, dim=-1)
    #since we have batch_size * seq_len * d_k (d_head) outputs from each attention head, we want to concatenate outputs along the d_k axis
    #Since d_k = d_model/num_heads, we can just append all the values from each attention head output to the corresponding batch_size*seq_len along the axis d_k.
    #that way, we get d_k*num_heads entries per token, giving us a batch_size*seq_len*d_model final tensor with all of the information from each attention head
    #concatenated together.
    return self.proj(concat_out) #we are already in the correct dimensionality space before projecting, so we could mathematically
    #just return our concatenated output of attention. However, this would just be attention head 1, followed by head 2, head 3, and so on
    #for each token, which doesn't mix what each head learned at all. The idea of this projection layer W_O is that we can learn how to
    #combine each of our attention head outputs.
