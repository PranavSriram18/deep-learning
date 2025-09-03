import torch

def exercise0(x: torch.Tensor) -> torch.Tensor:
    """
    Write a function mask_future_positions(x: torch.Tensor) -> torch.Tensor that
    creates a lower triangular attention mask for a transformer.
    Specifically:

    Input x is a batch of sequences with shape (batch_size, sequence_length)
    Output should be a mask tensor of shape (batch_size, sequence_length, sequence_length)
    For each position i,j in the mask:

    If j ≤ i, mask[b,i,j] = 1.0 (can attend)
    If j > i, mask[b,i,j] = 0.0 (cannot attend)


    The mask should broadcast properly across the batch dimension
    Try to do it without using loops!
    """
    B, C = x.shape
    row_idxs = torch.arange(0, C).reshape(C, 1)
    col_idxs = torch.arange(0, C).reshape(1, C)
    diff = row_idxs - col_idxs  # CxC, (i,j) entry is i-j
    mask = torch.where(diff >= 0, 1, 0).unsqueeze(0)  # 1xCxC
    return mask.repeat(B, 1, 1)  # BxCxC

def exercise1(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Write a function gather_embeddings(
    embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor that:

    Takes a embedding matrix of shape (vocab_size, embedding_dim)
    Takes a tensor of indices of shape (batch_size, sequence_length)
    Returns a tensor of shape (batch_size, sequence_length, embedding_dim) 
    where each index has been replaced with its corresponding embedding vector
    """
    return embeddings[indices]

def exercise2(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Write a function batch_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor that computes a masked
    softmax over batched sequences where sequences might have different lengths.
    
    Specifically:

    x has shape (batch_size, sequence_length, sequence_length) and contains
    attention scores
    mask has shape (batch_size, sequence_length) and contains 1s for valid 
    positions and 0s for padding

    Return a tensor of same shape as x where:

    Each valid position (i,j) contains softmax probabilities only over valid 
    positions j. Padding positions should be set to 0 in the output
    The softmax for each query position i should only consider valid key 
    positions j
    The sum of probabilities for each query position should be 1 (where valid)
    """
    B, C, _ = x.shape

    query_mask = mask.unsqueeze(-1)    # B x C x 1
    key_mask = mask.unsqueeze(1)       # B x 1 x C
    attn_mask = query_mask * key_mask  # B x C x C

    logits = torch.where(attn_mask > 0, x, -torch.inf)  # apply mask to x. BxCxC
    sm = torch.nn.functional.softmax(logits, dim=-1)  # BxCxC
    out = torch.where(attn_mask > 0, sm, 0)  # set invalid positions to 0. BxCxC
    return out

def exercise3(seqs: torch.Tensor, lengths: torch.Tensor):
    """
    Given a batch of padded sequences and their true lengths:
    B, T = 3, 5
    sequences = torch.randn(B, T)  # padded sequences
    lengths = torch.tensor([3, 5, 2])  # true lengths of each sequence

    # Create a tensor containing only the valid elements (based on lengths)
    # Then reshape it so sequences are concatenated end-to-end
    # Expected shape: (sum(lengths),)
    """
    B, T = seqs.shape
    mask = torch.arange(0, T).unsqueeze(0).repeat(B, 1)  # B x T
    lengths = lengths.reshape(B, 1)
    mask = mask < lengths  # broadcast each length across the seq dimension. B x T
    return seqs[mask]  # automatically flattens to 1D

def exercise4(x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (B, N, N) containing B square matrices
    Create a function that extracts all triangular matrices (upper and lower)
    and stacks them in a specific way:
    - Upper triangles should come first, then lower triangles
    - Each triangle should be flattened
    - Result should have shape (2*B, (N*(N+1))//2)

    Args:
        x (torch.Tensor): Input of shape (B, N, N)
    Returns:
        torch.Tensor: Output of shape (2*B, (N*N)//2)
    """
    B, N, _ = x.shape
    row_idxs = torch.arange(0, N).reshape(N, 1)  # Nx1
    col_idxs = torch.arange(0, N).reshape(1, N)  # 1xN
    l_mask = row_idxs >= col_idxs  # NxN
    u_mask = col_idxs >= row_idxs  # NxN
    upper_elems = x[:, u_mask].reshape(B, -1)  # BxM, where M = N*(N+1)/2
    lower_elems = x[:, l_mask].reshape(B, -1)  # BxM
    return torch.cat([upper_elems, lower_elems], dim=0)  # 2BxM

def exercise5(x: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    """
    1. A tensor x of shape (B, N, N) containing B square matrices
    2. A tensor idxs of shape (B, K) containing K valid indices for each batch item
    
    Create a function that:
    1. For each batch item, extracts a KxK submatrix using the corresponding indices
    2. Masks out (sets to zero) elements below the diagonal in each submatrix
    3. Returns the result stacked in a new tensor of shape (B, K, K)
    """
    # TODO - check
    B, N, _ = x.shape
    _, K = idxs.shape

    batch_idx = torch.arange(0, B).unsqueeze(-1)  # Bx1
    row_idx = idxs.unsqueeze(-1)  # BxKx1
    col_idx = idxs.unsqueeze(1)  # Bx1xK

    # select rows 
    x_r = x[batch_idx, row_idx, :]  # B x K x N

    # select cols
    x_rc = x_r[batch_idx, :, col_idx]  # B x K x K


def exercise6(
        data: torch.Tensor, 
        segment_starts: torch.Tensor,
        segment_lens: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
    """
    Given:
    1. data: a tensor of shape (B, T, D) representing B sequences of length T with dimension D
    2. segment_starts: tensor of shape (B, S) containing S valid start indices for each batch
    3. segment_lens: tensor of shape (B, S) containing the length of each segment
    4. max_len: maximum length to extract for any segment (segments longer than this should be truncated)
    
    Task:
    Extract all segments for each batch into a single tensor, where each segment:
    - Starts at the given start index
    - Has the specified length (or max_len if shorter)
    - Preserves all D channels
    - Is padded with zeros if shorter than max_len
    
    Return:
    A tensor of shape (B*S, max_len, D) containing all segments
        Args:
            data: Input sequences (B, T, D)
            segment_starts: Start indices for each segment (B, S)
            segment_lens: Length of each segment (B, S)
            max_len: Maximum length to extract
        Returns:
            Tensor of shape (B*S, max_len, D) containing all segments
    """
    B, T, D = data.shape
    _, S = segment_starts.shape

    # construct the index for the batch dim
    batch_idx = torch.arange(B).unsqueeze(1)  # Bx1

    # adjust segment_lens to truncate to max_len
    segment_lens = torch.where(segment_lens <= max_len, segment_lens, max_len)

    # add a dummy entry to each sequence
    dummy = torch.zeros(B, 1, D)
    data = torch.cat([data, dummy], dim=1)  # Bx(T+1)xD

    # (i, j, k) entry of seq_idxs will be position of 
    # kth elem for jth segment for ith data point
    # given a segment_start position pos for batch b, and seq s, 
    # we want its last dim to look like
    # [pos, pos+1, ..., pos+segment_lens[b, s], T, T, ..., T]
    # the last several entries will allow us to access the dummy entry of the
    # sequence and hence pad with 0s. To do this we'll add an offset tensor
    # to the initial segment_starts tensor
    seq_idxs = segment_starts.reshape(B, S, 1).repeat(1, 1, max_len)

    offsets = torch.arange(max_len).reshape(1, 1, max_len)
    offsets = offsets.repeat(B, S, 1)  # B x S x max_len

    # Create a view of segment_lens that is B x S x max_len
    segment_lens = segment_lens.reshape(B, S, 1).expand(B, S, max_len)
    offsets = torch.where(offsets < segment_lens, offsets, T)

    # apply the offsets to seq_idxs, and squish indices > T to T
    seq_idxs += offsets  # B x S x max_len
    seq_idxs = torch.where(seq_idxs <= T, seq_idxs, T)
    seq_idxs = seq_idxs.reshape(B, -1)  # B x (S*max_len)

    slices = data[batch_idx, seq_idxs, :]  # B x (S*max_len) x D
    return slices.reshape(-1, max_len, D)  # (B*S) x max_len x D