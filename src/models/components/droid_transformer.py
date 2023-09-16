"""Some classes to describe transformer architectures.
adapted from https://github.com/rodem-hep/pcdroid/blob/master/src/utils/transformers.py
"""

import math
from copy import deepcopy
from functools import partial
from typing import Callable, Mapping, Optional, Union

import torch as T
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention, softmax


def merge_masks(
    kv_mask: Union[T.BoolTensor, None],
    attn_mask: Union[T.BoolTensor, None],
    attn_bias: Union[T.Tensor, None],
    query: T.Size,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding information
    and the bias terms.

    New philosophy is just to define a kv_mask, and let the q_mask be
    ones. Let the padded nodes receive what they want! Their outputs
    dont matter and they don't add to computation anyway!!!
    """

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, expand the attention mask such that padded tokens
    # are never attended to
    if kv_mask is not None:
        merged_mask = kv_mask.unsqueeze(-2).expand(-1, query.shape[-2], -1)

    # If attention mask exists, create
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    # If the attention bias exists, convert to a float and add to the mask
    if attn_bias is not None:
        if merged_mask is not None:
            merged_mask = T.where(merged_mask, 0, -T.inf).type(query.dtype)
            merged_mask = merged_mask + attn_bias.permute(0, 3, 1, 2)
        else:
            merged_mask = attn_bias.permute(0, 3, 1, 2)

    return merged_mask


def my_scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: T.Tensor | None = None,
    attn_bias: T.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_act: callable = partial(softmax, dim=-1),
    pad_val: float = -float("inf"),
) -> T.Tensor:
    """Computes the scaled dot product attention using the given query, key,
    and value tensors.

    Parameters
    ----------
    query : T.Tensor
        The query tensor.
    key : T.Tensor
        The key tensor.
    value : T.Tensor
        The value tensor.
    attn_mask : T.Tensor | None, optional
        The attention mask tensor, by default None.
    dropout_p : float, optional
        The dropout probability, by default 0.0.
    is_causal : bool, optional
        Whether to use causal attention, by default False.
    attn_act : callable, optional
        The attention activation function, by default partial(softmax, dim=-1).
    pad_val : float, optional
        The padding value for the attention mask, by default -float("inf").

    Returns
    -------
    T.Tensor
        The result of the scaled dot product attention operation.

    Notes
    -----
    This function is a Pytorch equivalent operation of scaled_dot_product_attention but
    here we have freedom to use any activation we want as long as attn_act(pad_val) = 0.
    """

    # Get the shapes
    L = query.shape[-2]
    S = key.shape[-2]

    # Build the attention mask as a float
    if is_causal:
        attn_mask = T.ones(L, S, dtype=T.bool).tril(diagonal=0)
    elif attn_mask is not None and attn_mask.dtype == T.bool:
        attn_mask = attn_mask.float().masked_fill(~attn_mask, pad_val)
    else:
        attn_mask = 0.0

    # Apply the attention operation using the mask as a bias
    attn_weight = attn_act((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask)
    attn_weight = T.dropout(attn_weight, dropout_p, train=dropout_p > 0.0)

    return attn_weight @ value


class MultiHeadedAttentionBlock(nn.Module):
    """Generic Multiheaded Attention.

    Takes in three sequences with dim: (batch, sqeuence, features)
    - q: The primary sequence queries (determines output sequence length)
    - k: The attending sequence keys (determines incoming information)
    - v: The attending sequence values

    In a message passing sense you can think of q as your receiver nodes, v and k
    are the information coming from the sender nodes.

    When q == k(and v) this is a SELF attention operation
    When q != k(and v) this is a CROSS attention operation

    ===

    Block operations:

    1) Uses three linear layers to project the sequences.
    - q = q_linear * q
    - k = k_linear * k
    - v = v_linear * v

    2) Outputs are reshaped to add a head dimension, and transposed for matmul.
    - features = model_dim = head_dim * num_heads
    - dim becomes: batch, num_heads, sequence, head_dim

    3) Passes these through to the attention module (message passing)
    - In standard transformers this is the scaled dot product attention
    - Also takes additional dropout param to mask the attention

    4) Flatten out the head dimension and pass through final linear layer
    - Optional layer norm before linear layer using `do_layer_norm=True`
    - The output can also be zeroed on init using `init_zeros=True`
    - results are same as if attention was done seperately for each head and concat
    - dim: batch, q_seq, head_dim * num_heads
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
        drp: float = 0,
        init_zeros: bool = False,
        do_selfattn: bool = False,
        do_layer_norm: bool = False,
        attn_act: Callable | None = None,
    ) -> None:
        """
        Args:
            model_dim: The dimension of the model
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
            drp: The dropout probability used in the MHA operation
            init_zeros: If the final linear layer is initialised with zero weights
            do_selfattn: Only self attention should only be used if the
                q, k, v are the same, this allows slightly faster matrix multiplication
                at the beginning
            do_layer_norm: If a layernorm is applied before the output final linear
                projection (Only really needed with deep models)
        """
        super().__init__()

        # Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.do_selfattn = do_selfattn
        self.drp = drp
        self.do_layer_norm = do_layer_norm
        self.attn_act = attn_act

        # Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        # Initialise the weight matrices (only 1 for do self attention)
        if do_selfattn:
            self.all_linear = nn.Linear(model_dim, 3 * model_dim)
        else:
            self.q_linear = nn.Linear(model_dim, model_dim)
            self.k_linear = nn.Linear(model_dim, model_dim)
            self.v_linear = nn.Linear(model_dim, model_dim)

        # The optional (but advised) layer normalisation
        if do_layer_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

        # Set the output linear layer weights and bias terms to zero
        self.out_linear = nn.Linear(model_dim, model_dim)
        if init_zeros:
            self.out_linear.weight.data.fill_(0)
            self.out_linear.bias.data.fill_(0)

    def forward(
        self,
        q: T.Tensor,
        k: T.Tensor | None = None,
        v: T.Tensor | None = None,
        kv_mask: Optional[T.BoolTensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        attn_bias: T.Tensor | None = None,
    ) -> T.Tensor:
        """
        Args:
            q: The main sequence queries (determines the output length)
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
            attn_bias: Extra bias term for the attention matrix (eg: edge features)
        """

        # Store the batch size, useful for reshaping
        b_size, seq, feat = q.shape

        # If only q is provided then we automatically apply self attention
        if k is None:
            k = q
        if v is None:
            v = k

        # Work out the masking situation, with padding, no peaking etc
        merged_mask = merge_masks(kv_mask, attn_mask, attn_bias, q)

        # Generate the q, k, v projections
        if self.do_selfattn:
            q_out, k_out, v_out = self.all_linear(q).chunk(3, -1)
        else:
            q_out = self.q_linear(q)
            k_out = self.k_linear(k)
            v_out = self.v_linear(v)

        # Break final dim, transpose to get dimensions: B,H,Seq,Hdim
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q_out = q_out.view(shape).transpose(1, 2)
        k_out = k_out.view(shape).transpose(1, 2)
        v_out = v_out.view(shape).transpose(1, 2)

        # Calculate the new sequence values
        if self.attn_act:
            a_out = my_scaled_dot_product_attention(
                q_out,
                k_out,
                v_out,
                attn_mask=merged_mask,
                dropout_p=self.drp if self.training else 0,
                attn_act=self.attn_act,
            )
        else:
            a_out = scaled_dot_product_attention(
                q_out,
                k_out,
                v_out,
                attn_mask=merged_mask,
                dropout_p=self.drp if self.training else 0,
            )

        # Concatenate the all of the heads together to get shape: B,Seq,F
        a_out = a_out.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        # Pass through the optional normalisation layer
        if self.do_layer_norm:
            a_out = self.layer_norm(a_out)

        # Pass through final linear layer
        return self.out_linear(a_out)


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style
    arcitecture.

    We choose a cross between Normformer and FoundationTransformers as they have often
    proved to be the most stable to train
    https://arxiv.org/abs/2210.06423
    https://arxiv.org/abs/2110.09456

    It contains:
    - Multihead(self)Attention block
    - A dense network

    Layernorm is applied before each operation
    Residual connections are used to bypass each operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Mapping | None = None,
        dense_config: Mapping | None = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimension of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(model_dim, do_selfattn=True, **mha_config)
        self.dense = DenseNetwork(model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config)

        # The pre MHA and pre FFN layer normalisations
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.self_attn(
            self.norm1(x), kv_mask=mask, attn_mask=attn_mask, attn_bias=attn_bias
        )
        x = x + self.dense(self.norm2(x), ctxt)
        return x


class TransformerCrossAttentionLayer(nn.Module):
    """A transformer cross attention layer.

    It contains:
    - cross-attention-block
    - A feed forward network

    Does not allow for attn masks/biases
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Mapping | None = None,
        dense_config: Mapping | None = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimension of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.cross_attn = MultiHeadedAttentionBlock(model_dim, do_selfattn=False, **mha_config)
        self.dense = DenseNetwork(model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config)

        # The two pre MHA and pre FFN layer normalisations
        self.norm0 = nn.LayerNorm(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        kv_mask: Optional[T.BoolTensor] = None,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        q_seq = q_seq + self.cross_attn(self.norm1(q_seq), self.norm0(kv_seq), kv_mask=kv_mask)
        q_seq = q_seq + self.dense(self.norm2(q_seq), ctxt)

        return q_seq


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final
    normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Mapping | None = None,
        dense_config: Mapping | None = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)


class FullTransformerEncoder(nn.Module):
    """A transformer encoder with added input and output embedding networks.

    Sequence -> Sequence
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        edge_dim: int = 0,
        ctxt_dim: int = 0,
        te_config: Mapping | None = None,
        node_embd_config: Mapping | None = None,
        outp_embd_config: Mapping | None = None,
        edge_embd_config: Mapping | None = None,
        ctxt_embd_config: Mapping | None = None,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of of the final output vector
            edge_dim: Dim. of the input edge features
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            te_config: Keyword arguments to pass to the TVE constructor
            node_embd_config: Keyword arguments for node dense embedder
            outp_embd_config: Keyword arguments for output dense embedder
            edge_embd_config: Keyword arguments for edge dense embedder
            ctxt_embd_config: Keyword arguments for context dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim
        te_config = deepcopy(te_config) or {}
        node_embd_config = deepcopy(node_embd_config) or {}
        outp_embd_config = deepcopy(outp_embd_config) or {}
        edge_embd_config = deepcopy(edge_embd_config) or {}

        # By default we would like the dense networks in this model to double the width
        if "model_dim" in te_config.keys():
            model_dim = te_config["model_dim"]
            if "hddn_dim" not in node_embd_config.keys():
                node_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in ctxt_embd_config.keys():
                ctxt_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in outp_embd_config.keys():
                outp_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in te_config["dense_config"].keys():
                te_config["dense_config"]["hddn_dim"] = 2 * model_dim

        # Initialise the context embedding network (optional)
        if self.ctxt_dim:
            self.ctxt_emdb = DenseNetwork(
                inpt_dim=self.ctxt_dim,
                **ctxt_embd_config,
            )
            self.ctxt_out = self.ctxt_emdb.outp_dim
        else:
            self.ctxt_out = 0

        # Initialise the TVE, the main part of this network
        self.te = TransformerEncoder(**te_config, ctxt_dim=self.ctxt_out)
        self.model_dim = self.te.model_dim

        # Initialise all embedding networks
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

        # Initialise the edge embedding network (optional)
        if self.edge_dim:
            self.edge_embd = DenseNetwork(
                inpt_dim=self.edge_dim,
                outp_dim=self.te.layers[0].self_attn.num_heads,
                ctxt_dim=self.ctxt_out,
                **edge_embd_config,
            )

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        if self.edge_dim:
            attn_bias = self.edge_embd(attn_bias, ctxt)
        x = self.node_embd(x, ctxt)
        x = self.te(x, mask=mask, ctxt=ctxt, attn_bias=attn_bias, attn_mask=attn_mask)
        x = self.outp_embd(x, ctxt)
        return x


class CrossAttentionEncoder(nn.Module):
    """A type of encoder which includes uses cross attention to move to and
    from the original sequence. Self attention is used in the learned sequence
    steps.

    Sequence -> Squence

    It is non resizing, so model_dim must be used for inputs and outputs
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_tokens: int = 4,
        num_layers: int = 5,
        mha_config: Mapping | None = None,
        dense_config: Mapping | None = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature size for input, output, and all intermediate sequences
            num_tokens: The number of global tokens to use,
            num_layers: Number of there and back cross attention layers
            mha_config: Keyword arguments for all multiheaded attention layers
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_tokens = num_tokens

        # Initialise the learnable perceiver tokens as random values
        self.global_tokens = nn.Parameter(T.randn((1, num_tokens, model_dim)))

        # The cross attention layers going from our original sequence
        self.from_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )

        # The cross attention layers going to our original sequence
        self.to_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        seq: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: T.Tensor | None = None,
    ) -> Union[T.Tensor, tuple]:
        """Pass the input through all layers sequentially."""

        # Make sure the class tokens are expanded to batch size
        # Use shape not len as it is ONNX safe!
        global_tokens = self.global_tokens.expand(seq.shape[0], self.num_tokens, self.model_dim)

        # Pass through the layers of there and back cross attention
        for from_layer, to_layer in zip(self.from_layers, self.to_layers):
            global_tokens = from_layer(global_tokens, seq, mask, ctxt)
            seq = to_layer(seq, global_tokens, None, ctxt)

        return seq


class FullCrossAttentionEncoder(nn.Module):
    """A cross attention encoder with added input and output embedding
    networks.

    Sequence -> Sequence
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        cae_config: Mapping | None = None,
        node_embd_config: Mapping | None = None,
        outp_embd_config: Mapping | None = None,
        ctxt_embd_config: Mapping | None = None,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of each element of output sequence
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            cae_config: Keyword arguments to pass to the CrossAttentionEncoder
            node_embd_config: Keyword arguments for node dense embedder
            outp_embd_config: Keyword arguments for output dense embedder
            ctxt_embd_config: Keyword arguments for context dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        cae_config = deepcopy(cae_config) or {}
        node_embd_config = deepcopy(node_embd_config) or {}
        outp_embd_config = deepcopy(outp_embd_config) or {}

        # By default we would like the dense networks in this model to double the width
        if "model_dim" in cae_config.keys():
            model_dim = cae_config["model_dim"]
            if "hddn_dim" not in node_embd_config.keys():
                node_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in ctxt_embd_config.keys():
                ctxt_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in outp_embd_config.keys():
                outp_embd_config["hddn_dim"] = 2 * model_dim
            if "hddn_dim" not in cae_config["dense_config"].keys():
                cae_config["dense_config"]["hddn_dim"] = 2 * model_dim

        # Initialise the context embedding network (optional)
        if self.ctxt_dim:
            self.ctxt_emdb = DenseNetwork(
                inpt_dim=self.ctxt_dim,
                **ctxt_embd_config,
            )
            self.ctxt_out = self.ctxt_emdb.outp_dim
        else:
            self.ctxt_out = 0

        # Initialise the TVE, the main part of this network
        self.cae = CrossAttentionEncoder(**cae_config, ctxt_dim=self.ctxt_out)
        self.model_dim = self.cae.model_dim

        # Initialise all embedding networks
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        x = self.node_embd(x, ctxt)
        x = self.cae(x, mask=mask, ctxt=ctxt)
        x = self.outp_embd(x, ctxt)
        return x


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock.

        Parameters
        ----------
        inpt_dim : int
            The number of features for the input layer
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        n_layers : int, optional
            The number of transform layers in this block, by default 1
        act : str, optional
            A string indicating the name of the activation function, by default "lrlu"
        nrm : str, optional
            A string indicating the name of the normalisation, by default "none"
        drp : float, optional
            The dropout probability, 0 implies no dropout, by default 0
        do_res : bool, optional
            Add to previous output, only if dim does not change, by default 0
        init_zeros : bool, optional,
            If the final layer weights and bias values are set to zero
        """
        super().__init__()

        # Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # If this layer includes an additive residual connection
        self.do_res = do_res and (inpt_dim == outp_dim)

        # Initialise the block layers as a module list
        self.block = nn.ModuleList()
        for n in range(n_layers):
            # Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            # Linear transform, activation, normalisation, dropout
            self.block.append(nn.Linear(lyr_in, outp_dim))

            # Initialise the final layer with zeros
            if init_zeros and n == n_layers - 1:
                self.block[-1].weight.data.fill_(0)
                self.block[-1].bias.data.fill_(0)

            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        # Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError("Was expecting contextual information but none has been provided!")
        temp = T.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt

        # Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        # Add the original inputs again for the residual connection
        if self.do_res:
            temp = temp + inpt

        return temp

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the
        block."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and
    context injection layers."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        drp_on_output: bool = False,
        nrm_on_output: bool = False,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        output_init_zeros: bool = False,
    ) -> None:
        """Initialise the DenseNetwork.

        Parameters
        ----------
        inpt_dim : int
            The number of input neurons
        outp_dim : int, optional
            The number of output neurons. If none it will take from inpt or hddn,
            by default 0
        ctxt_dim : int, optional
            The number of context features. The context feature use is determined by
            ctxt_type, by default 0
        hddn_dim : Union[int, list], optional
            The width of each hidden block. If a list it overides depth, by default 32
        num_blocks : int, optional
            The number of hidden blocks, can be overwritten by hddn_dim, by default 1
        n_lyr_pbk : int, optional
            The number of transform layers per hidden block, by default 1
        act_h : str, optional
            The name of the activation function to apply in the hidden blocks,
            by default "lrlu"
        act_o : str, optional
            The name of the activation function to apply to the outputs,
            by default "none"
        do_out : bool, optional
            If the network has a dedicated output block, by default True
        nrm : str, optional
            Type of normalisation (layer or batch) in each hidden block, by default "none"
        drp : float, optional
            Dropout probability for hidden layers (0 means no dropout), by default 0
        do_res : bool, optional
            Use resisdual-connections between hidden blocks (only if same size),
            by default False
        ctxt_in_inpt : bool, optional
            Include the ctxt tensor in the input block, by default True
        ctxt_in_hddn : bool, optional
            Include the ctxt tensor in the hidden blocks, by default False
        output_init_zeros : bool, optional
            Initialise the output layer weights as zeros

        Raises
        ------
        ValueError
            If the network was given a context input but both ctxt_in_inpt and
            ctxt_in_hddn were False
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_hddn and not ctxt_in_inpt:
                raise ValueError("Network has context inputs but nowhere to use them!")

        # We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        # Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h,
            nrm=nrm,
            drp=drp,
        )

        # All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm,
                        drp=drp,
                        do_res=do_res,
                    )
                )

        # Output block
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
                outp_dim=self.outp_dim,
                act=act_o,
                init_zeros=output_init_zeros,
                nrm=nrm if nrm_on_output else "none",
                drp=drp if drp_on_output else 0,
            )

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Pass through all layers of the dense network."""

        # Reshape the context if it is available. Equivalent to performing
        # multiple ctxt.unsqueeze(1) until the dim matches the input.
        # Batch dimension is kept the same.
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        # Pass through each hidden block
        for h_block in self.hidden_blocks:  # Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  # block was initialised with a ctxt dim

        # Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i+1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.outp_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.outp_dim)
        return string
