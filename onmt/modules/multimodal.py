""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable

import onmt
import onmt.inputters
from onmt.utils.misc import aeq, sequence_mask
from onmt.utils.loss import NMTLossCompute, shards

class MultiModalNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, bridge, decoder, imgw=False):
        super(MultiModalNMTModel, self).__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder
        self.imgw = imgw

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, img_feats=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        assert img_feats is not None
        dec_in = tgt[:-1]  # exclude last target from inputs

        if self.imgw:
            enc_state, memory_bank, lengths = self.encoder(src, img_feats, lengths=lengths)
            # expand indices to account for image "word"
            src = torch.cat([src[0:1,:,:], src], dim=0)
        else:
            enc_state, memory_bank, lengths = self.encoder(src, lengths=lengths)
        if self.bridge is not None:
            memory_bank = self.bridge(memory_bank, img_feats, src.size(0))

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)


class MultiModalMemoryBankGate(nn.Module):
    def __init__(self, bank_size, img_feat_size, add=0):
        super(MultiModalMemoryBankGate, self).__init__()
        self.bank_to_gate = nn.Linear(
            bank_size, bank_size, bias=False)
        self.feat_to_gate = nn.Linear(
            img_feat_size, bank_size, bias=True)
        #nn.init.constant_(self.feat_to_gate.bias, 1.0) # newer pytorch
        nn.init.constant_(self.feat_to_gate.bias, 1.0)
        self.add = add

    def forward(self, bank, img_feats, n_time):
        feat_to_gate = self.feat_to_gate(img_feats)
        feat_to_gate = feat_to_gate.expand(n_time, -1, -1)
        bank_to_gate = self.bank_to_gate(bank)
        gate = torch.sigmoid(feat_to_gate + bank_to_gate) + self.add
        gate = gate / (1. + self.add)
        return bank * gate

class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator, img_feat_size, add=0, use_hidden=False):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        self.vocab_size = self.linear.weight.size(0)
        self.gate = nn.Linear(img_feat_size, self.vocab_size, bias=True)
        #nn.init.constant_(self.gate.bias, 1.0) # newer pytorch
        nn.init.constant_(self.gate.bias, 1.0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.add = add
        self.use_hidden = use_hidden
        if use_hidden:
            self.hidden_to_gate = nn.Linear(
                self.linear.weight.size(1), self.vocab_size, bias=False)

    def forward(self, hidden, img_feats, n_time):
        proj = self.linear(hidden)
        pre_sigmoid = self.gate(img_feats)
        if self.use_hidden:
            pre_sigmoid = pre_sigmoid.repeat(n_time, 1)
            hidden_to_gate = self.hidden_to_gate(hidden)
            gate = torch.sigmoid(pre_sigmoid + hidden_to_gate) + self.add
        else:
            gate = torch.sigmoid(pre_sigmoid) + self.add
            gate = gate.repeat(n_time, 1)
        gate = gate / (1. + self.add)
        return self.logsoftmax(proj * gate)

class MultiModalLossCompute(NMTLossCompute):
    def __call__(self,
                 batch,
                 output,
                 attns,
                 img_feats,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state, img_feats=img_feats)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard, img_feats=img_feats)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _compute_loss(self, batch, output, target, img_feats, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):
        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output, img_feats, output.size(0))

        ### Copypasta from superclass
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)
            loss += align_loss
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats
        ### Copypasta ends






       
