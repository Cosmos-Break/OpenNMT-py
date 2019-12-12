""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable

import onmt
import onmt.inputters
from onmt.utils.misc import aeq, sequence_mask
from onmt.utils.loss import NMTLossCompute

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
        nn.init.constant(self.feat_to_gate.bias, 1.0)
        self.add = add

    def forward(self, bank, img_feats, n_time):
        feat_to_gate = self.feat_to_gate(img_feats)
        feat_to_gate = feat_to_gate.expand(n_time, -1, -1)
        bank_to_gate = self.bank_to_gate(bank)
        #bank_to_gate = bank_to_gate.view(-1, bank_to_gate.size(2))
        #print('bank_to_gate flat', bank_to_gate.shape)
        gate = F.sigmoid(feat_to_gate + bank_to_gate) + self.add
        gate = gate / (1. + self.add)
        return bank * gate

class MultiModalGenerator(nn.Module):
    def __init__(self, old_generator, img_feat_size, add=0, use_hidden=False):
        super(MultiModalGenerator, self).__init__()
        self.linear = old_generator[0]
        self.vocab_size = self.linear.weight.size(0)
        self.gate = nn.Linear(img_feat_size, self.vocab_size, bias=True)
        #nn.init.constant_(self.gate.bias, 1.0) # newer pytorch
        nn.init.constant(self.gate.bias, 1.0)
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
            gate = F.sigmoid(pre_sigmoid + hidden_to_gate) + self.add
        else:
            gate = F.sigmoid(pre_sigmoid) + self.add
            gate = gate.repeat(n_time, 1)
        gate = gate / (1. + self.add)
        return self.logsoftmax(proj * gate)

class MultiModalLossCompute(NMTLossCompute):
    def _compute_loss(self, batch, output, target, img_feats, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):
        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output, img_feats, output.size[0])

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






       
