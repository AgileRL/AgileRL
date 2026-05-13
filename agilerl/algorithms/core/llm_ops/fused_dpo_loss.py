try:
    from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
    from liger_kernel.chunked_loss.fused_linear_preference import (
        LigerFusedLinearPreferenceBase,
    )

    class _LigerDPOWithAlpha(LigerFusedLinearPreferenceBase):
        """Thin wrapper that exposes ``alpha`` for NLL scaling.

        ``LigerFusedLinearDPOFunction`` passes ``compute_nll_loss`` as a bool
        but never forwards ``alpha`` to the base class (which defaults to 1.0).
        This subclass reuses the DPO preference loss and adds ``alpha`` so the
        fused kernel correctly scales the NLL component.
        """

        preference_loss_fn = staticmethod(
            LigerFusedLinearDPOFunction.preference_loss_fn
        )

        @classmethod
        def forward(
            cls,
            ctx,
            _input,
            weight,
            target,
            bias=None,
            ref_input=None,
            ref_weight=None,
            ref_bias=None,
            ignore_index=-100,
            beta=0.1,
            alpha=1.0,
            compute_nll_loss=True,
            compiled=True,
            use_ref_model=True,
            average_log_prob=False,
            chunk_size=1,
            loss_type="sigmoid",
        ):
            return LigerFusedLinearPreferenceBase.forward(
                cls=cls,
                ctx=ctx,
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ignore_index=ignore_index,
                alpha=alpha,
                beta=beta,
                compute_nll_loss=compute_nll_loss,
                compiled=compiled,
                use_ref_model=use_ref_model,
                ref_input=ref_input,
                ref_weight=ref_weight,
                ref_bias=ref_bias,
                average_log_prob=average_log_prob,
                chunk_size=chunk_size,
                loss_type=loss_type,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
            return (*grads, *(None,) * 12)

except ImportError:
    _LigerDPOWithAlpha = None
