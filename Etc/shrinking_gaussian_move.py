import numpy as np
from emcee.moves import MHMove, GaussianMove
from emcee.moves.gaussian import _diagonal_proposal, _isotropic_proposal, _proposal


class ShrinkingGaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function where the given covariance shrinks every time a proposal is generated. 
    Based heavily on emcee.moves.GaussianMove

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general respectively.
        nsteps: ...
        z (Optional): Factor by which the covariance should have decreased after nsteps. Defaults to 100.
        mode (Optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn.
        factor (Optional[float]): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode.

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(self, cov, nsteps, z = 100, mode="vector", factor=None):
        self.cov = cov

        # Parse the proposal type.
        try:
            float(self.cov)

        except TypeError:
            self.cov = np.atleast_1d(self.cov)
            if len(self.cov.shape) == 1:
                # A diagonal proposal was given.
                ndim = len(self.cov)
                proposal = _diagonal_proposal(np.sqrt(self.cov), factor, mode)

            elif len(self.cov.shape) == 2 and self.cov.shape[0] == self.cov.shape[1]:
                # The full, square self.covariance matrix was given.
                ndim = self.cov.shape[0]
                proposal = _proposal(self.cov, factor, mode)

            else:
                raise ValueError("Invalid proposal scale dimensions")

        else:
            # This was a scalar proposal.
            ndim = None
            proposal = _isotropic_proposal(np.sqrt(self.cov), factor, mode)

        self.cov_factor = (cov - cov/z) / (nsteps)
        self.factor = factor
        self.mode = mode

        super().__init__(proposal, ndim=ndim)


    def _change_proposal(self):
        try:
            float(self.cov)

        except TypeError:
            self.cov = np.atleast_1d(self.cov)
            if len(self.cov.shape) == 1:
                # A diagonal proposal was given.
                proposal = _diagonal_proposal(np.sqrt(self.cov), self.factor, self.mode)

            elif len(self.cov.shape) == 2 and self.cov.shape[0] == self.cov.shape[1]:
                # The full, square self.covariance matrix was given.
                proposal = _proposal(self.cov, self.factor, self.mode)

            else:
                raise ValueError("Invalid proposal scale dimensions")

        else:
            # This was a scalar proposal.
            proposal = _isotropic_proposal(np.sqrt(self.cov), self.factor, self.mode)
        
        self.get_proposal = proposal # change the proposal function in the super i.e. MHmove according with the new covariance

    def propose(self, model, state):
        new_state, accepted = super().propose(model, state)
        self.cov -= self.cov_factor
        self._change_proposal()

        return new_state, accepted

