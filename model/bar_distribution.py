import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalRegressionHead(nn.Module):
    def __init__(self, hidden_size, num_bins, x_min=0., x_max=1.):
        """
        Args:
            hidden_size: Dimension of input features.
            num_bins: Total ordinal bins (e.g., 4).
        """
        super().__init__()
        self.num_bins = num_bins
        self.fc = nn.Linear(hidden_size, 1)  # Maps to scalar z
        self.base = nn.Parameter(torch.tensor(0.0))
        self.delta = nn.Parameter(torch.zeros(num_bins - 1))  # Increments for thresholds
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, hidden_size)
        Returns:
            bin_probs: Tensor of shape (batch_size, num_bins) representing the probability for each bin.
        """
        B, T, d = x.shape
        z = self.fc(x.reshape(B*T, d))  # (batch_size, 1)
        # Compute ordered thresholds: b_i = base + cumulative sum of softplus(delta)
        thresholds = self.base + torch.cumsum(F.softplus(self.delta), dim=0)  # (num_bins-1,)
        # Cumulative probabilities: F(i)=σ(b_i - z)
        cum_probs = torch.sigmoid(thresholds.unsqueeze(0) - z)  # (batch_size, num_bins-1)
        # Derive bin probabilities:
        p1 = cum_probs[:, :1]                           # P(y=1)=F(1)
        p_diff = cum_probs[:, 1:] - cum_probs[:, :-1]     # P(y=k)=F(k)-F(k-1)
        p_last = 1 - cum_probs[:, -1:]                    # P(y=num_bins)=1-F(num_bins-1)
        bin_probs = torch.cat([p1, p_diff, p_last], dim=1)
        return bin_probs.reshape(B, T, self.num_bins)

    def predict(self, x):
        """
        Returns predicted ordinal bin via argmax.
        """
        bin_probs = self.forward(x)
        return torch.argmax(bin_probs, dim=1)


    def unbin(self, bin_idx):
        """
        Maps a bin index back to a value in [x_min, x_max] assuming uniform binning.
        
        Args:
            bin_idx: Tensor of shape (batch_size,) containing bin indices (0 to num_bins-1).
        
        Returns:
            values: Tensor of shape (batch_size,) with corresponding values in [x_min, x_max].
        """
        # Convert bin index to a float in [0, 1]
        normalized_pos = (bin_idx.float() + 0.5) / self.num_bins  # Center of the bin
        # Linearly map [0, 1] → [x_min, x_max]
        values = self.x_min + (self.x_max - self.x_min) * normalized_pos
        return values

class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor, smoothing=.0, ignore_nan_targets=True, distance_weighting_sigma=None): # here borders should start with min and end with max, where all values lie in (min,max) and are sorted
        '''
        :param borders:
        :param smoothing:
        :param append_mean_pred: Whether to predict the mean of the other positions as a last output in forward,
        is enabled when additionally y has a sequence length 1 shorter than logits, i.e. len(logits) == 1 + len(y)
        '''
        super().__init__()
        assert len(borders.shape) == 1
        self.register_buffer('borders', borders)
        self.register_buffer('smoothing', torch.tensor(smoothing))
        self.register_buffer('bucket_widths', self.borders[1:] - self.borders[:-1])
        full_width = self.bucket_widths.sum()

        assert (1 - (full_width / (self.borders[-1] - self.borders[0]))).abs() < 1e-2, f'diff: {full_width - (self.borders[-1] - self.borders[0])} with {full_width} {self.borders[-1]} {self.borders[0]}'
        assert (self.bucket_widths >= 0.0).all() , "Please provide sorted borders!" # This also allows size zero buckets
        self.num_bars = len(borders) - 1
        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

        # Precompute bin centers
        bin_centers = (self.borders[:-1] + self.borders[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)

        # Distance-based weighting
        self.distance_weighting_sigma = distance_weighting_sigma
        if distance_weighting_sigma is not None:
            # Compute pairwise distances between bin centers
            distances = torch.abs(bin_centers.unsqueeze(1) - bin_centers.unsqueeze(0))
            # Apply Gaussian kernel
            weights = torch.exp(-0.5 * (distances / distance_weighting_sigma) ** 2)
            # Normalize weights so each row sums to 1 (optional)
            weights = weights / weights.sum(dim=1, keepdim=True)
            self.register_buffer('distance_weights', weights)
        else:
            self.register_buffer('distance_weights', torch.eye(self.num_bars))

        self.to(borders.device)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('append_mean_pred', False)

    def map_to_bucket_idx(self, y):
        #target_sample = torch.searchsorted(self.borders, y) - 1
        #target_sample[y == self.borders[0]] = 0
        #target_sample[y == self.borders[-1]] = self.num_bars - 1
        #return target_sample
        # Ensure values on boundaries are assigned to the correct bin
        return torch.clamp(
            torch.searchsorted(self.borders, y, side='right') - 1,
            0, self.num_bars - 1
        )

    def ignore_init(self, y):
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any():
            if not self.ignore_nan_targets: raise ValueError(f'Found NaN in target {y}')
            # print_once("A loss was ignored because there was nan target.")

        y = y.long()
        ignore_loss_mask = (y < 0) | (y >= self.num_bars)
        y[ignore_loss_mask] = self.borders[0].long()
        return ignore_loss_mask

    # def ignore_init(self, y):
    #     ignore_loss_mask = torch.isnan(y)
    #     if ignore_loss_mask.any():
    #         if not self.ignore_nan_targets: raise ValueError(f'Found NaN in target {y}')
    #         # print_once("A loss was ignored because there was nan target.")
    #     y[ignore_loss_mask] = self.borders[0] # this is just a default value, it will be ignored anyway
    #     return ignore_loss_mask

    def compute_scaled_log_probs(self, logits): # logits: seq-1 x Batch x n_bins
        # this is equivalent to log(p(y)) of the density p
        bucket_log_probs = torch.log_softmax(logits, -1)
        #scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        # Only scale if bins are non-uniform
        if not torch.allclose(self.bucket_widths, self.bucket_widths[0]):
            scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        else:
            scaled_bucket_log_probs = bucket_log_probs
        return scaled_bucket_log_probs

    def forward(self, logits, y, mean_prediction_logits=None, gamma=0.5, alpha=None, model=None):
        """
        Computes the negative log-density (loss). 
        - y: [T, B] (or some shape compatible with logits.shape[:-1])
        - logits: [T, B, self.num_bars]
        The function returns a loss tensor of shape [T, B].
        """

        # 1) Ensure y is cloned & contiguous, then reshape to match logits up to the last dimension
        y = y.clone().contiguous().view(*logits.shape[:-1])  # remove any trailing one-dimension
    
        # 2) Identify positions to ignore (if you have an ignore_init method for certain time steps)
        ignore_loss_mask = self.ignore_init(y)

        # 3) Convert continuous y-values to bucket indices (internally uses torch.searchsorted)
        target_sample = self.map_to_bucket_idx(y)  # seq-1 x Batch
        # target_sample = model(y.unsqueeze(-1).permute(1,0,2)).argmax(dim=-1).permute(1,0)
        # 4) Check that all targets are within the valid bin range
        assert (target_sample >= 0).all() and (target_sample < self.num_bars).all(), (
            f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        )
        assert logits.shape[-1] == self.num_bars, f"{logits.shape[-1]} vs {self.num_bars}"

        # 5) Compute scaled log-probs for each bucket
        # scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)  # seq-1 x batch x n_bins
        
        # Compute probabilities (needed for Focal Loss)
        bucket_probs = torch.softmax(logits, dim=-1)  # [T, B, num_bars]
        true_bin_probs = bucket_probs.gather(-1, target_sample[..., None]).squeeze(-1)  # [T, B]

        # ADD 
        if self.distance_weighting_sigma is not None:
            # Create smoothed target distribution using distance weights
            target_dist = F.one_hot(target_sample, num_classes=self.num_bars).float()
            smoothed_target_dist = target_dist @ self.distance_weights
            # Compute KL divergence between predicted and smoothed target
            smoothed_target_log_dist = torch.log_softmax(smoothed_target_dist, dim=-1)
            nll_loss = F.kl_div(
                #scaled_bucket_log_probs,
                torch.log_softmax(logits, -1), 
                smoothed_target_log_dist, 
                reduction='none', 
                log_target=True
            ).sum(-1)
        else:
            # Standard cross-entropy -> step 6)
            #nll_loss = -scaled_bucket_log_probs.gather(-1, target_sample[..., None]).squeeze(-1)  # seq-1 x batch
            
            # Use Focal Loss
            nll_loss = -((1 - true_bin_probs) ** gamma) * torch.log(true_bin_probs + 1e-10)  # [T, B]
            if alpha is not None:
                alpha_weight = alpha[target_sample]  # [T, B]
                nll_loss *= alpha_weight

        # 6) Negative log-likelihood for the correct bucket at each time-step/sample
        #    => shape [T, B]
        #nll_loss = -scaled_bucket_log_probs.gather(-1, target_sample[..., None]).squeeze(-1)

        # 7) Optional mean_prediction_logits term (non-myopic BO extension)
        if mean_prediction_logits is not None:
            if not self.training:
                print('Calculating loss incl. mean prediction loss for nonmyopic BO.')
            scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
            # e.g. combine the mean-prediction loss with the NLL
            nll_loss = torch.cat((nll_loss, self.mean_loss(logits, scaled_mean_log_probs)), dim=0)

        # 8) Label smoothing
        # smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)         # average negative log-prob
        smooth_loss = -torch.log_softmax(logits, -1).mean(dim=-1)
        smoothing = self.smoothing if self.training else 0.0
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

        # 9) Zero out the loss for ignored steps
        loss[ignore_loss_mask] = 0.0

        return loss


    def mean_loss(self, logits, scaled_mean_logits):
        assert (len(logits.shape) == 3) and (len(scaled_mean_logits.shape) == 2), \
            (len(logits.shape), len(scaled_mean_logits.shape))
        means = self.mean(logits).detach()  # T x B
        target_mean = self.map_to_bucket_idx(means).clamp_(0, self.num_bars - 1)  # T x B
        return -scaled_mean_logits.gather(1, target_mean.T).mean(1).unsqueeze(0)  # 1 x B

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths/2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def median(self, logits):
        return self.icdf(logits, 0.5)

    def icdf(self, logits, left_prob):
        """
        Implementation of the quantile function
        :param logits: Tensor of any shape, with the last dimension being logits
        :param left_prob: float: The probability mass to the left of the result.
        :return: Position with `left_prob` probability weight to the left.
        """
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = torch.searchsorted(cumprobs, left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device))\
            .squeeze(-1).clamp(0, cumprobs.shape[-1] - 1)  # this might not do the right for outliers
        cumprobs = torch.cat([torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1)

        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx+1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(-1, idx[..., None]).squeeze(-1)

    def quantile(self, logits, center_prob=.682):
        side_probs = (1.-center_prob)/2
        return torch.stack((self.icdf(logits, side_probs), self.icdf(logits, 1.-side_probs)),-1)

    def ucb(self, logits, best_f, rest_prob=(1-.682)/2, maximize=True):
        """
        UCB utility. Rest Prob is the amount of utility above (below) the confidence interval that is ignored.
        Higher rest_prob is equivalent to lower beta in the standard GP-UCB formulation.
        :param logits: Logits, as returned by the Transformer.
        :param rest_prob: The amount of utility above (below) the confidence interval that is ignored.
        The default is equivalent to using GP-UCB with `beta=1`.
        To get the corresponding `beta`, where `beta` is from
        the standard GP definition of UCB `ucb_utility = mean + beta * std`,
        you can use this computation: `beta = math.sqrt(2)*torch.erfinv(torch.tensor(2*(1-rest_prob)-1))`.
        :param maximize:
        :return: utility
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits):
        mode_inds = logits.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths/2
        return bucket_means[mode_inds]

    def ei(self, logits, best_f, maximize=True): # logits: evaluation_points x batch x feature_dim
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[...,0].shape, best_f, device=logits.device)

        best_f = best_f[..., None].repeat(*[1]*len(best_f.shape), logits.shape[-1])
        clamped_best_f = best_f.clamp(self.borders[:-1], self.borders[1:])

        #bucket_contributions = (best_f[...,None] < self.borders[:-1]).float() * bucket_means
        # true bucket contributions
        bucket_contributions = ((self.borders[1:]**2-clamped_best_f**2)/2 - best_f*(self.borders[1:] - clamped_best_f))/bucket_diffs

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def pi(self, logits, best_f, maximize=True):# logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[...,0].shape, best_f, device=logits.device)
        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1. - ((best_f[...,None] - self.borders[:-1]) / border_widths).clamp(0., 1.)
        return (p * factor).sum(-1)


    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (left_borders.square() + right_borders.square() + left_borders*right_borders)/3.
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits):
        return self.mean_of_square(logits) - self.mean(logits).square()

    def pi(self, logits, best_f, maximize=True):# logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        p = torch.softmax(logits, -1)
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1. - ((best_f - self.borders[:-1]) / border_widths).clamp(0., 1.)
        return (p * factor).sum(-1)


    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (left_borders.square() + right_borders.square() + left_borders*right_borders)/3.
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits):
        return self.mean_of_square(logits) - self.mean(logits).square()


class FullSupportBarDistribution(BarDistribution):
    @staticmethod
    def halfnormal_with_p_weight_before(range_max,p=.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.)).icdf(torch.tensor(p))
        return torch.distributions.HalfNormal(s)


    def forward(self, logits, y, mean_prediction_logits=None): # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        assert self.num_bars > 1
        y = y.clone().view(len(y),-1) # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y) # alters y
        target_sample = self.map_to_bucket_idx(y) # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert logits.shape[-1] == self.num_bars, f'{logits.shape[-1]} vs {self.num_bars}'
        assert (target_sample >= 0).all() and (target_sample < self.num_bars).all(), \
            f'y {y} not in support set for borders (min_y, max_y) {self.borders}'
        assert logits.shape[-1] == self.num_bars, f'{logits.shape[-1]} vs {self.num_bars}'
        # ignore all position with nan values


        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        assert len(scaled_bucket_log_probs) == len(target_sample), (len(scaled_bucket_log_probs), len(target_sample))
        log_probs = scaled_bucket_log_probs.gather(-1, target_sample.unsqueeze(-1)).squeeze(-1)

        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]), self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))


        log_probs[target_sample == 0] += side_normals[0].log_prob((self.borders[1]-y[target_sample == 0]).clamp(min=.00000001)) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars-1] += side_normals[1].log_prob((y[target_sample == self.num_bars-1]-self.borders[-2]).clamp(min=.00000001)) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs

        if mean_prediction_logits is not None:
            assert not ignore_loss_mask.any(), "Ignoring examples is not implemented with mean pred."
            if not self.training:
                print('Calculating loss incl mean prediction loss for nonmyopic BO.')
            if not torch.is_grad_enabled():
                print("Warning: loss is not correct in absolute terms, only the gradient is right, when using `append_mean_pred`.")
            scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
            nll_loss = torch.cat((nll_loss, self.mean_loss(logits, scaled_mean_log_probs)), 0)
            #ignore_loss_mask = torch.zeros_like(nll_loss, dtype=torch.bool)

        if self.smoothing:
            smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
            smoothing = self.smoothing if self.training else 0.
            nll_loss = (1. - smoothing) * nll_loss + smoothing * smooth_loss

        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.

        return nll_loss

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means.to(logits.device)

    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (left_borders.square() + right_borders.square() + left_borders*right_borders)/3.
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        bucket_mean_of_square[0] = side_normals[0].variance + (-side_normals[0].mean + self.borders[1]).square()
        bucket_mean_of_square[-1] = side_normals[1].variance + (side_normals[1].variance + self.borders[-2]).square()
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def pi(self, logits, best_f, maximize=True):# logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer (evaluation_points x batch x feature_dim)
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[...,0].shape, best_f, device=logits.device) # evaluation_points x batch
        assert best_f.shape == logits[...,0].shape, f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"
        p = torch.softmax(logits, -1) # evaluation_points x batch
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1. - ((best_f[...,None] - self.borders[:-1]) / border_widths).clamp(0., 1.) # evaluation_points x batch x num_bars

        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        position_in_side_normals = (-(best_f - self.borders[1]).clamp(max=0.), (best_f - self.borders[-2]).clamp(min=0.)) # evaluation_points x batch
        factor[...,0] = 0.
        factor[...,0][position_in_side_normals[0] > 0.] = side_normals[0].cdf(position_in_side_normals[0][position_in_side_normals[0] > 0.])
        factor[...,-1] = 1.
        factor[...,-1][position_in_side_normals[1] > 0.] = 1. - side_normals[1].cdf(position_in_side_normals[1][position_in_side_normals[1] > 0.])
        return (p * factor).sum(-1)


    def ei_for_halfnormal(self, scale, best_f, maximize=True):
        """
        This is the EI for a standard normal distribution with mean 0 and variance `scale` times 2.
        Which is the same as the half normal EI.
        I tested this with MC approximation:
        ei_for_halfnormal = lambda scale, best_f: (torch.distributions.HalfNormal(torch.tensor(scale)).sample((10_000_000,))- best_f ).clamp(min=0.).mean()
        print([(ei_for_halfnormal(scale,best_f), FullSupportBarDistribution().ei_for_halfnormal(scale,best_f)) for scale in [0.1,1.,10.] for best_f in [.1,10.,4.]])
        :param scale:
        :param best_f:
        :param maximize:
        :return:
        """
        assert maximize
        mean = torch.tensor(0.)
        u = (mean - best_f) / scale
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        try:
            ucdf = normal.cdf(u)
        except ValueError:
            print(f"u: {u}, best_f: {best_f}, scale: {scale}")
            raise
        updf = torch.exp(normal.log_prob(u))
        normal_ei = scale * (updf + u * ucdf)
        return 2*normal_ei

    def ei(self, logits, best_f, maximize=True): # logits: evaluation_points x batch x feature_dim
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[...,0].shape, best_f, device=logits.device)
        assert best_f.shape == logits[...,0].shape, f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"


        best_f_per_logit = best_f[..., None].repeat(*[1]*len(best_f.shape), logits.shape[-1])
        clamped_best_f = best_f_per_logit.clamp(self.borders[:-1], self.borders[1:])

        # true bucket contributions
        bucket_contributions = ((self.borders[1:]**2-clamped_best_f**2)/2 - best_f_per_logit*(self.borders[1:] - clamped_best_f))/bucket_diffs

        # extra stuff for continuous
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        position_in_side_normals = (-(best_f - self.borders[1]).clamp(max=0.),
                                    (best_f - self.borders[-2]).clamp(min=0.))  # evaluation_points x batch

        bucket_contributions[...,-1] = self.ei_for_halfnormal(side_normals[1].scale, position_in_side_normals[1])

        bucket_contributions[...,0] = self.ei_for_halfnormal(side_normals[0].scale, torch.zeros_like(position_in_side_normals[0])) \
                                  - self.ei_for_halfnormal(side_normals[0].scale, position_in_side_normals[0])

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)


def get_bucket_limits(num_outputs:int, full_range:tuple=None, ys:torch.Tensor=None, verbose:bool=False):
    assert (ys is None) != (full_range is None), 'Either full_range or ys must be passed.'

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        if len(ys) % num_outputs: ys = ys[:-(len(ys) % num_outputs)]
        print(f'Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys.')
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min() and full_range[1] >= ys.max(), f'full_range {full_range} not in range of ys {ys.min(), ys.max()}'
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (ys_sorted[ys_per_bucket-1::ys_per_bucket][:-1]+ys_sorted[ys_per_bucket::ys_per_bucket])/2
        if verbose:
            print(f'Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys.')
            print(full_range)
        bucket_limits = torch.cat([full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],0)

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat([full_range[0] + torch.arange(num_outputs).float()*class_width, torch.tensor(full_range[1]).unsqueeze(0)], 0)

    assert len(bucket_limits) - 1 == num_outputs, f'len(bucket_limits) - 1 == {len(bucket_limits) - 1} != {num_outputs} == num_outputs'
    assert full_range[0] == bucket_limits[0], f'{full_range[0]} != {bucket_limits[0]}'
    assert full_range[-1] == bucket_limits[-1], f'{full_range[-1]} != {bucket_limits[-1]}'

    return bucket_limits


def get_custom_bar_dist(borders, criterion):
    # Tested that a bar_dist with borders 0.54 (-> softplus 1.0) yields the same bar distribution as the passed one.
    borders_ = torch.nn.functional.softplus(borders) + 0.001
    borders_ = (torch.cumsum(torch.cat([criterion.borders[0:1], criterion.bucket_widths]) * borders_, 0))
    criterion_ = criterion.__class__(borders=borders_, handle_nans=criterion.handle_nans)
    return criterion_


