"""smoothing.py — Advanced probability smoothing for streaming channel features.

Replaces Laplace additive smoothing with Witten-Bell for sharper unseen handling.
"""
import math


class WittenBellSmoother:
    """Witten-Bell smoothing for streaming count caches.

    Unlike Laplace: P(unseen) = U/(N+U) where U = distinct items seen.
    Seen items: P(t) = count(t)/(N+U)

    This dynamically adapts: repetitive contexts get near-zero unseen
    probability; diverse contexts get smooth backoff.

    Also tracks unique items via a set for the current context window.
    """
    def __init__(self, V):
        self.V = V
        self._seen_items = set()

    def reset(self):
        self._seen_items = set()

    def add(self, item):
        self._seen_items.add(item)

    def smooth(self, count, total_count, uniform_logp):
        """Return Witten-Bell smoothed log-probability.
        
        Args:
            count: count of the specific item
            total_count: total count of all items in this context
            uniform_logp: log uniform probability = -log(V)
        
        Returns:
            log probability under Witten-Bell smoothing
        """
        N = total_count
        U = len(self._seen_items)
        if N <= 0:
            return uniform_logp
        
        # P(unseen) = U / (N + U)
        # P(seen_item) = count / (N + U)
        if count > 0:
            p = count / (N + U)
        else:
            # Item not seen — allocate from unseen mass
            unseen_mass = U / (N + U)
            # Distribute unseen mass among unseen items
            unseen_items_count = max(1, self.V - U)
            p = unseen_mass / unseen_items_count
        
        return math.log(max(p, 1e-12))


class KNInterpolatedSmoother:
    """Kneser-Ney style interpolated smoothing for bigram/trigram caches.
    
    Interpolates between higher-order and lower-order estimates:
    P_kn(token|ctx) = max(count - D, 0)/N + λ * P_lower(token)
    
    where D is a discount constant (0.75 typical) and λ ensures sum=1.
    """
    def __init__(self, discount=0.75):
        self.discount = discount

    def interpolate(self, high_count, high_total, low_logp, uniform_logp, V):
        """KN-interpolated log probability.

        Args:
            high_count: count of this (context, token) pair
            high_total: total count for this context
            low_logp: log probability from lower-order (e.g., unigram)
            uniform_logp: uniform fallback log prob
        
        Returns:
            log probability under KN interpolation
        """
        if high_total <= 0:
            return low_logp if low_logp > uniform_logp else uniform_logp
        
        D = self.discount
        discounted = max(high_count - D, 0) / high_total
        
        # λ: mass from discounted counts redistributed to lower order
        n1 = sum(1 for c in [high_count] if c == 1)  # simplified: use count directly
        lambda_ = (D / high_total) * (1 if high_count > 0 else 1)
        
        lower_p = math.exp(low_logp) if low_logp > -1e10 else 1.0 / V
        
        p = discounted + lambda_ * lower_p
        return math.log(max(p, 1e-12))
