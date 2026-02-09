#!/usr/bin/env python3
"""Module for learning rate decay using inverse time decay."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha decays
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed further

    Returns:
        Updated value for alpha
    """
    # Calculate the decay step (stepwise fashion)
    epoch = global_step // decay_step
    
    # Inverse time decay formula
    updated_alpha = alpha / (1 + decay_rate * epoch)
    
    return updated_alpha
```

**Shpjegim:**

1. **Stepwise decay**: Përdorim `//` (floor division) për të marrë numrin e plotë të epochs
   - Për `global_step` 0-9 me `decay_step=10` → epoch = 0
   - Për `global_step` 10-19 → epoch = 1
   - Etj.

2. **Inverse time decay formula**: 
```
   alpha_new = alpha / (1 + decay_rate * epoch)
