#!/usr/bin/env python3
"""
Moduli për Deep RNN Forward Propagation
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Kryen forward propagation për një Deep RNN.

    Parametrat:
    - rnn_cells: listë me instanca të RNNCell
    - X: të dhënat (t, m, i)
    - h_0: gjendja fillestare (l, m, h)

    Kthehet: H, Y
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Përcaktojmë dimensionin e daljes 'o' nga qeliza e fundit
    o = rnn_cells[-1].by.shape[1]

    # H ka përmasat (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    # Y ka përmasat (t, m, o)
    Y = np.zeros((t, m, o))

    # Vendosim gjendjen fillestare h_0 në hapin 0
    H[0] = h_0

    # Cikli për çdo hap kohor (Time Steps)
    for step in range(t):
        # Për shtresën e parë, hyrja është data X
        current_input = X[step]

        # Cikli për çdo shtresë (Layers)
        for layer in range(l):
            cell = rnn_cells[layer]

            # Marrim gjendjen e mëparshme për këtë shtresë specifike
            h_prev = H[step, layer]

            # Ekzekutojmë forward të qelizës
            h_next, y = cell.forward(h_prev, current_input)

            # Ruajmë h_next në matricën H për hapin tjetër kohor
            H[step + 1, layer] = h_next

            # Hyrja për shtresën sipër është gjendja e fshehur e shtresës poshtë
            current_input = h_next

        # Dalja finale në këtë hap kohor është y i shtresës së fundit
        Y[step] = y

    return H, Y
