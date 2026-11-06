"""Pacote de modelos probabilísticos para a Mega da Virada.

Contém:
- M0: Uniforme (piso)
- M1: Plackett–Luce marginalizado (modelo primário)
- M2: Fisher (sensibilidade)
- M3: Produto posicional normalizado (ablação)
- M4: Plackett–Luce neural sequencial (ablação opcional)
- M5: LSTM de 6 cabeças como M3 / pseudo-likelihood (ablação)
"""

from __future__ import annotations

from lstm_virada.models.fisher import FisherError, FisherModel
from lstm_virada.models.lstm_seis_cabecas import (
    VARIANTE_CCE,
    VARIANTE_SET,
    LSTMSeisCabecas,
    LSTMSeisCabecasError,
)
from lstm_virada.models.plackett_luce import PlackettLuce, PlackettLuceError
from lstm_virada.models.positional import PositionalError, PositionalModel
from lstm_virada.models.seq_pl import (
    NUM_CLASSES,
    NUM_DEZENAS_JOGO,
    SeqPLError,
    SeqPLModel,
)

__all__ = [
    "NUM_CLASSES",
    "NUM_DEZENAS_JOGO",
    "VARIANTE_CCE",
    "VARIANTE_SET",
    "FisherError",
    "FisherModel",
    "LSTMSeisCabecas",
    "LSTMSeisCabecasError",
    "PlackettLuce",
    "PlackettLuceError",
    "PositionalError",
    "PositionalModel",
    "SeqPLError",
    "SeqPLModel",
]
