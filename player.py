import os
import sys
import importlib.util
import logging
import yaml
import chess
import random
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from chess_tournament.players import Player


REPO_ID = "Jochemvkem/magnusbot"

logger = logging.getLogger(__name__)


def _load_module_from_path(module_name: str, file_path: str):
    """
    Load a Python module directly from a file path without mutating sys.path.
    This keeps the import fully scoped and avoids global side effects.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransformerPlayer(Player):
    """
    A chess player that uses a Transformer model downloaded from HuggingFace
    to score and select legal moves.

    Use the factory method `TransformerPlayer.from_hub()` to construct an instance.
    Direct instantiation via `__init__` expects pre-loaded components and is
    intended for testing or dependency injection.

    Note on inference cost: `score_legal_moves` runs O(move_length) forward
    passes per group of same-length moves. This is intentional autoregressive
    scoring and is acceptable for single-game inference throughput.
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        temperature: float = 1.0,
    ):
        super().__init__(name)
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = device
        self.temperature = temperature

    # -------------------------
    # Factory
    # -------------------------

    @classmethod
    def from_hub(
        cls,
        name: str = "MagnusBot",
        temperature: float = 1.0,
        repo_id: str = REPO_ID,
    ) -> "TransformerPlayer":
        """
        Download scripts, config, and weights from HuggingFace and return a
        fully initialised TransformerPlayer.

        Artifacts are cached locally by `hf_hub_download`; subsequent calls
        are fast when the cache is warm. Pass `local_files_only=True` (via
        environment variable HF_HUB_OFFLINE=1) to enforce offline use.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Step 1: download scripts from HF ──────────────────────────
        tok_path  = hf_hub_download(repo_id=repo_id, filename="scripts/tokenizer.py")
        arch_path = hf_hub_download(repo_id=repo_id, filename="scripts/architecture.py")

        # ── Step 2: import without mutating sys.path ───────────────────
        # The tokenizer module is loaded first under its own isolated name.
        tokenizer_mod = _load_module_from_path("magnusbot.tokenizer", tok_path)

        # architecture.py contains `from tokenizer import ChessTokenizer`, a bare
        # import that only resolves if a module named "tokenizer" exists in
        # sys.modules. We register our already-loaded module there temporarily so
        # the import succeeds, then remove it immediately to avoid polluting the
        # global module namespace for anything else in the process.
        _TOKENIZER_ALIAS = "tokenizer"
        sys.modules[_TOKENIZER_ALIAS] = tokenizer_mod
        try:
            architecture_mod = _load_module_from_path("magnusbot.architecture", arch_path)
        finally:
            sys.modules.pop(_TOKENIZER_ALIAS, None)

        ChessTokenizer = tokenizer_mod.ChessTokenizer
        Transformer    = architecture_mod.Transformer

        # ── Step 3: download config and weights ───────────────────────
        config_path  = hf_hub_download(repo_id=repo_id, filename="opt-configs.yml")
        weights_path = hf_hub_download(repo_id=repo_id, filename="magnusbot.pth")

        with open(config_path) as f:
            settings = yaml.safe_load(f)

        # ── Step 4: build tokenizer ────────────────────────────────────
        tokenizer = ChessTokenizer()

        # ── Step 5: build model ────────────────────────────────────────
        model = Transformer(
            src_vocab_size = tokenizer.vocab_size,
            tgt_vocab_size = tokenizer.vocab_size,
            d_model        = settings["d_model"],
            num_heads      = settings["num_heads"],
            num_layers     = settings["num_layers"],
            d_ff           = settings["d_ff"],
            max_seq_length = 100,
            dropout        = settings["dropout"],
        ).to(device)

        # ── Step 6: load weights ───────────────────────────────────────
        # `weights_only=True` is a security measure: it prevents arbitrary code
        # execution that can occur when loading untrusted pickled checkpoints.
        state = torch.load(weights_path, map_location=device, weights_only=True)

        # Strip _orig_mod. prefix if the model was saved from a torch.compile'd run.
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

        model.load_state_dict(state)
        model.eval()

        logger.info("[%s] Ready on %s.", name, device)
        print(f"[{name}] Ready on {device}.")

        return cls(name=name, model=model, tokenizer=tokenizer, device=device, temperature=temperature)

    # -------------------------
    # Log-prob scoring
    # -------------------------

    def _encode_moves(self, legal_moves: list[str]) -> dict[str, list[int]]:
        """
        Encode each UCI move string into a list of token ids.

        Raises ValueError if any character in a move is absent from the
        tokenizer vocabulary, which would silently corrupt scoring otherwise.
        """
        encoded = {}
        for move in legal_moves:
            tokens = []
            for c in move:
                if c not in self.tokenizer.char_to_int:
                    raise ValueError(
                        f"Character {c!r} in move {move!r} is not in the tokenizer "
                        f"vocabulary. The model may be incompatible with this position."
                    )
                tokens.append(self.tokenizer.char_to_int[c])
            encoded[move] = tokens
        return encoded

    def score_legal_moves(self, src_tensor: torch.Tensor, legal_moves: list[str]) -> dict[str, float]:
        """
        Score each legal move by summing the log-probabilities of its character
        tokens under the model.

        Moves are grouped by token length so a single batched forward pass is
        used per character position within each length group, reducing the total
        number of forward passes from O(n_moves × move_length) to
        O(n_unique_lengths × max_move_length).
        """
        encoded = self._encode_moves(legal_moves)

        # Group moves by length for batched scoring.
        by_length: dict[int, list[str]] = {}
        for move, tokens in encoded.items():
            by_length.setdefault(len(tokens), []).append(move)

        scores: dict[str, float] = {}

        with torch.no_grad():
            for length, group in by_length.items():
                # Decoder input rows: [BOS, char0, char1, ...]
                tgt_ids   = torch.tensor(
                    [[1] + encoded[m] for m in group], dtype=torch.long
                ).to(self.device)
                src_batch = src_tensor.expand(len(group), -1)

                log_probs = torch.zeros(len(group), device=self.device)

                for i in range(length):
                    # Feed BOS + characters seen so far; predict the next character.
                    output         = self.model(src_batch, tgt_ids[:, :i + 1])
                    step_log_probs = F.log_softmax(output[:, -1, :] / self.temperature, dim=-1)
                    next_token     = tgt_ids[:, i + 1].unsqueeze(1)
                    log_probs     += step_log_probs.gather(1, next_token).squeeze(1)

                for move, lp in zip(group, log_probs.tolist()):
                    scores[move] = lp

        return scores

    # -------------------------
    # Main API
    # -------------------------

    def get_move(self, fen: str) -> str | None:
        board       = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        encoded_fen = self.tokenizer.encode(fen, is_target=False)
        src_tensor  = torch.tensor(encoded_fen, dtype=torch.long).unsqueeze(0).to(self.device)

        try:
            scores = self.score_legal_moves(src_tensor, legal_moves)
            return max(scores, key=scores.get)
        except (ValueError, RuntimeError) as exc:
            # ValueError: tokenizer vocab mismatch (logged as a warning — indicates
            #   an unexpected position or model/tokenizer version mismatch).
            # RuntimeError: GPU/tensor errors during the forward pass.
            # Other exceptions (e.g. KeyboardInterrupt, OOM) are intentionally
            # allowed to propagate so they are not silently swallowed.
            logger.warning(
                "[%s] Falling back to random move due to scoring error: %s",
                self.name, exc,
            )
            return random.choice(legal_moves)
