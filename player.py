import os
import sys
import importlib.util
import logging
import yaml
import chess
import random
import time
import torch
import torch.nn.functional as F

from collections import defaultdict
from huggingface_hub import hf_hub_download
from chess_tournament.players import Player
from typing import Optional


REPO_ID = "Jochemvkem/magnusbot"

logger = logging.getLogger(__name__)

# ── Time budget ────────────────────────────────────────────────────────────────
MOVE_TIME_BUDGET = 5.0

# ── Heuristic bonuses (added to model score) ──────────────────────────────────
BONUS_CHECKMATE       = 1000.0  # virtually guarantees checkmate is always chosen
BONUS_QUEEN_PROMOTION =   20.0  # strongly prefer promoting to queen
BONUS_FREE_CAPTURE    =    2.0  # multiplied by piece value for undefended captures
BONUS_SACRIFICE       =    3.0  # capturing a more-valuable piece with a lesser one

# ── Heuristic penalties (subtracted from model score) ─────────────────────────
PENALTY_DRAW          =   10.0  # moves that lead to repetition / 50-move draw
PENALTY_HANG          =    5.0  # moves that walk our piece into an attacked square

# ── Endgame weight ─────────────────────────────────────────────────────────────
# The endgame heuristic score (king proximity + pawn advancement + captures) is
# multiplied by this and added to the model score when in an endgame position.
ENDGAME_WEIGHT        =    1.5

# ── Move history / loop penalties ─────────────────────────────────────────────
HISTORY_DEPTH  = 6
PENALTY_RECENT = 4.0
PENALTY_OLDER  = 2.0

# ── Material values ────────────────────────────────────────────────────────────
PIECE_VALUES: dict[int, int] = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


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
    to score and select legal moves, combined with a set of heuristic bonuses
    and penalties to improve move quality.

    Scoring pipeline:
        score = model_log_prob
              + BONUS_CHECKMATE        if the move delivers checkmate
              + BONUS_QUEEN_PROMOTION  if the move promotes to queen
              + BONUS_FREE_CAPTURE     × captured piece value  (undefended pieces)
              + BONUS_SACRIFICE        if captured piece > moving piece in value
              + ENDGAME_WEIGHT         × endgame_heuristic     if in endgame
              - PENALTY_DRAW           if the move leads to a draw
              - PENALTY_HANG           if the move walks our piece into an attack
              - loop_penalty           based on move history and position counts

    Initialise directly: `TransformerPlayer("MyBot")` — weights and config are
    downloaded automatically from HuggingFace Hub inside `__init__`.
    """

    def __init__(
        self,
        name: str = "MagnusBot",
        temperature: float = 1.0,
        repo_id: str = REPO_ID,
    ):
        super().__init__(name)
        self.temperature = temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.tokenizer = ChessTokenizer()

        # ── Step 5: build model ────────────────────────────────────────
        # Infer num_layers from the checkpoint so architecture always matches.
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        indices    = {int(k.split(".")[1]) for k in state if k.startswith("encoder_layers")}
        num_layers = max(indices) + 1 if indices else settings["num_layers"]

        self.model = Transformer(
            src_vocab_size = self.tokenizer.vocab_size,
            tgt_vocab_size = self.tokenizer.vocab_size,
            d_model        = settings["d_model"],
            num_heads      = settings["num_heads"],
            num_layers     = num_layers,
            d_ff           = settings["d_ff"],
            max_seq_length = 100,
            dropout        = settings["dropout"],
        ).to(self.device)

        # ── Step 6: load weights ───────────────────────────────────────
        # `weights_only=True` is a security measure: it prevents arbitrary code
        # execution that can occur when loading untrusted pickled checkpoints.
        self.model.load_state_dict(state)
        self.model.eval()

        self._position_counts = defaultdict(int)
        self._move_history    = []

        logger.info("[%s] Ready on %s.", name, self.device)
        print(f"[{name}] Ready on {self.device}.")

    def reset_game(self):
        """Reset per-game state (position counts and move history)."""
        self._position_counts.clear()
        self._move_history.clear()

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
    # Heuristic adjustments
    # -------------------------

    @staticmethod
    def _piece_value(piece: Optional[chess.Piece]) -> int:
        if piece is None:
            return 0
        return PIECE_VALUES.get(piece.piece_type, 0)

    def _heuristic_adjustment(self, board: chess.Board, move: chess.Move) -> float:
        """
        Compute the total heuristic adjustment for a single move.

        Positive values make the move more attractive; negative values less so.
        All signals are summed and returned as a single float to be added to the
        model log-prob score.

        Signals:
          +BONUS_CHECKMATE        — move delivers immediate checkmate
          +BONUS_QUEEN_PROMOTION  — move promotes a pawn to queen
          +BONUS_FREE_CAPTURE × v — undefended enemy piece of value v captured
          +BONUS_SACRIFICE        — captured piece is more valuable than ours
          +ENDGAME_WEIGHT × h     — endgame positional heuristic h
          -PENALTY_DRAW           — move leads to threefold repetition / 50-move
          -PENALTY_HANG           — move places our piece on an attacked square
          -loop_penalty           — move repeats recent history or revisits position
        """
        adjustment = 0.0
        opponent   = not board.turn

        # ── Checkmate bonus ────────────────────────────────────────────────
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return BONUS_CHECKMATE  # dominates everything; return early

        # ── Draw penalty ───────────────────────────────────────────────────
        board.push(move)
        if board.is_repetition(count=3) or board.can_claim_fifty_moves():
            adjustment -= PENALTY_DRAW
        board.pop()

        # ── Queen promotion bonus ──────────────────────────────────────────
        if move.promotion == chess.QUEEN:
            adjustment += BONUS_QUEEN_PROMOTION

        # ── Capture bonuses ────────────────────────────────────────────────
        if board.is_capture(move):
            our_piece = board.piece_at(move.from_square)

            # En-passant: the captured pawn is not on to_square.
            if board.is_en_passant(move):
                captured_value = PIECE_VALUES[chess.PAWN]
            else:
                captured_value = self._piece_value(board.piece_at(move.to_square))

            # Free-capture bonus: scale with the value of the undefended piece.
            board.push(move)
            if not board.is_attacked_by(opponent, move.to_square):
                adjustment += BONUS_FREE_CAPTURE * captured_value
            board.pop()

            # Sacrifice bonus: we give up a lesser piece to take a greater one.
            if captured_value > self._piece_value(our_piece):
                adjustment += BONUS_SACRIFICE

        # ── Hang penalty ───────────────────────────────────────────────────
        board.push(move)
        if board.is_attacked_by(opponent, move.to_square):
            adjustment -= PENALTY_HANG
        board.pop()

        # ── Endgame heuristic bonus ────────────────────────────────────────
        if self._is_endgame(board):
            adjustment += ENDGAME_WEIGHT * self._endgame_heuristic(board, move)

        # ── Loop / repetition penalty ──────────────────────────────────────
        adjustment -= self._loop_penalty(move.uci(), board, move)

        return adjustment

    # -------------------------
    # Endgame helpers
    # -------------------------

    def _is_endgame(self, board: chess.Board) -> bool:
        """Return True when 6 or fewer major/minor pieces remain."""
        major_pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        count = sum(
            len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
            for pt in major_pieces
        )
        return count <= 6

    def _endgame_heuristic(self, board: chess.Board, move: chess.Move) -> float:
        """
        Positional score for endgame moves. Three components:
          1. Captures    — 10× piece value
          2. King proximity — reward closing in on the enemy king
          3. Pawn advancement — reward pushing pawns toward promotion

        board.turn flips after push(), so 'our' colour is `not board.turn` inside.
        """
        score = 0.0
        board.push(move)

        if board.is_capture(move):
            score += self._piece_value(board.piece_at(move.to_square)) * 10

        our_king_sq   = board.king(not board.turn)
        enemy_king_sq = board.king(board.turn)
        if our_king_sq and enemy_king_sq:
            score += 14 - chess.square_distance(our_king_sq, enemy_king_sq)

        our_color = not board.turn
        for sq in board.pieces(chess.PAWN, our_color):
            rank = chess.square_rank(sq)
            score += (rank if our_color == chess.WHITE else 7 - rank) * 0.5

        board.pop()
        return score

    # -------------------------
    # Loop prevention
    # -------------------------

    @staticmethod
    def _position_key(board: chess.Board) -> str:
        return board.board_fen()

    def _loop_penalty(self, uci: str, board: chess.Board, move: chess.Move) -> float:
        """
        Penalty from two independent signals:
          1. Move history  — penalises recently repeated UCI move strings,
                             catching A→B→A oscillations.
          2. Position counts — penalises returning to positions seen this game,
                               catching loops that travel different move paths.
        """
        penalty = 0.0

        recent = self._move_history[-HISTORY_DEPTH:]
        if uci in recent[-2:]:
            penalty += PENALTY_RECENT
        elif uci in recent:
            penalty += PENALTY_OLDER

        board.push(move)
        penalty += self._position_counts[self._position_key(board)] * 2.0
        board.pop()

        return penalty

    # -------------------------
    # Main API
    # -------------------------

    def get_move(self, fen: str) -> str | None:
        t0    = time.time()
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        self._position_counts[self._position_key(board)] += 1

        uci_moves   = [m.uci() for m in legal_moves]
        src_tensor  = torch.tensor(
            self.tokenizer.encode(fen, is_target=False), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # ── Model scoring ──────────────────────────────────────────────────
        try:
            scores_dict = self.score_legal_moves(src_tensor, uci_moves)
            scores      = [scores_dict[uci] for uci in uci_moves]
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "[%s] Falling back to random move due to scoring error: %s",
                self.name, exc,
            )
            return random.choice(uci_moves)

        # ── Heuristic adjustments ──────────────────────────────────────────
        adjusted = [
            score + self._heuristic_adjustment(board, move)
            for score, move in zip(scores, legal_moves)
        ]

        best_idx = max(range(len(adjusted)), key=lambda i: adjusted[i])
        chosen   = uci_moves[best_idx]

        self._move_history.append(chosen)
        logger.info("[%s] Move: %s | Time: %.2fs", self.name, chosen, time.time() - t0)
        print(f"[{self.name}] Move: {chosen} | Time: {time.time() - t0:.2f}s")
        return chosen
