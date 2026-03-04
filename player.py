"""
TransformerPlayer — Magnus Carlsen-style chess player.
Fine-tuned Qwen2.5-0.5B on Magnus Carlsen's Lichess games.

Scores all legal moves by log-probability under the model.
Key optimisation: all legal moves are scored in a single batched
forward pass, so inference is O(1) in the number of legal moves
instead of O(N).  This keeps the tournament clock happy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
get_move() is structured as a layered early-exit pipeline:

  1.  Checkmate grab          — hard rule, bypasses model entirely
  2.  Queen promotion         — hard rule, bypasses model entirely
  3.  Draw avoidance          — filters candidate move list
  4.  Tactical override       — returns best free capture if one exists
  5.  Endgame heuristics      — bypasses model when few pieces remain
  6.  Hang filter             — filters candidate move list
  7.  Time budget check       — emergency bail-out before inference
  8.  Model scoring (batched) — core LM inference
  9.  Loop prevention         — post-scoring penalty on recently played moves

Each layer either returns a move immediately (early exit) or narrows
the candidate list passed to the next layer.  The model is only ever
asked to score moves that have already survived the earlier filters —
a smaller, cleaner question: "among tactically reasonable moves, which
looks most like something Magnus would play?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import time
import chess
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player
from collections import defaultdict

# ── Time budget ────────────────────────────────────────────────────────────────
# Maximum seconds allowed per move. The time-budget check in get_move fires at
# 80% of this value, leaving a safety margin before the tournament clock expires.
# Increase if running on GPU; decrease on slow CPU hardware.
MOVE_TIME_BUDGET = 5.0

# ── Move history penalty ───────────────────────────────────────────────────────
# Controls how strongly recently played moves are discouraged.
# HISTORY_DEPTH  — how many of our own past moves to look back through.
# PENALTY_RECENT — penalty applied to a move played in the last 1-2 turns.
# PENALTY_OLDER  — penalty applied to a move played 3-HISTORY_DEPTH turns ago.
# Raise these values if loops persist; lower them if the bot becomes too
# reluctant to repeat a genuinely good move.
HISTORY_DEPTH  = 6
PENALTY_RECENT = 4.0
PENALTY_OLDER  = 2.0


class TransformerPlayer(Player):

    HF_MODEL_ID: str = "Jochemvkem/magnusbot-qwen"
    # Qwen2.5-0.5B fine-tuned on Magnus Carlsen's Lichess games.
    # At 0.5B parameters it is fast enough for batched CPU inference but
    # has a limited tactical ceiling — hence all the heuristic overrides.

    def __init__(self, name: str = "MagnusBot"):
        super().__init__(name)
        self._model           = None
        self._tokenizer       = None
        self._device          = None

        # Tracks how many times each piece arrangement has been reached this
        # game (keyed by board_fen, which is side-to-move agnostic).
        # Used by the position-based repetition penalty alongside move history.
        self._position_counts = defaultdict(int)

        # Ordered list of UCI move strings we have played this game.
        # Used by the move-history penalty to directly penalise recently
        # repeated moves, catching A→B→A oscillations without needing to
        # inspect board state.
        self._move_history    = []

    def reset_game(self):
        """
        Clear all per-game state.
        Call this at the start of every new game so history from a previous
        game does not bleed into the next one.
        """
        self._position_counts.clear()
        self._move_history.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────────────────────
    def _load(self):
        """
        Load the tokeniser and model on first call only.
        Deferred loading avoids paying the ~10-30s startup cost at import time
        and keeps tournament setup fast.  Subsequent calls are no-ops due to
        the early return guard.
        """
        if self._model is not None:
            return

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self.name}] Loading model on {self._device} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )
        # Qwen2 was not trained with an explicit pad token.
        # We alias it to the EOS token so left-padding in batched inference
        # doesn't introduce an unknown token ID.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # float16 halves memory usage on CUDA with negligible accuracy loss.
        # float32 is used on CPU because some builds produce NaNs in float16.
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[{self.name}] Ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Core inference — batched log-prob scoring
    # ──────────────────────────────────────────────────────────────────────────
    def _score_moves_batched(self, prompt: str, uci_moves: list) -> list:
        """
        Score every move in uci_moves in a SINGLE forward pass.

        Why batched?
        ────────────
        A naive implementation would call the model once per legal move.
        With ~20-35 legal moves in a typical position that would be 20-35
        serial forward passes — far too slow for tournament play.
        By stacking all (prompt + move) sequences into one batch we pay the
        forward-pass cost exactly once, regardless of how many moves there are.

        How it works
        ────────────
        For each candidate move we construct:
            full_text = "<FEN string> <uci_move>"

        All sequences are left-padded to the same length so that the final
        tokens (the move tokens) align across the batch.  Left-padding is
        important here: the model reads left-to-right, so we want the move
        tokens to sit at the same positions in every row of the batch tensor.

        After one forward pass we read off, for each move, the sum of
        log P(token_t | tokens_0..t-1) over just the move tokens.
        This gives a log-probability score: how likely is the model to
        generate this move given the FEN prompt?

        A higher (less negative) score = the model thinks Magnus would be
        more likely to play this move.
        """
        import torch
        import torch.nn.functional as F

        tok = self._tokenizer

        # Tokenise the prompt once — reused for every move.
        # add_special_tokens=False avoids a leading BOS token that would
        # shift the prompt/move boundary and corrupt the score calculation.
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        n_prompt   = len(prompt_ids)

        # Build (prompt + move) token sequences and record how many tokens
        # each move contributes so we know which tokens to score later.
        full_ids_list = []
        move_lengths  = []
        for uci in uci_moves:
            full_text = prompt + " " + uci
            ids = tok.encode(full_text, add_special_tokens=False)
            full_ids_list.append(ids)
            move_lengths.append(len(ids) - n_prompt)

        # ── Left-pad all sequences to the same length ──────────────────────
        # The attention mask zeros out pad positions so they don't influence
        # the model's attention over real tokens.
        max_len    = max(len(ids) for ids in full_ids_list)
        pad_id     = tok.pad_token_id
        input_ids  = []
        attn_masks = []
        for ids in full_ids_list:
            pad_len = max_len - len(ids)
            input_ids.append([pad_id] * pad_len + ids)
            attn_masks.append([0] * pad_len + [1] * len(ids))

        input_tensor = torch.tensor(input_ids,  dtype=torch.long).to(self._device)
        attn_tensor  = torch.tensor(attn_masks, dtype=torch.long).to(self._device)

        # ── Single forward pass ────────────────────────────────────────────
        with torch.no_grad():
            logits    = self._model(input_tensor, attention_mask=attn_tensor).logits
            log_probs = F.log_softmax(logits, dim=-1)
        # log_probs shape: (batch, seq_len, vocab_size)
        # log_probs[b, t, v] = log P(token v | tokens 0..t) for batch item b

        # ── Safety check ───────────────────────────────────────────────────
        # If a move tokenises to only 1 token, move_start = max_len - 1 and
        # we'd read a log_prob from a padding position for short prompts.
        for i, (uci, mv_len) in enumerate(zip(uci_moves, move_lengths)):
            move_start = max_len - mv_len
            assert move_start > 0, (
                f"Move '{uci}' tokenises to {mv_len} token(s), "
                f"leaving no room before padding boundary (max_len={max_len}). "
                f"Prompt may be too short or move too long."
            )

        # ── Extract per-move scores ────────────────────────────────────────
        # For each move, sum log P(move_token_t | everything before t)
        # over all tokens that belong to the move (not the prompt).
        scores = []
        for b, (ids, mv_len) in enumerate(zip(full_ids_list, move_lengths)):
            score      = 0.0
            move_start = max_len - mv_len
            for t in range(move_start, max_len):
                target_token = input_tensor[b, t].item()
                # log_probs[b, t-1, target] = log P(token at t | tokens 0..t-1)
                score += log_probs[b, t - 1, target_token].item()
            scores.append(score)

        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # Sequential fallback
    # ──────────────────────────────────────────────────────────────────────────
    def _score_move_single(self, prompt: str, move_uci: str) -> float:
        """
        Score a single move with its own forward pass.

        Fallback if batched scoring raises an exception (e.g. OOM).
        Slower — O(N) in move count — but robust.
        """
        import torch

        full_text  = prompt + " " + move_uci
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        full_ids   = self._tokenizer.encode(
            full_text, add_special_tokens=False, return_tensors="pt"
        ).to(self._device)
        n_prompt   = len(prompt_ids)

        with torch.no_grad():
            logits    = self._model(full_ids).logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        score = 0.0
        for i in range(n_prompt - 1, full_ids.shape[1] - 1):
            score += log_probs[i, full_ids[0, i + 1].item()].item()
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Tactical helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _is_draw_move(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Return True if playing this move leads to a draw position.

        Checks threefold repetition and the fifty-move rule.
        Used to filter the candidate list before model scoring so the model
        never voluntarily steps into a known draw when winning alternatives exist.
        """
        board.push(move)
        is_draw = board.is_repetition(count=3) or board.can_claim_fifty_moves()
        board.pop()
        return is_draw

    def _get_tactical_override(self, board: chess.Board, legal_moves: list) -> Optional[chess.Move]:
        """
        Return the best free capture available, or None.

        Scans all captures for undefended enemy pieces — if we capture and the
        opponent cannot recapture, the piece was hanging and we take it for free.
        Returns the highest-value such capture, bypassing model scoring entirely.

        Exists because the model was trained on games that mostly end by
        resignation and may pass up free material in favour of positional play.
        """
        opponent = not board.turn

        hanging_captures = []
        for move in legal_moves:
            if board.is_capture(move):
                captured_sq = move.to_square
                board.push(move)
                recapturable = board.is_attacked_by(opponent, captured_sq)
                board.pop()

                if not recapturable:
                    captured_piece = board.piece_at(captured_sq)
                    value = self._piece_value(captured_piece)
                    hanging_captures.append((value, move))

        if hanging_captures:
            hanging_captures.sort(key=lambda x: x[0], reverse=True)
            return hanging_captures[0][1]

        return None

    @staticmethod
    def _piece_value(piece) -> int:
        """
        Standard material values used by tactical and endgame heuristics.
        King is assigned 0 — it cannot be captured in legal play.
        """
        if piece is None:
            return 0
        values = {
            chess.PAWN:   1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK:   5,
            chess.QUEEN:  9,
            chess.KING:   0,
        }
        return values.get(piece.piece_type, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # Endgame helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _is_endgame(self, board: chess.Board) -> bool:
        """
        Return True when 6 or fewer major/minor pieces remain in total.

        The model is bypassed in endgames because the training data is heavily
        weighted toward middlegame positions — most Lichess games end by
        resignation before a true endgame.  The explicit heuristic in
        _endgame_score encodes the three key endgame principles far more
        reliably than the model can.
        """
        major_pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        count = sum(
            len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
            for pt in major_pieces
        )
        return count <= 6

    def _endgame_score(self, board: chess.Board, move: chess.Move) -> float:
        """
        Heuristic scoring for endgame moves. Three components:

        1. Captures (10× piece value) — material gain dominates everything.
        2. King proximity (14 - Chebyshev distance) — reward closing in on
           the enemy king, which is essential for delivering checkmate.
        3. Pawn advancement (0.5 × rank) — reward pushing pawns toward
           promotion.

        Note: board.turn is flipped after board.push(), so "our" colour is
        `not board.turn` inside the push/pop block.
        """
        score = 0.0
        board.push(move)

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            score += self._piece_value(captured) * 10

        our_king_sq   = board.king(not board.turn)
        enemy_king_sq = board.king(board.turn)
        if our_king_sq and enemy_king_sq:
            dist = chess.square_distance(our_king_sq, enemy_king_sq)
            score += (14 - dist)

        our_color = not board.turn
        for sq in board.pieces(chess.PAWN, our_color):
            rank = chess.square_rank(sq)
            advancement = rank if our_color == chess.WHITE else (7 - rank)
            score += advancement * 0.5

        board.pop()
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Loop prevention
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _position_key(board: chess.Board) -> str:
        """
        Return a position key that is independent of whose turn it is.

        board.fen() includes side-to-move, halfmove clock, and fullmove number,
        meaning the same piece arrangement on White's turn and Black's turn
        would produce different keys — breaking cross-turn repetition tracking.
        board.board_fen() returns only the piece placement portion, which is
        the same regardless of turn or clock fields.
        """
        return board.board_fen()

    def _loop_penalty(self, uci: str, board: chess.Board, move: chess.Move) -> float:
        """
        Combined loop-prevention penalty from two independent signals:

        1. MOVE HISTORY (move-literal)
           ───────────────────────────
           Checks whether this UCI string appears in our recent move history.
           Directly catches A→B→A→B oscillations without inspecting board state.
           Simple and fast — just a list lookup, no push/pop required.
             - Played in last 1-2 of our turns → PENALTY_RECENT
             - Played further back in HISTORY_DEPTH window → PENALTY_OLDER

        2. POSITION COUNTS (board-state aware)
           ─────────────────────────────────────
           Checks how many times the resulting position (by piece arrangement,
           ignoring whose turn it is) has been seen this game.
           Catches loops that use different move sequences to reach the same
           position, which the move-literal check would miss.
           Penalty = visit_count × 2.0

        Both penalties are summed. Using both gives loop detection at the move
        level and the position level simultaneously.

        Why soft penalties rather than hard bans?
        A hard ban could eliminate all moves in a position the bot legitimately
        needs to revisit.  The soft penalty makes repetition increasingly
        unattractive the more times it has occurred, while still allowing it
        if all alternatives score worse.
        """
        penalty = 0.0

        # Signal 1: move history
        recent = self._move_history[-HISTORY_DEPTH:]
        if uci in recent[-2:]:
            penalty += PENALTY_RECENT
        elif uci in recent:
            penalty += PENALTY_OLDER

        # Signal 2: position counts
        board.push(move)
        penalty += self._position_counts[self._position_key(board)] * 2.0
        board.pop()

        return penalty

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────
    def get_move(self, fen: str) -> Optional[str]:
        """
        Main entry point — return the best UCI move string for the given FEN.

        Executes the layered early-exit pipeline described in the module
        docstring.  Each stage either returns immediately or passes a narrowed
        candidate list to the next stage.
        """
        t0 = time.time()
        self._load()

        board       = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Record current position for the position-count signal in _loop_penalty.
        # Keyed by board_fen (piece arrangement only, side-to-move agnostic).
        self._position_counts[self._position_key(board)] += 1

        # ── Step 1: Checkmate grab ─────────────────────────────────────────
        # Hard rule — return any move that immediately ends the game.
        # Bypasses all scoring. The training data underrepresents checkmate
        # delivery (most Lichess games end by resignation) so the model cannot
        # be trusted to find it reliably.
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # ── Step 2: Queen promotion ────────────────────────────────────────
        # Hard rule — always promote to queen if the option exists.
        # Promotions are rare in the training data and the model may miss them.
        for move in legal_moves:
            if move.promotion == chess.QUEEN:
                return move.uci()

        # ── Step 3: Draw avoidance ─────────────────────────────────────────
        # Filter out moves that lead to threefold repetition or the 50-move
        # rule, unless ALL moves lead to draws (forced draw situation).
        non_draw_moves = [m for m in legal_moves if not self._is_draw_move(board, m)]
        if non_draw_moves:
            legal_moves = non_draw_moves

        # ── Step 4: Tactical override — free captures ──────────────────────
        # Take the highest-value undefended enemy piece if one exists.
        # Fires before model scoring so the model never has to decide whether
        # to take free material.
        tactical = self._get_tactical_override(board, legal_moves)
        if tactical:
            return tactical.uci()

        # ── Step 5: Endgame heuristics ─────────────────────────────────────
        # Bypass the model entirely when few pieces remain, using explicit
        # heuristics for king activity, captures, and pawn advancement.
        if self._is_endgame(board):
            endgame_scores = [self._endgame_score(board, m) for m in legal_moves]
            best_idx = max(range(len(endgame_scores)), key=lambda i: endgame_scores[i])
            return legal_moves[best_idx].uci()

        # ── Step 6: Hang filter ────────────────────────────────────────────
        # Remove moves that walk our own pieces into attacked squares.
        # Falls back to the full list if all moves are unsafe.
        opponent = not board.turn

        def hangs_piece(move: chess.Move) -> bool:
            board.push(move)
            result = board.is_attacked_by(opponent, move.to_square)
            board.pop()
            return result

        safe_moves = [m for m in legal_moves if not hangs_piece(m)]
        if safe_moves:
            legal_moves = safe_moves

        # ── Step 7: Time budget check ──────────────────────────────────────
        # Skip model inference if 80% of the time budget has already elapsed.
        if time.time() - t0 > MOVE_TIME_BUDGET * 0.8:
            print(f"[{self.name}] Time budget nearly exceeded before inference, picking first safe move.")
            return legal_moves[0].uci()

        # ── Step 8: Model scoring ──────────────────────────────────────────
        # Score each remaining candidate by log-probability under the LM.
        # All moves are scored in a single batched forward pass.
        prompt    = fen
        uci_moves = [m.uci() for m in legal_moves]

        try:
            scores = self._score_moves_batched(prompt, uci_moves)
        except Exception as e:
            print(f"[{self.name}] Batched scoring failed ({e}), falling back to sequential.")
            scores = [self._score_move_single(prompt, uci) for uci in uci_moves]

        # ── Step 9: Loop prevention penalty ───────────────────────────────
        # Subtract a combined penalty from each model score based on:
        #   - whether this UCI move appears in our recent move history
        #   - how many times the resulting position has been seen this game
        # The final move is the argmax of (model_score - loop_penalty).
        penalties = [
            self._loop_penalty(uci, board, move)
            for uci, move in zip(uci_moves, legal_moves)
        ]
        adjusted  = [s - p for s, p in zip(scores, penalties)]
        best_idx  = max(range(len(adjusted)), key=lambda i: adjusted[i])
        chosen    = uci_moves[best_idx]

        # Record chosen move in history before returning.
        self._move_history.append(chosen)

        print(f"[{self.name}] Move: {chosen} | Time: {time.time() - t0:.2f}s")
        return chosen