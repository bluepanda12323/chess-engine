# Bigbotturtle

A chess engine built from scratch as a portfolio project for the Deep Learning course (Data & AI, KdG).
The search is written in Rust and positions are evaluated by a neural network (NNUE) trained with PyTorch.
The engine plays on Lichess as **bigbotturtle**.

## About the bot

| | |
|---|---|
| Lichess account | [@bigbotturtle](https://lichess.org/@/bigbotturtle) |
| Current version | **v27-tuned** |
| Lichess rating | **~2410 blitz** (after 147 rated games, November 2026) |
| Engine | Rust, UCI protocol, runs with 4 threads |
| Evaluation | NNUE (768 piece inputs + 384 pawn-structure inputs → 1024×2 → 1). First trained on ~90M positions from the Lichess evaluation database, then fine-tuned on ~143M positions from that database plus 50M positions from rated Lichess games (evaluation + game result) |

### Play against it

When the bot is online you can challenge it on Lichess:

- standard chess only
- bullet, blitz, rapid and classical (up to 30 minutes + 20 seconds increment)
- rated or casual
- one game at a time

The bot runs on my own laptop, so it is not always online.

### Strength

- **Lichess:** about 2410 in blitz, mostly against other bots.
- **Lichess Stockfish AI:** beat level 4 ([game](https://lichess.org/EVhTNDbK)) and level 6 ([game](https://lichess.org/olaWvcx0)) with black at 3+2 (version v14-search).
- **Stockfish 19 with limited strength** (60+0.6, version v15-history):

| Stockfish setting | Result for Bigbotturtle |
|---|---|
| Elo 2000 | +35 =0 −1 |
| Elo 2400 | +31 =0 −4 |
| Elo 2800 | +14 =12 −8 |

Stockfish's limited-strength Elo is only an approximation, so these results are a rough indication, not an official rating.

## Version history

★ = has been used by the Lichess bot.

| Version | What changed |
|---|---|
| v1-mlp | First Rust engine: alpha-beta search with a small neural network (MLP) as evaluation |
| v2-quiescence | Quiescence search: keeps looking at captures so the engine never stops in the middle of an exchange |
| v3-tt | Transposition table: remembers positions that were already searched |
| v4-killers | Killer moves for better move ordering |
| v5-nullmove | Null-move pruning |
| v6-lmr | Late move reductions |
| v7-time | Time management: how long to think per move |
| v8-nnue | First NNUE evaluation, replacing the MLP |
| v9-no-tanh | NNUE without tanh on the output |
| v10-syzygy | Syzygy endgame tablebases: perfect play with 5 pieces or fewer |
| v11-nnue1024 ★ | Bigger network (1024 hidden neurons), hyperparameters tuned with Optuna |
| v12-mobility | Network with extra inputs for piece counts and mobility |
| v13-mobility-fast | Much faster NNUE code, still with the v12 network |
| v14-search ★ | Many search improvements: draw detection, check extensions, futility pruning, late move pruning, SEE, aspiration windows |
| v15-history ★ | Move ordering and pruning based on history tables |
| v16-smp ★ | Multithreading (Lazy SMP), shared hash table, pondering, correction history |
| v17-time | Different time management |
| v17-falling | Thinks longer when the evaluation drops |
| v18-fix | Bug fix (engine could hang while pondering in a lost position); all pruning margins became settings |
| v19-speed ★ | Faster network updates and smarter move generation |
| v20-tt | Hash table with buckets, singular extensions added (off by default) |
| v21-screlu | New network with SCReLU activation, trained longer |
| v22-datagen | Self-play data generator for training future networks |
| v23-combined ★ | v20 with singular extensions and "think longer when the evaluation drops" turned on |
| v24-int16 ★ | Faster network evaluation with 16-bit integers and AVX2 instructions |
| v25-pawns ★ | New network: extra inputs for pawn structure (passed, isolated and doubled pawns), fine-tuned from v11 on more data, including positions from real Lichess games with their result |
| v26-l2-8, v26-l2-16, v26-l2-24 | v25 network with an extra hidden layer of 8, 16 or 24 neurons |
| v27-tuned ★ | v25 with its 12 search settings tuned automatically by SPSA (3000 game pairs of self-play) |
