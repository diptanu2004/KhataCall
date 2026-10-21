# Getting the eval's background noise

The eval's noise conditions need a few minutes of realistic shop-street noise.
We use **field recordings of Indian streets and markets from Freesound**,
licensed **CC0** (public domain), downloaded in their original lossless form.
Why not a YouTube video: see DECISIONS.md D-033.

## Steps (about 5 minutes)

1. Make a free account at [freesound.org](https://freesound.org). Downloading
   originals requires one.
2. Open each page below and press **Download**. Keep the file names exactly as
   they come: they start with the sound's ID, which is how the import traces
   each file back to its page.

   | Sound | Why |
   |---|---|
   | [516655](https://freesound.org/s/516655/): Delhi street, traffic, horns, Hindi voices, bazaar | closest to a kirana counter on a busy road |
   | [263635](https://freesound.org/s/263635/): Trivandrum street market, voices, tuktuks, honking | market chatter, a second city |
   | [552692](https://freesound.org/s/552692/): flea market in India | crowd without much traffic |

   All three were CC0 when checked on Oct 22, 2026. Two is enough; three gives
   the eval more variety.
3. Put the downloaded files in `eval/noise/incoming/`.
4. Run:

   ```bash
   uv run python -m scripts.get_noise
   ```

   It converts each to mono 16 kHz WAV in `eval/noise/` and records the source
   and licence in `eval/noise/SOURCES.md` (committed; the audio is not).

## Other files

Anything else can be imported the same way, but the script will not guess a
licence: pass it explicitly.

```bash
uv run python -m scripts.get_noise --source https://example.org/clip --license "CC BY 4.0"
```

Your own recording of a local street is fine too (`--license "own recording"`).
Avoid clearly audible, identifiable conversations.

## Optional: a benchmark reference

`uv run python -m scripts.get_noise --demand PCAFETER` fetches one environment
of DEMAND (CC BY 4.0), the noise used in the VoiceBank-DEMAND benchmark. It is
European, so it is reported as a separate reference condition, not as the shop.
Zenodo served it at about 55 KB/s when tried, so expect about half an hour.
