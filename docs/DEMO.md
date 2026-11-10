# Recording the 60-second demo

The story: a shop owner on a noisy mobile line puts Rs 150 on Sharma ji's
account. KhataCall reads the amount back digit by digit, the owner says "haan",
and the ledger updates. Then the chart shows the measured result behind it.

## Before you start (about 15 minutes, most of it Evon)

1. **Start Evon on Kaggle** ([EVON_KAGGLE.md](EVON_KAGGLE.md), about 10 minutes)
   and put its new address in `.env`. Then run the demo's lines through it as text
   (no credits, about a minute) and check every scenario says `ok`:

   ```bash
   uv run python -m scripts.demo_check
   ```

2. **A fresh ledger** (Sharma ji owes Rs 400, Mehra ji Rs 200):

   ```bash
   uv run python -m scripts.seed --reset --db scripts/out/demo.db
   ```

3. **The server**, on the demo ledger. `--reload-include .env` restarts it when
   `.env` is saved, so a new Evon address needs no restart by hand:

   ```bash
   DB_PATH=scripts/out/demo.db uv run uvicorn app.main:app --port 8010 --reload --reload-include .env
   ```

4. **An https address for the phone** (phones allow the microphone only over https).
   Use Cloudflare's quick tunnel (install once with `winget install --id Cloudflare.cloudflared`,
   then open a new terminal):

   ```bash
   cloudflared tunnel --url http://127.0.0.1:8010    # prints https://<random>.trycloudflare.com
   ```

   Not localhost.run (`ssh -R 80:127.0.0.1:8010 nokey@localhost.run`): it needs nothing
   installed, but on Nov 10 its free tier delivered the reply audio slower than it plays.
   The page then holds each reply until it has arrived, which adds seconds. The Log panel
   says how fast each reply arrived ("reply audio: 4.0 s arrived in 8.1 s (0.5x real time)").

   The address is public and has no login: anyone who has it can spend credits
   and change the ledger. Stop it (Ctrl+C) as soon as the recording is done.
5. **On the phone:** open the address and tap **Tap to turn on the mic** (allow the
   microphone). The dot at the bottom turns green. Do one rehearsal turn, then reset (below).

## How to film it

- **Best: film the phone with a second device** (a laptop webcam or another phone).
  It records your voice, KhataCall's reply and the screen together, and needs nothing
  special on the phone.
- **Or use the phone's screen recorder with device audio only.** Your voice will not
  be in the recording, but the transcript is on screen and KhataCall's replies are.
  "Device audio and microphone" can take the microphone away from the browser on
  some phones: if you want it, test one take first.
- Keep the page in the foreground for the whole take: KhataCall turns the microphone
  off when the page goes to the background.

## The script

| Time | On screen | Said |
|---|---|---|
| 0:00-0:06 | The app. **Khata** shows Sharma ji: Rs 400 | *Voice-over:* "This is KhataCall. A kirana owner keeps the udhaar book by talking, in Hinglish, on a phone." |
| 0:06-0:12 | Open **Phone-line mode**, switch it on: shop noise **10 dB**, packet loss **0%**. The panel shows the measured result for that setting: 55% correct, 5% wrong | *Voice-over:* "Now a simulated mobile call from a noisy shop, the same channel our eval measured." |
| 0:12-0:24 | Hold the mic | *You:* "Sharma ji ka dedh sau udhaar likho."<br>*KhataCall:* "Sharma ji ke khate mein ek paanch shunya, 150 rupaye udhaar likh doon?" |
| 0:24-0:29 | Tap **▶ Hear what Prisma heard** | The noisy 8 kHz clip plays. *Voice-over:* "That is what the speech model actually heard." |
| 0:29-0:37 | Hold the mic | *You:* "Haan."<br>*KhataCall:* "Likh diya. Sharma ji ka ab 550 rupaye baaki hai."<br>Sharma ji's row goes from Rs 400 to Rs 550 |
| 0:37-0:43 | Stay on the ledger | *Voice-over:* "Nothing is written until the owner says yes. The amount is parsed by code, not by the model, and on a phone line it is read back digit by digit." |
| 0:43-0:55 | The chart: `eval/results/m4_2026-11-08_heldout_fixes.png` | *Voice-over:* "We played 180 recorded requests through 12 phone-line conditions. On the held-out set, our fixes halve wrong entries on a clean line, and cut them from 65% to 48% on the harshest. When the agent is unsure, it asks again instead of writing a wrong amount." |
| 0:55-1:00 | Back to the app | *Voice-over:* "Prisma hears, Evon decides, Timbre speaks. KhataCall." |

The chart is a separate shot: screen-record it on the laptop, or open it on the
phone from the repo, and join the clips.

**To show a multi-step instruction instead,** swap 0:12-0:37 for the Mehra ji bill
below. With phone-line mode on, the read-back gives all three numbers and spells out
the one that will be written: "Mehra ji ka bill 650 rupaye, 500 rupaye mile. Baaki ek
paanch shunya, 150 rupaye udhaar likh doon?" Then "haan" gives "Likh diya. Mehra ji ka
ab 350 rupaye baaki hai." *Voice-over:* "The arithmetic is done in code, from the
owner's own words, and the owner hears all three numbers before anything is written."

## More lines to try

Each line below gets exactly the reply shown, pinned by a test
(`tests/test_demo_scenarios.py`). Start from a fresh ledger and say each one once,
in any order; the two Mehra ji lines are alternatives, so say only one of them. With
phone-line mode on, the amount that will be written is also spelled out digit by digit.

| Say | KhataCall answers | Then |
|---|---|---|
| "Mehra ji ka bill saadhe chhe sau ka hua, paanch sau diye, baaki khate mein likh do" | "Mehra ji ka bill 650 rupaye, 500 rupaye mile. Baaki 150 rupaye udhaar likh doon?" | "haan ji" → "Likh diya. Mehra ji ka ab 350 rupaye baaki hai." |
| "Mehra ji ne paanch sau diye, bill saadhe chhe sau ka tha, baaki khate mein likh do" | the same read-back: the order does not matter, and "ne ... diye" does not turn it into a payment | "haan" → "Likh diya. Mehra ji ka ab 350 rupaye baaki hai." |
| "Yadav ji ka kitna baaki hai?" | "Yadav ji ka 1,500 rupaye baaki hai." | |
| "Gupta aunty ka hisaab batao" | "Gupta aunty ka hisaab chukta hai." | |
| "Joshi uncle ka kitna baaki hai?" | "Joshi uncle ke 100 rupaye jama hain." (he paid ahead) | |
| "Chauhan bhai ne ek hazaar diye" | "Chauhan bhai ke khate mein 1,000 rupaye jama kar doon?" | "haan" → "Likh diya. Chauhan bhai ka ab 1,500 rupaye baaki hai." |
| "Verma bhaiya ka do sau likho" | "Verma bhaiya ke khate mein 200 rupaye udhaar likh doon?" | "nahi" → "Theek hai, nahi likha." Then "Verma bhaiya ka kitna baaki hai?" → still "350 rupaye baaki hai." |
| "Kapoor ji ka teen sau udhaar likho" | "Kapoor ji naya customer hai. Kapoor ji ke khate mein 300 rupaye udhaar likh doon?" | "haan" → "Likh diya. Kapoor ji ka ab 300 rupaye baaki hai." |
| "Maggi kitni bachi hai?" | "Maggi 12 packet bachi hai." | |
| "Kya kya khatam hone wala hai?" | "Ye kam hain: tel 9 litre, Maggi 12 packet." | |
| "Parle-G ke bees packet aaye" | "Parle-G ke 20 packet aaye, likh doon?" | "haan" → "Likh diya. Parle-G ab 68 packet." |
| "Paanch kilo chini bik gayi" | "chini ke 5 kilo nikal gaye, likh doon?" | "haan" → "Likh diya. chini ab 25 kilo." |

The tests pin what the code says once Evon has proposed the call. Whether Evon
proposes it is what `scripts.demo_check` checks against the real model: all 13
lines passed on Nov 10, twice in a row.

## If a take goes wrong

- **"Amount samajh nahi aaya. Kitne rupaye?"** or **"Naam phir se boliye."** The noise
  hid a word and the agent asked again instead of guessing. That is the safety
  feature working, but for the video, reset and take it again.
- **Takes keep failing at 10 dB:** use 20 dB (59% correct when measured) or no noise
  (71%). Your phone's microphone is not the one the eval was recorded on.
- **The dot turns red, or the reply never comes:** check Evon
  (`uv run python -m scripts.evon_check`); the Kaggle session may have ended.
- **The read-back has the wrong amount or name:** say "nahi". Nothing is written
  ("Theek hai, nahi likha"), and that can be a good extra shot.

## Reset between takes

Close the tab on the phone first (an open page keeps the ledger file open), then:

```bash
uv run python -m scripts.seed --reset --db scripts/out/demo.db
```

Open the address again; Sharma ji is back at Rs 400.
