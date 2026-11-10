# KhataCall: written explanation

Submission for Gnani AI's Great Indian AI Internship Challenge 2026.
Repository: https://github.com/diptanu2004/KhataCall

**Models used:** Gnani Prisma v2.5 (speech to text), Gnani Evon v3.3 (LLM, tool
calls, self-hosted), Gnani Timbre v2.5 (text to speech). No other provider's model
is in the product.

## Short version

KhataCall lets a kirana shop owner keep the udhaar book by speaking Hinglish into
a phone: "Sharma ji ka dedh sau udhaar likho." Prisma transcribes it, Evon proposes
a tool call, and deterministic code parses the amount, reads it back and writes
nothing until the owner says "haan". To test it on the audio shop owners actually
have, we played 180 recorded requests through 12 simulated phone-line conditions.
On held-out recordings, it gets 72% of requests fully right on clean audio and 55%
on a mobile call with shop noise at 10 dB, and almost all of the rest it asks again
rather than guessing. Five fixes in code cut wrong entries by 37-56% on a quiet
line. Replies start 2.0 s after the owner stops speaking (p50).

## The problem

A kirana store runs on credit: neighbours buy now and pay later, and the owner
writes it in a notebook between customers. Speaking is the natural way to record
it. The owner's hands are busy, the phone is already in their pocket, and Hinglish
is how the shop talks. But a voice agent that handles money has one job above all:
**never write a wrong amount with confidence.** Asking again costs a few seconds. A
wrong entry costs money and the owner's trust. A busy shop on a mobile line is also
the hardest audio for speech recognition.

## What KhataCall does

Six tools: add credit, record a payment, check a balance, update stock, check
stock, and list what is running low. The owner holds a button and speaks. The reply
is spoken back and the ledger on screen updates live. The page is built for a
phone, and a **phone-line mode** runs the owner's live speech through the same
simulated mobile call the evaluation used, so a judge can hear what Prisma heard.

## Which Gnani model does what

- **Prisma (speech to text)** with `en-IN`, which covers code-mixed Hinglish. The
  live path streams to Prisma's realtime WebSocket while the owner speaks, which
  made that leg 17x faster than REST. Phone-line mode and the evaluation send
  8 kHz telephone audio to Prisma REST, which accepts it natively.
- **Evon (LLM)** reads the transcript and proposes one tool call, with the amount
  *as spoken* ("dedh sau"). Gnani does not host Evon v3.3, so we serve the open
  weights ourselves: a 4-bit GGUF under llama.cpp on Kaggle's two T4 GPUs, with
  native tool calls. Given perfect text, it gets 92% of our corpus right.
- **Timbre (text to speech)** speaks the read-back and the answer in Hinglish
  (voice Poorvi, `hi-en`), streamed to the browser. Its socket is opened when the
  button is pressed, so the handshake hides behind the owner's speech.

## How it keeps the book correct

- **The LLM never writes to the database.** Evon proposes; an executor validates
  the proposal and holds it as a pending action.
- **Amounts are parsed by code, not the LLM.** A parser with over 200 test cases
  handles irregular Hindi numbers, *sau / hazaar / lakh*, the fraction words
  (*sava, saadhe, paune, dedh, dhai*), English and digits. It returns "ask again" on
  anything ambiguous and never guesses. Money is integer paise everywhere.
- **The read-back is built from the values that will be saved,** and a yes must be
  only a yes. "Gupta ji ka do sau likho", said while Sharma ji's entry waits, is a
  new instruction, not agreement.
- **On a phone line the amount is spelled out** ("ek paanch shunya"), because Prisma
  exposes no confidence score to decide when it misheard.

## How we evaluated it

- **Corpus:** 180 scripted Hinglish requests across the six intents, all names made
  up, recorded by one speaker. 60 were the **dev set**, whose failures we read to
  design fixes. The other 120 were recorded afterwards and are the **held-out set**,
  used only to measure.
- **A telephone channel simulator,** applied in the order a real call degrades
  audio. First, shop noise (CC0 street recordings from Indian markets) is mixed in at
  a target SNR measured on the speech. Then the audio is band-limited and resampled
  to 8 kHz, encoded with GSM 06.10 then G.711, and hit with bursty packet loss. That
  gives 12 conditions, from clean audio to 0 dB noise with 10% loss.
- **The headline metric is task outcome, split three ways:**
  - **correct:** the intent, the customer and the amount are all right;
  - **caught:** the agent asks again and writes nothing;
  - **wrong:** the agent would read back a wrong value with confidence.

  The outcome is scored with the agent's own parser and name matcher. Word error
  rate is reported alongside.
- **Cost:** every result is cached by audio hash, condition and model, so re-runs
  are free. The whole project used Rs 102 of the Rs 5,000 credit grant.

## Results (held-out set, 120 takes per condition)

| Condition | Correct | Wrong (before → after fixes) |
|---|---|---|
| clean audio | 71.7% | 6.7% → 3.3% |
| mobile call | 70.8% | 9.2% → 5.8% |
| mobile + 10% packet loss | 65.8% | 13.3% → 8.3% |
| mobile + shop noise 10 dB | 55.0% | 10.0% → 5.0% |
| mobile + shop noise 0 dB | 24.2% | 41.7% → 35.0% |
| 0 dB noise + 10% loss | 15.0% | 65.0% → 47.5% |

**Latency** with all three real models: 2.03 s from the end of speech to the first
word of the reply (p50; p90 2.27 s). A "haan" confirming an entry takes 0.60 s,
because yes/no is answered in code without calling Evon.

## What we learned

1. **Background noise hurts; the phone network barely does.** A mobile codec with
   10% packet loss moves word error rate from 16% to 19%; noise at 0 dB moves it to
   52%.
2. **Most of what is lost on clean audio is transcription, not reasoning.** Evon gets
   92% right from perfect text and the pipeline gets 72% from clean audio. The gap
   is number words: Prisma hears *pachchees* (25) as *pachas* (50), and romanised
   spellings vary from take to take.
3. **The fixes buy safety more than accuracy.** Five checks in code, such as "the
   number Evon passes must be one the owner said", cut wrong answers in every
   condition: by 37-56% on a quiet line and 16-29% in heavy noise. Correct answers
   rose only 2.5-4.2 points, because no code can recover words the line destroyed.
4. **The held-out set kept us honest.** On the dev set the fixes took clean accuracy
   from 70% to 80%; on held-out recordings, from 68% to 72%.

## Limitations and the path to production

- **Limitations:** push-to-talk with no barge-in, one speaker in the corpus, a
  simulated (not carrier) phone channel, and Evon on Kaggle as a demo host.
- **Path to production:**
  - real calls through a telephony media stream, which delivers the 8 kHz μ-law
    audio the simulator models;
  - end-of-speech detection from Prisma's VAD;
  - many concurrent calls, with session state in a shared store;
  - per-shop tenancy on Postgres;
  - Evon on a dedicated GPU, which removes the tunnel's 0.4 s.

The README has the full details: <https://github.com/diptanu2004/KhataCall#readme>.
