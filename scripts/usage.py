"""What we have spent with Gnani, from the local meter.

    uv run python -m scripts.usage                 # summary
    uv run python -m scripts.usage --budget 5000   # against the programme grant
    uv run python -m scripts.usage --by-day

The programme grant is finite and non-renewable, so this is the number that
matters, not the call count. Rates and the 1 credit = Rs 1 reading are in
`app/adapters/usage.py` and DECISIONS.md D-020. This reflects only calls made
from this machine; the Gnani dashboard is the real authority.
"""

import argparse
from collections import defaultdict

from app.adapters.usage import (
    STT_INR_PER_HOUR,
    TTS_INR_PER_10K_CHARS,
    read_log,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=float, default=5000.0, help="programme grant (default 5000)")
    parser.add_argument(
        "--already-spent",
        type=float,
        default=5.0,
        help="credits used before metering existed (5 as of Oct 8, per the dashboard)",
    )
    parser.add_argument("--by-day", action="store_true")
    args = parser.parse_args()

    records = read_log()
    if not records:
        print("No usage recorded yet.")
        return 0

    stt = [r for r in records if r.service == "stt"]
    tts = [r for r in records if r.service == "tts"]
    stt_seconds = sum(r.units for r in stt)
    tts_chars = sum(r.units for r in tts)
    stt_cost = sum(r.cost_inr for r in stt)
    tts_cost = sum(r.cost_inr for r in tts)
    total = stt_cost + tts_cost

    print(f"Rates: STT Rs {STT_INR_PER_HOUR:.2f}/hour of audio, TTS Rs {TTS_INR_PER_10K_CHARS:.2f}/10k chars\n")
    print(f"  STT  {len(stt):5d} calls  {stt_seconds / 60:8.1f} min of audio   Rs {stt_cost:8.2f}")
    print(f"  TTS  {len(tts):5d} calls  {tts_chars:8.0f} characters   Rs {tts_cost:8.2f}")
    print(f"  {'':5}{'':12}{'':25} Rs {total:8.2f}  total\n")
    overall = total + args.already_spent
    print(f"  metered here      Rs {total:8.2f}")
    print(f"  before metering   Rs {args.already_spent:8.2f}  (dashboard, not in the log)")
    print(
        f"  spent {overall:.2f} of {args.budget:.0f}  ->  {args.budget - overall:.2f} left "
        f"({overall / args.budget * 100:.2f}% used)"
    )

    if args.by_day:
        per_day: dict[str, float] = defaultdict(float)
        for r in records:
            per_day[r.at[:10]] += r.cost_inr
        print("\n  by day:")
        for day in sorted(per_day):
            print(f"    {day}  Rs {per_day[day]:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
