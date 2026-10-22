"""The eval's condition grid and its seeding."""

import numpy as np
import pytest
import soundfile as sf

from eval.conditions import CONDITIONS, by_id, load_noise_bank, noise_for, rng_for


def test_the_grid_is_the_planned_ten():
    assert [c.id for c in CONDITIONS] == [
        "clean", "tel", "tel_loss1", "tel_loss5", "tel_loss10",
        "tel_snr20", "tel_snr10", "tel_snr5", "tel_snr0", "worst",
    ]
    assert len({c.id for c in CONDITIONS}) == len(CONDITIONS)


def test_only_clean_stays_wideband():
    assert [c.id for c in CONDITIONS if not c.telephone] == ["clean"]


def test_worst_case_is_noise_plus_loss():
    worst = by_id("worst")
    assert (worst.snr_db, worst.loss_pct) == (5, 5)


def test_unknown_condition_names_the_known_ones():
    with pytest.raises(KeyError, match="tel_snr10"):
        by_id("nope")


def test_seeds_are_stable_and_independent():
    """Same utterance and condition, same randomness; anything else, different."""
    a = rng_for("001", "tel_snr10").random(4)
    assert np.array_equal(a, rng_for("001", "tel_snr10").random(4))
    assert not np.array_equal(a, rng_for("001", "tel_snr5").random(4))
    assert not np.array_equal(a, rng_for("002", "tel_snr10").random(4))


def test_an_utterance_gets_the_same_noise_in_every_condition(tmp_path):
    for name in ("freesound_1", "freesound_2", "freesound_3"):
        sf.write(tmp_path / f"{name}.wav", np.full(16_000, 0.1, dtype=np.float32), 16_000)
    bank = load_noise_bank(tmp_path)
    assert set(bank) == {"freesound_1", "freesound_2", "freesound_3"}
    assert noise_for("017", bank)[0] == noise_for("017", bank)[0]
    used = {noise_for(f"{i:03d}", bank)[0] for i in range(1, 61)}
    assert used == set(bank), "60 utterances should spread across every noise file"


def test_noise_that_was_not_imported_properly_is_refused(tmp_path):
    sf.write(tmp_path / "freesound_9.wav", np.zeros((8_000, 2), dtype=np.float32), 8_000)
    with pytest.raises(ValueError, match="re-import"):
        load_noise_bank(tmp_path)


def test_no_noise_files_is_an_explained_error():
    with pytest.raises(ValueError, match="NOISE.md"):
        noise_for("001", {})


def test_the_real_noise_bank_loads_if_present():
    bank = load_noise_bank()
    if not bank:
        pytest.skip("noise not imported on this machine")
    assert all(len(audio) > 20 * 16_000 for audio in bank.values())
