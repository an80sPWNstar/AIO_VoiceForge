"""Tests for heartbeat progress reporting in audio cleanup."""

import unittest

import webui_audio_cleanup as audio_cleanup


class TestDurationFormatter(unittest.TestCase):
    """Test the _format_duration helper."""

    def test_format_seconds_under_minute(self):
        """Format seconds under 60 as just seconds."""
        self.assertEqual(audio_cleanup._format_duration(8), "8s")
        self.assertEqual(audio_cleanup._format_duration(1), "1s")
        self.assertEqual(audio_cleanup._format_duration(59), "59s")

    def test_format_minutes_and_seconds(self):
        """Format elapsed time with both minutes and seconds."""
        self.assertEqual(audio_cleanup._format_duration(75), "1m 15s")
        self.assertEqual(audio_cleanup._format_duration(125), "2m 5s")
        self.assertEqual(audio_cleanup._format_duration(3725), "62m 5s")

    def test_format_exact_minutes(self):
        """Format exact minutes without seconds component."""
        self.assertEqual(audio_cleanup._format_duration(60), "1m")
        self.assertEqual(audio_cleanup._format_duration(120), "2m")
        self.assertEqual(audio_cleanup._format_duration(600), "10m")


class TestCleanupProgressHeartbeat(unittest.TestCase):
    """Test CleanupProgress dataclass and heartbeat field."""

    def test_default_heartbeat_false(self):
        """CleanupProgress defaults heartbeat to False."""
        progress = audio_cleanup.CleanupProgress(
            stage="test",
            message="test message"
        )
        self.assertFalse(progress.heartbeat)

    def test_heartbeat_can_be_set_true(self):
        """CleanupProgress heartbeat field can be explicitly set to True."""
        progress = audio_cleanup.CleanupProgress(
            stage="test",
            message="test message",
            heartbeat=True
        )
        self.assertTrue(progress.heartbeat)

    def test_existing_constructions_unaffected(self):
        """Existing code constructing CleanupProgress still works."""
        # Without fraction
        p1 = audio_cleanup.CleanupProgress(
            stage="Denoise",
            message="Removing noise"
        )
        self.assertEqual(p1.stage, "Denoise")
        self.assertEqual(p1.message, "Removing noise")
        self.assertIsNone(p1.fraction)
        self.assertFalse(p1.heartbeat)

        # With fraction
        p2 = audio_cleanup.CleanupProgress(
            stage="Dereverb",
            message="Remove reverb and echo: processing",
            fraction=0.44
        )
        self.assertEqual(p2.stage, "Dereverb")
        self.assertEqual(p2.message, "Remove reverb and echo: processing")
        self.assertEqual(p2.fraction, 0.44)
        self.assertFalse(p2.heartbeat)

    def test_heartbeat_carries_same_fraction(self):
        """A heartbeat event reuses the fraction from the last real event."""
        # Simulate a real event
        real_event = audio_cleanup.CleanupProgress(
            stage="Dereverb",
            message="Remove reverb and echo: processing",
            fraction=0.44,
            heartbeat=False
        )

        # Construct a heartbeat with the same fraction
        heartbeat = audio_cleanup.CleanupProgress(
            stage=real_event.stage,
            message="Remove reverb and echo: processing (4m 10s)",
            fraction=real_event.fraction,
            heartbeat=True
        )

        # Verify the fraction hasn't moved
        self.assertEqual(real_event.fraction, 0.44)
        self.assertEqual(heartbeat.fraction, 0.44)
        self.assertEqual(heartbeat.fraction, real_event.fraction)
        self.assertTrue(heartbeat.heartbeat)
        self.assertFalse(real_event.heartbeat)


if __name__ == "__main__":
    unittest.main()
