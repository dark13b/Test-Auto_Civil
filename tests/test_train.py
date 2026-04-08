import io
import unittest
from unittest.mock import patch

import train


class FailingEncodingStream:
    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.buffer = io.BytesIO()
        self.flush_called = False

    def write(self, text: str) -> int:
        raise UnicodeEncodeError("charmap", text, 0, 1, "character maps to <undefined>")

    def flush(self) -> None:
        self.flush_called = True


class TrainLoggingTests(unittest.TestCase):
    def test_log_status_falls_back_to_ascii_safe_output_when_stream_cannot_encode_unicode(self) -> None:
        stream = FailingEncodingStream()

        with patch("sys.stdout", stream):
            train.log_status("Unicode warning \u26a0 and arabic \u0627")

        emitted = stream.buffer.getvalue().decode("ascii")
        self.assertIn("\\u26a0", emitted)
        self.assertIn("\\u0627", emitted)
        self.assertTrue(stream.flush_called)


if __name__ == "__main__":
    unittest.main()
