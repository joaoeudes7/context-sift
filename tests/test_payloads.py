import base64
import json
import unittest

from compact_dataset.payloads import compact_base64


class PayloadTests(unittest.TestCase):
    def test_compacts_long_base64_with_stable_metadata(self) -> None:
        encoded = base64.b64encode(b"binary-image" * 100).decode()
        output = compact_base64(f"image={encoded}")
        self.assertRegex(output, r"^image=\[base64 bytes=1200 sha256=[0-9a-f]{16}\]$")

    def test_preserves_uuid_and_short_or_invalid_text(self) -> None:
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        invalid = "A" * 129
        self.assertEqual(compact_base64(f"trace={uuid} payload={invalid}"), f"trace={uuid} payload={invalid}")

    def test_json_remains_valid(self) -> None:
        encoded = base64.b64encode(b"x" * 256).decode()
        output = compact_base64(json.dumps({"id": "550e8400-e29b-41d4-a716-446655440000", "data": encoded}))
        parsed = json.loads(output)
        self.assertEqual(parsed["id"], "550e8400-e29b-41d4-a716-446655440000")
        self.assertTrue(parsed["data"].startswith("[base64 bytes=256 sha256="))
