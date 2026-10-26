"""Publication guard regressions using constructed, synthetic examples only."""
import unittest

from scripts.audit_public import sensitive_rules


class PublicationAuditTests(unittest.TestCase):
    def setUp(self):
        self.patterns = dict(sensitive_rules())

    def test_ipv4_with_literal_dots_is_detected(self):
        address = b".".join([b"203", b"0", b"113", b"42"])
        self.assertIsNotNone(self.patterns["IPv4 address"].search(address))

    def test_three_part_version_is_not_an_ipv4_address(self):
        self.assertIsNone(self.patterns["IPv4 address"].search(b"vllm 0.8.5"))

    def test_python_and_json_credential_assignments(self):
        for key in [b"password", b"passwd", b"api_key", b"access_token",
                    b"auth_token", b"client_secret"]:
            for separator in [b" = ", b"\t=\t", b"\n:\n"]:
                for quote in [b'"', b"'"]:
                    for quoted_key in [key, quote + key + quote]:
                        value = quoted_key + separator + quote + b"synthetic-value" + quote
                        with self.subTest(key=key, separator=separator, quote=quote):
                            self.assertIsNotNone(
                                self.patterns["credential assignment"].search(value))

    def test_environment_lookup_is_not_a_literal_credential(self):
        value = b"api_key" + b' = os.environ["SERVICE_KEY"]'
        self.assertIsNone(self.patterns["credential assignment"].search(value))


if __name__ == "__main__":
    unittest.main()
