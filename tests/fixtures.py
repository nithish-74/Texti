"""Fake documents for tests. Every number here is invented or generated; none is real."""

from texti.privacy.patterns import luhn_valid, verhoeff_check_digit


def make_aadhaar(prefix: str = "23456789012") -> str:
    """A 12-digit number that passes the Verhoeff checksum (not a real Aadhaar)."""
    assert len(prefix) == 11
    return prefix + verhoeff_check_digit(prefix)


def make_card(prefix: str = "411111111111111") -> str:
    """A 16-digit number that passes the Luhn checksum (a standard test pattern)."""
    for check in "0123456789":
        if luhn_valid(prefix + check):
            return prefix + check
    raise AssertionError("unreachable")


AADHAAR = make_aadhaar()
AADHAAR_SPACED = f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}"
CARD = make_card()
CARD_SPACED = " ".join(CARD[i : i + 4] for i in range(0, 16, 4))
PAN = "ABCPE1234F"
PASSPORT = "K1234567"
IFSC = "SBIN0001234"
ACCOUNT = "123456789012345"
PHONE = "98765 43210"
EMAIL = "student.demo@example.com"

LECTURE_TEXT = """Lecture 4: Binary Search Trees
Learning outcomes: by the end of this unit you can define a BST and insert nodes.
Definition. A binary search tree keeps smaller keys on the left of each node.
Theorem. Searching a balanced BST takes O(log n) comparisons.
Assignment: implement insert and delete. References: the course textbook, chapter 12.
"""

BANK_STATEMENT_TEXT = f"""State Bank Demo - Bank Statement
Account No: {ACCOUNT}   IFSC: {IFSC}
01 Jan  Salary credit        45,000.00
05 Jan  Rent payment        -15,000.00
Closing balance 30,000.00
"""

ID_CARD_TEXT = f"""Government of Demo
Name: Test Person
Aadhaar number: {AADHAAR_SPACED}
Permanent Account Number: {PAN}
"""

PRIVATE_NOTE_TEXT = "Shopping list: milk, bread, tomatoes. Call mom on Sunday."
