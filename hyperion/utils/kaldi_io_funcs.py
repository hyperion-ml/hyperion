"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Low-level helpers for reading/writing Kaldi token and scalar fields.
"""
import struct
from typing import Any, IO


def init_kaldi_output_stream(f: IO[Any], binary: bool) -> None:
    """Write the Kaldi binary stream marker when needed.

    Args:
        f: Open writable file handle.
        binary: If ``True``, writes the Kaldi binary marker ``b"\\0B"``.
    """
    if binary:
        f.write(b"\0B")


def init_kaldi_input_stream(f: IO[Any]) -> bool:
    """Detect and consume the Kaldi binary stream marker.

    Args:
        f: Open readable file handle.

    Returns:
        ``True`` if the next two bytes are the Kaldi binary marker
        and they were consumed, otherwise ``False``.
    """
    if peek(f, True, 2) == b"\0B":
        f.read(2)
        return True
    return False


def check_token(token: str) -> None:
    """Validate that a token can be written in Kaldi token format.

    Args:
        token: Token string to validate.

    Raises:
        AssertionError: If ``token`` contains spaces.
    """
    assert token.find(" ") == -1, "Token %s is not valid" % token


def is_token(token: str) -> bool:
    """Return whether ``token`` is valid in Kaldi token format.

    A valid token is non-empty, printable, and contains no spaces.

    Args:
        token: Token string to test.

    Returns:
        ``True`` if the token is valid, else ``False``.
    """
    if len(token) == 0:
        return False
    if not token.isprintable():
        return False
    if " " in token:
        return False
    return True


def read_token(f: IO[Any], binary: bool) -> str:
    """Read one ASCII token delimited by a space or EOF.

    Args:
        f: Open readable file handle.
        binary: If ``False``, leading spaces are skipped before reading.

    Returns:
        Decoded token string (ASCII).
    """
    if not binary:
        while f.peek(1) == b" ":
            f.read(1)
        token = b""
    else:
        token = b""
    while 1:
        c = f.read(1)
        if c == b" " or c == b"":
            break
        token += c

    return token.decode("ascii")


def write_token(f: IO[Any], binary: bool, token: str) -> None:
    """Write a token followed by a trailing space.

    Args:
        f: Open writable file handle.
        binary: If ``True``, writes bytes; otherwise writes text.
        token: Token string to write. Must not contain spaces.
    """
    check_token(token)
    token = "%s " % token
    if binary:
        token = token.encode("ascii")
    f.write(token)


def peek(f: IO[Any], binary: bool, num_bytes: int = 1) -> bytes:
    """Peek bytes from the input stream without consuming them.

    Args:
        f: Open readable file handle supporting ``peek``.
        binary: If ``False``, leading spaces are skipped before peeking.
        num_bytes: Number of bytes to peek.

    Returns:
        A bytes object with up to ``num_bytes`` bytes (normally exact unless
        near end-of-stream).
    """
    if not binary:
        while f.peek(1)[0] == " ":
            f.read(1)
    p = f.peek(num_bytes)[:num_bytes]
    peek_bytes = len(p)
    if peek_bytes < num_bytes:
        f.read(peek_bytes)
        delta_bytes = num_bytes - peek_bytes
        p = p + f.peek(delta_bytes)[:delta_bytes]
        f.seek(-peek_bytes, 1)
    return p


def read_int32(f: IO[Any], binary: bool) -> int:
    """Read one 32-bit integer from the stream.

    Args:
        f: Open readable file handle.
        binary: If ``True``, expects Kaldi binary int32 format
            (size byte + little-endian payload). Otherwise reads ASCII digits
            terminated by a space.

    Returns:
        Parsed integer value.
    """
    if binary:
        size = int(struct.unpack("b", f.read(1))[0])
        assert size == 4, "Wrong size %d" % size
        val = struct.unpack("<i", f.read(4))[0]
        return val
    while f.peek(1) == " ":
        f.read(1)
    token = ""
    while 1:
        c = f.read(1)
        if c == " ":
            break
        token += c

    return int(token)


def write_int32(f: IO[Any], binary: bool, val: int) -> None:
    """Write one 32-bit integer to the stream.

    Args:
        f: Open writable file handle.
        binary: If ``True``, writes Kaldi binary int32 format
            (size byte + little-endian payload). Otherwise writes text with
            a trailing space.
        val: Integer value to write.
    """
    if binary:
        f.write(struct.pack("b", 4))
        f.write(struct.pack("<i", val))
    else:
        f.write("%d " % val)
