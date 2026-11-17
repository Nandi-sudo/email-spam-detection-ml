from __future__ import annotations

import email
import email.policy
import io
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen


CORPORA = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
}


def download_and_extract(url: str, destination: Path) -> None:
    """Download a SpamAssassin tarball and safely extract it into destination."""
    request = Request(url, headers={"User-Agent": "email-spam-detection-ml/1.0"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as archive:
        for member in archive.getmembers():
            # Avoid writing archive entries outside the requested data directory.
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                continue
            target = (destination / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def extract_text(message: email.message.Message) -> str:
    """Return subject and readable text/plain body from an email message."""
    pieces: list[str] = []
    subject = message.get("subject", "")
    if subject:
        pieces.append(str(subject))

    if message.is_multipart():
        parts = message.walk()
    else:
        parts = [message]

    for part in parts:
        if part.get_content_type() != "text/plain":
            continue
        try:
            body = part.get_content()
        except (LookupError, UnicodeDecodeError, AttributeError):
            payload = part.get_payload(decode=True) or b""
            body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if isinstance(body, str) and body.strip():
            pieces.append(body)

    return "\n".join(pieces).strip()


def load_corpus(data_dir: Path, download: bool = True) -> tuple[list[str], list[str]]:
    """Load SpamAssassin email messages, optionally downloading the corpus."""
    if download:
        for folder, url in CORPORA.items():
            if not (data_dir / folder).exists():
                print(f"Downloading {folder} corpus...")
                download_and_extract(url, data_dir)

    texts: list[str] = []
    labels: list[str] = []
    for folder, label in (("easy_ham", "ham"), ("spam", "spam")):
        corpus_dir = data_dir / folder
        if not corpus_dir.exists():
            continue
        for path in sorted(corpus_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("!"):
                continue
            try:
                raw = path.read_bytes()
                message = email.message_from_bytes(raw, policy=email.policy.default)
                text = extract_text(message)
            except (OSError, email.errors.MessageError, ValueError):
                continue
            if text:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise RuntimeError(
            f"No email messages found in {data_dir}. Check your internet connection "
            "or remove the data directory and try again."
        )
    if len(set(labels)) < 2:
        raise RuntimeError("Both ham and spam examples are required to train the model.")
    return texts, labels
