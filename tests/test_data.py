from email.message import EmailMessage

from src.data import extract_text


def test_extract_text_includes_subject_and_plain_body():
    message = EmailMessage()
    message["Subject"] = "Important account notice"
    message.set_content("Please review your account details.")

    text = extract_text(message)

    assert "Important account notice" in text
    assert "Please review your account details." in text
