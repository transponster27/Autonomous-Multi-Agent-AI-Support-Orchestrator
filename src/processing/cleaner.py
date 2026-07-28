import re


class TextCleaner:

    @staticmethod
    def clean(text: str):
        if not text:
            return ""

        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        text = re.sub(
            r'Page\s+\d+',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'\n+',
            '\n',
            text
        )

        text = text.strip()

        return text