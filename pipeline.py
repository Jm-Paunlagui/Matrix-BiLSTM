import re


class TextPreprocessing:
    """Text preprocessing class."""

    def __init__(self, text: str):
        self.text = text

    def remove_punctuation(self) -> str:
        """
        Remove punctuation from the text.

        :return: The text without punctuation
        """
        return re.sub(r'[^\w\s]', '', self.text)

    def remove_numbers(self) -> str:
        """
        Remove numbers from the text.

        :return: The text without numbers
        """
        return re.sub(r'\d+', '', self.text)

    def remove_non_ascii_characters(self) -> str:
        """
        Remove non-ASCII characters from the text.

        :return: The text without non-ASCII characters
        """
        return self.text.encode("ascii", "ignore").decode()

    def remove_tabs_carriage_newline(self) -> str:
        """
        Remove tabs, carriage and newline characters from the text.

        :return: The text without tabs, carriage and newline characters
        """
        return re.sub(r'[\t\r\n]', '', self.text)

    def remove_whitespace(self) -> str:
        """
        Remove whitespace from the text.

        :return: The text without whitespace
        """
        return " ".join(self.text.split())

    def remove_special_characters(self) -> str:
        """
        Remove special characters from the text.

        :return: The text without special characters
        """
        return re.sub(r'[^\w\s]', '', self.text)

    def remove_multiple_whitespaces(self) -> str:
        """
        Remove multiple whitespaces from the text.

        :return: The text without multiple whitespaces
        """
        return re.sub(r'\s+', ' ', self.text)

    def remove_urls(self) -> str:
        """
        Remove URLs from the text.

        :return: The text without URLs
        """
        return re.sub(r'http\S+', '', self.text)

    def remove_emails(self) -> str:
        """
        Remove emails from the text.

        :return: The text without emails
        """
        return re.sub(r'\S+@\S+', '', self.text)

    def remove_html_tags(self) -> str:
        """
        Remove HTML tags

        :return: The text without HTML tags
        """
        return re.sub(r'<.*?>', '', self.text)

    def remove_whitespace_at_beginning_and_end(self) -> str:
        """
        Remove whitespace at the beginning and end of the text.

        :return: The text without whitespace at the beginning and end
        """
        return self.text.strip()

    def clean_text(self) -> str:
        """
        Clean the text.

        :return: The cleaned text
        """
        self.text = self.remove_punctuation()
        self.text = self.remove_numbers()
        self.text = self.remove_non_ascii_characters()
        self.text = self.remove_tabs_carriage_newline()
        self.text = self.remove_whitespace()
        self.text = self.remove_special_characters()
        self.text = self.remove_multiple_whitespaces()
        self.text = self.remove_urls()
        self.text = self.remove_emails()
        self.text = self.remove_html_tags()
        self.text = self.remove_whitespace_at_beginning_and_end()
        return self.text