class ResponseValidator:

    @staticmethod
    def validate(
        response
    ):

        if not response:

            return False

        if len(response) < 5:

            return False

        return True