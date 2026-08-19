class InsufficientHistoryError(Exception):
    def __init__(self, symbol: str, have_days: int, need_days: int):
        self.symbol = symbol
        self.have_days = have_days
        self.need_days = need_days
        super().__init__(
            f"{symbol}: only {have_days} cached trading day(s) available, "
            f"need at least {need_days}. Run again later (or on consecutive "
            f"days) to let the local cache accumulate more history."
        )


class DataFetchError(Exception):
    def __init__(self, symbol: str, message: str):
        self.symbol = symbol
        super().__init__(f"{symbol}: {message}")


class MissingCredentialsError(Exception):
    def __init__(self, provider: str, env_vars: list):
        self.provider = provider
        self.env_vars = env_vars
        vars_str = ", ".join(env_vars)
        super().__init__(
            f"{provider} credentials not configured -- missing env var(s): {vars_str}. "
            f"Copy .env.example to .env and fill them in, or export them in your shell."
        )
