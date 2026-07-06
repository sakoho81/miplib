import sys
import time

from miplib.processing.to_string import time_to_str


class ProgressBar:
    """Text-based terminal progress bar.

    Usage::

        bar = ProgressBar(0, 100)
        for i in range(100):
            bar(i)

    """

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = 100,
        total_width: int = 80,
        prefix: str = "",
        show_percentage: bool = True,
    ) -> None:
        self.min = min_value
        self.max = max_value
        self.span = max_value - min_value or 1  # avoid zero-division
        self.width = total_width
        self.show_percentage = show_percentage
        self.prefix = prefix

        self.amount = 0.0
        self.starting_amount: float | None = None
        self.start_time: float | None = None
        self.current_time = time.time()

        self._comment = ""
        self._comment_last = ""
        self._bar = ""
        self._bar_last = ""

        self.update_amount(0)

    def update_comment(self, comment: str) -> None:
        self._comment = comment

    def update_amount(self, new_amount: float = 0) -> None:
        """Update the progress bar with the new amount."""
        if self.starting_amount is None:
            self.starting_amount = new_amount
            self.start_time = time.time()

        new_amount = max(self.min, min(new_amount, self.max))
        self.amount = new_amount

        percent_done = int(round((self.amount - self.min) / self.span * 100))

        all_full = self.width - 2
        num_hashes = int(round(percent_done / 100 * all_full))

        if num_hashes == 0:
            self._bar = f"[>{' ' * (all_full - 1)}]"
        elif num_hashes == all_full:
            self._bar = f"[{'=' * all_full}]"
        else:
            self._bar = f"[{'=' * (num_hashes - 1)}>{' ' * (all_full - num_hashes)}]"

        if self.show_percentage:
            percent_place = int(len(self._bar) / 2 - len(str(percent_done)))
            label = f"{percent_done}%"
        else:
            percent_place = int(len(self._bar) / 2 - len(str(percent_done)))
            label = f"{int(self.amount)}/{self.span}"

        self._bar = (
            self._bar[:percent_place] + label + self._bar[percent_place + len(label) :]
        )

        if self.starting_amount is not None:
            amount_diff = self.amount - self.starting_amount
            if amount_diff:
                self.current_time = time.time()
                elapsed = self.current_time - self.start_time
                eta = elapsed * (self.max - self.amount) / amount_diff
                self._bar += f" ETA:{time_to_str(eta)}"

    def __str__(self) -> str:
        return self._bar

    def __call__(self, value: float) -> None:
        """Update the amount and write the progress bar to stdout."""
        self.update_amount(value)
        if self._bar_last == self._bar and self._comment == self._comment_last:
            return
        sys.stdout.write(f"\r {self.prefix}{self}{self._comment} ")
        sys.stdout.flush()
        self._bar_last = self._bar
        self._comment_last = self._comment
