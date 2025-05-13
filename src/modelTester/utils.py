from colorama import Fore, Back, Style

def _custom_print(message: str, colorama_settings: list[str]) -> None:
    format_mess = f"{''.join(colorama_settings)}{message}"
    print(format_mess)

def print_top_level(message: str) -> None:
    _custom_print(
        message=f"\n{message}\n",
        colorama_settings=[Fore.BLACK, Back.WHITE]
    )

def print_success(message: str) -> None:
    _custom_print(
        message=message,
        colorama_settings=[Fore.GREEN]
    )

def print_failure(message: str) -> None:
    _custom_print(
        message=message,
        colorama_settings=[Fore.RED]
    )