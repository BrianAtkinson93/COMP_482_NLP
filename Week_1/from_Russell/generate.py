import re
from typing import List, Optional


def gen_word(a_in: List[str], n_in: int, r_in: int) -> str:
    """
    Generate a word based on the list a_in, using binary numbers.

    :param a_in: List of characters to construct the word from.
                 The list should contain exactly 2 elements.
    :param n_in: Length of the word to be generated.
    :param r_in: Remainder to use for word generation.
    :return: Generated word.
    """
    word = ""
    for j in range(n_in - 1, -1, -1):
        bit = 0  # bit value chooses letter
        # r now represents remainder
        if r_in // 2 ** j > 0:
            r_in = r_in % 2 ** j
            bit = 1
        word += a_in[bit]
    return word


def print_match(m: Optional[re.Match], word: str) -> None:
    """
    Print the match result alongside the word.

    :param m: Match object, could be None.
    :param word: The original word.
    """
    if m is not None:
        print(m.group(0) + " (match)")
    else:
        print(word)


def main(n: int, a: List[str], pattern: re.Pattern) -> None:
    """
    The main flow of the program.

    :param n: The number of combinations we would like.
    :param a: List containing the values we are willing to use.
    :param pattern: Compiled regular expression pattern to match against.
    """
    for i in range(2 ** n):
        word = gen_word(a, n, i)
        m = re.match(pattern, word)
        print_match(m, word)


if __name__ == "__main__":
    number = 3
    str_list = ['a', 'b']
    compiled_pattern = re.compile(r'^a*ba*$')

    main(number, str_list, compiled_pattern)
