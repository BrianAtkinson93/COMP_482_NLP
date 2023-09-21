"""
The Minimum Edit Distance algorithm is a technique used in natural language processing,
computational biology, and other fields to measure the "distance" between two sequences.
The distance is calculated as the minimum number of edit operations required to transform
one sequence into another. The basic edit operations are:

Insertion: Insert a character into a string.
Deletion: Remove a character from a string.
Substitution: Replace a character in a string with another character.

Dynamic Programming Approach

The most common way to solve this problem is by using dynamic programming.
The idea is to build a matrix where the cell at (i, j) represents the minimum edit
distance between the first i characters of string A and the first j characters of string B.

Time Complexity
The time complexity of this algorithm is O(m×n), where m and n are the lengths of the two strings.
The space complexity is also O(m×n).
"""
import sys


def min_edit_distance_recursive(input_a: str, input_b: str, m: int, n: int) -> int:
    """
    Calculate the minimum edit distance between two strings recursively.

    Parameters:
    - input_a: The first string to compare.
    - input_b: The second string to compare.
    - m: The length of input_a.
    - n: The length of input_b.

    Returns:
    - The minimum edit distance between input_a and input_b.
    """

    # Base Cases: If one string is empty, the minimum edit distance is the length of the other string.
    if m == 0:
        return n
    if n == 0:
        return m

    # Recursive Cases: Calculate the minimum edit distance based on the last characters of the strings.
    # If the last characters are the same, no operation is needed.
    if input_a[m - 1] == input_b[n - 1]:
        return min_edit_distance_recursive(input_a, input_b, m - 1, n - 1)

    # Otherwise, consider all three operations: insertion, deletion, and substitution.
    insertion = min_edit_distance_recursive(input_a, input_b, m, n - 1) + 1
    deletion = min_edit_distance_recursive(input_a, input_b, m - 1, n) + 1
    substitution = min_edit_distance_recursive(input_a, input_b, m - 1, n - 1) + 1

    # Return the minimum cost among the three operations.
    return min(insertion, deletion, substitution)


def min_edit_distance(in_a: str, in_b: str) -> int:
    """
    Calculate the minimum edit distance between two strings using dynamic programming.

    Parameters:
    - in_a: The first string to compare.
    - in_b: The second string to compare.

    Returns:
    - The minimum edit distance between in_a and in_b.
    """
    # Get the lengths of the input strings.
    m, n = len(in_a), len(in_b)

    # Initialize a 2D list (m+1 x n+1) with zeros.
    # This will serve as our dynamic programming table.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    print(f'Instantiated')
    for row in dp:
        print(row)

    # Initialization: Filling in the first row and first column.
    # The first row represents the cost of converting a substring of input_a to an empty string.
    # The first column represents the cost of converting an empty string to a substring of input_b.
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    print(f'\nPopulated')
    print(f'The first row represents the cost of converting a substring of the first string to an empty string')
    print(f'The first column represents the cost of converting an empty string to a substring of the second string')
    for row in dp:
        print(row)

    # Main Loop: Filling in the rest of the dp table.
    # We start from index 1 because index 0 is already initialized.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the cost of insertion.
            insertion = dp[i][j - 1] + 1

            # Calculate the cost of deletion.
            deletion = dp[i - 1][j] + 1

            # Calculate the cost of substitution.
            # If the characters are the same, the cost is 0. Otherwise, it's 1.
            substitution = dp[i - 1][j - 1] + (0 if in_a[i - 1] == in_b[j - 1] else 1)

            # Store the minimum cost among insertion, deletion, and substitution.
            dp[i][j] = min(insertion, deletion, substitution)

    print(f'\nFilled')
    print(f'The value at dp[i][j] represents the minimum edit distance between the first i characters of the '
          f'first string and the first j characters of the second string.\n'
          f'The value is calculated as the minimum cost among insertion, deletion, and substitution.')

    " This is an example of how to do formatted the pythonic way "
    " You can use < for left, ^ for middle, > for right"
    # Define the format string
    format_string = "{:<3} " * len(dp[0])  # Assuming all rows have the same length
    # format_string = "{:^3} " * len(dp[0])  # Assuming all rows have the same length
    # format_string = "{:>3} " * len(dp[0])  # Assuming all rows have the same length

    # Loop through each row in dp
    for row in dp:
        # Format the row and print it
        formatted_row = format_string.format(*row)
        print(formatted_row)

    print(f'\nThe value at the bottom-right corner (dp[{len(dp) - 1}][{len(dp[0]) - 1}] = {dp[-1][-1]}) is the minimum '
          f'edit distance between the two full strings')

    # The value at dp[m][n] contains the minimum edit distance between input_a and input_b.
    return dp[m][n]


if __name__ == "__main__":
    # Test the function
    A = "#inention"
    B = "#extention"

    print("\nMinimum Edit Distance:", min_edit_distance(A, B))

    print("\nRecursive example:")
    m, n = len(A), len(B)
    print("Minimum Edit Distance:", min_edit_distance_recursive(A, B, m, n))
