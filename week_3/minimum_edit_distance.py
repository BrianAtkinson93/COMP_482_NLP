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


def min_edit_distance(input_a: str, input_b: str) -> int:
    """
    Calculate the minimum edit distance between two strings using dynamic programming.

    Parameters:
    - input_a: The first string to compare.
    - input_b: The second string to compare.

    Returns:
    - The minimum edit distance between input_a and input_b.
    """
    # Get the lengths of the input strings.
    m, n = len(input_a), len(input_b)

    # Initialize a 2D list (m+1 x n+1) with zeros.
    # This will serve as our dynamic programming table.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    print(f'dp: {dp}')

    # Initialization: Filling in the first row and first column.
    # The first row represents the cost of converting a substring of input_a to an empty string.
    # The first column represents the cost of converting an empty string to a substring of input_b.
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

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
            substitution = dp[i - 1][j - 1] + (0 if input_a[i - 1] == input_b[j - 1] else 1)

            # Store the minimum cost among insertion, deletion, and substitution.
            dp[i][j] = min(insertion, deletion, substitution)

    # The value at dp[m][n] contains the minimum edit distance between input_a and input_b.
    return dp[m][n]


if __name__ == "__main__":
    # Test the function
    A = "kitten"
    B = "sitting"
    print("Minimum Edit Distance:", min_edit_distance(A, B))
