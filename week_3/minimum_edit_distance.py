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


def min_edit_distance(A, B):
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialization
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Main Loop
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            insertion = dp[i][j - 1] + 1
            deletion = dp[i - 1][j] + 1
            substitution = dp[i - 1][j - 1] + (0 if A[i - 1] == B[j - 1] else 1)

            dp[i][j] = min(insertion, deletion, substitution)

    return dp[m][n]


if __name__ == "__main__":
    # Test the function
    A = "kitten"
    B = "sitting"
    print("Minimum Edit Distance:", min_edit_distance(A, B))
