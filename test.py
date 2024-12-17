

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        while left < right:
            mid = (left + right) >> 1
            x, y = divmod(mid, n)
            if matrix[x][y] >= target:
                right = mid
            else:
                left = mid + 1
        return matrix[left // n][left % n] == target

test_case_generator_results = []
solution = Solution()
for _ in range(100):
    m = random.randint(1, 100)
    n = random.randint(1, 100)
    matrix = [[random.randint(-10000, 10000) for _ in range(n)] for _ in range(m)]
    target = random.randint(-10000, 10000)
    expected_result = solution.searchMatrix(matrix, target)
    
    test_case = f"assert solution.searchMatrix({matrix}, {target}) == {expected_result}"
    test_case_generator_results.append(test_case)
