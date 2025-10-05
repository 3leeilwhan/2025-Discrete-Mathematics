import time

# 행렬 출력
def matrixout(mx, size):
    print("\u250c" + "        " * size + " \u2510")
    for i in range(size):
        print("|", end=" ")
        for j in range(size):
            print("%7.4f" %mx[i][j], end=" ")
        print("|")
    print("\u2514" + "        " * size + " \u2518")


# 전치 행렬 구하기
def transposeMatrix(m):
    return[[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


# 소행렬 구하기
def getMatrixMinor(m, i, j):
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]


# 행렬식 계산 (재귀)
def getMatrixDeterminant(m):
    if len(m) == 1:
        return m[0][0]
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * getMatrixDeterminant(getMatrixMinor(m, 0, c))
    return determinant


# 역행렬이 존재하는지 확인, 행렬이 존재하지 않을 경우(행렬식이 0인 경우) 오류 메시지를 출력
def has_inverse(m):
    determinant = getMatrixDeterminant(m)
    if abs(determinant) < 1e-10:     # 부동소수점 오차 고려
        print('오류! 해당 행렬의 행렬식은 0입니다')
        print('역행렬 계산은 불가능합니다!')
        return False
    return True

# 행렬식을 이용한 역행렬 계산
def getMatrixInverse(m):
    determinant = getMatrixDeterminant(m)

    if len(m) == 1:
        return [[1.0 / m[0][0]]]
    
    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]
    
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * getMatrixDeterminant(minor))
        cofactors.append(cofactorRow)

    adjugate = transposeMatrix(cofactors)

    for r in range(len(adjugate)):
        for c in range(len(adjugate)):
            adjugate[r][c] = adjugate[r][c] / determinant

    return adjugate


# 가우스-조던 소거법으로 역행렬 구하기
def inverse_gauss_jordan(m, eps: float = 1e-12, debug: bool = False):
    n = len(m)
    # [A | I] 만들기
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(m)]

    # 추가기능 1: 가우스-조던 소거법 단계별로 어떻게 계산되는지 보기
    if debug:
        print_augmented(aug, n, title="\n[초기 첨가행렬 [A | I]]")

    for col in range(n):
        # 피벗 선택(부분 피벗팅: 절댓값 최대 행)
        best_r = col
        best_val = abs(aug[col][col])
        for r in range(col + 1, n):
            val = abs(aug[r][col])
            if val > best_val:
                best_val = val
                best_r = r
        pivot_row = best_r

        # 피벗 유효성, 역행렬이 존재하지 않는 경우 예외 처리
        if abs(aug[pivot_row][col]) < eps:
            raise ZeroDivisionError("가우스-조던: 피벗이 0이라 역행렬이 존재하지 않습니다.")

        # 행 교환
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            if debug:
                print_augmented(aug, n, title=f"\n[Step {col+1}] 행 교환: row {col} <-> row {pivot_row}")

        # 피벗을 1로
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        if debug:
            print_augmented(aug, n, title=f"[Step {col+1}] 피벗 정규화 (pivot=1 @ ({col},{col}))")

        # 다른 행의 해당 열을 0으로
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) > eps:
                for j in range(2 * n):
                    aug[r][j] -= factor * aug[col][j]
                if debug:
                    print_augmented(aug, n, title=f"[Step {col+1}] 소거: row {r} -= {factor:.4g} * row {col}")

    if debug:
        print_augmented(aug, n, title="[최종] [I | A^{-1}]")

    # [I | A^{-1}] -> 오른쪽 절반이 역행렬
    inv = [row[n:] for row in aug]
    return inv



# 두 행렬이 같은지 확인
def matrices_same(A, B, tol: float = 1e-8):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    n, m = len(A), len(A[0])
    for i in range(n):
        for j in range(m):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


# 추가기능 1: 가우스-조던 소거법 단계별로 어떻게 계산되는지 보기
def print_augmented(aug, n, ndigits=4, title=None):
    if title:
        print(title)
    for i in range(n):
        left = " ".join(f"{aug[i][j]: .{ndigits}f}" for j in range(n))
        right = " ".join(f"{aug[i][j]: .{ndigits}f}" for j in range(n, 2*n))
        print(f"[ {left} | {right} ]")



# 추가기능 3: A x A⁻¹ = I인지 아닌지를 계산하여 역행렬이 잘 구해졌는지 검증
def matmul(A, B):
    m, k, n = len(A), len(A[0]), len(B[0])
    assert k == len(B), "행렬 곱 차원 불일치"
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]


def max_abs_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def verify_inverse(A, inv, rtol=1e-8, atol=1e-10):

    n = len(A)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    prod = matmul(A, inv)
    err = max_abs_diff(prod, I)              # ||A·A^{-1} - I||_max
    ok = err <= (atol + rtol)                # 대각 1 기준의 간단한 임계값
    return ok






# 메인 실행부
def main():
    try:
        k = int(input("정방행렬의 차수를 입력하시오: "))
        if k <= 0:
            print("차수는 양의 정수여야 합니다.")
            return

        print(f"{k}x{k} 정방행렬 A를 입력하세요 (각 행을 공백으로 구분하여 한 줄씩 입력): ")
        matrix_a = []
        for i in range(k):
            while True:
                try:
                    row_input = input(f"{i+1}행: ").strip()
                    row_values = [float(x) for x in row_input.split()]
                    if len(row_values) != k:
                        print(f"입력 오류: 정확히 {k}개의 값을 입력해야 합니다.")
                        continue
                    matrix_a.append(row_values)
                    break
                except ValueError:
                    print("입력 오류: 숫자만 입력해주세요.")

        # (1) 행렬식·여인수 방식 (행렬식이 0이면 건너뜀) 
        inv_det = None
        t0 = t1 = None
        if has_inverse(matrix_a):
            t0 = time.perf_counter()
            inv_det = getMatrixInverse(matrix_a)
            t1 = time.perf_counter()

        # (2) 가우스-조던 소거법 (피벗 실패로 비가역 판단)
        dbg_in = input("가우스-조던 단계 출력(debug) 모드로 볼까요? (y/n): ").strip().lower()
        debug_mode = (dbg_in == "y")

        inv_gj = None
        t2 = t3 = None
        try:
            t2 = time.perf_counter()
            inv_gj = inverse_gauss_jordan(matrix_a, debug=debug_mode)
            t3 = time.perf_counter()
        except ZeroDivisionError as e:
            print("\n[가우스-조던] 실패:", e)

        # 출력 (adjugate) 
        if inv_det is not None:
            print("\n[행렬식·여인수 방식] 역행렬:")
            matrixout(inv_det, k)
            ok1 = verify_inverse(matrix_a, inv_det)
            print("[검증: adjugate]: A x A⁻¹ = I ->", "검증 성공" if ok1 else "검증 실패")
        else:
            print("\n[행렬식·여인수 방식] 행렬식이 0이라 역행렬을 계산하지 않았습니다.")

        # 출력 (Gauss–Jordan) 
        if inv_gj is not None:
            print("\n[가우스-조던 소거법] 역행렬:")
            matrixout(inv_gj, k)
            ok2 = verify_inverse(matrix_a, inv_gj)
            print("[검증: Gauss-Jordan]: A x A⁻¹ = I ->", "검증 성공" if ok2 else "검증 실패")
        else:
            print("\n[가우스-조던 소거법] 역행렬을 얻지 못했습니다.")

        # 두 방식 결과 비교 
        if inv_det is not None and inv_gj is not None:
            same = matrices_same(inv_det, inv_gj, tol=1e-8)
            print("\n[비교] 두 방식의 결과가", "동일합니다." if same else "일치하지 않습니다.")
        else:
            print("\n[비교] 두 결과 중 하나가 없어 비교할 수 없습니다.")

        # 추가기능 2: 실행시간 비교
        print("\n[실행시간 비교]")
        if t0 is not None and t1 is not None:
            print(f" - adjugate(행렬식·여인수): {(t1 - t0) * 1000:.3f} ms")
        else:
            print(" - adjugate(행렬식·여인수): 행렬식 0으로 미실행")
        if t2 is not None and t3 is not None:
            print(f" - Gauss-Jordan           : {(t3 - t2) * 1000:.3f} ms")
        else:
            print(" - Gauss-Jordan           : 피벗 실패로 미완료")

    except ValueError:
        print("입력 오류: 정수를 입력해주세요. ")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

    print()

    

# 프로그램 시작점
if __name__ == "__main__":
    main()




