# 공통 유틸
def print_matrix(matrix, title=""):
    if title != "":
        print(title)
    for row in matrix:
        print(" ".join(str(x) for x in row))
    print()

def print_relation(matrix, A):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                print(f"({A[i]}, {A[j]})", end=" ")
    print()


# 성질 판별
def is_reflexive(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] != 1:
            return False
    return True

def is_symmetric(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_transitive(matrix):
    # (i,k)=1, (k,j)=1 이면 (i,j)=1 이어야 함
    n = len(matrix)
    for i in range(n):
        for k in range(n):
            if matrix[i][k] == 1:
                for j in range(n):
                    if matrix[k][j] == 1 and matrix[i][j] == 0:
                        return False
    return True

def print_property_report(matrix):
    r = is_reflexive(matrix)
    s = is_symmetric(matrix)
    t = is_transitive(matrix)

    print(f"- 반사(reflexive): {'예' if r else '아니오'}")
    print(f"- 대칭(symmetric): {'예' if s else '아니오'}")
    print(f"- 추이(transitive): {'예' if t else '아니오'}")

    if r and s and t:
        print("=> 동치 관계입니다.\n")
        return True
    else:
        print("=> 동치 관계가 아닙니다.\n")
        return False


# 폐포 (closure)
def reflexive_closure(matrix):
    n = len(matrix)
    R = [row[:] for row in matrix]
    for i in range(n):
        R[i][i] = 1
    return R

def symmetric_closure(matrix):
    n = len(matrix)
    R = [row[:] for row in matrix]
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                R[j][i] = 1
    return R

def transitive_closure(matrix):
    # Warshall과 같은 삼중루프 방식
    n = len(matrix)
    R = [row[:] for row in matrix]
    for k in range(n):
        for i in range(n):
            if R[i][k] == 1:
                for j in range(n):
                    if R[k][j] == 1:
                        R[i][j] = 1
    return R

def equivalence_closure(matrix):
    # 가장 작은 동치관계: 반사 -> 대칭 -> 추이 순서로 적용
    return transitive_closure(symmetric_closure(reflexive_closure(matrix)))



# 동치류
def print_equivalence_classes(matrix, A, title):
    print(title)
    n = len(matrix)
    for i in range(n):
        cls = []
        for j in range(n):
            if matrix[i][j] == 1:
                cls.append(A[j])
        # 집합 형태로 보이게 출력
        print(f"[{A[i]}] = "+"{ "+", ".join(str(x) for x in cls)+" }")
    print()



# 폐포 전/후 비교
def show_closure_compare(name, beforeM, afterM, A):
    print(f"▶ {name} 폐포 변환 전/후 비교")
    print_matrix(beforeM, "변환 전 행렬:")
    print("[변환 전 관계 R] =", end=" ")
    print_relation(beforeM, A)
    print_property_report(beforeM)

    print_matrix(afterM, "변환 후 행렬:")
    print("[변환 후 관계 R] =", end=" ")
    print_relation(afterM, A)
    ok = print_property_report(afterM)
    if ok:
        print_equivalence_classes(afterM, A, f"{name} 폐포(변환 후)에서의 동치류")
    print("-"*60)



# 추가기능 - 성질 위배 원인 자세히 보여주기
def explain_violations(matrix, A):
    n = len(matrix)
    print("▶ 성질 위배 원인 분석")

    # 1) 반사 위배: (i,i)가 0인 경우들
    bad_ref = []
    for i in range(n):
        if matrix[i][i] == 0:
            bad_ref.append((A[i], A[i]))
    if bad_ref:
        print("  - 반사 위배:", ", ".join(f"({x},{y})" for x, y in bad_ref),
              "자리에 1이 없습니다.")

    # 2) 대칭 위배: (i,j) != (j,i) 인 쌍
    bad_sym = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                bad_sym.append((A[i], A[j], matrix[i][j], matrix[j][i]))
    if bad_sym:
        print("  - 대칭 위배:")
        for x, y, v1, v2 in bad_sym:
            print(f"    ({x}, {y}) = {v1}, ({y}, {x}) = {v2} 로 서로 다릅니다.")

    # 3) 추이 위배: (i,k)=1, (k,j)=1인데 (i,j)=0 인 경우
    max_show = 5  # 너무 많으면 최대 몇 개만 보여주기
    cnt = 0
    for i in range(n):
        for k in range(n):
            if matrix[i][k] == 1:
                for j in range(n):
                    if matrix[k][j] == 1 and matrix[i][j] == 0:
                        if cnt == 0:
                            print("  - 추이 위배 (대표 위배 사례들):")
                        if cnt < max_show:
                            print(f"    ({A[i]}, {A[k]})와 ({A[k]}, {A[j]})는 1인데,"
                                  f" ({A[i]}, {A[j]})가 0입니다.")
                        cnt += 1
    if not (bad_ref or bad_sym or cnt):
        print("  - 별도의 위배 사례를 찾지 못했습니다.")

    print()



# 메인 함수
def main():
    # 관계 행렬 입력 기능
    A = [1, 2, 3, 4, 5]
    n = len(A)
    matrix = []

    print("5x5 관계 행렬을 행 단위로 입력하세요: ")
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)

    print()
    print_matrix(matrix, "입력한 관계 행렬:")
    print("[관계 R에 포함된 순서쌍] =", end=" ")
    print_relation(matrix, A)

    ok = print_property_report(matrix)

    if ok:
        print_equivalence_classes(matrix, A, "동치류:")
    else:
        print("※ 현재 관계는 동치가 아니므로, 각 폐포를 적용하여 전/후를 비교합니다.\n")
        explain_violations(matrix, A)

    

    # 반사/대칭/추이 폐포 각각 전/후 비교
    RC = reflexive_closure(matrix)
    show_closure_compare("반사", matrix, RC, A)

    SC = symmetric_closure(matrix)
    show_closure_compare("대칭", matrix, SC, A)

    TC = transitive_closure(matrix)
    show_closure_compare("추이", matrix, TC, A)


    # 동치 폐포 (반사->대칭->추이) - 사용자 선택
    choice = input("동치 폐포를 한 번에 생성/확인할까요? (y/n) [y]: ").strip().lower()

    # 동치 폐포 (반사->대칭->추이)
    if choice in ("", "y", "yes"):
        EC = equivalence_closure(matrix)
        print("▶ 동치 폐포(반사->대칭->추이) 결과")
        print_matrix(EC, "동치 폐포 행렬:")
        print("[동치 폐포 관계] =", end=" ")
        print_relation(EC, A)
        ok_equiv = print_property_report(EC)
        if ok_equiv:
            print_equivalence_classes(EC, A, "동치 폐포에서의 동치류")

    else:
        print("동치 폐포 계산을 건너뜁니다.\n")

    print("프로그램 종료.")

if __name__ == "__main__":
    main()
