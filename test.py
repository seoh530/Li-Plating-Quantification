T = int(input())

for tc in range(1, T+1):
    N, M, K = map(int, input().split())

    germ_info = []

    for i in range(K):
        X, Y, germ_amount, direction = map(int, input().split())
        germ_info.append((X, Y, germ_amount, direction - 1))

    # 상 = 0, 하 = 1, 좌 = 2, 우 = 3
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]
    opposite = [1, 0, 3, 2]

    total_germ = 0

    for t in range(M):
        moved = {}

        for X, Y, germ_amount, direction in germ_info:
            nr = X + dr[direction]
            nc = Y + dc[direction]

            if nr == 0 or nr == N-1 or nc == 0 or nc == N-1:
                germ_amount //= 2
                direction = opposite[direction]

            if (nr, nc) not in moved:
                # [총 미생물 수, 가장 큰 군집 크기, 방향]
                moved[(nr, nc)] = [germ_amount, germ_amount, direction]
            else:
                moved[(nr, nc)][0] += germ_amount

                if germ_amount > moved[(nr, nc)][1]:
                    moved[(nr, nc)][1] = germ_amount
                    moved[(nr, nc)][2] = direction

        germ_info = []

        for (r, c), info in moved.items():
            germ_info.append((r, c, info[0], info[2]))

    for r, c, germ_amount, direction in germ_info:
        total_germ += germ_amount

    print(f"#{tc} {total_germ}")