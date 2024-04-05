from game import *


def actions():
    ...


def demo():
    p1 = Player('AA', 0)
    p2 = Player('BB', 1)
    p3 = Player('CC', 2)
    p4 = Player('DD', 3)

    p1.move('right')
    p1.place_bomb()
    p1.move('left')
    p1.move('down')
    time.sleep(4*TIME_CONST)
    p1.move('up')
    p1.move('right')
    p1.move('right')
    p1.place_bomb()
    p1.move('left')
    p1.move('down')
    time.sleep(4*TIME_CONST)

    p1.move('up')
    p1.move('right')
    p1.move('right')
    p1.place_bomb()
    p1.move('left')
    p1.move('down')
    time.sleep(4 * TIME_CONST)

    p1.move('up')
    p1.move('right')
    p1.move('right')
    p1.place_bomb()
    p1.move('left')
    p1.move('down')
    time.sleep(4 * TIME_CONST)

    p1.move('up')
    p1.move('right')
    # p1.move('right')
    p1.move('right')
    p1.place_bomb()
    p1.move('left')
    p1.move('left')
    p1.move('down')
    time.sleep(4 * TIME_CONST)
    print_grid()
    [print(p) for p in [p1, p2, p3, p4]]


if __name__ == '__main__':
    demo()
