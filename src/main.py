from game import *
import random
import numpy as np

TRAINING = True

def getch():
    import sys, termios
    # return input("")
    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)

    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ICANON
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0

    try:
        termios.tcsetattr(fd, termios.TCSAFLUSH, new)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, orig)

def demo2():
    p1 = Player('AA')
    p2 = Player('BB')
    p3 = Player('CC')
    p4 = Player('DD')
    try:
        while True:
            key = getch()
            match key:
                case 'w':
                    p1.move('up')
                case 's':
                    p1.move('down')
                case 'a':
                    p1.move('left')
                case 'd':
                    p1.move('right')
                case ' ':
                    p1.place_bomb()

    except KeyboardInterrupt:
        print('interrupted!')


if __name__ == '__main__':
    demo2()
