import pygame
import math
import csv

width = 1000
height = 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Neat Metro Net')
background_color = (252, 252, 252)

c = []
with open('/Users/student/desktop/metnet/a.csv', 'rb') as f:
    reader = csv.reader(f)
    s_c = list(reader)
    for i in s_c:
        c.append([float(i[0]), float(i[1]), (42, 42, 42)])

def draw(x):
    screen.fill(background_color)
    for i in c:
        pygame.draw.circle(screen, i[2], (int(i[0]), int(i[1])), 2)
    pygame.display.update()

line = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    draw(line);
