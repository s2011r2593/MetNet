import numpy as np
import pygame
import math
import csv
import utm

width = 1000.0
height = 700.0
screen = pygame.display.set_mode((int(width), int(height)))
pygame.display.set_caption('Neat Metro Net')

qpi = math.pi/4

background_color = (252, 252, 252)

xm = 38.0 # x margin
ym = 130.0 # y margin\

x = []
y = []
c = []

with open('/Users/student/desktop/metnet/1.csv', 'rb') as f:
    reader = csv.reader(f)
    s_c = list(reader)
    for i in s_c:
        cc = utm.from_latlon(float(i[0]), float(i[1]))
        xx = cc[1]
        yy = cc[0]
        x.append(xx)
        y.append(yy)
        c.append([xx, yy, (42, 42, 42)])

xmin = min(x)
ymin = min(y)
xmax = max(x)
ymax = max(y)
xr = xmax - xmin

scale = (width - (2*xm))/xr

for i in range(len(c)):
    c[i][0] = ((c[i][0] - xmin) * scale) + xm
    c[i][1] = ((c[i][1] - ymin) * scale) + ym

with open ('/Users/student/desktop/metnet/a.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    for i in c:
        wr.writerow(i[:-1])

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
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                line = 1

    draw(line);


# xrange: 36.769548 ~ 37.948124
# yrange: 126.616732 ~ 127.148937

# reducedxrange: 1.178576
# reducedyrange: 0.532205

# x = (longitude+180) * (mapWidth/360)
# latRad = latitude*math.pi/180
# mercN = np.log(math.tan((math.pi/4)+(latRad/2)))
# y = (mapHeight/2)-(mapWidth*mercN/(2*math.pi))
